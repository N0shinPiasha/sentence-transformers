

import io
import json
import os
import re
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import requests
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score

SEED = 42
MAX_CLAUSES = 6000        # memory cap for CUAD all-pairs (6000 -> 18M pairs)
os.makedirs("results", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")
BATCH = 256 if DEVICE == "cuda" else 32
MODEL_NAME = "sentence-transformers/bert-base-nli-mean-tokens"
print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
model = SentenceTransformer(MODEL_NAME, device=DEVICE)


def encode_unique(texts):
    """Encode each distinct text once. Returns (embeddings, index_of_each_text)."""
    uniq = list(dict.fromkeys(texts))
    pos = {t: i for i, t in enumerate(uniq)}
    print(f"  encoding {len(uniq):,} unique texts ...")
    emb = model.encode(uniq, batch_size=BATCH, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    return emb, np.array([pos[t] for t in texts])


def pair_cosine(s1, s2):
    emb, idx = encode_unique(list(s1) + list(s2))
    ia, ib = idx[:len(s1)], idx[len(s1):]
    out = np.empty(len(s1), dtype=np.float32)
    for k in range(0, len(s1), 50_000):                 # chunked to save memory
        out[k:k + 50_000] = (emb[ia[k:k + 50_000]] * emb[ib[k:k + 50_000]]).sum(1)
    return out


def all_pairs_auc(texts, labels):
    """AUC over EVERY pair: same label = similar (1), different = 0."""
    emb, _ = encode_unique(texts)          # texts are already unique here
    sim = emb @ emb.T
    iu = np.triu_indices(len(texts), k=1)
    lab = np.array(labels)
    y = (lab[iu[0]] == lab[iu[1]]).astype(np.int8)
    return roc_auc_score(y, sim[iu]), len(y)


# ------------------------------------------------------------ datasets
def run_biosses():
    ds = load_dataset("mteb/biosses-sts", split="test")
    pred = pair_cosine(ds["sentence1"], ds["sentence2"])
    return "Spearman", spearmanr(pred, ds["score"]).correlation * 100, len(pred)


def run_qqp():
    ds = load_dataset("nyu-mll/glue", "qqp", split="validation")
    pred = pair_cosine(ds["question1"], ds["question2"])
    return "ROC-AUC", roc_auc_score(ds["label"], pred) * 100, len(pred)


def run_fpb():
    path = hf_hub_download("takala/financial_phrasebank",
                           "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith("Sentences_AllAgree.txt")][0]
        text = z.read(name).decode("latin-1")
    sents = {}
    for line in text.splitlines():
        if "@" in line:
            s, lab = line.rsplit("@", 1)
            sents.setdefault(s.strip(), lab.strip())        # de-duplicate
    auc, n = all_pairs_auc(list(sents), list(sents.values()))
    print(f"  {len(sents):,} sentences")
    return "ROC-AUC", auc * 100, n


def run_cuad():
    r = requests.get("https://github.com/TheAtticusProject/cuad/raw/main/data.zip", timeout=300)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()
    name = next((n for n in names if n.endswith("test.json")),
                next(n for n in names if n.endswith(".json")))
    data = json.loads(z.read(name))
    skip = {"Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date"}
    clauses = {}
    for doc in data["data"]:
        for para in doc["paragraphs"]:
            for qa in para["qas"]:
                m = re.search(r'related to "(.+?)"', qa["question"])
                if not m or m.group(1) in skip:
                    continue
                for ans in qa["answers"]:
                    t = " ".join(ans["text"].split())
                    if 8 <= len(t.split()) <= 120:
                        clauses.setdefault(t, m.group(1))
    items = list(clauses.items())
    if len(items) > MAX_CLAUSES:                           # memory safety
        rng = np.random.default_rng(SEED)
        items = [items[i] for i in rng.choice(len(items), MAX_CLAUSES, replace=False)]
    print(f"  {len(items):,} clauses, {len(set(c for _, c in items))} clause types")
    auc, n = all_pairs_auc([t for t, _ in items], [c for _, c in items])
    return "ROC-AUC", auc * 100, n


def run_esci():
    gain = {"Exact": 3, "Substitute": 2, "Complement": 1, "Irrelevant": 0}
    ds = load_dataset("tasksource/esci", split="test", streaming=True)
    q, t, y = [], [], []
    for row in ds:
        if row["product_locale"] == "us" and row["esci_label"] in gain:
            q.append(row["query"])
            t.append(row["product_title"])
            y.append(gain[row["esci_label"]])
    print(f"  {len(y):,} US pairs collected")
    pred = pair_cosine(q, t)
    pd.DataFrame({"query": q, "title": t, "gold": y, "cosine": pred}).to_csv(
        "results/full_pred_esci.csv.gz", index=False)
    return "Spearman", spearmanr(pred, y).correlation * 100, len(y)


DATASETS = {
    "biosses": ("BIOSSES", "Biomedical", run_biosses),
    "qqp": ("Quora Question Pairs", "Everyday questions", run_qqp),
    "fpb": ("Financial PhraseBank", "Finance", run_fpb),
    "cuad": ("CUAD", "Legal contracts", run_cuad),
    "esci": ("Amazon ESCI", "E-commerce search", run_esci),
}

chosen = [a.lower() for a in sys.argv[1:]] or ["all"]
keys = list(DATASETS) if "all" in chosen else [k for k in chosen if k in DATASETS]

rows = []
for key in keys:
    name, domain, fn = DATASETS[key]
    print(f"\n=== {name} ({domain}) ===")
    t0 = time.time()
    try:
        metric, score, n = fn()
        mins = (time.time() - t0) / 60
        print(f"  pairs={n:,}  {metric}={score:.2f}  ({mins:.1f} min)")
        rows.append([name, domain, n, metric, round(score, 2), f"{mins:.1f} min", "ok"])
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        rows.append([name, domain, 0, "", "", "", f"FAILED: {type(e).__name__}"])

out = pd.DataFrame(rows, columns=["dataset", "domain", "pairs", "metric", "score", "time", "status"])
fname = "results/full_results.csv" if "all" in chosen else f"results/full_results_{'_'.join(keys)}.csv"
out.to_csv(fname, index=False)
print("\n" + "=" * 75)
print(f"FULL evaluation on {DEVICE}  |  reference: STS-B = 76.99 Spearman")
print(out.to_string(index=False))
print(f"Saved {fname}")
