
import io
import json
import os
import random
import re
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

N_PAIRS = 2000          # pairs per dataset (BIOSSES only has 100). Lower = faster.
SEED = 42
random.seed(SEED)
os.makedirs("results", exist_ok=True)

MODEL_NAME = "sentence-transformers/bert-base-nli-mean-tokens"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():      # Apple Silicon (your M3)
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Loading {MODEL_NAME} on {DEVICE} ...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)


def cosine(s1, s2):
    a = model.encode(s1, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    b = model.encode(s2, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    return (a * b).sum(1)


def pairs_from_groups(groups, n):
    """groups: {label: [texts]}. Returns half same-label pairs (1), half different (0)."""
    labels = [g for g, t in groups.items() if len(t) >= 2]
    s1, s2, y = [], [], []
    for i in range(n):
        if i % 2 == 0:                                   # same group -> 1
            g = random.choice(labels)
            a, b = random.sample(groups[g], 2)
            y.append(1)
        else:                                            # different groups -> 0
            g1, g2 = random.sample(labels, 2)
            a, b = random.choice(groups[g1]), random.choice(groups[g2])
            y.append(0)
        s1.append(a)
        s2.append(b)
    return s1, s2, np.array(y)


# ------------------------------------------------------------ loaders
def load_biosses():
    ds = load_dataset("mteb/biosses-sts", split="test")
    return list(ds["sentence1"]), list(ds["sentence2"]), np.array(ds["score"]), "graded"


def load_qqp():
    ds = load_dataset("nyu-mll/glue", "qqp", split="validation")
    ds = ds.shuffle(seed=SEED).select(range(N_PAIRS))
    return list(ds["question1"]), list(ds["question2"]), np.array(ds["label"]), "binary"


def load_financial_phrasebank():
    # The HF loading script is no longer supported, so read the original zip directly.
    path = hf_hub_download("takala/financial_phrasebank",
                           "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith("Sentences_AllAgree.txt")][0]
        text = z.read(name).decode("latin-1")
    groups = {}
    for line in text.splitlines():
        if "@" in line:
            sent, label = line.rsplit("@", 1)
            groups.setdefault(label.strip(), []).append(sent.strip())
    s1, s2, y = pairs_from_groups(groups, N_PAIRS)
    return s1, s2, y, "binary"


def load_cuad():
    # Official data from the CUAD GitHub repo (SQuAD-style JSON).
    r = requests.get("https://github.com/TheAtticusProject/cuad/raw/main/data.zip", timeout=180)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()
    name = next((n for n in names if n.endswith("test.json")),
                next(n for n in names if n.endswith(".json")))
    data = json.loads(z.read(name))
    skip = {"Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date"}
    groups = {}
    for doc in data["data"]:
        for para in doc["paragraphs"]:
            for qa in para["qas"]:
                m = re.search(r'related to "(.+?)"', qa["question"])
                if not m or m.group(1) in skip:
                    continue
                for ans in qa["answers"]:
                    t = " ".join(ans["text"].split())
                    if 8 <= len(t.split()) <= 120:
                        groups.setdefault(m.group(1), []).append(t)
    groups = {k: v for k, v in groups.items() if len(v) >= 10}
    s1, s2, y = pairs_from_groups(groups, N_PAIRS)
    return s1, s2, y, "binary"


def load_esci():
    gain = {"Exact": 3, "Substitute": 2, "Complement": 1, "Irrelevant": 0}
    ds = load_dataset("tasksource/esci", split="test", streaming=True)
    s1, s2, y = [], [], []
    for row in ds:
        if row["product_locale"] != "us" or row["esci_label"] not in gain:
            continue
        s1.append(row["query"])
        s2.append(row["product_title"])
        y.append(gain[row["esci_label"]])
        if len(y) >= N_PAIRS:
            break
    return s1, s2, np.array(y), "graded"


DATASETS = [
    ("BIOSSES", "Biomedical", load_biosses),
    ("Quora Question Pairs", "Everyday questions", load_qqp),
    ("Financial PhraseBank", "Finance", load_financial_phrasebank),
    ("CUAD", "Legal contracts", load_cuad),
    ("Amazon ESCI", "E-commerce search", load_esci),
]

# ------------------------------------------------------------ run
rows = []
for name, domain, loader in DATASETS:
    print(f"\n=== {name} ({domain}) ===")
    t0 = time.time()
    try:
        s1, s2, y, kind = loader()
        pred = cosine(s1, s2)
        if kind == "graded":
            metric, score = "Spearman", spearmanr(pred, y).correlation * 100
        else:
            metric, score = "ROC-AUC", roc_auc_score(y, pred) * 100
        safe = name.lower().replace(" ", "_")
        pd.DataFrame({"text_a": s1, "text_b": s2, "gold": y, "cosine": pred}).to_csv(
            f"results/pred_{safe}.csv", index=False)
        print(f"  pairs={len(y)}  {metric}={score:.2f}  ({time.time() - t0:.0f}s)")
        rows.append([name, domain, len(y), metric, f"{score:.2f}", "ok"])
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        rows.append([name, domain, 0, "", "", f"FAILED: {type(e).__name__}"])

out = pd.DataFrame(rows, columns=["dataset", "domain", "pairs", "metric", "score", "status"])
out.to_csv("results/existing_datasets_results.csv", index=False)
print("\n" + "=" * 70)
print("Reference: same model on STS-B (general English) = 76.99 Spearman")
print(out.to_string(index=False))