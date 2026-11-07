"""
Robustness checks for two of the five existing datasets.

    python robustness_checks.py selftest   # checks the metric code on toy data (no model, no internet)
    python robustness_checks.py cuad       # CPU is fine: about 2-5 minutes
    python robustness_checks.py esci       # use a GPU (Google Colab T4): about 25 minutes

cuad  A CUAD clause can carry several clause types. run_full_datasets.py keeps only the
      first type of each clause, so two clauses that share a second type count as
      "different". This re-scores the same clauses with the multi-label rule
      "similar = the two clauses share at least one clause type".
esci  run_full_datasets.py pools all US test query-product pairs into one Spearman
      correlation. ESCI is normally evaluated per query (how well the products returned
      for one query are ranked). This adds that view: mean within-query Spearman and
      nDCG@10 with the official ESCI gains (Exact 1.0, Substitute 0.1, Complement 0.01,
      Irrelevant 0), compared with a random ranking of the same products.

Outputs: results/robustness_cuad.txt, results/robustness_esci.txt
         (and results/full_pred_esci.csv.gz with every ESCI prediction)
"""

import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODEL_NAME = "sentence-transformers/bert-base-nli-mean-tokens"
ESCI_GAIN = {3: 1.0, 2: 0.1, 1: 0.01, 0: 0.0}       # KDD Cup 2022 Task 1 gains for E, S, C, I
SEED = 42


# ---------------------------------------------------------------- metric code (model-free)
def ndcg_at_k(gains_in_ranked_order, k=10):
    """nDCG@k for one query, given the gains of its products in the model's ranked order."""
    g = np.asarray(gains_in_ranked_order, dtype=float)
    disc = 1.0 / np.log2(np.arange(2, min(k, len(g)) + 2))
    ideal = (np.sort(g)[::-1][:k] * disc).sum()
    return (g[:k] * disc).sum() / ideal if ideal > 0 else np.nan


def per_query_metrics(query, gold, score, n_random=20, seed=SEED):
    """Mean within-query Spearman and nDCG@10 (model and random ranking) over all queries."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"q": query, "y": gold, "s": score})
    rhos, ndcg_model, ndcg_random = [], [], []
    for _, g in df.groupby("q", sort=False):
        y, s = g["y"].to_numpy(), g["s"].to_numpy()
        if len(np.unique(y)) > 1:
            rhos.append(spearmanr(s, y).correlation)
        gains = np.vectorize(ESCI_GAIN.get)(y)
        if gains.max() > 0:
            ndcg_model.append(ndcg_at_k(gains[np.argsort(-s, kind="stable")]))
            ndcg_random.append(np.mean([ndcg_at_k(rng.permutation(gains)) for _ in range(n_random)]))
    return {"queries": df["q"].nunique(), "queries_with_mixed_labels": len(rhos),
            "mean_within_query_spearman_x100": float(np.nanmean(rhos) * 100),
            "ndcg10_model": float(np.nanmean(ndcg_model)), "ndcg10_random": float(np.nanmean(ndcg_random))}


def first_and_multi_targets(label_lists):
    """For every pair i<j: y_first (same first label) and y_multi (share any label)."""
    first = np.array([labels[0] for labels in label_lists])
    vocab = sorted({lab for labels in label_lists for lab in labels})
    col = {lab: k for k, lab in enumerate(vocab)}
    onehot = np.zeros((len(label_lists), len(vocab)), dtype=np.float32)
    for i, labels in enumerate(label_lists):
        for lab in labels:
            onehot[i, col[lab]] = 1
    iu = np.triu_indices(len(label_lists), k=1)
    y_first = (first[iu[0]] == first[iu[1]]).astype(np.int8)
    y_multi = ((onehot @ onehot.T)[iu] > 0).astype(np.int8)
    return iu, y_first, y_multi


def selftest():
    assert abs(ndcg_at_k([1.0, 0.1, 0.0]) - 1.0) < 1e-9                      # already ideal
    assert ndcg_at_k([0.0, 0.1, 1.0]) < ndcg_at_k([1.0, 0.0, 0.1]) < 1.0      # worse orders score lower
    m = per_query_metrics(["a"] * 4 + ["b"] * 3, [3, 2, 1, 0, 3, 1, 0], [0.9, 0.8, 0.7, 0.1, 0.9, 0.2, 0.1])
    assert m["queries"] == 2 and abs(m["mean_within_query_spearman_x100"] - 100) < 1e-6
    assert abs(m["ndcg10_model"] - 1.0) < 1e-9 and m["ndcg10_random"] < 1.0
    iu, y1, ym = first_and_multi_targets([["A"], ["A", "B"], ["B"], ["C"]])
    pairs = dict(zip(zip(iu[0], iu[1]), zip(y1, ym)))
    assert pairs[(0, 1)] == (1, 1)        # same first label
    assert pairs[(1, 2)] == (0, 1)        # different first label, but they share "B"
    assert pairs[(2, 3)] == (0, 0)
    print("selftest passed: nDCG, per-query metrics and multi-label targets behave as expected")


# ---------------------------------------------------------------- model + data
def load_model():
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    return SentenceTransformer(MODEL_NAME, device=device), (256 if device == "cuda" else 32)


def encode_unique(model, batch, texts):
    uniq = list(dict.fromkeys(texts))
    pos = {t: i for i, t in enumerate(uniq)}
    print(f"  encoding {len(uniq):,} unique texts ...")
    emb = model.encode(uniq, batch_size=batch, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    return emb, np.array([pos[t] for t in texts])


def run_cuad():
    import requests
    r = requests.get("https://github.com/TheAtticusProject/cuad/raw/main/data.zip", timeout=300)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()
    name = next((n for n in names if n.endswith("test.json")), next(n for n in names if n.endswith(".json")))
    data = json.loads(z.read(name))
    skip = {"Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date"}
    clauses = {}                                   # same filters as run_full_datasets.py
    for doc in data["data"]:
        for para in doc["paragraphs"]:
            for qa in para["qas"]:
                m = re.search(r'related to "(.+?)"', qa["question"])
                if not m or m.group(1) in skip:
                    continue
                for ans in qa["answers"]:
                    t = " ".join(ans["text"].split())
                    if 8 <= len(t.split()) <= 120:
                        labels = clauses.setdefault(t, [])
                        if m.group(1) not in labels:
                            labels.append(m.group(1))
    texts, label_lists = list(clauses), list(clauses.values())
    model, batch = load_model()
    emb, _ = encode_unique(model, batch, texts)
    iu, y_first, y_multi = first_and_multi_targets(label_lists)
    sim = (emb @ emb.T)[iu]
    out = [f"CUAD test clauses: {len(texts):,}  ({sum(len(x) > 1 for x in label_lists)} carry 2+ clause types)",
           f"pairs: {len(y_first):,}  (pairs whose label changes under the multi-label rule: {int((y_first != y_multi).sum()):,})",
           f"ROC-AUC, first type only (as in run_full_datasets.py) = {roc_auc_score(y_first, sim) * 100:.2f}",
           f"ROC-AUC, similar = share any clause type             = {roc_auc_score(y_multi, sim) * 100:.2f}"]
    return out, "robustness_cuad.txt"


def run_esci():
    from datasets import load_dataset
    gain = {"Exact": 3, "Substitute": 2, "Complement": 1, "Irrelevant": 0}
    ds = load_dataset("tasksource/esci", split="test", streaming=True)
    q, t, y = [], [], []
    for row in ds:                                 # same filters as run_full_datasets.py
        if row["product_locale"] == "us" and row["esci_label"] in gain:
            q.append(row["query"])
            t.append(row["product_title"])
            y.append(gain[row["esci_label"]])
    print(f"  {len(y):,} US pairs collected")
    model, batch = load_model()
    emb, idx = encode_unique(model, batch, q + t)
    ia, ib = idx[:len(q)], idx[len(q):]
    pred = np.concatenate([(emb[ia[k:k + 50_000]] * emb[ib[k:k + 50_000]]).sum(1)
                           for k in range(0, len(q), 50_000)])
    y = np.array(y)
    pd.DataFrame({"query": q, "title": t, "gold": y, "cosine": pred}).to_csv(
        RESULTS / "full_pred_esci.csv.gz", index=False)
    m = per_query_metrics(q, y, pred)
    means = pd.Series(pred).groupby(y).mean()
    out = [f"ESCI US test pairs: {len(y):,}  queries: {m['queries']:,}",
           f"pooled Spearman x100 (as in run_full_datasets.py) = {spearmanr(pred, y).correlation * 100:.2f}",
           f"mean within-query Spearman x100                 = {m['mean_within_query_spearman_x100']:.2f}"
           f"  ({m['queries_with_mixed_labels']:,} queries with 2+ labels)",
           f"nDCG@10, SBERT ranking                          = {m['ndcg10_model']:.4f}",
           f"nDCG@10, random ranking of the same products    = {m['ndcg10_random']:.4f}",
           "mean cosine by label: " + ", ".join(f"{k}={means[v]:.3f}" for k, v in gain.items() if v in means)]
    return out, "robustness_esci.txt"


def main():
    task = sys.argv[1].lower() if len(sys.argv) > 1 else "selftest"
    if task == "selftest":
        selftest()
        return
    if task not in {"cuad", "esci"}:
        sys.exit(__doc__)
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    out, fname = run_cuad() if task == "cuad" else run_esci()
    out.append(f"time: {(time.time() - t0) / 60:.1f} min")
    (RESULTS / fname).write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"Saved results/{fname}")


if __name__ == "__main__":
    main()
