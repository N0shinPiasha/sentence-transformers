import csv
import os
import platform
import sys
import time

import numpy as np
import torch
import transformers
import sentence_transformers
from datasets import load_dataset
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from transformers import AutoModel, AutoTokenizer

os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def log_environment():
    print("=" * 70)
    print("ENVIRONMENT")
    print(f"  Python                : {sys.version.split()[0]}")
    print(f"  Platform              : {platform.platform()}")
    print(f"  torch                 : {torch.__version__}")
    print(f"  transformers          : {transformers.__version__}")
    print(f"  sentence-transformers : {sentence_transformers.__version__}")
    gpu = f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""
    print(f"  device                : {device}{gpu}")
    print("=" * 70)


# ---------------------------------------------------------------- encoders
def st_encoder(name):
    """Released Sentence-Transformers checkpoint."""
    model = SentenceTransformer(name, device=device)
    return lambda sents: model.encode(sents, batch_size=64, convert_to_numpy=True), model


def raw_bert_encoder(pooling):
    """Plain bert-base-uncased, no fine-tuning (paper's BERT baselines)."""
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()

    @torch.no_grad()
    def encode(sents):
        out = []
        for i in range(0, len(sents), 64):
            b = tok(sents[i:i + 64], padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to(device)
            h = bert(**b).last_hidden_state
            if pooling == "cls":
                e = h[:, 0]
            else:  # mean over real tokens only
                m = b["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1)
            out.append(e.cpu().numpy())
        return np.vstack(out)

    return encode, None


def cosine(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return (a * b).sum(1)


# ---------------------------------------------------------------- runs
# (label, factory, published STSb score from Table 1)
RUNS = [
    ("Avg. GloVe embeddings*", lambda: st_encoder(
        "sentence-transformers/average_word_embeddings_glove.6B.300d"), 58.02),
    ("Avg. BERT embeddings", lambda: raw_bert_encoder("mean"), 46.35),
    ("BERT CLS-vector", lambda: raw_bert_encoder("cls"), 20.16),
    ("SBERT-NLI-base", lambda: st_encoder(
        "sentence-transformers/bert-base-nli-mean-tokens"), 77.03),
    ("SBERT-NLI-large", lambda: st_encoder(
        "sentence-transformers/bert-large-nli-mean-tokens"), 79.23),
]


def main():
    log_environment()
    ds = load_dataset("sentence-transformers/stsb", split="test")
    s1, s2 = list(ds["sentence1"]), list(ds["sentence2"])
    gold = np.array(ds["score"], dtype=float)  # 0-1 scale
    print(f"STS-B test pairs: {len(gold)} (full test split, no subsampling)\n")

    rows = []
    for label, factory, published in RUNS:
        t0 = time.time()
        try:
            encode, st_model = factory()
            pred = cosine(encode(s1), encode(s2))
            rho = spearmanr(pred, gold).correlation * 100
            status = f"{rho:6.2f}"
            rows.append([label, f"{rho:.2f}", f"{published:.2f}",
                         f"{rho - published:+.2f}", f"{time.time() - t0:.0f}s"])

            if label == "SBERT-NLI-base":
                # Cross-check with the library's own evaluator
                ev = EmbeddingSimilarityEvaluator(s1, s2, gold.tolist(), name="sts-test")
                res = ev(st_model)
                lib = {k: v for k, v in res.items() if "spearman_cosine" in k}
                print(f"  Library evaluator cross-check: {lib}")

                # Save every prediction + show worst failures
                err = np.abs(pred - gold)
                with open("results/sbert_predictions.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["sentence1", "sentence2", "gold_0to5", "cosine", "abs_error"])
                    for i in np.argsort(-err):
                        w.writerow([s1[i], s2[i], f"{gold[i]*5:.2f}",
                                    f"{pred[i]:.3f}", f"{err[i]:.3f}"])
                print("  Largest disagreements (gold 0-5 | model cosine):")
                for i in np.argsort(-err)[:5]:
                    print(f"    {gold[i]*5:.2f} | {pred[i]:.2f} | {s1[i]}  ||  {s2[i]}")
        except Exception as e:  # record failures honestly instead of crashing
            status = f"FAILED: {type(e).__name__}: {e}"
            rows.append([label, "FAILED", f"{published:.2f}", "", ""])
        print(f"{label:<26} {status}   (paper: {published:.2f})\n")

    print("=" * 70)
    print(f"{'Model':<26}{'Ours':>8}{'Paper':>8}{'Diff':>8}{'Time':>8}")
    for r in rows:
        print(f"{r[0]:<26}{r[1]:>8}{r[2]:>8}{r[3]:>8}{r[4]:>8}")
    print("* Paper used GloVe 840B.300d; this uses 6B.300d, so expect a gap.")

    with open("results/table1_replication.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "ours", "paper", "diff", "time"])
        w.writerows(rows)


if __name__ == "__main__":
    main()