"""
Extra statistics computed from the prediction files already saved in results/.
No model and no internet are needed, so this runs anywhere in under a minute:

    python extra_analysis.py

What it does
  1. STS-B (results/sbert_predictions.csv, all 1,379 test pairs)
       - bootstrap 95% CI of SBERT's Spearman correlation
       - how much Spearman varies on random 150-pair subsets (the size of the new ULO set)
       - a TF-IDF (word-overlap) baseline, for comparison with the ULO data
       - the distribution of the gold scores (compared with the ULO labels in the report)
  2. BIOSSES (results/pred_biosses.csv, all 100 pairs): bootstrap CI and TF-IDF baseline
  3. Amazon ESCI (results/pred_amazon_esci.csv, the first 2,000 US test rows from the
     earlier sample run): shows why that sample was not representative. Per-query
     ranking (nDCG@10) on the FULL data is in robustness_checks.py (needs a GPU).
  4. SBERT versus a TF-IDF word-overlap baseline on exactly the same pairs, for the
     2,000-pair sample files of QQP, Financial PhraseBank, CUAD and ESCI.

Output: results/extra_analysis.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
SEED = 42
lines = []


def say(text=""):
    print(text)
    lines.append(text)


def rho(x, y):
    return spearmanr(x, y).correlation * 100


def boot_ci(x, y, n_boot=10_000, seed=SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    vals = [rho(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))]
    return np.percentile(vals, [2.5, 97.5])


def tfidf_cosine(a, b):
    vec = TfidfVectorizer(lowercase=True, stop_words="english").fit(list(dict.fromkeys(list(a) + list(b))))
    A, B = vec.transform(a), vec.transform(b)
    return np.asarray(A.multiply(B).sum(1)).ravel()


# ---------------------------------------------------------------- 1. STS-B
sts = pd.read_csv(RES / "sbert_predictions.csv")
gold, cos = sts["gold_0to5"].values, sts["cosine"].values
lo, hi = boot_ci(cos, gold)
say("=" * 72)
say(f"1. STS-B test set ({len(sts)} pairs)")
say(f"  SBERT-NLI-base Spearman = {rho(cos, gold):.2f}  [95% CI {lo:.1f}, {hi:.1f}]  "
    "(cosines saved to 3 decimals; full-precision value 76.99)")
rng = np.random.default_rng(SEED)
for size in (150, 500):                      # the sizes of the ULO dataset (v1 and v1+v2)
    sub = [rho(cos[i], gold[i]) for i in (rng.choice(len(sts), size, replace=False) for _ in range(10_000))]
    say(f"  random {size}-pair subsets: mean {np.mean(sub):.2f}, middle 95% [{np.percentile(sub, 2.5):.1f}, "
        f"{np.percentile(sub, 97.5):.1f}]")
tf = tfidf_cosine(sts["sentence1"], sts["sentence2"])
lo, hi = boot_ci(tf, gold)
say(f"  TF-IDF cosine baseline  = {rho(tf, gold):.2f}  [95% CI {lo:.1f}, {hi:.1f}]")
bins = np.clip(np.rint(gold), 0, 5).astype(int)
dist = pd.Series(bins).value_counts().reindex(range(6), fill_value=0)
say("  gold scores rounded to whole points: " + ", ".join(f"{k}: {v}" for k, v in dist.items()))
say(f"  share of pairs with gold >= 3.5: {np.mean(gold >= 3.5) * 100:.1f}%")

# ---------------------------------------------------------------- 2. BIOSSES
bio = pd.read_csv(RES / "pred_biosses.csv")
lo, hi = boot_ci(bio["cosine"], bio["gold"])
say("\n" + "=" * 72)
say(f"2. BIOSSES ({len(bio)} pairs)")
say(f"  SBERT-NLI-base Spearman = {rho(bio['cosine'], bio['gold']):.2f}  [95% CI {lo:.1f}, {hi:.1f}]")
tf = tfidf_cosine(bio["text_a"], bio["text_b"])
lo, hi = boot_ci(tf, bio["gold"])
say(f"  TF-IDF cosine baseline  = {rho(tf, bio['gold']):.2f}  [95% CI {lo:.1f}, {hi:.1f}]")

# ---------------------------------------------------------------- 3. ESCI sample
esci = pd.read_csv(RES / "pred_amazon_esci.csv")
say("\n" + "=" * 72)
say(f"3. Amazon ESCI - first {len(esci):,} US test rows (earlier sample run)")
per_query = esci.groupby("text_a").size()
say(f"  distinct queries: {len(per_query)}  (median {per_query.median():.0f} products per query; "
    f"the stream is ordered by product, so this is not a random sample)")
say(f"  distinct products: {esci['text_b'].nunique()}")
say("  label distribution (3=Exact .. 0=Irrelevant): "
    + ", ".join(f"{k}: {v}" for k, v in esci["gold"].value_counts().sort_index(ascending=False).items()))
say(f"  pooled Spearman over these rows = {rho(esci['cosine'], esci['gold']):.2f}  (full test set: 15.05)")

# ---------------------------------------------------------------- 4. SBERT vs TF-IDF, same pairs
say("\n" + "=" * 72)
say("4. SBERT vs TF-IDF on identical pairs (2,000-pair sample files)")
for fname, label, metric in [("pred_quora_question_pairs.csv", "QQP (random sample)", "auc"),
                             ("pred_financial_phrasebank.csv", "Financial PhraseBank (balanced)", "auc"),
                             ("pred_cuad.csv", "CUAD (balanced)", "auc"),
                             ("pred_amazon_esci.csv", "Amazon ESCI (first rows)", "rho")]:
    df = pd.read_csv(RES / fname)
    tf = tfidf_cosine(df["text_a"].astype(str), df["text_b"].astype(str))
    if metric == "auc":
        s, t, name = roc_auc_score(df["gold"], df["cosine"]) * 100, roc_auc_score(df["gold"], tf) * 100, "ROC-AUC"
    else:
        s, t, name = rho(df["cosine"], df["gold"]), rho(tf, df["gold"]), "Spearman"
    say(f"  {label:<34} {name:<8} SBERT {s:6.2f}   TF-IDF {t:6.2f}   gap {s - t:+6.2f}")

(RES / "extra_analysis.txt").write_text("\n".join(lines) + "\n")
print(f"\nSaved {RES / 'extra_analysis.txt'}")
