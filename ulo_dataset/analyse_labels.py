"""
Merge the pasted LLM labels (llm_labels.csv) with pairs.csv, run sanity checks and add SBERT cosines.
Run from inside ulo_dataset/:   python analyse_labels.py               (reuses saved SBERT cosines)
                                python analyse_labels.py --recompute   (re-encodes with SBERT)
SBERT cosines do not depend on the labels, so the cosines saved in ulo_dataset_labelled.csv are
reused when every pair has one; --recompute (or a missing file) encodes the pairs again.
Outputs: ulo_dataset_labelled.csv, label_report.txt
"""

import argparse
import io

import pandas as pd
from scipy.stats import spearmanr

parser = argparse.ArgumentParser(description="Merge LLM labels with pairs.csv and add SBERT cosines.")
parser.add_argument("--recompute", action="store_true", help="re-encode the pairs with SBERT")
args = parser.parse_args()

report = []


def say(line=""):
    print(line)
    report.append(line)


# ---- load LLM labels (tolerates repeated headers / blank lines from pasting)
raw = open("llm_labels.csv").read().splitlines()
clean_lines = ["pair_id,llm_score,llm_reason"] + [
    l for l in raw if l.strip() and not l.lower().startswith("pair_id")]
llm = pd.read_csv(io.StringIO("\n".join(clean_lines)), on_bad_lines="skip")
llm["pair_id"] = pd.to_numeric(llm["pair_id"], errors="coerce")
llm["llm_score"] = pd.to_numeric(llm["llm_score"], errors="coerce")
llm = llm.dropna(subset=["pair_id", "llm_score"])
llm = llm[llm["llm_score"].between(0, 5)].drop_duplicates("pair_id", keep="last")
llm["pair_id"] = llm["pair_id"].astype(int)

pairs = pd.read_csv("pairs.csv").drop(columns=["llm_score", "llm_reason"], errors="ignore")
data = pairs.merge(llm, on="pair_id", how="left")

# ---- 1. coverage
labelled = data["llm_score"].notna().sum()
say(f"1. LLM labels: {labelled} / {len(data)} pairs labelled")
missing = data.loc[data["llm_score"].isna(), "pair_id"].tolist()
if missing:
    say(f"   Missing pair_ids (re-run these): {missing[:30]}{' ...' if len(missing) > 30 else ''}")
say("   Score distribution: " + ", ".join(
    f"{int(k)}: {v}" for k, v in data["llm_score"].value_counts().sort_index().items()))

# ---- 2. sanity check by pair type
say("\n2. Mean LLM score by pair type (expect same_unit > same_discipline > cross):")
for t, m in data.groupby("pair_type")["llm_score"].mean().sort_values(ascending=False).items():
    say(f"   {t:<17} {m:.2f}")

# ---- 3. human vs LLM agreement (full statistics are in evaluate_ulo.py)
for fname, name in [("my_ratings.csv", "Round 1"), ("my_ratings_blind.csv", "Round 2 (blind)")]:
    try:
        mine = pd.read_csv(fname)
    except FileNotFoundError:
        say(f"\n3. {fname} not found.")
        continue
    mine["my_score"] = pd.to_numeric(mine["my_score"], errors="coerce")
    # match on the two outcome texts, not pair_id: pair ids change whenever pairs.csv is regenerated
    both = mine.dropna(subset=["my_score"]).drop(columns="pair_id").merge(
        data[["text_a", "text_b", "llm_score"]].dropna(), on=["text_a", "text_b"])
    if len(both) >= 5:
        rho = spearmanr(both["my_score"], both["llm_score"]).correlation
        mad = (both["my_score"] - both["llm_score"]).abs().mean()
        exact = (both["my_score"] == both["llm_score"]).mean() * 100
        say(f"\n3. You vs LLM, {name}, {len(both)} pairs: Spearman = {rho:.2f}, "
            f"mean difference = {mad:.2f} points, exact match = {exact:.0f}%")
    else:
        say(f"\n3. Fill in my_score in {fname} to measure human-LLM agreement.")

# ---- 4. SBERT cosines (reused from the previous run unless --recompute)
d = data.dropna(subset=["llm_score"])
cos, source = None, ""
if not args.recompute:
    try:
        old = pd.read_csv("ulo_dataset_labelled.csv", usecols=["text_a", "text_b", "sbert_cosine"])
        old = old.dropna().drop_duplicates(["text_a", "text_b"])
        cos = d[["text_a", "text_b"]].merge(old, on=["text_a", "text_b"], how="left")["sbert_cosine"]
        cos, source = (None, "") if cos.isna().any() else (cos.to_numpy(), "saved cosines")
    except (FileNotFoundError, ValueError):                 # no earlier file, or it has no cosines
        cos = None
if cos is None:
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")
        a = model.encode(d["text_a"].tolist(), normalize_embeddings=True)
        b = model.encode(d["text_b"].tolist(), normalize_embeddings=True)
        cos, source = (a * b).sum(1), "freshly encoded with SBERT-NLI-base"
    except Exception as e:
        say(f"\n4. SBERT cosines skipped: {e}")
if cos is not None:
    data.loc[d.index, "sbert_cosine"] = cos
    rho = spearmanr(data.loc[d.index, "sbert_cosine"], d["llm_score"]).correlation * 100
    say(f"\n4. PREVIEW - SBERT vs LLM labels on {len(d)} ULO pairs: Spearman = {rho:.2f} ({source})")
    say("   (for comparison, SBERT on STS-B = 76.99)")

data.to_csv("ulo_dataset_labelled.csv", index=False)
with open("label_report.txt", "w") as f:
    f.write("\n".join(report) + "\n")
say("\nSaved: ulo_dataset_labelled.csv, label_report.txt")