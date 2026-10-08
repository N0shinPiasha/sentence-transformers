import io

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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

# ---- 3. human vs LLM agreement
try:
    mine = pd.read_csv("my_ratings.csv")
    mine["my_score"] = pd.to_numeric(mine["my_score"], errors="coerce")
    both = mine.dropna(subset=["my_score"]).merge(llm, on="pair_id")
    if len(both) >= 5:
        rho = spearmanr(both["my_score"], both["llm_score"]).correlation
        mad = (both["my_score"] - both["llm_score"]).abs().mean()
        exact = (both["my_score"] == both["llm_score"]).mean() * 100
        say(f"\n3. You vs LLM on {len(both)} blind pairs: Spearman = {rho:.2f}, "
            f"mean difference = {mad:.2f} points, exact match = {exact:.0f}%")
    else:
        say("\n3. Fill in my_score in my_ratings.csv to measure human-LLM agreement.")
except FileNotFoundError:
    say("\n3. my_ratings.csv not found.")

# ---- 4. SBERT preview
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")
    d = data.dropna(subset=["llm_score"])
    a = model.encode(d["text_a"].tolist(), normalize_embeddings=True)
    b = model.encode(d["text_b"].tolist(), normalize_embeddings=True)
    data.loc[d.index, "sbert_cosine"] = (a * b).sum(1)
    rho = spearmanr(data.loc[d.index, "sbert_cosine"], d["llm_score"]).correlation * 100
    say(f"\n4. PREVIEW - SBERT vs LLM labels on {len(d)} ULO pairs: Spearman = {rho:.2f}")
    say("   (for comparison, SBERT on STS-B = 76.99)")
except Exception as e:
    say(f"\n4. SBERT preview skipped: {e}")

data.to_csv("ulo_dataset_labelled.csv", index=False)
with open("label_report.txt", "w") as f:
    f.write("\n".join(report) + "\n")
say("\nSaved: ulo_dataset_labelled.csv, label_report.txt")