
import random
import re

import pandas as pd

PAIRS_PER_TYPE = 50      # v1: 3 types x 50 = 150 random pairs (v2 adds 350 more below)
MY_SAMPLE_SIZE = 20      # pairs you will rate yourself
MIN_WORDS = 4
SEED = 42
random.seed(SEED)

# ---------------------------------------------------------------- 1. CLEAN
df = pd.read_csv("ulos.csv")
log = [f"Raw outcomes collected            : {len(df)}"]


def clean(text):
    text = str(text)
    text = re.sub(r"^\s*ULO\s*\d+\s*[:.\-]?\s*", "", text)  # stray "ULO1:" prefixes
    text = " ".join(text.split())                          # extra spaces / newlines
    return text.strip()


df["ulo_text"] = df["ulo_text"].map(clean)

before = len(df)
df = df[df["ulo_text"].str.split().str.len() >= MIN_WORDS]
log.append(f"Removed too short (<{MIN_WORDS} words)      : {before - len(df)}")

before = len(df)
df["_key"] = df["ulo_text"].str.lower().str.rstrip(".")
df = df.drop_duplicates("_key").drop(columns="_key")
log.append(f"Removed duplicates (same wording) : {before - len(df)}")
log.append(f"Clean outcomes kept               : {len(df)}")

df.to_csv("ulos_clean.csv", index=False)
rows = df.to_dict("records")

# ---------------------------------------------------------------- 2. PAIRS
def sample_pairs(condition, n):
    candidates = [(a, b) for i, a in enumerate(rows) for b in rows[i + 1:]
                  if condition(a, b)]
    random.shuffle(candidates)
    return candidates[:n]


groups = {
    "same_unit": lambda a, b: a["unit_code"] == b["unit_code"],
    "same_discipline": lambda a, b: a["unit_code"] != b["unit_code"]
                                    and a["discipline"] == b["discipline"],
    "cross_discipline": lambda a, b: a["discipline"] != b["discipline"],
}

pairs = []
for pair_type, cond in groups.items():
    chosen = sample_pairs(cond, PAIRS_PER_TYPE)
    log.append(f"Pairs made ({pair_type:<16}) : {len(chosen)}")
    for a, b in chosen:
        pairs.append({
            "ulo_id_a": a["ulo_id"], "unit_a": a["unit_code"],
            "discipline_a": a["discipline"], "text_a": a["ulo_text"],
            "ulo_id_b": b["ulo_id"], "unit_b": b["unit_code"],
            "discipline_b": b["discipline"], "text_b": b["ulo_text"],
            "pair_type": pair_type,
            "llm_score": "", "llm_reason": "",   # filled in Step 3
        })

random.shuffle(pairs)
out = pd.DataFrame(pairs)
out.insert(0, "pair_id", range(1, len(out) + 1))
out["sample"] = "v1_random_by_type"

# ---------------------------------------------------------------- 3. EXTRA PAIRS (v2)
# Random pairs are almost never near-equivalent (v1 had no score of 5 and one 4), so v2
# adds 350 pairs drawn from bands of TF-IDF word overlap, taking every one of the rare
# high-overlap pairs. v1 keeps pair ids 1-150, so earlier labels and ratings still match.
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

V2_TOTAL = 350
BANDS = [("v2_overlap_0.30+", 0.30, 1.01, None),        # None = every pair in the band
         ("v2_overlap_0.20-0.30", 0.20, 0.30, 70),
         ("v2_overlap_0.10-0.20", 0.10, 0.20, 70),
         ("v2_overlap_0.00-0.10", 1e-9, 0.10, 70),
         ("v2_overlap_none", -1.0, 1e-9, "rest")]         # fills up to V2_TOTAL
rng = random.Random(SEED)
tfidf = TfidfVectorizer(lowercase=True, stop_words="english").fit_transform([r["ulo_text"] for r in rows])
sim = (tfidf @ tfidf.T).toarray()
used = {frozenset((a, b)) for a, b in zip(out["ulo_id_a"], out["ulo_id_b"])}
candidates = [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows))
              if frozenset((rows[i]["ulo_id"], rows[j]["ulo_id"])) not in used]
extra = []
for name, lo, hi, n in BANDS:
    band = [(i, j) for i, j in candidates if lo <= sim[i, j] < hi]
    rng.shuffle(band)
    take = len(band) if n is None else (V2_TOTAL - len(extra) if n == "rest" else n)
    for i, j in band[:take]:
        a, b = rows[i], rows[j]
        pair_type = ("same_unit" if a["unit_code"] == b["unit_code"] else
                     "same_discipline" if a["discipline"] == b["discipline"] else "cross_discipline")
        extra.append({"ulo_id_a": a["ulo_id"], "unit_a": a["unit_code"], "discipline_a": a["discipline"],
                      "text_a": a["ulo_text"], "ulo_id_b": b["ulo_id"], "unit_b": b["unit_code"],
                      "discipline_b": b["discipline"], "text_b": b["ulo_text"], "pair_type": pair_type,
                      "llm_score": "", "llm_reason": "", "sample": name})
    log.append(f"Pairs made ({name:<20}) : {min(take, len(band))} of {len(band)}")
rng.shuffle(extra)                     # mix the bands so batches are not ordered by overlap
extra = pd.DataFrame(extra)
extra.insert(0, "pair_id", range(len(out) + 1, len(out) + len(extra) + 1))
v1 = out
out = pd.concat([out, extra], ignore_index=True)
out.to_csv("pairs.csv", index=False)
out = v1                               # the hand-rating sample below is drawn from v1 as before

# blind sample for your own ratings (no pair_type, so you are not biased)
mine = out.sample(n=min(MY_SAMPLE_SIZE, len(out)), random_state=SEED)
try:                                   # never overwrite ratings that were already entered
    already_rated = pd.read_csv("my_ratings.csv")["my_score"].notna().any()
except (FileNotFoundError, KeyError):
    already_rated = False
if already_rated:
    print("my_ratings.csv already contains ratings - left unchanged.")
else:
    mine[["pair_id", "text_a", "text_b"]].assign(my_score="").to_csv(
        "my_ratings.csv", index=False)

log.append(f"TOTAL pairs                       : {len(out) + len(extra)} "
           f"(v1 {len(out)} random by type + v2 {len(extra)} by word-overlap band)")
log.append(f"Pairs for you to rate by hand     : {len(mine)}")

report = "\n".join(log)
print(report)
with open("preprocessing_log.txt", "w") as f:
    f.write(report + "\n")
print("\nSaved: ulos_clean.csv, pairs.csv, my_ratings.csv, preprocessing_log.txt")