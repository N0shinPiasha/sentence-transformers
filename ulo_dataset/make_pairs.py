
import random
import re

import pandas as pd

PAIRS_PER_TYPE = 50      # 3 types x 50 = 150 pairs (raise later towards 500)
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
out.to_csv("pairs.csv", index=False)

# blind sample for your own ratings (no pair_type, so you are not biased)
mine = out.sample(n=min(MY_SAMPLE_SIZE, len(out)), random_state=SEED)
mine[["pair_id", "text_a", "text_b"]].assign(my_score="").to_csv(
    "my_ratings.csv", index=False)

log.append(f"TOTAL pairs                       : {len(out)}")
log.append(f"Pairs for you to rate by hand     : {len(mine)}")

report = "\n".join(log)
print(report)
with open("preprocessing_log.txt", "w") as f:
    f.write(report + "\n")
print("\nSaved: ulos_clean.csv, pairs.csv, my_ratings.csv, preprocessing_log.txt")