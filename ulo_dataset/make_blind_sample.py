"""
Make a second, fully blind rating sheet for the human-vs-LLM agreement check.

    python ulo_dataset/make_blind_sample.py            # 50 new pairs, stratified by pair type
    python ulo_dataset/rate_pairs.py my_ratings_blind.csv

Rate WITHOUT any AI help and without opening llm_labels.csv or pairs.csv.
The sheet hides the pair type and the LLM score, and skips the 20 pairs already
rated in my_ratings.csv. Afterwards run
    python ulo_dataset/evaluate_ulo.py
which reports the agreement for this round next to the first one.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
N_PAIRS = 50
SEED = 7                      # different from the seed used to draw the first 20 pairs

pairs = pd.read_csv(HERE / "pairs.csv")
rated = pd.read_csv(HERE / "my_ratings.csv")
done = set(zip(rated["text_a"], rated["text_b"]))          # match on texts: pair ids can change
pool = pairs[[(a, b) not in done for a, b in zip(pairs["text_a"], pairs["text_b"])]]

types = sorted(pool["pair_type"].unique())
per_type = {t: N_PAIRS // len(types) + (1 if i < N_PAIRS % len(types) else 0) for i, t in enumerate(types)}
sample = pd.concat([pool[pool["pair_type"] == t].sample(n=min(n, (pool["pair_type"] == t).sum()), random_state=SEED)
                    for t, n in per_type.items()])
sample = sample.sample(frac=1, random_state=SEED)          # shuffle so pair types are mixed

out = HERE / "my_ratings_blind.csv"
if out.exists():
    raise SystemExit(f"{out.name} already exists - delete it first if you really want a new sample.")
sample[["pair_id", "text_a", "text_b"]].assign(my_score="").to_csv(out, index=False)
print(f"Saved {out.name}: {len(sample)} pairs {sample['pair_type'].value_counts().to_dict()}")
print("Now run:  python ulo_dataset/rate_pairs.py my_ratings_blind.csv")
