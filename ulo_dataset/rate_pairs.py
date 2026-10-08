
import textwrap

import pandas as pd

GUIDE = """
  5 = same thing            4 = almost the same
  3 = similar skill/topic   2 = share one aspect
  1 = loosely related       0 = unrelated
  (judge the MEANING, not shared words)   q = save and quit
"""

df = pd.read_csv("my_ratings.csv")
df["my_score"] = pd.to_numeric(df["my_score"], errors="coerce")
todo = df.index[df["my_score"].isna()].tolist()
print(f"{len(df) - len(todo)} already rated, {len(todo)} to go.")
print(GUIDE)

for n, i in enumerate(todo, start=1):
    row = df.loc[i]
    print("=" * 70)
    print(f"Pair {n}/{len(todo)}  (id {row.pair_id})\n")
    print("A: " + textwrap.fill(str(row.text_a), 66, subsequent_indent="   "))
    print()
    print("B: " + textwrap.fill(str(row.text_b), 66, subsequent_indent="   "))
    while True:
        ans = input("\nYour score (0-5): ").strip().lower()
        if ans == "q" or ans in {"0", "1", "2", "3", "4", "5"}:
            break
        print("Please type a whole number 0-5, or q to quit.")
    if ans == "q":
        break
    df.loc[i, "my_score"] = int(ans)
    df.to_csv("my_ratings.csv", index=False)     
df["my_score"] = df["my_score"].astype("Int64")
df.to_csv("my_ratings.csv", index=False)
done = df["my_score"].notna().sum()
print(f"\nSaved. {done}/{len(df)} pairs rated.")