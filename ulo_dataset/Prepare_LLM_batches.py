
import pandas as pd

BATCH_SIZE = 50

PROMPT = """You are an expert in university curriculum design. You will rate how similar \
in MEANING two university Unit Learning Outcomes (ULOs) are.

Use this scale (0-5). Judge what the student must be able to do, not shared words.
5 = Equivalent: same skill, same subject, same level. Could replace each other.
4 = Mostly equivalent: same skill and subject; small differences in scope or detail.
3 = Roughly similar: same general skill OR same specific topic, but clear differences.
2 = Partly related: share one aspect (e.g. both about data analysis, or both about \
communication) but different overall goals.
1 = Loosely related: different topics; only a very general connection.
0 = Unrelated: nothing meaningful in common.

Rules:
- Rate every pair independently.
- Output ONLY CSV, no other text, with this exact header:
pair_id,llm_score,llm_reason
- llm_score must be a whole number 0-5.
- llm_reason must be under 20 words and must not contain commas.

PAIRS:
"""

pairs = pd.read_csv("pairs.csv")
with open("judge_prompt.txt", "w") as f:
    f.write(PROMPT)

n_batches = 0
for start in range(0, len(pairs), BATCH_SIZE):
    chunk = pairs.iloc[start:start + BATCH_SIZE]
    lines = [f"[{r.pair_id}] A: {r.text_a}\n      B: {r.text_b}" for r in chunk.itertuples()]
    n_batches += 1
    with open(f"batch_{n_batches}.txt", "w") as f:
        f.write(PROMPT + "\n".join(lines) + "\n")
    print(f"batch_{n_batches}.txt  ->  pairs {chunk.pair_id.min()}-{chunk.pair_id.max()}")

# empty file where you paste the LLM's answers
with open("llm_labels.csv", "w") as f:
    f.write("pair_id,llm_score,llm_reason\n")
print(f"\nCreated {n_batches} batches, judge_prompt.txt and an empty llm_labels.csv")