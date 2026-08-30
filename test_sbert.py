from sentence_transformers import SentenceTransformer, util

# 1. Load the pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Define test sentences
sentences = [
    "A man is playing the guitar.",
    "A musician is playing a song on his acoustic guitar.",
    "The weather is very sunny in Sydney today."
]

# 3. Compute vector embeddings
embeddings = model.encode(sentences)

# 4. Compute cosine similarities
sim_1_2 = util.cos_sim(embeddings[0], embeddings[1])
sim_1_3 = util.cos_sim(embeddings[0], embeddings[2])

# 5. Output results
print(f"Similarity (Sentence 1 vs 2): {sim_1_2.item():.4f}")
print(f"Similarity (Sentence 1 vs 3): {sim_1_3.item():.4f}")