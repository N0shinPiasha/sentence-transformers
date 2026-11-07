"""
OPTIONAL: train SBERT-NLI-base yourself from plain BERT with the paper's recipe, then
evaluate it on the STS-B test set (the released checkpoint scored 76.99 in this project,
77.03 in the paper).

Recipe (Reimers & Gurevych, 2019, Sec. 4): bert-base-uncased + MEAN pooling, SNLI + MultiNLI,
softmax classification objective on (u, v, |u - v|), batch size 16, 1 epoch,
Adam with learning rate 2e-5 and linear warm-up over 10% of the training data.

Needs a GPU - run it on Google Colab (Runtime > Change runtime type > T4 GPU):
    python train_sbert_nli.py --max-pairs 100000   # quick check, roughly 10-20 minutes on a T4
    python train_sbert_nli.py                      # full AllNLI (~943k pairs), roughly 1-2 hours
The times are rough estimates. Output: results/training_log.txt (+ the model in output/).
"""

import argparse
import time
from pathlib import Path

import torch
from datasets import load_dataset

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments

try:                                       # sentence-transformers v6 (this fork)
    from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
    from sentence_transformers.sentence_transformer.losses import SoftmaxLoss
except ImportError:                        # sentence-transformers v3-v5
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    from sentence_transformers.losses import SoftmaxLoss

ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--max-pairs", type=int, default=0, help="use only the first N training pairs (0 = all)")
parser.add_argument("--base-model", default="google-bert/bert-base-uncased")
args = parser.parse_args()

if not torch.cuda.is_available():
    print("WARNING: no GPU found - training on CPU will take many hours. Use Colab with a T4 GPU.")

model = SentenceTransformer(args.base_model)          # plain BERT -> mean pooling is added automatically
train = load_dataset("sentence-transformers/all-nli", "pair-class", split="train")
if args.max_pairs:
    train = train.shuffle(seed=42).select(range(min(args.max_pairs, len(train))))
print(f"Training pairs: {len(train):,}")

loss = SoftmaxLoss(model=model, embedding_dimension=model.get_embedding_dimension(), num_labels=3)
targs = SentenceTransformerTrainingArguments(
    output_dir=str(ROOT / "output" / "sbert-nli-base-replica"),
    num_train_epochs=1,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=0.1,                     # fraction of training steps (transformers >= 5)
    fp16=torch.cuda.is_available(),
    eval_strategy="no",
    save_strategy="no",
    logging_steps=500,
    seed=42,
    report_to="none",
)
t0 = time.time()
SentenceTransformerTrainer(model=model, args=targs, train_dataset=train, loss=loss).train()
minutes = (time.time() - t0) / 60

test = load_dataset("sentence-transformers/stsb", split="test")
evaluator = EmbeddingSimilarityEvaluator(test["sentence1"], test["sentence2"], test["score"], name="sts-test")
scores = evaluator(model)
rho = [v for k, v in scores.items() if k.endswith("spearman_cosine")][0] * 100

model.save(str(ROOT / "output" / "sbert-nli-base-replica" / "final"))
report = (f"Own training of SBERT-NLI-base: {len(train):,} AllNLI pairs, 1 epoch, batch 16, lr 2e-5\n"
          f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}  "
          f"training time: {minutes:.1f} min\n"
          f"STS-B test Spearman x100 = {rho:.2f}   (released checkpoint: 76.99, paper: 77.03)\n")
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "training_log.txt").write_text(report)
print(report)
