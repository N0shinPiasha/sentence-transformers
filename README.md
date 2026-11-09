# COMP8240 project: Sentence-BERT Revisited

**Noshin Anjum Malik Piasha (61158445), Macquarie University, Session 2 2026**

This repository is a fork of [sentence-transformers](https://github.com/UKPLab/sentence-transformers),
the official code of *Sentence-BERT* (Reimers & Gurevych, EMNLP-IJCNLP 2019). The library
code is unchanged; the project adds the scripts and data listed below to:

1. replicate the paper's STS-B results (Table 1),
2. test the same model on five existing datasets from other domains, and
3. build and evaluate a new dataset of Macquarie University unit learning outcomes (ULOs)
   labelled by an LLM judge and checked against human ratings.

## Results at a glance

| Experiment | Data | Metric | Result |
|---|---|---|---|
| Replication, SBERT-NLI-base | STS-B test, 1,379 pairs | Spearman x100 | **76.99** (paper 77.03) |
| Replication, SBERT-NLI-large | STS-B test, 1,379 pairs | Spearman x100 | **79.20** (paper 79.23) |
| Quora Question Pairs | GLUE validation, 40,430 pairs | ROC-AUC x100 | 80.30 |
| CUAD (legal clauses) | 1,416 test clauses, 1.0M pairs | ROC-AUC x100 | 73.55 |
| BIOSSES (biomedical) | 100 pairs | Spearman x100 | 51.62 |
| Financial PhraseBank | 2,259 sentences, 2.55M pairs | ROC-AUC x100 | 52.09 |
| Amazon ESCI (product search) | 434,234 US test pairs | Spearman x100 | 15.05 |
| **New: ULO dataset** | 500 pairs, LLM-judge labels | Spearman x100 | **77.05** vs. LLM labels (58.21 vs. the author's blind ratings, 50 pairs) |

## Repository map

| Path | What it does | Main output |
|---|---|---|
| `sbert_replicate.py` | Replicates Table 1 of the paper on the full STS-B test set | `results/table1_replication.csv`, `results/replication_log.txt`, `results/sbert_predictions.csv` |
| `run_existing_datasets.py` | First pass on 2,000-pair samples (kept for transparency, superseded) | `results/existing_datasets_results.csv`, `results/pred_*.csv` |
| `run_full_datasets.py` | Full evaluation on the five existing datasets | `results/full_results.csv`, `results/full_log.txt` |
| `extra_analysis.py` | Confidence intervals, TF-IDF baselines, sample diagnostics (no model needed) | `results/extra_analysis.txt` |
| `robustness_checks.py` | Optional follow-up: CUAD multi-label and ESCI per-query (nDCG@10) checks; not part of the reported results | `results/robustness_*.txt` |
| `train_sbert_nli.py` | Optional: trains SBERT from plain BERT with the paper's recipe (GPU); not part of the reported results | `results/training_log.txt` |
| `ulo_dataset/collect_ulo1.py` | Scrapes ULOs from unitguides.mq.edu.au (1 s delay between requests) | `ulos.csv` |
| `ulo_dataset/make_pairs.py` | Cleans outcomes; samples 150 random pairs by type (v1) + 350 pairs by word-overlap band (v2), seed 42 | `ulos_clean.csv`, `pairs.csv`, `preprocessing_log.txt` |
| `ulo_dataset/Prepare_LLM_batches.py` | Writes the fixed judge prompt and ten 50-pair batches | `judge_prompt.txt`, `batch_*.txt` |
| `ulo_dataset/llm_labels.csv` | Final labels for all 500 pairs (score 0-5 + reason): Google Gemini, Pro model | |
| `ulo_dataset/llm_labels_claude.csv` | Second judge: Claude Opus 5.5 labels for the same 500 pairs (judge-vs-judge agreement) | |
| `ulo_dataset/llm_labels_round1.csv` | Gemini's first-round labels for pairs 1-150, October 2026 (consistency check) | |
| `ulo_dataset/rate_pairs.py`, `make_blind_sample.py` | Terminal tool for blind human ratings | `my_ratings.csv`, `my_ratings_blind.csv` |
| `ulo_dataset/analyse_labels.py` | Merges labels, sanity checks, SBERT cosines (reuses saved cosines; `--recompute` re-encodes) | `ulo_dataset_labelled.csv`, `label_report.txt` |
| `ulo_dataset/evaluate_ulo.py` | Agreement statistics, baselines, error analysis | `results/ulo_evaluation.txt`, `results/ulo_scores.csv` |
| `report/make_figures.py` | Draws the report's ULO figure (the report itself is submitted separately) | `report/figures/` |

Files from the proposal stage (500-pair STS-B subset): `results_comparison.md`, `test_sbert_gotfromfirstproposal.py`, `output.txt`, `comparison_output.png`.

## How to run

Tested in GitHub Codespaces (CPU, Python 3.14) and Google Colab (T4 GPU).

```bash
pip install -e .                         # this fork of sentence-transformers
pip install -r requirements-project.txt

python sbert_replicate.py                # about 30 min on a 2-core CPU
python run_full_datasets.py biosses qqp fpb cuad   # CPU is fine
python run_full_datasets.py esci         # 434k pairs: use a GPU (about 24 min on a T4)
python extra_analysis.py                 # seconds, no model needed
python robustness_checks.py cuad         # CPU is fine
python robustness_checks.py esci         # GPU
cd ulo_dataset && python analyse_labels.py && cd ..   # merges the labels; reuses saved SBERT cosines
python ulo_dataset/evaluate_ulo.py       # about 2 min; add --models to re-encode with all baselines
python report/make_figures.py
```

The ULO scripts inside `ulo_dataset/` that predate the final report (`collect_ulo1.py`,
`make_pairs.py`, `Prepare_LLM_batches.py`, `analyse_labels.py`) use relative paths, so run them from that
folder (`cd ulo_dataset`). Re-running them does not overwrite the saved ratings or LLM labels.
The scraped data (collected 2026-10-07) is saved in `ulos.csv`, so nothing needs re-scraping.

**On Colab:** `!git clone https://github.com/N0shinPiasha/sentence-transformers`, then
`%cd sentence-transformers`, `!pip install -e . -r requirements-project.txt`, then e.g.
`!python run_full_datasets.py all`.

## Data notes

- ULO texts are copyright Macquarie University. They were collected from public unit guide pages
  for this assessment only; every row keeps its source URL and collection date.
- LLM judge labels were produced with the fixed prompt in `ulo_dataset/judge_prompt.txt`, in batches of 50,
  blind to SBERT scores (which are computed afterwards by `analyse_labels.py`).
- Which judge made which labels: the final labels in `llm_labels.csv` come from Google Gemini (Pro model, web app)
  on 9 November 2026; each `batch_*.txt` file was pasted unchanged into Gemini and its CSV answer was pasted
  into the file. `llm_labels_claude.csv` holds Claude Opus 5.5's labels for the same 500 pairs (November 2026,
  the final labels before the Gemini run), kept as the second judge. `llm_labels_round1.csv` holds Gemini's
  first-round labels for pairs 1-150 (October 2026, reported in the project update). Claude had seen 30 of those
  first-round labels before labelling (`claude_seen_round1_ids.txt`), so agreement is also reported without them.
- Human ratings: `my_ratings.csv` (20 pairs) was rated with an AI assistant's help on applying the rubric, and
  Claude saw these ratings before labelling. `my_ratings_blind.csv` (50 pairs) was rated blind, without AI help,
  before the Gemini labels were made; neither LLM judge saw it.

---

*Everything below this line is the original sentence-transformers README.*

---

<!--- BADGES: START --->

[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-models-yellow)](https://huggingface.co/models?library=sentence-transformers)
[![GitHub - License](https://img.shields.io/github/license/huggingface/sentence-transformers?logo=github&style=flat&color=green)][#github-license]
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sentence-transformers?logo=pypi&style=flat&color=blue)][#pypi-package]
[![PyPI - Package Version](https://img.shields.io/pypi/v/sentence-transformers?logo=pypi&style=flat&color=orange)][#pypi-package]
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=docs&message=sentence-transformers)][#docs-package]

<!-- [![PyPI - Downloads](https://img.shields.io/pypi/dm/sentence-transformers?logo=pypi&style=flat&color=green)][#pypi-package] -->

<!--- BADGES: END --->

# Sentence Transformers: Embeddings, Retrieval, and Reranking

This framework provides an easy method to compute embeddings for accessing, using, and training state-of-the-art embedding and reranker models. It can be used to compute embeddings using Sentence Transformer models ([quickstart](https://sbert.net/docs/quickstart.html#sentence-transformer)), to calculate similarity scores using Cross-Encoder (a.k.a. reranker) models ([quickstart](https://sbert.net/docs/quickstart.html#cross-encoder)), to generate sparse embeddings using Sparse Encoder models ([quickstart](https://sbert.net/docs/quickstart.html#sparse-encoder)) or to compute token-level embeddings for ColBERT-style late-interaction retrieval using Multi-Vector Encoder models ([quickstart](https://sbert.net/docs/quickstart.html#multi-vector-encoder)). This unlocks a wide range of applications, including [semantic search](https://sbert.net/examples/applications/semantic-search/README.html), [semantic textual similarity](https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html), and [paraphrase mining](https://sbert.net/examples/applications/paraphrase-mining/README.html).

A wide selection of over [15,000 pre-trained Sentence Transformers models](https://huggingface.co/models?library=sentence-transformers) are available for immediate use on 🤗 Hugging Face, including many of the state-of-the-art models from the [Massive Text Embeddings Benchmark (MTEB) leaderboard](https://huggingface.co/spaces/mteb/leaderboard). Additionally, it is easy to train or finetune your own [embedding models](https://sbert.net/docs/sentence_transformer/training_overview.html), [reranker models](https://sbert.net/docs/cross_encoder/training_overview.html), [sparse encoder models](https://sbert.net/docs/sparse_encoder/training_overview.html) or [multi-vector encoder models](https://sbert.net/docs/multi_vector_encoder/training_overview.html) using Sentence Transformers, enabling you to create custom models for your specific use cases.

For the **full documentation**, see **[www.SBERT.net](https://www.sbert.net)**.

## Installation

We recommend **Python 3.10+**, **[PyTorch 2.2+](https://pytorch.org/get-started/locally/)**, and **[transformers v5.0+](https://github.com/huggingface/transformers)**.

```
pip install -U sentence-transformers
```

See [Installation](https://www.sbert.net/docs/installation.html) in the docs for uv, conda, source, and editable installs, CUDA setup, and extras (`[image]`, `[audio]`, `[video]`, `[train]`, `[onnx]`, `[openvino]`, `[dev]`).

## Getting Started

See [Quickstart](https://www.sbert.net/docs/quickstart.html) in our documentation.

### Embedding Models

First download a pretrained embedding a.k.a. Sentence Transformer model.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
```

Then provide some texts to the model.

```python
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# => (3, 384)
```

And that's already it. We now have numpy arrays with the embeddings, one for each text. We can use these to compute similarities.

```python
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])
```

### Reranker Models

First download a pretrained reranker a.k.a. Cross Encoder model.

```python
from sentence_transformers import CrossEncoder

# 1. Load a pretrained CrossEncoder model
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
```

Then provide some texts to the model.

```python
# The texts for which to predict similarity scores
query = "How many people live in Berlin?"
passages = [
    "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.",
    "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
]

# 2a. predict scores for pairs of texts
scores = model.predict([(query, passage) for passage in passages])
print(scores)
# => [8.607139 5.506266 6.352977]
```

And we're good to go. You can also use [`model.rank`](https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.model.CrossEncoder.rank) to avoid having to perform the reranking manually:

```python
# 2b. Rank a list of passages for a query
ranks = model.rank(query, passages, return_documents=True)

print("Query:", query)
for rank in ranks:
    print(f"- #{rank['corpus_id']} ({rank['score']:.2f}): {rank['text']}")
"""
Query: How many people live in Berlin?
- #0 (8.61): Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.
- #2 (6.35): In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.
- #1 (5.51): Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.
"""
```

### Sparse Encoder Models

First download a pretrained sparse embedding a.k.a. Sparse Encoder model.

```python

from sentence_transformers import SparseEncoder

# 1. Load a pretrained SparseEncoder model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate sparse embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 30522] - sparse representation with vocabulary size dimensions

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[   35.629,     9.154,     0.098],
#         [    9.154,    27.478,     0.019],
#         [    0.098,     0.019,    29.553]])

# 4. Check sparsity stats
stats = SparseEncoder.sparsity(embeddings)
print(f"Sparsity: {stats['sparsity_ratio']:.2%}")
# Sparsity: 99.84%
```

### Multi-Vector Encoder Models

First download a pretrained multi-vector a.k.a. late-interaction (ColBERT-style) model.

```python
from sentence_transformers import MultiVectorEncoder

# 1. Load a pretrained MultiVectorEncoder model
model = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")

queries = ["What is the capital of France?"]
documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
]

# 2. Encode queries and documents into sequences of token-level embeddings
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings[0].shape, document_embeddings[0].shape)
# (10, 128) (9, 128)  # one 128-dimensional vector per token

# 3. Score them with late interaction (MaxSim)
scores = model.similarity(query_embeddings, document_embeddings)
print(scores)
# tensor([[9.6037, 9.4055]])
```

## Pre-Trained Models

We provide a large list of pretrained models for more than 100 languages. Some models are general purpose models, while others produce embeddings for specific use cases.

- [Pretrained Sentence Transformer (Embedding) Models](https://sbert.net/docs/sentence_transformer/pretrained_models.html)
- [Pretrained Cross Encoder (Reranker) Models](https://sbert.net/docs/cross_encoder/pretrained_models.html)
- [Pretrained Sparse Encoder (Sparse Embeddings) Models](https://sbert.net/docs/sparse_encoder/pretrained_models.html)
- [Pretrained Multi-Vector Encoder (Late Interaction) Models](https://sbert.net/docs/multi_vector_encoder/pretrained_models.html)

## Training

> **Tip:** Using an AI coding agent (Claude Code, Codex, Cursor, Gemini CLI, ...)? Install the [`train-sentence-transformers`](skills/) Hugging Face Agent Skill via `hf skills add train-sentence-transformers [--claude] [--global]` and ask your agent to fine-tune a model on your data.

This framework allows you to fine-tune your own sentence embedding methods, so that you get task-specific sentence embeddings. You have various options to choose from in order to get perfect sentence embeddings for your specific task.

- Embedding Models
  - [Sentence Transformer > Training Overview](https://www.sbert.net/docs/sentence_transformer/training_overview.html)
  - [Sentence Transformer > Training Examples](https://www.sbert.net/docs/sentence_transformer/training/examples.html) or [training examples on GitHub](https://github.com/huggingface/sentence-transformers/tree/main/examples/sentence_transformer/training).
- Reranker Models
  - [Cross Encoder > Training Overview](https://www.sbert.net/docs/cross_encoder/training_overview.html)
  - [Cross Encoder > Training Examples](https://www.sbert.net/docs/cross_encoder/training/examples.html) or [training examples on GitHub](https://github.com/huggingface/sentence-transformers/tree/main/examples/cross_encoder/training).
- Sparse Embedding Models
  - [Sparse Encoder > Training Overview](https://www.sbert.net/docs/sparse_encoder/training_overview.html)
  - [Sparse Encoder > Training Examples](https://www.sbert.net/docs/sparse_encoder/training/examples.html) or [training examples on GitHub](https://github.com/huggingface/sentence-transformers/tree/main/examples/sparse_encoder/training).
- Multi-Vector (Late Interaction) Models
  - [Multi-Vector Encoder > Training Overview](https://www.sbert.net/docs/multi_vector_encoder/training_overview.html)
  - [Training examples on GitHub](https://github.com/huggingface/sentence-transformers/tree/main/examples/multi_vector_encoder/training).

Some highlights across the different types of training are:

- Support of various transformer networks including BERT, RoBERTa, XLM-R, DistilBERT, Electra, BART, ...
- Multilingual and multi-task learning
- Evaluation during training to find optimal model
- [20+ loss functions](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html) for embedding models, [10+ loss functions](https://www.sbert.net/docs/package_reference/cross_encoder/losses.html) for reranker models and [10+ loss functions](https://www.sbert.net/docs/package_reference/sparse_encoder/losses.html) for sparse embedding models, allowing you to tune models specifically for semantic search, paraphrase mining, semantic similarity comparison, clustering, triplet loss, contrastive loss, etc.

## Companion Blog Posts

The following Hugging Face blog posts complement this documentation with narrative walkthroughs and full training examples:

**Training guides:**

- [Training and Finetuning Embedding Models](https://huggingface.co/blog/train-sentence-transformers): end-to-end training of bi-encoder embedding models.
- [Training and Finetuning Reranker Models](https://huggingface.co/blog/train-reranker): training Cross Encoder models for the second stage of retrieve-and-rerank pipelines.
- [Training and Finetuning Sparse Embedding Models](https://huggingface.co/blog/train-sparse-encoder): training SPLADE and other sparse encoders.

**Multimodal:**

- [Multimodal Embedding & Reranker Models](https://huggingface.co/blog/multimodal-sentence-transformers): using text, image, audio, and video models through a single API.
- [Training and Finetuning Multimodal Embedding & Reranker Models](https://huggingface.co/blog/train-multimodal-sentence-transformers): training multimodal models, with a Visual Document Retrieval walkthrough.

**Efficiency techniques:**

- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): variable-size embeddings that can be truncated with minimal quality loss.
- [Train 400x faster Static Embedding Models](https://huggingface.co/blog/static-embeddings): CPU-friendly embedding models without attention.
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization): post-training compression of embedding vectors.

## Application Examples

You can use this framework for:

- **Computing Sentence Embeddings**

  - [Dense Embeddings](https://www.sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html)
  - [Sparse Embeddings](https://www.sbert.net/examples/sparse_encoder/applications/computing_embeddings/README.html)

- **Semantic Textual Similarity**

  - [Dense STS](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html)
  - [Sparse STS](https://www.sbert.net/examples/sparse_encoder/applications/semantic_textual_similarity/README.html)

- **Semantic Search**

  - [Dense Search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
  - [Sparse Search](https://www.sbert.net/examples/sparse_encoder/applications/semantic_search/README.html)

- **Retrieve & Re-Rank**

  - [Dense only Retrieval](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
  - [Sparse/Dense/Hybrid Retrieval](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)

- [Clustering](https://www.sbert.net/examples/sentence_transformer/applications/clustering/README.html)

- [Paraphrase Mining](https://www.sbert.net/examples/sentence_transformer/applications/paraphrase-mining/README.html)

- [Translated Sentence Mining](https://www.sbert.net/examples/sentence_transformer/applications/parallel-sentence-mining/README.html)

- [Multilingual Image Search, Clustering & Duplicate Detection](https://www.sbert.net/examples/sentence_transformer/applications/image-search/README.html)

and many more use-cases.

For all examples, see [examples/sentence_transformer/applications](https://github.com/huggingface/sentence-transformers/tree/main/examples/sentence_transformer/applications).

## Development setup

After cloning the repo (or a fork) to your machine, in a virtual environment, run:

```
python -m pip install -e ".[dev]"

pre-commit install
```

To test your changes, run:

```
pytest
```

## Citing & Authors

If you find this repository helpful, feel free to cite our publication [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://huggingface.co/papers/1908.10084):

```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

If you use one of the multilingual models, feel free to cite our publication [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://huggingface.co/papers/2004.09813):

```bibtex
@inproceedings{reimers-2020-multilingual-sentence-bert,
    title = "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2004.09813",
}
```

Please have a look at [Publications](https://www.sbert.net/docs/publications.html) for our different publications that are integrated into SentenceTransformers.

### Maintainers

Maintainer: [Tom Aarsen](https://github.com/tomaarsen), 🤗 Hugging Face

Don't hesitate to open an issue if something is broken (and it shouldn't be) or if you have further questions.

---

This project was originally developed by the [Ubiquitous Knowledge Processing (UKP) Lab](https://www.ukp.tu-darmstadt.de/) at TU Darmstadt. We're grateful for their foundational work and continued contributions to the field.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

[#docs-package]: https://www.sbert.net/
[#github-license]: https://github.com/huggingface/sentence-transformers/blob/main/LICENSE
[#pypi-package]: https://pypi.org/project/sentence-transformers/
