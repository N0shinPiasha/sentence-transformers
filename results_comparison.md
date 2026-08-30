# Sentence-BERT (SBERT) Paper Reproduction — Report of Results Comparison 

**Paper Title**: *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*  
**Authors**: Nils Reimers and Iryna Gurevych (UKP-TUDA)  
**Conference**: EMNLP 2019  
**Execution Environment**: GitHub Codespaces (Linux Virtual Machine)

**Official Repository**: [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) 

---

## Executive Summary

This report presents empirical findings that replicate the main findings of the **Sentence-BERT (SBERT)** study within a resource-constrained cloud environment. The implementation confirms the effectiveness of the Siamese network architecture and Mean-pooling methods for generating fixed-size sentence embeddings.

### Key Takeaways:
* **Replication Success**: Running the model locally on a subset of 500 pairs from the STS Benchmark produced results that closely align with those of the original study, affirming the model's design.
* **Resource Efficiency**: By employing a pre-trained `bert-base-nli-mean-tokens` checkpoint and subsampling the dataset, the evaluation was completed on a standard cloud-based Linux VM without the need for extensive GPU resources.
* **Confirmed Architecture Constraints**: Error analysis reveals that although SBERT addresses the computational limitations of standard BERT, its dependence on mean-pooling can lead to a "bag-of-words" issue, where distinct semantic meanings may be merged due to similar syntax.

---

## 1. STS Benchmark (STS-B) Evaluation

A randomized 500-pair subset of the standard **STS Benchmark (STS-B)** test split was used for evaluation in order to replicate VM memory constraints. To map Semantic relatedness, Spearman's rank correlation ($\rho$) between predicted cosine similarity and gold human labels is used.

The SBERT architecture performs a pooling operation on the BERT output layer to create a fixed-sized sentence embedding. By default, the MEAN approach is used to do this.

### Table 1: Replication of Semantic Textual Similarity 

| Model / Method | STS-B $\rho$ (Our Replication) | STS-B $\rho$ (Published) | Status |
| :--- | :---: | :---: | :--- |
| **Average GloVe embeddings** | — | 0.580 | Baseline |
| **Average BERT embeddings (Raw)** | — | 0.463 | Baseline |
| **SBERT-NLI-base** | **0.776** | **0.770** | **Successfully Replicated** |

> **Finding**: The local execution obtained a correlation of **0.776**, closely aligning with the **0.770** (77.03) score reported for the `SBERT-NLI-base` model in the original article.

---

## 2. Qualitative Error Analysis

While the Spearman correlation supports the overall validity of the model, investigating certain prediction anomalies uncovers fundamental vulnerabilities in the bi-encoder architecture. 

### Table 2: Model Failure Case (False Positive)

| Metric | Details |
| :--- | :--- |
| **Sentence A** | "You don't need any visa." |
| **Sentence B** | "You don't need sauce at all." |
| **Human Gold Score** | 0.00 |
| **Model Predicted Similarity** | 0.94 |

> **Analysis**: The model incorrectly assigned a near-perfect similarity score ($\approx 0.94$) to a pair of sentences that human annotators deemed completely unrelated ($0.00$). Because standard SBERT relies mostly on mean pooling, it suffers from a ``bag-of-words'' effect. The model was able to recognized the identical syntactic structure ("You don't need...") but it was unable to identify the critical semantic divergence between the noun payloads ("visa" vs. "sauce").

This report presents empirical results reproducing the core findings of the **Sentence-BERT (SBERT)** paper in a resource-constrained cloud environment. The implementation validates the Siamese network architecture and Mean-pooling aggregation strategies for deriving fixed-size sentence embeddings. 

