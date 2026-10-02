# Research: SCRBM — Semantic-Collaborative Restricted Boltzmann Machine

## SCRBM overview
SCRBM is my first-author recommender-systems paper: "Semantic-Collaborative Restricted Boltzmann Machine (SCRBM) for Recommender Systems: Architecture, Graph-Enhanced Extension, and Cross-Domain Evaluation". Co-authors: Ayush, Sanjana Verma and Dr. Govind Kumar Jha (all GEC Munger). It is under peer review at the Journal of Ambient Intelligence and Humanized Computing (Springer, Q1) after a major revision.

It tackles two problems that keep coming up in real recommendation services: interaction matrices are extremely sparse, and new users get poor recommendations until enough data accumulates (the cold-start problem).

## How SCRBM works
- SCRBM conditions the hidden units of a Restricted Boltzmann Machine (RBM) directly on Sentence-BERT (SBERT, all-MiniLM-L6-v2) item embeddings inside a single energy function, so item meaning is part of the generative model rather than a post-hoc add-on. I did not find this combination anywhere in prior literature.
- Instead of contrastive divergence, I train it with a rank-discounted Bayesian Personalised Ranking (BPR) objective, which optimises ranking quality directly for top-K recommendation.
- **SCRBM+** extends it with two-hop graph-smoothed user representations (via randomised SVD), a density-adaptive four-way fusion gate that routes each user toward the most reliable information source, and a rating-proportional BPR loss.
- SBERT is used only to pre-compute item embeddings offline, so no large language model runs at inference. That makes it far cheaper than LLM-based recommenders and deployable on commodity hardware.

## SCRBM results
Evaluated on four public benchmarks: MovieLens-100K, MovieLens-1M, Yelp2020 and MovieLens-10M.
- MovieLens-1M (leave-one-out, 100 negatives): HR@10 = 0.7367 ± 0.0051 and NDCG@10 = 0.5513 ± 0.0021 across five independent seeds, ahead of LightGCN by about 5.7% (paired t-test p < 0.0001, Cohen's d = 9.06).
- Yelp2020, full-ranking controlled comparison: SCRBM+ reaches Recall@10 = 0.0566 and NDCG@10 = 0.0281, beating LightGCL (ICLR 2023) by 7.8% and 9.0%.
- MovieLens-10M: SCRBM+ shows its largest margin, +34.9% HR@10 over LightGCN.
- Cold start: up to 87.1% better HR@10 than NCF for users with 20 or fewer interactions.
- All experiments ran on Kaggle P100/T4 GPUs within 12-hour session limits, so I designed checkpointing around that constraint.

## SCRBM limitations
SCRBM needs item text metadata; the rating matrix is stored densely, so ML-10M is roughly the single-GPU limit of the current implementation; the graph representation is computed offline and does not update in real time; and all four datasets are explicit star ratings. Future work: dynamic graph updates, sparse indexing for industrial scale, and implicit-feedback datasets.
