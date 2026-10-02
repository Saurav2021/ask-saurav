# Research: SecSDAE — Sequential Recommendation with Gated Fusion of Identity and Text

## SecSDAE overview
SecSDAE is my first-author paper, co-authored with Dr. Govind Kumar Jha: "SecSDAE: A Semantic-Aware Contrastive Sequential Denoising Autoencoder for Cross-Modal Recommendation". It is currently under review at the Journal of Ambient Intelligence and Humanized Computing (Springer), the same journal reviewing my SCRBM paper. It is the successor to my SCRBM work and moves from static to sequential recommendation, i.e. predicting a user's next interaction from their chronological history.

The problem: almost all sequential recommenders treat items as anonymous ID tokens and throw away the text that describes them. That breaks down for newly added items and for users with short histories.

## How SecSDAE works
- Each user sequence is encoded through two parallel pathways: a collaborative identity pathway (trainable ID embeddings) and a frozen Sentence-BERT semantic pathway built from item text. The encoder combines a GRU for local transitions with two causal Transformer blocks for longer-range dependencies.
- A position-wise sigmoid gate blends the two streams at every step, trained jointly with no supervision about how to balance them.
- A symmetric cross-modal InfoNCE contrastive loss aligns the identity view and the semantic view of the same sequence, instead of two corrupted copies of one signal as in CL4SRec or DuoRec.
- A denoising reconstruction branch regularises the encoder. The ablation shows it is the single largest contributor: removing it costs 1.62% HR@10.
- Explanations come for free: gate weights, semantic similarity and counterfactual removal scores combine into compact evidence sets, evaluated with ERASER sufficiency and comprehensiveness metrics.

## SecSDAE results
- MovieLens-1M: HR@10 = 0.8218 ± 0.0030 (p < 0.003 against all evaluated baselines).
- MovieLens-100K: HR@10 = 0.7336 ± 0.0047 (p = 0.003 vs DuoRec).
- Yelp: HR@10 = 0.7574, +14.67% over S3-Rec.
- Amazon Digital Music: HR@10 = 0.5410, +19.98% over S3-Rec.
- It led 11 baselines across the four datasets, with gains from about 1.2% on the densest dataset to about 20% on the sparsest.
- Single-user inference runs in about 12.84 ms on CPU.
- The gate learned domain-specific balances without being told: 69.6% semantic for movies and 70.3% identity for local businesses, analysed over 1,000 users. That emergent behaviour is the result I find most interesting.
- Robustness: 96.3% of HR@10 retained with only five history items, and 96.6% under 40% noise.

## SecSDAE limitations
Five-seed evaluation was only feasible on the MovieLens datasets; cold-item results are mixed (it loses cold-item HR@10 by 2.7% to CL4SRec while winning NDCG@10 by 15.4%); explanations have not yet been validated against human judgement; and the text side uses short descriptors only. I report where it fails as openly as where it wins.
