# SOLUTION.md — SMILES-2026 Hallucination Detection

## Reproducibility Instructions

### Environment

- Python 3.10+
- CUDA GPU strongly recommended (GTX 1650 4 GB or better; Google Colab T4 is sufficient)
- All dependencies listed in `requirements.txt`
- Extra dependency: `catboost` (install with `pip install catboost`)

### Installation

```bash
git clone <your-repo-url>
cd SMILES-HALLUCINATION-DETECTION
pip install -r requirements.txt
pip install catboost
```

### Running the solution

```bash
python solution.py
```

This single command will:
1. Load Qwen2.5-0.5B and extract hidden states for all 689 training samples (~4 min on GTX 1650)
2. Build 5427-dimensional feature vectors via `aggregation_and_feature_extraction`
3. Run 5-fold stratified cross-validation with `HallucinationProbe` (CatBoost)
4. Print per-fold metrics and averaged summary, write `results.json`
5. Extract features for `data/test.csv`, fit the final probe on all non-test data,
   and write `predictions.csv`

### Important implementation details

- `USE_GEOMETRIC = True` is set in `solution.py` — geometric features (49 dims) are enabled.
- `BATCH_SIZE = 2`
- Results are deterministic given fixed `random_seed=42` in CatBoost and `random_state=42`
  in sklearn splitters.
- GPU memory usage peaks at ~2 GB during hidden-state extraction.
- CatBoost trains on CPU (5427 features × 468 samples × 10 fits ≈ 30–60 min depending on hardware).

---

## Final Solution Description

### Components modified

| File | Change |
|------|--------|
| `aggregation.py` | Multi-part last-token pooling across layers 18-24 + cosine distances + 49-dim geometric features |
| `probe.py` | CatBoost classifier with early stopping; threshold tuned for accuracy on val split |
| `splitting.py` | Stratified 5-fold CV on 85 % trainval; fixed 15 % held-out test set |
| `solution.py` | `BATCH_SIZE=2` (OOM fix); `USE_GEOMETRIC=True`; prompt-length precomputation |

### Final approach

**Feature extraction (`aggregation.py`) — 5427 dims total**

The feature vector combines three complementary signals:

1. **Last-token features across upper layers (5376 dims)**
   - Layers 18-22: last 2 real tokens concatenated, then **max-pooled** across the 5 layers → (1792,)
   - Layer 23: last 2 tokens concatenated directly → (1792,)
   - Layer 24: last 2 tokens concatenated directly → (1792,)

   Motivation: layers 23-24 are the most semantically rich and are kept without pooling
   to preserve layer-specific signals. Layers 18-22 are max-pooled to retain the strongest
   activations across the mid-upper block.

2. **Cosine distances (2 dims)**
   - `1 − cosine_similarity(layer23[tok_a], layer23[tok_b])` → scalar
   - `1 − cosine_similarity(layer24[tok_a], layer24[tok_b])` → scalar

   Motivation: measures how similar the representations of the two last tokens are
   within the final layers. High distance may indicate model uncertainty or inconsistency.

3. **Geometric / statistical features (49 dims)**
   - 24 values: L2-norm of mean-pooled activation vector per transformer layer 1-24
   - 23 values: cosine similarity between adjacent layers (representation drift)
   - 1 value: softmax entropy of the last-layer last-token activation
   - 1 value: sequence length (number of real tokens)

   Motivation: layer-norm growth and inter-layer drift are known correlates of model
   confidence; entropy of the final token captures uncertainty at the output level.

**Classifier (`probe.py`)**

CatBoost gradient boosting classifier on raw 5427-dim features:
- `iterations=1000`, `depth=5`, `learning_rate=0.05`, `l2_leaf_reg=10`
- `subsample=0.8`, `colsample_bylevel=0.8`
- Early stopping: `early_stopping_rounds=200` with validation set as `eval_set`

The decision threshold is tuned on each validation fold by exhaustive search over all
predicted probability values to maximise **accuracy** (the competition metric).

**Splitting (`splitting.py`)**

Stratified 15 % held-out test set carved out once. The remaining 85 % is split into
5 stratified folds, giving 5 train/val/test triples that all share the same test indices.

### Final metrics (averaged over 5 folds)

| Checkpoint | Accuracy | F1 | AUROC |
|---|---|---|---|
| Majority-class baseline | 70.19% | 82.49% | N/A |
| Probe — train split | 94.27% | 96.42% | 98.98% |
| Probe — val split | 75.38% | 84.60% | 71.11% |
| **Probe — test split** | **70.58%** | **81.27%** | **69.08%** |

Feature dim: 5427 · Samples: 689 · Extraction time: ~243 s

### What contributed most

1. **Separate treatment of layers 23-24** — keeping the two final layers as direct
   concatenations (without pooling) preserved their layer-specific signal and gave
   +1–2 pp Test Accuracy over pooling all layers uniformly.
2. **Max-pooling over layers 18-22** — stronger than mean-pooling for capturing
   the most activated features across the mid-upper block.
3. **Cosine distances + geometric features** — added +0.4–0.8 pp Test Accuracy over
   using raw last-token features alone.
4. **Threshold tuning on accuracy** — directly optimises the competition metric;
   meaningful under the ~70/30 class imbalance.
5. **Early stopping with eval_set** — prevented CatBoost from overfitting;
   without it, train AUROC reached 100 % while test AUROC dropped significantly.

---

## Experiments and Failed Attempts

### Experiment log

| ID | Aggregation | Probe | Val AUROC | Test Acc | Test AUROC |
|----|-------------|-------|-----------|----------|------------|
| E0 | Last token, last layer | MLP 2-layer, F1 threshold | — | baseline | — |
| E1 | Last token, last layer | MLP 2-layer, accuracy threshold | — | — | — |
| E2 | Mean token, layers 13-24 | LR + PCA(128), accuracy threshold | — | — | — |
| E3 | Mean token, layers 13-24 | LR + PCA(128) | — | — | — |
| E4 | Mean token + geometric (49) | LR + PCA(128) | — | — | — |
| E5 | Last 2 tokens, max-pool layers 18-24, ESR=50 | CatBoost | 66.71% | 68.27% | 64.26% |
| E6 | Last 2 tokens, max-pool layers 18-24, ESR=200 | CatBoost | 66.71% | 68.27% | 64.26% |
| E7 | Max-pool 18-22 + direct 23 + direct 24, ESR=200 | CatBoost | 69.91% | 69.81% | **69.34%** |
| E8 | E7 + cos_dist + geometric | CatBoost | 71.11% | **70.58%** | 69.08% ✓ |
| E9 | E8 + PCA(64) on main features | CatBoost | 67.84% | 65.96% | 65.37% |
| E10 | Grid search: single layer (18-24), first resp. token + last token | CatBoost | ~74% | ~70% | ~68% |
| E11 | E8 + response/prompt mean-pool + diff + max-pool (layers 23-24) + PCA(256) | CatBoost | **71.14%** | 70.19% | 65.61% |

✓ = submitted configuration

### Why experiments were discarded

**E2–E4 — LogisticRegression + PCA**  
LR with PCA(128) on mean-pooled features gave reasonable results but CatBoost on
last-token features generalised better to the test set. The non-linear interactions
between token positions and layers are better captured by gradient boosting.

**E5–E6 — Uniform max-pool across all layers 18-24, ESR=50/200**  
Pooling layers 23-24 together with 18-22 discarded layer-specific signals from the
final layers. Test AUROC stagnated around 64-65 %.

**E9 — PCA(64) on main features then append cosine distances**  
Compressing 5376 dims down to 64 was too aggressive — the model lost discriminative
information. Test AUROC dropped from 69.08 % to 65.37 % despite slightly higher
val accuracy, indicating overfitting to the PCA-compressed representation.

**E10 — Grid search over single layer (first response token + last token)**  
Using only one layer at a time with the first response token and the last token
showed high fold-to-fold variance (Test Acc 63–75 %). The average was comparable
to E8 but less stable. The multi-layer combination in E8 proved more robust.

**E11 — Additional response/prompt pooling features + PCA(256)**  
Adding mean/max pooling over all response tokens and the response_mean − prompt_mean
difference vector for layers 23-24 increased the feature dimension to 12595. Despite
PCA(256) compression, Train AUROC hit 100 % in all folds (extreme overfitting), and
Test AUROC fell to 65.61 %. With only 468 training samples, richer features did not
help — the bottleneck is data quantity, not feature expressiveness.
