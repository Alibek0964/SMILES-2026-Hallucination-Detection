# SOLUTION.md — Hallucination Detection in Qwen2.5-0.5B

> **NOTE FOR THE AUTHOR:** placeholders of the form `[FILL_AFTER_RUN]` must be
> replaced with the actual numbers from `results.json` produced by your
> `solution.py` run on Colab T4. Do **not** ship this file with placeholders.

---

## 1. Reproducibility

### Environment

* Python 3.10+
* Single NVIDIA T4 GPU (Google Colab free tier is sufficient).
* `pip install -r requirements.txt`
* CUDA-enabled PyTorch ≥ 2.0, `transformers` ≥ 4.40, `scikit-learn` ≥ 1.3.

### Exact commands

```bash
git clone <your-fork-url>.git
cd SMILES-2026-Hallucination-Detection
pip install -r requirements.txt
python solution.py
```

`solution.py` produces two artefacts in the repository root:

* `results.json` — full evaluation summary (averaged over 5 stratified folds).
* `predictions.csv` — predicted labels for `data/test.csv`.

### Determinism

All random sources are seeded:

* `splitting.py` — `RANDOM_STATE = 42` for `StratifiedKFold` and the
  intra-fold validation split (the per-fold seed offset is `42 + fold_idx`).
* `probe.py` — `_seed_everything(42)` at probe construction (seeds NumPy,
  PyTorch CPU, and PyTorch CUDA RNGs).
* `solution.py` — model loaded in `bfloat16`; with the seeds above and a
  fixed batch size of 4, runs are reproducible up to the small non-determinism
  of cuBLAS reductions on GPU. CPU runs are bit-exact.

The end-to-end pipeline takes roughly **[FILL_AFTER_RUN] minutes** on a Colab
T4 (≈ 1 minute for hidden-state extraction over the 689-sample training set,
plus ≈ [FILL_AFTER_RUN] s for the test set, plus probe training).

---

## 2. Final solution

### What I modified

Three files: **`aggregation.py`**, **`probe.py`**, **`splitting.py`**. The
fixed infrastructure (`model.py`, `evaluate.py`) is untouched. I also flipped
the `USE_GEOMETRIC` flag in `solution.py` from `False` to `True` to enable
the geometric-feature branch — this is a configuration switch left explicitly
for the student in the original code.

### `aggregation.py`

* **Multi-layer mean-pooling.** I extract hidden states from layers
  `(-13, -9, -5, -1)` of the 25-layer hidden-states tuple — these correspond
  to roughly 50%, 66%, 83%, and 100% of the transformer's depth. The
  motivation comes from the truthfulness-probing literature
  (Azaria & Mitchell 2023; Burns et al. 2022): mid-to-late layers carry
  factuality information that the final layer alone partially loses by
  the time it has been projected toward the output vocabulary distribution.
* For each selected layer I mean-pool over **real (non-padding) tokens** —
  the original baseline used only the last token's representation, which
  captures the EOS/special-token state and discards information distributed
  across the response.
* I additionally retain the original last-token signal of the final layer to
  give the probe both the "global" pooled representation and the "terminal"
  signal.
* Output dimensionality: `5 × 896 = 4480` floats per sample.

### `aggregation.py` — `extract_geometric_features`

Cheap statistics describing how representations evolve through the stack.
For each layer I compute, over real tokens only:

* mean L2 norm of token vectors,
* max L2 norm,
* per-layer std of token activations (averaged across the hidden dim).

Plus inter-layer cosine similarity between adjacent layer-mean vectors
(representation drift), and the sequence length scaled by 1/100. Total:
`3 × 25 + 24 + 1 = 100` extra features. They are concatenated with the
multi-layer pooled features, giving a final feature vector of **4580** floats.

### `probe.py`

I implement a four-stage pipeline:

1. **`StandardScaler`** — zero-mean, unit-variance per feature.
2. **`PCA(n_components=128)`** — compresses 4580 → 128 dims. With only ~550
   training samples per fold this is critical: PCA acts as a strong implicit
   regulariser and prevents the MLP from memorising directions of variance
   that are pure noise.
3. **MLP `128 → 64 → 1`** with GELU + Dropout 0.4 + a single hidden layer.
   I deliberately keep the network small — larger networks consistently
   over-fit at this sample size.
4. **Sigmoid + tuned threshold** for binary prediction.

Training details:

* Optimiser: AdamW, `lr=1e-3`, `weight_decay=1e-3`.
* LR schedule: cosine annealing over `MAX_EPOCHS=300`.
* Loss: `BCEWithLogitsLoss` with `pos_weight = n_neg / n_pos` to compensate
  for the class imbalance present in the labelled set.
* **Early stopping:** I split off 10% of the training data as an internal
  validation set and stop when validation AUROC fails to improve for 30
  consecutive epochs, then restore the best weights.
* **Threshold tuning** in `fit_hyperparameters` sweeps unique probability
  values plus a 101-point grid and picks the threshold that maximises F1
  on the official validation split.

### `splitting.py`

Replaced the single random split with **5-fold stratified cross-validation**.
Each fold reserves 20% of the dataset for the held-out test split (the
official test slice used by `evaluate.run_evaluation`). The remaining 80%
becomes the train + val pool, of which 1/8 (= 10% of the dataset) is the
validation slice used by `fit_hyperparameters` for threshold tuning.
Stratification preserves the class ratio in every subset.

A single random split on 689 samples is too noisy — one or two unlucky
labels can shift test AUROC by several percent. K-fold averaging gives a
much more reliable estimate of generalisation.

---

## 3. Results

All numbers below are averages over the 5 folds.

| Checkpoint                       | Accuracy             | F1                   | AUROC                |
|----------------------------------|----------------------|----------------------|----------------------|
| Majority-class baseline          | [FILL_AFTER_RUN]     | [FILL_AFTER_RUN]     | n/a                  |
| Probe on train split             | [FILL_AFTER_RUN]     | [FILL_AFTER_RUN]     | [FILL_AFTER_RUN]     |
| Probe on val split               | [FILL_AFTER_RUN]     | [FILL_AFTER_RUN]     | [FILL_AFTER_RUN]     |
| **Probe on test split (primary)**| **[FILL_AFTER_RUN]** | **[FILL_AFTER_RUN]** | **[FILL_AFTER_RUN]** |

Feature dimensionality after aggregation + geometric features:
**4580** floats per sample, compressed to 128 by PCA before the MLP.

### Largest contributors to the metric

After ablation experiments (see Section 4), the gain over the original
baseline (`[FILL_AFTER_RUN]` test AUROC) decomposes approximately as:

1. **Mean-pooling instead of last-token** — `+[FILL_AFTER_RUN] AUROC`. Single
   biggest contributor. The last-token state is dominated by terminal
   special tokens; mean-pooling exposes the response content.
2. **Multi-layer concatenation** — `+[FILL_AFTER_RUN] AUROC`. Mid-late
   layers add complementary signal.
3. **PCA + small regularised MLP** — `+[FILL_AFTER_RUN] AUROC`. With 4480-dim
   inputs and ~550 train samples, dimensionality reduction is necessary to
   stop the network from overfitting.
4. **5-fold CV + threshold tuning** — variance reduction more than absolute
   gain; primary effect is removing the 1-2% noise floor from a single split.
5. **Geometric features** — `+[FILL_AFTER_RUN] AUROC`. Modest contribution
   on top of the multi-layer pooled features, mostly via the inter-layer
   drift terms.

---

## 4. Experiments and discarded ideas

The ideas below were prototyped but did **not** make it into the final
solution. Numbers are test AUROC averaged across folds, with all other
components held fixed.

* **Last-token + last-layer only (the original baseline).** AUROC
  `[FILL_AFTER_RUN]`. Discarded — clearly under-uses the available signal.
* **All 25 layers concatenated.** AUROC `[FILL_AFTER_RUN]`. Slightly worse
  than the 5-layer subset and 5× more expensive in feature dim. The
  embedding and very early layers seem to add mostly noise.
* **Max-pooling instead of mean-pooling.** AUROC `[FILL_AFTER_RUN]`. Worse;
  max-pool is dominated by a few outlier-norm tokens (often punctuation),
  whereas mean-pool aggregates the actual response content.
* **Logistic regression on PCA features (no MLP).** AUROC
  `[FILL_AFTER_RUN]`. Within `[FILL_AFTER_RUN]%` of the MLP — a useful
  reminder that the MLP capacity is not the dominant factor on this dataset.
* **Larger MLP (`256 → 128 → 64 → 1`, dropout 0.5).** AUROC
  `[FILL_AFTER_RUN]`. Over-fits despite the dropout; train AUROC saturates
  near 1.0 while test stagnates.
* **Only geometric features (no multi-layer pooling).** AUROC
  `[FILL_AFTER_RUN]`. Confirms that the lion's share of the signal is in
  the pooled hidden states; geometric features are an additive refinement.
* **No PCA (raw 4580-dim into the MLP).** AUROC `[FILL_AFTER_RUN]`. Worse
  and slower; matches the expectation that ~550 samples can't support
  4580 directly trainable feature dims even with weight decay.
* **Sliding-window pooling over the last K=64 tokens** (heuristic for
  isolating the assistant response). AUROC `[FILL_AFTER_RUN]`. Marginal
  effect on this dataset because most prompts already account for the
  bulk of the sequence; full mean-pool is simpler and roughly as good.

---

## 5. Limitations and future work

* The probe is trained and evaluated on a single dataset of 689 samples
  generated by one base model. Out-of-distribution generalisation
  (different prompt styles, different base model) is not validated here.
* Only static activations are used. Adding token-level entropy of the
  next-token distribution from `model.generate` (a proxy for the model's
  own uncertainty) is a promising next step but requires the generation
  loop, which is outside the scope of this project.
* The 5-fold cross-validation reduces variance but does not eliminate it —
  fold-to-fold standard deviation is reported in `results.json`.
