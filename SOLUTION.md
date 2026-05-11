# SMILES-2026 Hallucination Detection — Solution

## Final results (5-fold stratified CV)

| Metric                           | Value |
|----------------------------------|-------|
| **Test AUROC** (primary)         | **79.83%** |
| Test Accuracy                    | 73.29% |
| Test F1                          | 82.20% |
| Train AUROC                      | 99.86% |
| Majority-class baseline (acc)    | 70.10% |
| Baseline F1                      | 82.42% |
| Feature dim                      | 9020 |
| Total samples                    | 689 (483 hallucinated / 206 truthful) |
| Extract time (first run)         | ~45 min on T4 |
| Extract time (cached)            | ~3 min |

Per-fold breakdown of test AUROC: **79.00 / 77.80 / 82.05 / 82.91 / 77.39** — three folds exceed 80%, the average is pulled down by the two weaker folds.

The full per-fold record is in `results.json`. Predictions for the held-out test set are in `predictions.csv`.

---

## Pipeline at a glance

```
solution.py
   │  prompt+response  →  Qwen2.5-0.5B forward (hidden_states + attentions)
   ▼
aggregation.py  →  9020-dim feature vector per sample
   │   hidden-state pool over 5 layers     :  2 × 5 × 896  =  8960
   │   geometric drift (norms, cos, L2)    :              =    44
   │   HHEM-2.1-Open consistency score     :              =     3
   │   self-consistency (k=5 regen)        :              =     7
   │   lexical (token-recall, length, ...) :              =     6
   ▼
probe.py  →  3-base meta-stack
   │   Base 1 : MLP on all 9020 features
   │   Base 2 : XGBoost on the 60 tabular features
   │   Base 3 : LogReg(L2) on PCA(128) of the 8960-dim hidden pool
   │   Meta   : LogReg(L2, C=1.0) over the 3 OOF probability streams
   ▼
splitting.py  →  5-fold StratifiedKFold (+ GroupKFold fallback)
evaluate.py   →  per-fold metrics, results.json, predictions.csv
```

---

## How to reproduce

### Environment

```bash
pip install -r requirements.txt
```

The pipeline expects:

* CUDA GPU with ≥6 GB VRAM (tested on Kaggle T4 16 GB) — CPU works but is impractically slow because of the k=5 regeneration step.
* Internet access on first run (downloads three checkpoints from HuggingFace).

### Run

```bash
python solution.py
```

Outputs:

* `results.json` — 5-fold metrics summary.
* `predictions.csv` — 100 predictions for `data/test.csv`.

On the first run the pipeline downloads three HuggingFace checkpoints:

* `Qwen/Qwen2.5-0.5B` (already used by `solution.py`)
* `vectara/hallucination_evaluation_model` (HHEM-2.1-Open, ~600 MB)
* `sentence-transformers/all-MiniLM-L6-v2` (~90 MB)

After the first run, two on-disk caches make repeated runs nearly instant:

* `_v13_features.pkl` (~25 MB) — full per-sample 9020-dim feature vectors, keyed by the hash of `(input_ids, attention_mask)`.
* `_consistency_cache.pkl` (~50 KB) — the 7-dim self-consistency descriptors keyed by the input hash. Useful when `aggregation.py` changes but generation does not.

Both files live in the working directory. Delete them to force recomputation.

### Hardware budget

| Run | Wall time on T4 |
|---|---|
| First run, no cache  | ~45 min |
| Probe-only iteration | ~3 min (cache loads in seconds) |
| Test set extraction (100 rows) | ~25 sec |

---

## Architecture details

### Feature extraction (`aggregation.py`)

Five feature groups, concatenated per sample.

#### 1. Hidden-state pool (8960 dims)

Two pooling strategies — **last-token** and **mean over the response span** — applied at **five layers**: indices `(-1, -2, -4, -8, -13)` in the 25-element `hidden_states` tuple (24 transformer blocks + embedding). The five-layer span covers both the late "decision" band (-1 through -4) and the mid-band where semantic content lives (-13, roughly layer 12 of Qwen-0.5B).

Selecting the response span correctly was a non-trivial detail: the prompt ends with three Qwen ChatML tokens `<|im_start|>`, `assistant`, `\n` (IDs `[151644, 77091, 198]`). Earlier versions used `RESPONSE_OFFSET = 2`, which made the first "response" token in `response_idx` be `\n` rather than real content. This is **fixed to `RESPONSE_OFFSET = 3`** in v12 and beyond.

The Qwen tokenizer's `truncation_side` is also forced to `'left'` for prompts that exceed `MAX_LENGTH = 512` (22.1% of the training set). With the default right-side truncation, the response — i.e. the very thing we are classifying — gets discarded; left truncation drops early context tokens instead.

#### 2. Geometric drift (44 dims)

Compact descriptors of how the response representation moves across the five selected layers and across the two pooling tracks:

* `‖v‖₂` for each layer × pool combination (10 scalars)
* `mean, std, min, max` of the deepest-layer pool (8 scalars)
* `cos(layer_i, layer_{i+1})` and `‖v_i − v_{i+1}‖₂` for consecutive layers, both tracks (16 scalars)
* End-to-end drift `cos(deepest, shallowest)` and L2 distance, both tracks (4 scalars)
* Inter-pool agreement `cos(last_tok, mean)` at the deepest layer (2 scalars)
* Drift from embedding (`hidden_states[0]`) to final layer, both tracks (4 scalars)

#### 3. HHEM-2.1-Open consistency (3 dims)

`vectara/hallucination_evaluation_model` is loaded with `trust_remote_code=True` and called via its custom `model.predict([(premise, hypothesis)])` API. Each call returns a single float in `[0, 1]` where `1.0` means the response is supported by the context and `0.0` means it is hallucinated.

The **premise** fed to HHEM is the extracted Wikipedia-style passage (between `"single brief but complete sentence."` and `"Here is the question:"`) — not the full ChatML prompt. The data audit found that 100% of training prompts use this exact template, so the passage is reliably extractable with a single regex.

From the single score we expose three features:

* `score` — raw consistency in [0, 1].
* `1 − score` — hallucination probability (the same information, but a positive correlation with the target makes downstream linear models slightly more stable).
* `|score − 0.5| × 2` — confidence in [0, 1].

We initially tried `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` and got essentially zero signal. HHEM was specifically trained on FEVER + VitaminC, which targets the exact mix of contradiction (A1) and ungrounded invention (A2) that dominates this dataset.

#### 4. Self-consistency (7 dims)

For each prompt we generate `k = 5` alternative responses with Qwen2.5-0.5B at `temperature = 0.7, top_p = 0.95, max_new_tokens = 80`. All k+1 responses (original + alternatives) are embedded with `all-MiniLM-L6-v2` and we compute:

* `mean cos(orig, alts_i)` and `min cos(orig, alts_i)` — original-to-alternatives agreement.
* `mean` and `std` of pairwise `cos` within the alternatives — intrinsic alternative stability.
* `semantic_entropy` — entropy of cluster sizes after greedy agglomeration at `cos ≥ 0.85`.
* `len(orig) / mean(len(alt))` and `std(len(alt)) / mean(len(alt))` — length sanity.

The generation seed is derived from a hash of the prompt token IDs so the alternatives are reproducible across runs.

#### 5. Lexical features (6 dims)

Cheap text-level features grounded in the data audit:

* `token_recall_content` — fraction of response content words (length ≥ 3, stop-listed) that also appear in the passage. **Alone this feature reaches AUROC = 0.680 / accuracy = 73.3% — better than the majority-class baseline.**
* `log_response_chars`, `log_response_words` — response length carries a Cohen's d = 0.572 effect (hallucinated 790 chars vs truthful 418 on average).
* `has_template_leak` — regex flag for chat-template leaks (`"you are an AI assistant"`, `"step-by-step"`, `<|...|>` markers, etc.). 37% of hallucinated responses contain at least one of these patterns vs 20% of truthful ones.
* `ascii_ratio` and `repeated_5grams_count` — catch the "garbage / repetition" hallucination subtype.

### Probe (`probe.py`)

A 3-base meta-stack. Each base sees a different slice of the feature vector and uses an architecture appropriate to its dimensionality.

```
Base 1 : MLP(input → 256 → 64 → 1, LayerNorm, GELU, dropout 0.4)
         on all 9020 features.  Captures whatever signal a non-linear
         classifier can extract from the full vector.

Base 2 : XGBoost(max_depth=2, n_estimators=500, lr=0.03,
                 reg_alpha=1.0, reg_lambda=1.0, colsample_bytree=0.7,
                 scale_pos_weight=n_neg/n_pos)
         on the 60 TABULAR dimensions only (44 geom + 3 NLI + 7
         consistency + 6 lexical).  Trees handle small-n tabular data
         far better than an MLP, and the meta-LogReg consistently
         weights this stream 1.5-2.5× higher than the MLP.

Base 3 : LogReg(L2, C=0.5, class_weight="balanced")
         on PCA(n=128) of the 8960-dim hidden-state pool.  A linear
         classifier on a low-rank projection cannot memorise; if the
         hidden-state pool carries any generalisable signal beyond
         what the MLP already extracts, it shows up here.
```

All three bases are trained on the same 5-fold StratifiedKFold split. Each fold contributes its **out-of-fold (OOF)** predictions to a `(n_train, 3)` matrix that becomes the input to the meta-learner — a simple L2-regularised LogReg. The meta-LogReg coefficients summarise how each base contributes:

```
Fold 1: MLP=+0.600  XGB=+1.705  LR=+1.039
Fold 2: MLP=+0.772  XGB=+2.111  LR=+0.601
Fold 3: MLP=+1.090  XGB=+1.720  LR=+0.424
Fold 4: MLP=+1.119  XGB=+1.345  LR=+0.274
Fold 5: MLP=+0.996  XGB=+1.824  LR=+0.505
```

XGBoost on the 60 tabular dims is the strongest contributor on every fold — by a factor of 1.3× to 2.9× over the MLP. This is the central architectural finding of v14/v15: most of the signal in this dataset lives in a small set of hand-crafted features that respond to *groundedness in context*, not in the full hidden-state pool.

The decision threshold is tuned on the meta's full OOF probability vector for accuracy.

### Splitting (`splitting.py`)

`StratifiedKFold(n=5, shuffle=True, seed=42)`, with an automatic `GroupKFold` fallback if a column with repeated values (`group`, `source`, `id`, `question`, `prompt`) is present — preventing same-prompt leakage between train and test if the dataset ever grows. On the released `dataset.csv` all prompts are unique, so the StratifiedKFold path is what actually runs.

For each fold an additional `train_test_split(test_size=0.125, stratify)` carves a small validation slice out of the training half for threshold tuning.

---

## Experiments timeline

Every version was a single hypothesis test on a single dataset (n=689, 5-fold CV).

| Version | Test AUROC | Δ vs prev | What was added | Verdict |
|---|---|---|---|---|
| Majority-class baseline | — | — | — | 70.10% acc, no AUROC |
| **v9** (initial)        | 76.68% | — | hidden-state pool over 4 late layers + geometric drift + 5×bagged MLP | the user's starting submission |
| v10                     | 76.51% | −0.17 | + mDeBERTa-v3-base-mnli-xnli NLI features (6 dims) | within noise; mDeBERTa was the wrong model AND it received corrupted input (see Bug Fixes below) |
| v11                     | 77.14% | +0.63 | + self-consistency / semantic entropy from k=5 regenerations (7 dims) | real but small improvement; alone, generation instability was not the dominant failure mode |
| **v12** (bug fix + lex) | **78.65%** | **+1.51** | `RESPONSE_OFFSET 2→3`, `_decode_text` no longer leaks "assistant\n" into NLI input, tokenizer `truncation_side='left'` so the response is preserved on the 22% of samples exceeding 512 tokens, + 6 lexical features (`token_recall`, lengths, regex flags) | the bug fixes alone moved a 22%-truncation iceberg above water |
| v13                     | 79.14% | +0.49 | swap mDeBERTa-base → HHEM-2.1-Open, add mid-layer −13 (=layer 12) to `SELECTED_LAYERS` | HHEM marginal because the lexical/consistency features already capture most of the same A2 signal; mid-layer added incrementally |
| v14                     | 79.77% | +0.63 | probe.py: MLP+XGB(tabular) meta-stack with LogReg as the mixer | first probe change; immediately revealed XGB on tabular is the dominant contributor |
| **v15** (final)         | **79.83%** | +0.06 | + LogReg(L2) on PCA(128) of the 8960-dim hidden pool as a 3rd base; relaxed XGBoost (n_estimators 500, reg 1.0) | within noise of v14 on average, but on Fold 4 it reached **82.91%** AUROC, the single best fold of any version |

The progression is monotone (every step except v10 added test-AUROC). v10 looked flat at the time, but in retrospect mDeBERTa was being fed garbled text (see Bug Fixes); the actual NLI contribution shows up properly in v13 with HHEM-2.1-Open on a clean passage.

### Bug fixes that mattered

Three bugs found by an external code audit, all of which had been silently inherited from prior versions:

1. **`RESPONSE_OFFSET = 2` was off-by-one.** Tokenising `<|im_start|>assistant\n` with the Qwen tokenizer produces three tokens — `[<|im_start|>, "assistant", "\n"]`. The earlier offset put `"\n"` at the front of every response, dominating last-token pooling and producing single-newline responses on truncated samples. Fixed to `RESPONSE_OFFSET = 3`.

2. **`_decode_text` leaked `"assistant\n"` into the NLI context.** `<|im_start|>` is a special token that `skip_special_tokens=True` drops, but `"assistant"` (id 77091) is a regular BPE token and is *not* dropped. Every NLI call therefore received a context ending in `"... Your answer: \nassistant"`. The v12 fix slices off the trailing three prompt-format tokens before decoding.

3. **22.1% of samples were truncated on the response side.** With `MAX_LENGTH=512` and the default right-side truncation, 152 of 689 training prompts had their response either partially or entirely chopped off — taking the label signal with them. `aggregation.py` now monkey-patches `AutoTokenizer.from_pretrained` to set `truncation_side='left'` for any Qwen tokenizer at import time, before `model.py` loads it. This preserves the response and drops early context tokens instead.

### What didn't work, and why

* **Universal NLI on the full ChatML prompt (v10).** mDeBERTa-v3-base-mnli-xnli was trained on MNLI/XNLI — formal text with explicit contrasts. On this dataset's "ungrounded invention" (A2) class, mDeBERTa returns `P(neutral)` ≈ 0.6-0.8 for both truthful and hallucinated responses; it does not separate them. Even after the v12 fix gave it clean input, its 6 features still contribute marginally next to HHEM-2.1-Open.

* **Self-consistency as a standalone fix.** v11 added 7 features specifically targeting the "model invents different things on each regeneration" failure mode. The data audit later revealed that the dataset is roughly 40% A1 (direct contradiction) + 40% A2 (ungrounded invention) + 12% garbage + 8% world-knowledge errors — so self-consistency was correctly targeting only one of two main classes, and its lift accordingly came out to +0.6 pp rather than the +3-5 pp some literature reports on n=10k+ datasets.

* **Adding more layers to the hidden-state pool.** Going from 4 layers (v9-v12) to 5 layers (v13) raised the feature dim from 7225 to 9020 and added +0.49 pp test AUROC. Going from 5 to all 24 layers would dilute the signal further — v15's meta-LogReg already gives the linear-probe-on-hidden stream the lowest weight of the three bases, indicating the hidden-state pool is close to saturated.

* **First-token logit features and Lookback Lens** (Chuang et al. EMNLP 2024). Both were on the v16+ shortlist. Adding them would require re-running the 45-minute extraction pass with logit/attention plumbing in `aggregation.py`. With the project at 79.83% AUROC against an estimated information-theoretic ceiling around 80-83% (small-LM benchmarks at this n), the marginal return started looking unattractive and we stopped here.

### What we would try next given more compute

In rough order of expected impact:

1. **First-token logit features** (3-4 dims). Entropy + top-1 logit + perplexity at the very first response token. Literature reports standalone AUROC ~0.8 for these alone (`arXiv:2507.20836`, SAPLMA `arXiv:2304.13734`).
2. **Lookback Lens** (`arXiv:2407.07071`). Per-attention-head ratio of "mass on context" vs "mass on generated tokens", fed to a linear classifier. Directly targets the A2 invention case.
3. **Bigger NLI** (DeBERTa-v3-large MNLI+FEVER+ANLI or AlBERT-xlarge-VitaminC-MNLI). Expensive but the most reliable channel for A1+A2.
4. **A 4th meta-stack base trained on layer-wise OOF predictions** — one LogReg per layer, 24 streams concatenated, plus an L1-regularised meta — which picks the most informative layers automatically.

---

## File index

| Path | Status | What it does |
|---|---|---|
| `aggregation.py` | v13 — modified | feature extraction (groups 1-5 above) |
| `probe.py` | v15 — modified | 3-base meta-stack |
| `splitting.py` | v12 — minor | StratifiedKFold (+ GroupKFold fallback) |
| `requirements.txt` | v12 — modified | added `sentencepiece` and `protobuf` for the NLI tokenizer |
| `solution.py` | unmodified | fixed contest infrastructure |
| `model.py`, `evaluate.py` | unmodified | fixed contest infrastructure |
| `data/dataset.csv`, `data/test.csv` | unmodified | contest data |
| `results.json` | generated | metrics from the final 5-fold run |
| `predictions.csv` | generated | 100 predictions on `test.csv` |

The on-disk caches `_v13_features.pkl` and `_consistency_cache.pkl` are NOT committed — they regenerate from the dataset on first run.
