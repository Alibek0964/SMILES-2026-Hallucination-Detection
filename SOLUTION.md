# SOLUTION.md — Hallucination Detection in Qwen2.5-0.5B

**Final test AUROC (5-fold avg): 77.17 %**
**Test accuracy: 71.40 %** (majority-class baseline: 70.10 %)
**`predictions.csv` (publicly hosted):** <https://github.com/StarDust1508/smiles-2026-hallucination-detection/releases/download/v1.0/predictions.csv>

Per-fold breakdown is in `results.json`.

---

## 1. Reproducibility

### Environment
- Python 3.10+
- Single NVIDIA T4 GPU (Google Colab / Kaggle free tier is sufficient)
- `pip install -r requirements.txt`
- `xgboost` is recommended; if it is missing the probe falls back to scikit-learn `LogisticRegression(class_weight='balanced')` automatically.

### Exact commands

```bash
git clone https://github.com/StarDust1508/smiles-2026-hallucination-detection.git
cd smiles-2026-hallucination-detection
pip install -r requirements.txt
python solution.py
```

`solution.py` produces two artefacts in the repository root:
- `results.json` — full evaluation summary, averaged over 5 stratified folds.
- `predictions.csv` — predicted labels for `data/test.csv`.

### Determinism

All random sources are seeded with `RANDOM_STATE = 42`:
- `splitting.py` — StratifiedKFold seed and per-fold validation-split seed (offset by fold index).
- `probe.py` — XGBoost `random_state` and the implicit `pos_weight` derived from the training labels.
- The Qwen model is loaded in `bfloat16` with `attn_implementation="eager"`. The eager attention is required because v7's alignment features need the cross-attention map and Qwen's default SDPA backend silently returns `None` for `output_attentions=True`.

End-to-end runtime on a Kaggle T4: ≈ 3 minutes for hidden-state extraction over 689 train samples + ≈ 30 seconds for 100 test samples + ≈ 5 seconds for 5-fold probe training.

---

## 2. The journey — why the final solution looks the way it does

I want to be honest about how this solution was found, because the final architecture only makes sense in light of what didn't work before it. I ran six probing iterations that all hit the same ceiling at ~68 % test AUROC, then realised I was attacking the wrong problem.

### What I did before writing any code

The first thing was to study the model. Qwen2.5-0.5B has 24 transformer layers, hidden dim 896, 14 attention heads. The dataset is wrapped in ChatML: `<|im_start|>system ... <|im_end|>`, then user, then `<|im_start|>assistant\n` followed by the response and `<|endoftext|>`. The vocabulary is around 152 k tokens.

Then I opened the dataset and looked at the first sample:

> **PROMPT (context):** "This is the most common method of construction procurement... the architect or engineer acts as the project coordinator. His or her role is to design the works, prepare the specifications and produce construction drawings, administer the contract, tender the works, and manage the works..."
>
> **RESPONSE:** "An architect or engineer has a direct relationship with the subcontractor."
>
> **LABEL:** 1 (hallucinated)

I read it as "the response is wrong about how construction procurement works in real life." That initial framing was the seed of all six dead ends. In reality the response isn't false in the world — it just isn't supported by the context provided. That is a textbook *faithfulness* hallucination, not a *factuality* one. It took me until v6 to feel that distinction.

### v1 — multi-layer mean-pool + small MLP

Standard SAPLMA-style probing, following Azaria & Mitchell 2023: mid-to-late transformer layers should encode a "truthfulness direction" better than the final layer alone. I concatenated mean-pooled hidden states from four layers (~50 %, ~66 %, ~83 %, ~100 % depth) with hand-crafted geometric scalars (per-layer norms, inter-layer cosine drift, std), then passed it through a 128 → 64 → 1 MLP with dropout and weight decay.

**Test AUROC: 65.27 %.** Train AUROC: 94.03 %. The 30-point train-test gap was the first warning sign.

### v2 — linear probe (StandardScaler + PCA-64 + LogReg)

I cut capacity hard. Linear probe with PCA(64) and class-balanced logistic regression — closer to the textbook "linear probe" of Alain & Bengio 2016. With 482 training samples per fold, an MLP with thousands of parameters cannot help but overfit.

**Test AUROC: 68.52 %.** Better, gap shrank, but still nowhere near a usable classifier.

### v3 — pool over the response only (heuristic)

I noticed mean-pooling over the *whole sequence* averages a 200-token prompt with a 10-token response, drowning the response signal. I added a heuristic that pools only the last 40 % of real tokens.

**Test AUROC: 68.66 %.** A non-improvement. In hindsight the heuristic was too crude — for typical samples the last 40 % is mostly the prompt's tail, not the response.

### v4 — logit confidence features

This was the version I was most excited about. I added the model's own next-token uncertainty signals: mean / min / std of chosen-token probabilities, per-token entropy, top-1 vs top-2 margin, perplexity, mean log-prob, an attention-entropy scalar from the last layer. All of it computed only over the response tokens.

**Test AUROC: 69.13 %.** This was the best I would get for a long time. Train AUROC was now 100 % across folds — the classifier was perfectly memorising training, but val/test stayed locked at ~69 %.

### v5 — added PCA on top of v4

I tried regularising the v4 feature space with PCA before XGBoost.

**Test AUROC: 66.20 %.** Worse — PCA averaged out the discrete logit-feature signal. Reverted.

### v6 — XGBoost with class balancing, no PCA

I removed PCA, kept the full v4 features, added `scale_pos_weight` to handle the 70/30 hallucinated/truthful imbalance, kept tree depth conservative.

**Test AUROC: 68.52 %.** Same plateau. By this point I had tried logistic regression, MLP, and XGBoost — three very different classifier families — and all of them ended within 2 % of each other on test. That's a strong signal that the model wasn't the bottleneck. **The features were.**

### The pivot — reading about how hallucinations actually work

I went back and read about hallucinations from first principles. I found a piece that classified them by source:

- **Factuality** hallucinations: response is verifiably false in the world.
- **Faithfulness** hallucinations: response contradicts the context provided in the prompt.
- **Reasoning** hallucinations: facts are right but the conclusion doesn't follow.

It also said clearly: *LLMs are confidently wrong*. Their internal uncertainty (which is what logit entropy measures) is approximately the same when they hallucinate as when they answer truthfully — they don't natively encode an "I don't know" state. I had been comparing this to cognitive biases in humans: a person can be just as sure when they're misremembering as when they're recalling correctly. Same shape of failure.

That was the moment everything clicked. I went back to my construction-procurement example. The response wasn't false about reality. It introduced a relationship — "architect has a direct relationship with the subcontractor" — that simply isn't in the provided context. And I had been trying to detect *factuality* errors using the model's *self-confidence*, which was exactly the wrong tool for exactly the wrong problem.

### v7 — context-response alignment

Once I had the right framing, the feature design was almost obvious.

**Drop:** all logit-based confidence features (they measure self-confidence, not faithfulness).

**Add:** features that compare the *response* against the *context*.

1. **Lexical overlap** between context and response token sets — Jaccard, response coverage (fraction of response tokens appearing in context), BLEU-1, BLEU-2.
2. **Semantic alignment** — cosine similarity between the mean-pooled context embedding and the mean-pooled response embedding, computed at three layers (~50 %, ~83 %, ~100 % depth).
3. **Cross-attention grounding** — for each response token, look at where it attends inside the context. Mean of max-attention-to-context, attention entropy, total attention mass directed at context.
4. **Length** — response/context length ratio, normalised response length.

Alongside these 12 alignment features I kept the v3-style hidden-state mean-pool (3 layers × 896 = 2688 dims) as a baseline representation.

For the response/context split I stopped relying on heuristics and used the actual `<|im_start|>` token (ID 151644) to find where the assistant turn begins. This required modifying `solution.py` to also pass `input_ids` and the last-layer attention map into the aggregation function, and to reload the model with `attn_implementation="eager"` so `output_attentions=True` would actually produce attention tensors.

Probe: same XGBoost as v6 but with stronger regularisation (`max_depth=2`, `reg_alpha=reg_lambda=2.0`, `min_child_weight=5`, `colsample_bytree=0.6`) — alignment features capture mostly low-order interactions, deeper trees were memorising hidden-state noise.

**Test AUROC: 77.17 %.** A jump of +8.0 percentage points over v4.

For the first time the test accuracy (71.40 %) exceeded the majority-class baseline (70.10 %) — meaning the classifier was now actually distinguishing classes rather than coasting on the imbalance.

### Ablation — how much do the hidden states actually contribute?

After v7 worked I ran one more experiment: I dropped the 2688 hidden-state features entirely and trained the same XGBoost on the **12 alignment features alone**.

**Test AUROC: 73.59 %.** Test accuracy: 73.72 % (the highest of any version).

Interpretation:
- The 12 hand-crafted alignment features carry **the lion's share of the signal** — about 95 % of the AUROC reachable with the full feature set.
- The 2688 hidden-state dims add only +3.6 % AUROC, at the cost of train AUROC inflating from 86 % to 99.86 %. They are mostly *memorised*, not *generalised*.
- Test accuracy is actually *higher* in the alignment-only version — the hidden states add noise that confuses the threshold tuning.

I kept the full v7 (with hidden states) as the final submission because the contest's primary ranking metric is AUROC and 77.17 % > 73.59 %. But the alignment-only version is the more elegant scientific finding, and I would lean on it if the goal were a deployable, interpretable detector. The ablation switch is preserved in `aggregation.py` as `INCLUDE_HIDDEN_STATES = True` so the alignment-only run is one boolean flip away.

---

## 3. Final architecture — what's actually in the repository

### `aggregation.py`

`aggregate(...)` — mean-pools hidden states over the response tokens at three layers (`-13`, `-5`, `-1` in the 25-element hidden-states tuple). Response tokens are identified by finding the last `<|im_start|>` and skipping the "assistant\n" tokens that follow.

`extract_alignment_features(...)` — produces a 12-dim vector with the four lexical features, three layer-wise cosine-similarity features, three cross-attention grounding features, and two length features.

`aggregation_and_feature_extraction(...)` — entry point called from `solution.py`. Has a documented `INCLUDE_HIDDEN_STATES` switch (default `True`) for the ablation. Logits are accepted in the signature for backward compatibility but deliberately ignored — v7's premise is that they measure the wrong thing.

### `probe.py`

`HallucinationProbe` is a thin wrapper around `StandardScaler → XGBoost`, with a `LogisticRegression(class_weight='balanced')` fallback for environments without xgboost. `fit_hyperparameters` tunes the decision threshold to maximise F1 on the official validation slice. The XGBoost hyperparameters were chosen by light manual sweep on the v7 feature space; the regularisation (`max_depth=2`, `reg_alpha=2.0`, `reg_lambda=2.0`, `min_child_weight=5`) is intentionally aggressive for a 482-sample-per-fold setting.

### `splitting.py`

5-fold StratifiedKFold (seed 42). Each fold yields a roughly 76 % / 11 % / 13 % train / val / test split, with the validation slice carved out of the train+val pool while preserving the class ratio. A single random split was rejected because on 689 samples the test slice is only ~140 samples and one or two unlucky labels can swing AUROC by several percentage points.

### `solution.py`

Edits relative to the original template are minimal but necessary:
- `USE_GEOMETRIC = True` (the original `False` would silently skip alignment features).
- After `get_model_and_tokenizer()`, the model is re-loaded with `attn_implementation="eager"` because Qwen's default SDPA attention returns `None` when asked for attention weights.
- The aggregation call now passes `input_ids` and `last_layer_attentions` in addition to hidden states.

`model.py` and `evaluate.py` are untouched.

---

## 4. Results table

All metrics are 5-fold averages.

| Checkpoint                                | Accuracy    | F1          | AUROC       |
|-------------------------------------------|------------:|------------:|------------:|
| 1. Majority-class baseline                | 70.10 %     | 82.42 %     | n/a         |
| 2. Probe on train split                   | 83.28 %     | 89.55 %     | 99.86 %     |
| 3. Probe on val split                     | 72.75 %     | 83.24 %     | 73.51 %     |
| **4. Probe on test split (primary)**      | **71.40 %** | **82.45 %** | **77.17 %** |

Per-fold test AUROC: 77.32, 80.89, 77.70, 75.92, 74.03 (std ≈ 2.5 %).

Feature dimensionality: 2700 (2688 hidden-state pool + 12 alignment).

---

## 5. Experiments and discarded ideas

| Idea | Test AUROC | Why discarded |
|---|---|---|
| **v1.** Multi-layer mean-pool + MLP, geometric scalars | 65.27 % | Severe overfit (train 94 %, test 65 %). Too much capacity. |
| **v2.** Linear probe (StandardScaler + PCA-64 + LogReg) | 68.52 % | Improvement but still ceiling-bound. |
| **v3.** Pool only the last 40 % of real tokens | 68.66 % | Heuristic too coarse — 40 % of a 200-token sample is still mostly prompt. |
| **v4.** + logit confidence features (entropy, perplexity, top-1 margin, ...) | 69.13 % | Logit features measure self-confidence, not faithfulness — wrong signal for this dataset. |
| **v5.** PCA on top of v4 | 66.20 % | PCA destroyed the discrete logit-feature signal. |
| **v6.** XGBoost on v4, no PCA, class-balanced | 68.52 % | Same logit-feature ceiling. Confirmed bottleneck is features, not classifier. |
| **v7-ablation.** Alignment features only (12 dims) | 73.59 % | Beautiful result, 86 % train AUROC vs full v7's 99.86 %, but 3.6 % below v7-full on AUROC, the contest's primary metric. |

The plateau across v1–v6 (always within 2 % of each other across radically different classifier families) was the empirical signal that the bottleneck was the *features*, not the *model*. The hallucination-typology literature gave the conceptual reframe (faithfulness, not factuality) that suggested what features to build instead.

---

## 6. Limitations and what I'd try with another week

- **Single base model.** The whole pipeline is built around Qwen2.5-0.5B. Most of v7's signal is task-agnostic (lexical overlap, cosine similarity, attention grounding) and should transfer, but I haven't validated this.
- **No NLI baseline.** A small NLI model (e.g. DeBERTa-MNLI) could give context-response entailment scores directly. I avoided it here because it would add a second model to the inference chain, but it would likely outperform a single-pass probe.
- **Fold-to-fold variance is real.** Test AUROC ranges 74–81 % across folds (std ≈ 2.5 %). On 689 samples, small subsets matter.
- **The 99.86 % train AUROC is uncomfortable.** Even with the alignment ablation showing 86 % train AUROC (much healthier), the full version still memorises. A natural next step: drop hidden states from the *probe* but keep them as a feature for *threshold calibration* — letting the alignment features do the actual classification.
- **More principled response detection.** I currently slice by token offset from the last `<|im_start|>`. Reading the role token explicitly (`assistant`) would be more robust to malformed prompts.
- **Self-consistency proxy.** A second forward pass with the context masked, comparing how response token probabilities change without the context, would be a near-ground-truth signal for faithfulness — at the cost of doubling inference time.

---

## Acknowledgements

Versions v1 through v6 — the entire factuality-probing journey and the diagnosis of the plateau — were my own iterative work. The v7 reframe to faithfulness/alignment, the alignment feature design, and the final ablation were developed in collaboration with an AI pair-programmer; I made every architectural decision and validated every experiment myself, but the implementation pace would not have been the same on my own.
