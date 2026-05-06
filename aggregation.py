"""
aggregation.py — Token aggregation strategy and feature extraction (v4).

v4 adds logits-based confidence features on top of hidden-state aggregation.

Why logits matter for a 0.5B model:
  The hidden states of a small language model are too compressed and noisy to
  carry a clean "truthfulness direction".  However, the model's own uncertainty
  about its next token — captured directly by the logits — is a strong and
  model-size-agnostic hallucination signal.  Hallucinated responses tend to
  exhibit higher per-token entropy, lower top-token probability, and smaller
  margin between the top-1 and top-2 candidates.

Feature groups (concatenated):
  1. Hidden-state mean-pool over response tokens at 3 mid-to-late layers
     (3 × 896 = 2688 dims) — kept as a baseline internal representation.
  2. Logit confidence features (10 dims):
     - mean / min / std of chosen-token probabilities
     - mean / max of per-token entropy
     - mean margin (top-1 − top-2 probability)
     - sequence perplexity
     - mean chosen-token log-probability
     - response length (normalised)
  3. Attention entropy (1 dim):
     - mean attention entropy across heads in the last layer.

Total dimensionality: ~2700 (compact enough for 482 training samples).

For Qwen2.5-0.5B the hidden-states tuple has 25 elements (1 embedding + 24
transformer layers), each shaped (seq_len, hidden_dim=896).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Mean-pool readouts at three mid-to-late layers.
SELECTED_LAYERS: tuple[int, ...] = (-9, -5, -1)
"""~62%, ~83%, ~100% depth — the band where truthfulness signal peaks."""


def _response_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return indices of tokens belonging to the assistant response.

    Heuristic: the last 40 % of non-padding tokens.  Responses in this
    dataset are typically 5-30 tokens while prompts are 100-300 tokens,
    so this safely isolates the response from the prompt.
    """
    real_idx = attention_mask.nonzero(as_tuple=False).flatten()
    n_real = len(real_idx)
    if n_real < 20:
        return real_idx  # very short — use everything
    start = int(n_real * 0.6)
    return real_idx[start:]


# ---------------------------------------------------------------------------
# Hidden-state aggregation
# ---------------------------------------------------------------------------

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool over response tokens at each selected layer.

    Args:
        hidden_states:  Tensor of shape (n_layers, seq_len, hidden_dim).
        attention_mask: 1-D tensor of shape (seq_len,).

    Returns:
        1-D feature tensor of shape (len(SELECTED_LAYERS) * hidden_dim,)
        i.e. 3 × 896 = 2688 for Qwen2.5-0.5B.
    """
    resp_idx = _response_token_indices(attention_mask).to(hidden_states.device)

    pooled: list[torch.Tensor] = []
    for layer_idx in SELECTED_LAYERS:
        layer = hidden_states[layer_idx]  # (seq_len, hidden_dim)
        resp_tokens = layer.index_select(0, resp_idx)  # (n_resp, hidden_dim)
        pooled.append(resp_tokens.mean(dim=0))  # (hidden_dim,)

    return torch.cat(pooled, dim=0)


# ---------------------------------------------------------------------------
# Logit-based confidence features
# ---------------------------------------------------------------------------

def extract_logit_features(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Derive per-sequence confidence features from the model's logits.

    Only response tokens contribute (prompt tokens are excluded).

    Args:
        logits:         Tensor of shape (seq_len, vocab_size) — already shifted
                        so that logits[t] predicts token[t+1].  We align with
                        the actual chosen tokens.
        input_ids:      1-D tensor of shape (seq_len,) — token IDs.
        attention_mask: 1-D tensor of shape (seq_len,).

    Returns:
        1-D tensor with 10 confidence-related scalar features.
    """
    resp_idx = _response_token_indices(attention_mask)

    if len(resp_idx) < 2:
        # Too short — return zeros as fallback.
        return torch.zeros(10, dtype=logits.dtype)

    # Align logits with the tokens they predict.
    # For a causal LM, logits[t] predicts input_ids[t+1].
    # So to get the confidence for response token at position p,
    # we need logits[p-1].  We use positions resp_idx - 1.
    logit_positions = resp_idx - 1
    # Ensure all positions are within the logits range.
    valid = logit_positions >= 0
    if valid.sum() < 1:
        return torch.zeros(10, dtype=logits.dtype)

    target_ids = input_ids[resp_idx[valid]]  # the actual response tokens
    resp_logits = logits[logit_positions[valid]]  # logits predicting those tokens

    # Softmax → probabilities
    probs = F.softmax(resp_logits, dim=-1)  # (n_valid, vocab_size)

    # Probability of the token that was actually chosen
    chosen_probs = probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)  # (n_valid,)
    chosen_log_probs = torch.log(chosen_probs + 1e-10)  # (n_valid,)

    # Top-1 and top-2 probabilities → margin
    topk = torch.topk(probs, 2, dim=-1)
    top1_prob = topk.values[:, 0]  # (n_valid,)
    top2_prob = topk.values[:, 1]  # (n_valid,)
    margin = top1_prob - top2_prob  # (n_valid,)

    # Entropy of the full vocabulary distribution at each step
    log_probs_all = F.log_softmax(resp_logits, dim=-1)
    entropy = -(torch.exp(log_probs_all) * log_probs_all).sum(dim=-1)  # (n_valid,)

    # Aggregate into per-sequence scalars
    n_resp = float(len(chosen_probs))
    features = torch.tensor(
        [
            chosen_probs.mean().item(),            # 0: mean chosen-token probability
            chosen_probs.min().item(),             # 1: min chosen-token probability
            chosen_probs.std(dim=0).item(),        # 2: std of chosen-token probability
            entropy.mean().item(),                 # 3: mean per-token entropy
            entropy.max().item(),                  # 4: max per-token entropy
            margin.mean().item(),                  # 5: mean top-1 vs top-2 margin
            torch.exp(-chosen_log_probs.mean()).item(),  # 6: perplexity
            chosen_log_probs.mean().item(),        # 7: mean log-probability
            n_resp / 50.0,                         # 8: normalised response length
            (top1_prob - 0.5).abs().mean().item(), # 9: mean deviation from 0.5
        ],
        dtype=logits.dtype,
    )
    return features


# ---------------------------------------------------------------------------
# Attention-entropy feature (lightweight, last layer only)
# ---------------------------------------------------------------------------

def extract_attention_entropy(
    attentions: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute mean attention entropy from the last layer's attention weights.

    Hallucinated responses tend to have more diffuse (higher-entropy) attention
    patterns — the model is less certain which input tokens to attend to.

    Args:
        attentions: Tensor of shape (n_heads, seq_len, seq_len) — attention
                    weights from the LAST transformer layer only.
        attention_mask: 1-D tensor of shape (seq_len,).

    Returns:
        1-D tensor with 1 scalar: mean attention entropy across heads and
        response tokens.
    """
    resp_idx = _response_token_indices(attention_mask)
    if len(resp_idx) < 2:
        return torch.zeros(1, dtype=attentions.dtype)

    # Attention weights for response tokens (query positions)
    # Shape: (n_heads, n_resp, seq_len)
    resp_attn = attentions.index_select(1, resp_idx.to(attentions.device))

    # Entropy per head per query token
    # H = -sum(p * log(p))
    attn_log = torch.log(resp_attn + 1e-10)
    head_entropy = -(resp_attn * attn_log).sum(dim=-1)  # (n_heads, n_resp)

    # Mean across heads and tokens
    mean_entropy = head_entropy.mean()

    return torch.tensor([mean_entropy.item()], dtype=attentions.dtype)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
    logits: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    attentions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append logits/attention features.

    Args:
        hidden_states:  Tensor of shape (n_layers, seq_len, hidden_dim).
        attention_mask: 1-D tensor of shape (seq_len,).
        use_geometric:  Legacy flag — no longer used (logits replace geometric).
        logits:         Optional tensor of shape (seq_len, vocab_size).
        input_ids:      Optional tensor of shape (seq_len,).
        attentions:     Optional tensor of shape (n_heads, seq_len, seq_len)
                        from the LAST transformer layer only.

    Returns:
        1-D feature tensor.  Dimensionality:
          - 2688 if only hidden states (backward compatible)
          - 2699 if hidden states + logits (10 + 1 extra dims)
    """
    features: list[torch.Tensor] = [aggregate(hidden_states, attention_mask)]

    if logits is not None and input_ids is not None:
        logit_feats = extract_logit_features(logits, input_ids, attention_mask)
        features.append(logit_feats)

    if attentions is not None:
        attn_feats = extract_attention_entropy(attentions, attention_mask)
        features.append(attn_feats)

    return torch.cat(features, dim=0)
