"""
aggregation.py — Token aggregation strategy and feature extraction (v2).

This implementation closely follows the SAPLMA approach
(Azaria & Mitchell, "The Internal State of an LLM Knows When It's Lying", 2023):
the truthfulness signal is read out from the **last-token** hidden state at
several mid-to-late layers of the transformer.

We deliberately avoid mean-pooling over the full prompt+response sequence —
in this dataset the assistant response is short (often a single sentence)
while the prompt is long (typical context length 200-400 tokens). Mean-pool
dilutes the response-specific signal in a sea of prompt tokens.

For Qwen2.5-0.5B the hidden-states tuple has 25 elements (1 embedding + 24
transformer layers), each shaped (seq_len, hidden_dim=896).
"""

from __future__ import annotations

import torch


# Last-token readouts at three mid-to-late layers. Three (rather than five) keeps
# the effective parameter count of the downstream linear probe well below the
# number of training samples (~482).
SELECTED_LAYERS: tuple[int, ...] = (-9, -5, -1)
"""Selection rationale: ~66%, ~83%, ~100% depth — the band where the
truthfulness signal is strongest in 24-layer decoder-only models."""


def _last_real_position(attention_mask: torch.Tensor) -> int:
    """Return the index of the last non-padding token."""
    real_idx = attention_mask.nonzero(as_tuple=False).flatten()
    return int(real_idx[-1].item())


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Concatenate the last-token hidden state across SELECTED_LAYERS.

    Args:
        hidden_states:  Tensor of shape (n_layers, seq_len, hidden_dim).
        attention_mask: 1-D tensor of shape (seq_len,).

    Returns:
        1-D feature tensor of shape (len(SELECTED_LAYERS) * hidden_dim,)
        i.e. 3 * 896 = 2688 for Qwen2.5-0.5B.
    """
    last_pos = _last_real_position(attention_mask)
    pooled = [hidden_states[layer_idx][last_pos] for layer_idx in SELECTED_LAYERS]
    return torch.cat(pooled, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Minimal hand-crafted features.

    We keep only the two cheapest, highest-signal scalars:

      * Sequence length (responses to hallucinated questions tend to differ
        in length from truthful ones in this dataset).
      * Last-token L2 norm at the final layer (a coarse confidence proxy —
        smaller norm often correlates with low-confidence generations).

    Larger geometric feature sets were tried and discarded: they consistently
    inflated the train-test gap without improving validation AUROC.
    """
    real_idx = attention_mask.nonzero(as_tuple=False).flatten().to(hidden_states.device)
    last_pos = int(real_idx[-1].item())

    seq_len_feat = torch.tensor(
        [len(real_idx) / 100.0],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    last_token_norm = hidden_states[-1][last_pos].norm().unsqueeze(0)

    return torch.cat([seq_len_feat, last_token_norm], dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features."""
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
