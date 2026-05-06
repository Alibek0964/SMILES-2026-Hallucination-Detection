"""
aggregation.py — Token aggregation strategy and feature extraction.

This implementation uses two key insights from the truthfulness-probing literature
(Azaria & Mitchell, "The Internal State of an LLM Knows When It's Lying", 2023;
Burns et al., "Discovering Latent Knowledge in Language Models Without
Supervision", 2022):

  1. Mid-to-late transformer layers (~60-95% depth) encode factuality signals
     better than the final layer alone. We therefore concatenate features from
     four selected layers spanning that depth band, plus the final layer.

  2. Mean-pooling over real (non-padding) tokens is more stable than the
     last-token-only readout — the latter captures only EOS/special-token
     state and discards information distributed across the response.

For Qwen2.5-0.5B the hidden-states tuple has 25 elements (1 embedding + 24
transformer layers), each shaped (seq_len, hidden_dim=896).
"""

from __future__ import annotations

import torch


# Layer indices into hidden_states (length 25 for Qwen2.5-0.5B: embedding + 24
# transformer layers). Negative indexing keeps the code robust to model size
# changes. Selection rationale: ~50%, ~66%, ~83%, ~100% depth — a small set
# spanning the band where truthfulness signals tend to peak.
SELECTED_LAYERS: tuple[int, ...] = (-13, -9, -5, -1)


def _real_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return integer indices of non-padding tokens."""
    return attention_mask.nonzero(as_tuple=False).flatten()


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token, per-layer hidden states into a single feature vector.

    Strategy:
      For each layer in SELECTED_LAYERS, mean-pool over real tokens, then
      concatenate. We also append the last-token vector of the final layer —
      the original baseline signal — so the probe sees both the global
      response representation and the terminal state.

    Args:
        hidden_states:  Tensor of shape (n_layers, seq_len, hidden_dim).
        attention_mask: 1-D tensor of shape (seq_len,) with 1 for real tokens.

    Returns:
        1-D feature tensor of shape ((len(SELECTED_LAYERS) + 1) * hidden_dim,)
        i.e. 5 * 896 = 4480 for Qwen2.5-0.5B.
    """
    real_idx = _real_token_indices(attention_mask)
    last_pos = int(real_idx[-1].item())

    pooled_layers: list[torch.Tensor] = []
    for layer_idx in SELECTED_LAYERS:
        layer = hidden_states[layer_idx]               # (seq_len, hidden_dim)
        real_tokens = layer.index_select(0, real_idx)  # (n_real, hidden_dim)
        pooled_layers.append(real_tokens.mean(dim=0))  # (hidden_dim,)

    # Original signal: last real token of the last layer.
    last_token_feat = hidden_states[-1][last_pos]      # (hidden_dim,)
    pooled_layers.append(last_token_feat)

    return torch.cat(pooled_layers, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Cheap hand-crafted features describing the geometry of activations.

    Intuition: hallucinated and truthful responses often differ in how the
    representation evolves through the stack. Specifically:

      - Layer-wise mean / max norms: how "loud" each layer is.
      - Per-layer token-spread (std across tokens): high spread can signal
        heterogeneous / uncertain content.
      - Inter-layer cosine similarity: representation drift between adjacent
        layers — large drift can correlate with model uncertainty.
      - Sequence length: hallucinations are sometimes longer than truthful
        answers; cheap and worth including.

    All features are computed on real (non-padding) tokens only.
    Total dimensionality: 3 * n_layers + (n_layers - 1) + 1 = 100 for Qwen2.5-0.5B.
    """
    real_idx = _real_token_indices(attention_mask)

    # (n_layers, n_real, hidden_dim) — slice out only real tokens.
    real_states = hidden_states.index_select(1, real_idx)

    # Per-layer norms of token vectors. Shape: (n_layers, n_real).
    token_norms = real_states.norm(dim=-1)
    mean_norms = token_norms.mean(dim=-1)              # (n_layers,)
    max_norms = token_norms.max(dim=-1).values         # (n_layers,)

    # Per-layer std of token activations (averaged across hidden dim).
    # High std ≈ heterogeneous representations across the response.
    token_std = real_states.std(dim=1).mean(dim=-1)    # (n_layers,)

    # Mean per-layer pooled vector for cosine-similarity computation.
    layer_means = real_states.mean(dim=1)              # (n_layers, hidden_dim)
    layer_cos = torch.nn.functional.cosine_similarity(
        layer_means[:-1], layer_means[1:], dim=-1
    )                                                  # (n_layers - 1,)

    # Sequence length scaled to be O(1).
    seq_len_feat = torch.tensor(
        [len(real_idx) / 100.0], dtype=mean_norms.dtype
    )

    return torch.cat(
        [
            mean_norms,    # (n_layers,)
            max_norms,     # (n_layers,)
            token_std,     # (n_layers,)
            layer_cos,     # (n_layers - 1,)
            seq_len_feat,  # (1,)
        ],
        dim=0,
    )


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
