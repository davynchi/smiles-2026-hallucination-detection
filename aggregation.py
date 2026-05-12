"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).
"""

from __future__ import annotations

import torch


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start_pos: int | None = None,
) -> torch.Tensor:
    """Convert per-token hidden states into a flat feature vector.

    Strategy:
      - Layers 18-22: last 2 real tokens concatenated, max-pooled → (1792,)
      - Layer 23:     last 2 real tokens concatenated directly     → (1792,)
      - Layer 24:     last 2 real tokens concatenated directly     → (1792,)
      - Cosine distance between layer 23's two last tokens         →    (1,)
      - Cosine distance between layer 24's two last tokens         →    (1,)
      Total: 5378 dims.
    """
    import torch.nn.functional as F

    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(1)
    last2 = (real_positions[-2:] if len(real_positions) >= 2
             else real_positions.repeat(2)[-2:])

    # Layers 18-22: max-pool over last 2 tokens
    pool_features = []
    for layer in hidden_states[18:23]:
        tok_a = layer[last2[0]]
        tok_b = layer[last2[1]]
        pool_features.append(torch.cat([tok_a, tok_b]))       # (1792,)
    part_pool = torch.stack(pool_features).max(dim=0).values   # (1792,)

    # Layer 23 direct
    l23 = hidden_states[23]
    tok_a_23, tok_b_23 = l23[last2[0]], l23[last2[1]]
    part_23 = torch.cat([tok_a_23, tok_b_23])                  # (1792,)

    # Layer 24 direct
    l24 = hidden_states[24]
    tok_a_24, tok_b_24 = l24[last2[0]], l24[last2[1]]
    part_24 = torch.cat([tok_a_24, tok_b_24])                  # (1792,)

    # Cosine distances between the two last tokens within each layer
    cos_23 = 1.0 - F.cosine_similarity(tok_a_23.unsqueeze(0), tok_b_23.unsqueeze(0))
    cos_24 = 1.0 - F.cosine_similarity(tok_a_24.unsqueeze(0), tok_b_24.unsqueeze(0))

    return torch.cat([part_pool, part_23, part_24, cos_23, cos_24])  # (5378,)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features (49 total).

       - 24 values: L2-norm of mean-pooled vector per transformer layer 1-24
       - 23 values: cosine similarity between adjacent layers
       -  1 value : softmax entropy of the last-layer last-token activation
       -  1 value : sequence length
    """
    import torch.nn.functional as F

    device = hidden_states.device
    mask_float = attention_mask.float().unsqueeze(-1).to(device)
    denom = mask_float.sum().clamp(min=1.0)

    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())

    features: list[torch.Tensor] = []

    # 1. Per-layer L2 norms (layers 1-24)
    layer_means = []
    for layer in hidden_states[1:]:
        pooled = (layer * mask_float).sum(dim=0) / denom
        layer_means.append(pooled)
        features.append(pooled.norm().unsqueeze(0))

    # 2. Inter-layer cosine similarities (23 values)
    for i in range(len(layer_means) - 1):
        cos_sim = F.cosine_similarity(
            layer_means[i].unsqueeze(0),
            layer_means[i + 1].unsqueeze(0),
        )
        features.append(cos_sim)

    # 3. Softmax entropy of last-layer last-token activation
    last_vec = hidden_states[-1][last_pos]
    probs = torch.softmax(last_vec.abs(), dim=0)
    entropy = -(probs * (probs + 1e-8).log()).sum()
    features.append(entropy.unsqueeze(0))

    # 4. Sequence length
    features.append(attention_mask.float().sum().unsqueeze(0).to(device))

    return torch.cat(features, dim=0)   # (49,)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
    response_start_pos: int | None = None,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features."""
    agg_features = aggregate(hidden_states, attention_mask, response_start_pos)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
