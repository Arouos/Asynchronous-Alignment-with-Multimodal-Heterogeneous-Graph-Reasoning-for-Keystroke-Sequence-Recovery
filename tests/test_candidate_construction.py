# Tests for multi-scale candidate construction.
# Author: Haoyu Qiang.

from __future__ import annotations

import torch

from src.models.aam_hg.scale_aggregation import ScaleAdaptiveAggregator


def test_scale_softmax_sums_to_one() -> None:
    aggregator = ScaleAdaptiveAggregator(modalities=["acoustic"], feature_dim=8, scale_count=4, scale_embed_dim=4)
    features = torch.randn(2, 3, 4, 8)
    scale_indices = torch.arange(4)
    out = aggregator("acoustic", features, scale_indices)
    assert out.shape == (2, 3, 8)


def test_aggregator_handles_single_scale() -> None:
    aggregator = ScaleAdaptiveAggregator(modalities=["vibration"], feature_dim=6, scale_count=1, scale_embed_dim=3)
    features = torch.randn(1, 2, 1, 6)
    out = aggregator("vibration", features, torch.tensor([0]))
    assert torch.allclose(out, features.squeeze(-2))
