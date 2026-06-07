# Tests for the relation-aware GNN layer.
# Author: Haoyu Qiang.

from __future__ import annotations

import torch

from src.models.aam_hg.graph_construction import GraphState
from src.models.aam_hg.relation_aware_gnn import RelationAwareGNNLayer
from src.models.aam_hg.temporal_gating import TemporalGating


def _toy_state() -> tuple[GraphState, torch.Tensor]:
    z = torch.randn(1, 4, 8)
    t = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    slices = {"acoustic": slice(0, 2), "vibration": slice(2, 4)}
    intra = {"acoustic": torch.tensor([[[1], [0]]]), "vibration": torch.tensor([[[1], [0]]])}
    cross = {("acoustic", "vibration"): torch.rand(1, 2, 2)}
    masks = {("acoustic", "vibration"): torch.ones(1, 2, 2, dtype=torch.bool)}
    state = GraphState(
        z=z,
        timestamps=t,
        modality_ids=torch.zeros(1, 4, dtype=torch.long),
        scales=torch.zeros(1, 4, dtype=torch.long),
        actionness=torch.rand(1, 4),
        modality_slices=slices,
        intra_edges=intra,
        cross_weights=cross,
        cross_masks=masks,
    )
    scales_embed = torch.randn(1, 4, 4)
    return state, scales_embed


def test_layer_preserves_shape() -> None:
    gating = TemporalGating(node_dim=8, hidden_dim=8, scale_embed_dim=4)
    layer = RelationAwareGNNLayer(node_dim=8, modalities=["acoustic", "vibration"], gating=gating)
    state, scales_embed = _toy_state()
    new_state = layer(state, scales_embed=scales_embed, use_gating=True)
    assert new_state.z.shape == state.z.shape
