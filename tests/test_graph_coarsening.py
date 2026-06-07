# Tests for the graph coarsening soft assignment.
# Author: Haoyu Qiang.

from __future__ import annotations

import torch

from src.models.aam_hg.graph_coarsening import GraphCoarsening


def test_assignment_columns_sum_to_one_when_local_window_active() -> None:
    coarsener = GraphCoarsening(node_dim=8, anchor_count=3, anchor_window=10.0)
    z = torch.randn(1, 6, 8)
    times = torch.linspace(0, 6, 6).unsqueeze(0)
    actionness = torch.rand(1, 6)
    cross = torch.rand(1, 6)
    c, s, tau = coarsener(z, times, actionness, cross)
    assert c.shape == (1, 6, 3)
    column_sums = c.sum(dim=1).squeeze(0)
    assert torch.all(column_sums >= 0)
    assert s.shape == (1, 3, 8)
    assert tau.shape == (1, 3)
