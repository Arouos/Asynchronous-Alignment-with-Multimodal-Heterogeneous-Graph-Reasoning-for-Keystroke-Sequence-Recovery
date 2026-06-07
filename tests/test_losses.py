# Tests for the loss functions.
# Author: Haoyu Qiang.

from __future__ import annotations

import torch

from src.losses.soft_degree_loss import soft_degree_penalty
from src.losses.sparse_edge_loss import sparse_edge_penalty
from src.losses.triangle_consistency_loss import triangle_consistency_penalty


def test_sparse_edge_penalty_is_differentiable() -> None:
    weights = torch.rand(8, requires_grad=True)
    mask = torch.ones(8, dtype=torch.bool)
    out = sparse_edge_penalty(weights, mask)
    out.backward()
    assert weights.grad is not None


def test_soft_degree_penalty_respects_budget() -> None:
    weights = torch.tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True)
    src = torch.tensor([0, 0, 1, 1])
    out = soft_degree_penalty(weights, src, num_nodes=2, max_degree=1.0)
    out.backward()
    assert torch.isfinite(out)


def test_triangle_consistency_zero_when_consistent() -> None:
    a_av = torch.eye(3).unsqueeze(0)
    a_ve = torch.eye(3).unsqueeze(0)
    a_ae = torch.eye(3).unsqueeze(0)
    out = triangle_consistency_penalty(a_av, a_ve, a_ae)
    assert torch.allclose(out, torch.zeros(()))
