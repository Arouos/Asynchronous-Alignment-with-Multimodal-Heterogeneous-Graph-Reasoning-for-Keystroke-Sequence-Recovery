# Tests for cross-modal matching and sparsification.
# Author: Haoyu Qiang.

from __future__ import annotations

import torch

from src.models.aam_hg.sparsification import sparsify_cross_modal


def test_sparsifier_respects_max_degree() -> None:
    soft = torch.rand(2, 4, 6)
    sparse, mask = sparsify_cross_modal(soft, max_degree=3)
    assert mask.sum(dim=-1).max().item() == 3
    assert (sparse[~mask] == 0).all()


def test_sparsifier_zero_degree() -> None:
    soft = torch.rand(1, 2, 3)
    sparse, mask = sparsify_cross_modal(soft, max_degree=0)
    assert sparse.sum() == 0
    assert mask.sum() == 0
