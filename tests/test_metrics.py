# Tests for the evaluation metrics.
# Author: Haoyu Qiang.

from __future__ import annotations

import torch

from src.metrics.character_error_rate import character_error_rate
from src.metrics.sequence_recovery_accuracy import sequence_recovery_accuracy
from src.metrics.topk_accuracy import topk_accuracy


def test_cer_exact_match() -> None:
    assert character_error_rate([[1, 2, 3]], [[1, 2, 3]]) == 0.0


def test_cer_substitution_cost() -> None:
    assert character_error_rate([[1, 2, 3]], [[1, 2, 4]]) == 1.0 / 3.0


def test_sra_perfect() -> None:
    assert sequence_recovery_accuracy([[1, 2, 3]], [[1, 2, 3]]) == 1.0


def test_topk_accuracy() -> None:
    logits = torch.tensor([[0.1, 0.9], [0.4, 0.6]])
    targets = torch.tensor([1, 0])
    assert topk_accuracy(logits, targets, k=1) == 0.5
    assert topk_accuracy(logits, targets, k=2) == 1.0
