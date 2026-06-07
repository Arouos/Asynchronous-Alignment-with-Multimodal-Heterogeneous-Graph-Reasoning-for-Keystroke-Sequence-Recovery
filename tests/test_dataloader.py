# Tests for the variable-length collate function.
# Author: Haoyu Qiang.

from __future__ import annotations

import torch

from src.data.collate import collate_keystroke_batch


def _toy_item(length: int, digits: list[int]) -> dict:
    return {
        "signals": {"acoustic": torch.arange(length).float()},
        "timestamps": torch.tensor([0.0]),
        "digits": torch.tensor(digits, dtype=torch.long),
        "sample_rate": {"acoustic": 16000},
        "meta": {},
    }


def test_collate_pads_signals_and_digits() -> None:
    batch = collate_keystroke_batch([_toy_item(3, [1, 2]), _toy_item(5, [3, 4, 5])])
    assert batch["signals"]["acoustic"].shape == (2, 5)
    assert batch["digits"].shape == (2, 3)
    assert torch.equal(batch["digit_lengths"], torch.tensor([2, 3]))
    assert torch.equal(batch["digits"][0], torch.tensor([1, 2, 0]))
