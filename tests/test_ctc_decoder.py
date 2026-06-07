# Tests for the CTC decoders.
# Author: Haoyu Qiang.

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.evaluation.beam_search import beam_search_decode
from src.evaluation.ctc_decoder import greedy_ctc_decode


def test_greedy_collapses_repeats_and_blanks() -> None:
    logits = torch.tensor([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]])
    log_probs = F.log_softmax(logits, dim=-1)
    decoded = greedy_ctc_decode(log_probs, blank=2)
    assert decoded == [[0, 1]]


def test_beam_search_returns_valid_sequence() -> None:
    logits = torch.tensor([[[0.6, 0.3, 0.1]], [[0.1, 0.7, 0.2]]])
    log_probs = F.log_softmax(logits, dim=-1)
    decoded = beam_search_decode(log_probs, blank=2, beam_width=3)
    assert isinstance(decoded, list) and decoded
