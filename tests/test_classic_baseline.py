"""Unit test for classic._region_baseline straight-baseline fitting."""

from __future__ import annotations

from paperless_paddleocr.paddle_engine.classic import _region_baseline
from paperless_paddleocr.paddle_engine.hocr import Word


def test_region_baseline_flat_line_sits_on_box_bottom():
    words = [Word("a", (0, 10, 40, 30), 90), Word("b", (50, 10, 90, 30), 90)]
    slope, constant = _region_baseline(words, (0, 10, 90, 30))
    assert abs(slope) < 1e-9
    assert abs(constant) < 1e-9


def test_region_baseline_slopes_with_rising_word_bottoms():
    # word bottoms climb toward the right: 30, 26, 22
    words = [
        Word("a", (0, 10, 40, 30), 90),
        Word("b", (50, 6, 90, 26), 90),
        Word("c", (100, 2, 140, 22), 90),
    ]
    slope, _constant = _region_baseline(words, (0, 2, 140, 30))
    assert slope < 0  # baseline y decreases as x increases


def test_region_baseline_no_words_is_flat():
    assert _region_baseline([], (0, 0, 100, 20)) == (0.0, 0.0)
