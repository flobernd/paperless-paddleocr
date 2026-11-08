"""_blocks_to_text must read columns left-to-right, lines top-to-bottom."""

from __future__ import annotations

from paperless_paddleocr.ocrmypdf_plugin import _blocks_to_text
from paperless_paddleocr.paddle_engine.hocr import Word
from paperless_paddleocr.paddle_engine.layout import reading_blocks


def _text(words: list[Word]) -> str:
    return _blocks_to_text(reading_blocks(words, bbox_of=lambda w: w.box))


def test_words_to_text_single_column_top_to_bottom():
    words = [
        Word("beta", (10, 40, 90, 60), 90),
        Word("alpha", (10, 0, 90, 20), 90),
    ]
    assert _text(words) == "alpha\nbeta"


def test_words_to_text_reads_left_column_fully_before_right():
    words = [
        Word("left1", (0, 0, 160, 20), 90),
        Word("right1", (240, 0, 400, 20), 90),
        Word("left2", (0, 30, 160, 50), 90),
        Word("right2", (240, 30, 400, 50), 90),
        Word("left3", (0, 60, 160, 80), 90),
        Word("right3", (240, 60, 400, 80), 90),
    ]
    assert _text(words) == ("left1\nleft2\nleft3\n\nright1\nright2\nright3")


def test_words_to_text_empty_input():
    assert _text([]) == ""


def test_words_to_text_letter_header_reads_blockwise():
    words = []
    for i in range(3):
        y0, y1 = i * 30, i * 30 + 20
        words += [
            Word(f"sender{i + 1}", (0, y0, 160, y1), 90),
            Word(f"recipient{i + 1}", (240, y0, 400, y1), 90),
        ]
    words += [
        Word("body1", (0, 130, 400, 150), 90),
        Word("body2", (0, 160, 400, 180), 90),
        Word("body3", (0, 190, 400, 210), 90),
    ]
    assert _text(words) == (
        "sender1\nsender2\nsender3\n\nrecipient1\nrecipient2\nrecipient3\n\nbody1\nbody2\nbody3"
    )
