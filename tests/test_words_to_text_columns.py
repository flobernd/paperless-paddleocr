"""_blocks_to_text must read columns left-to-right, lines top-to-bottom."""

from __future__ import annotations

from paperless_paddleocr.ocrmypdf_plugin import _blocks_to_text
from paperless_paddleocr.paddle_engine.hocr import Word
from paperless_paddleocr.paddle_engine.layout import reading_blocks


def _text(words: list[Word]) -> str:
    return _blocks_to_text(reading_blocks(words, bbox_of=lambda w: w.box))


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
