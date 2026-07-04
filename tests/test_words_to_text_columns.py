"""_words_to_text must read columns left-to-right, lines top-to-bottom."""

from __future__ import annotations

from paperless_paddleocr.ocrmypdf_plugin import _Word, _words_to_text


def test_words_to_text_single_column_top_to_bottom():
    words = [
        _Word("beta", 10, 40, 90, 60, 90),
        _Word("alpha", 10, 0, 90, 20, 90),
    ]
    assert _words_to_text(words) == "alpha\nbeta"


def test_words_to_text_reads_left_column_fully_before_right():
    words = [
        _Word("left1", 0, 0, 160, 20, 90),
        _Word("right1", 240, 0, 400, 20, 90),
        _Word("left2", 0, 30, 160, 50, 90),
        _Word("right2", 240, 30, 400, 50, 90),
        _Word("left3", 0, 60, 160, 80, 90),
        _Word("right3", 240, 60, 400, 80, 90),
    ]
    assert _words_to_text(words) == ("left1\nleft2\nleft3\n\nright1\nright2\nright3")


def test_words_to_text_empty_input():
    assert _words_to_text([]) == ""


def test_words_to_text_letter_header_reads_blockwise():
    words = []
    for i in range(3):
        y0, y1 = i * 30, i * 30 + 20
        words += [
            _Word(f"sender{i + 1}", 0, y0, 160, y1, 90),
            _Word(f"recipient{i + 1}", 240, y0, 400, y1, 90),
        ]
    words += [
        _Word("body1", 0, 130, 400, 150, 90),
        _Word("body2", 0, 160, 400, 180, 90),
        _Word("body3", 0, 190, 400, 210, 90),
    ]
    assert _words_to_text(words) == (
        "sender1\nsender2\nsender3\n\nrecipient1\nrecipient2\nrecipient3\n\nbody1\nbody2\nbody3"
    )
