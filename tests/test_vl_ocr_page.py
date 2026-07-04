"""_ocr_page must never emit zero-height line boxes.

The old integer division (block_height // line_count) collapsed every line
of a block shorter in pixels than its line count.
"""

from __future__ import annotations

from paperless_paddleocr.paddle_engine.hocr import Page
from paperless_paddleocr.paddle_engine.vl import _ocr_page


def _page() -> Page:
    return Page(width=100, height=50, lang="en", ocr_system="test")


def test_ocr_page_line_boxes_are_never_degenerate():
    page = _page()
    block = {
        "block_label": "text",
        "block_content": "one\ntwo\nthree",
        "block_bbox": [0, 10, 100, 12],  # 2 px tall, 3 text lines
    }
    _ocr_page([block], page)
    lines = [line for blk in page.blocks for line in blk.lines]
    assert len(lines) == 3
    for line in lines:
        assert line.box[3] > line.box[1]


def test_ocr_page_line_boxes_partition_the_block_height():
    page = _page()
    block = {
        "block_label": "text",
        "block_content": "one\ntwo\nthree",
        "block_bbox": [0, 0, 100, 30],
    }
    _ocr_page([block], page)
    boxes = [line.box for blk in page.blocks for line in blk.lines]
    assert boxes == [(0, 0, 100, 10), (0, 10, 100, 20), (0, 20, 100, 30)]


def test_out_of_bounds_word_boxes_are_logged(caplog):
    import logging

    from paperless_paddleocr.paddle_engine.hocr import Block, Line, Page, Word
    from paperless_paddleocr.paddle_engine.vl import _warn_out_of_bounds

    page = Page(width=100, height=50, lang="en", ocr_system="test")
    word = Word("x", (0, 0, 300, 40), 95)  # x1 far beyond width=100
    page.blocks.append(Block(box=word.box, lines=[Line(word.box, 95, "x", [word])]))
    with caplog.at_level(logging.WARNING, logger="paperless.paddleocr.vl"):
        _warn_out_of_bounds(page)
    assert "outside the 100x50 page" in caplog.text
