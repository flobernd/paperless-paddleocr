"""The engine must emit hOCR blocks and sidecar in banded reading order.

build_page is stubbed with a letter-shaped page (two-column header above a
full-width body) whose blocks arrive interleaved, the way PaddleOCR reports
regions.
"""

from __future__ import annotations

import types

from PIL import Image

from paperless_paddleocr.paddle_engine import classic
from paperless_paddleocr.paddle_engine.engine import PaddleOCREngine
from paperless_paddleocr.paddle_engine.hocr import Block, Line, Page, Word


def _line_block(text: str, box) -> Block:
    words = [Word(text, box, 90)]
    return Block(box=box, lines=[Line(box=box, confidence=90, text=text, words=words)])


def _letter_page() -> Page:
    page = Page(width=400, height=300, lang="en", ocr_system="test")
    for i in range(3):
        y0, y1 = i * 30, i * 30 + 20
        page.blocks.append(_line_block(f"sender{i + 1}", (0, y0, 160, y1)))
        page.blocks.append(_line_block(f"recipient{i + 1}", (240, y0, 400, y1)))
    for i in range(3):
        y0, y1 = 130 + i * 30, 150 + i * 30
        page.blocks.append(_line_block(f"body{i + 1}", (0, y0, 400, y1)))
    return page


def test_generate_hocr_orders_blocks_and_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(classic, "build_page", lambda input_file, options: _letter_page())
    img = tmp_path / "page.png"
    Image.new("RGB", (400, 300), "white").save(img)
    out_hocr, out_text = tmp_path / "o.hocr", tmp_path / "o.txt"
    options = types.SimpleNamespace(languages=["en"], paddle_engine="classic-cpu")

    PaddleOCREngine._generate_hocr_classic(img, out_hocr, out_text, options)

    assert out_text.read_text(encoding="utf-8") == (
        "sender1\nsender2\nsender3\n\nrecipient1\nrecipient2\nrecipient3\n\nbody1\nbody2\nbody3"
    )
    hocr = out_hocr.read_text(encoding="utf-8")
    assert hocr.index("sender3") < hocr.index("recipient1") < hocr.index("body1")
