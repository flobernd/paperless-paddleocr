"""Regression test for ``generate_pdf`` after the ocrmypdf>=17 rewrite.

``generate_pdf`` parses the hOCR with ``HocrParser`` and renders a real
text-only PDF via ``Fpdf2PdfRenderer`` (ocrmypdf 17 removed the old
``ocrmypdf.hocrtransform.HocrTransform``). The renderer is *not* stubbed, so
this test fails loudly if the ocrmypdf hOCR→PDF surface drifts again.
"""

from __future__ import annotations

import types

from PIL import Image

from paperless_paddleocr.paddle_engine import PaddleOCREngine, classic
from paperless_paddleocr.paddle_engine.hocr import Word
from paperless_paddleocr.paddle_engine.layout import reading_blocks
from paperless_paddleocr.paddle_engine.merge import _merged_page


def _tiny_png(path):
    Image.new("RGB", (120, 40), "white").save(path, dpi=(300, 300))


def test_generate_pdf_renders_a_textonly_pdf(tmp_path, monkeypatch):
    img = tmp_path / "page.png"
    _tiny_png(img)
    out_pdf = tmp_path / "out.pdf"
    out_text = tmp_path / "out.txt"

    def fake_build_page(input_file, options):
        words = [
            Word("hallo", (4, 4, 54, 30), 95),
            Word("welt", (60, 4, 110, 30), 93),
        ]
        blocks = reading_blocks(words, bbox_of=lambda w: w.box)
        return _merged_page(blocks, 120, 40, hocr_lang="german")

    monkeypatch.setattr(classic, "build_page", fake_build_page)

    options = types.SimpleNamespace(languages=["german"], paddle_engine="classic-cpu")

    # Call exactly as ocrmypdf's textonly-pdf pipeline does: on the instance.
    engine = PaddleOCREngine()
    engine.generate_pdf(img, out_pdf, out_text, options)

    data = out_pdf.read_bytes()
    assert data.startswith(b"%PDF"), "generate_pdf must emit a real PDF"
    assert data.rstrip().endswith(b"%%EOF"), "PDF must be well-formed"
    assert out_text.read_text(encoding="utf-8") == "hallo welt"
