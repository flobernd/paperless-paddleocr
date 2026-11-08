"""Regression: ``_parse_hocr_words`` must parse engine-written hOCR.

The multi-language merge reads per-language hOCR back through
``_parse_hocr_words``. That path had no test coverage, which let an invalid
``lxml`` HTMLParser keyword argument (``resolve_entities``) sit unnoticed
until newer lxml started rejecting it with ``TypeError``. Pin the contract:
words written through ``_merged_page`` round-trip out with their box and
confidence, and an escaped entity decodes back to its character.
"""

from __future__ import annotations

from paperless_paddleocr.ocrmypdf_plugin import _merged_page, _parse_hocr_words
from paperless_paddleocr.paddle_engine.hocr import Word, write_document
from paperless_paddleocr.paddle_engine.layout import reading_blocks


def test_parse_hocr_words_roundtrip(tmp_path):
    out = tmp_path / "p.hocr"
    words = [
        Word("hallo", (10, 10, 90, 40), 88),
        Word("R&D", (100, 10, 200, 40), 91),  # '&' is written as &amp;
    ]
    blocks = reading_blocks(words, bbox_of=lambda w: w.box)
    page = _merged_page(blocks, 400, 300, hocr_lang="german")
    write_document(page, out, tmp_path / "p.txt", sidecar="")

    parsed = _parse_hocr_words(out)

    assert [(w.text, w.box, w.confidence) for w in parsed] == [
        ("hallo", (10, 10, 90, 40), 88),
        ("R&D", (100, 10, 200, 40), 91),
    ]
