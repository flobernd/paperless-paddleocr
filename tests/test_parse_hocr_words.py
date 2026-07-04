"""Regression: ``_parse_hocr_words`` must parse engine-written hOCR.

The multi-language merge reads per-language hOCR back through
``_parse_hocr_words``. That path had no test coverage, which let an invalid
``lxml`` HTMLParser keyword argument (``resolve_entities``) sit unnoticed
until newer lxml started rejecting it with ``TypeError``. Pin the contract:
words written by ``_write_merged_hocr`` round-trip out with their box and
confidence, and an escaped entity decodes back to its character.
"""

from __future__ import annotations

from paperless_paddleocr.ocrmypdf_plugin import (
    _parse_hocr_words,
    _Word,
    _write_merged_hocr,
)


def test_parse_hocr_words_roundtrip(tmp_path):
    out = tmp_path / "p.hocr"
    words = [
        _Word("hallo", 10, 10, 90, 40, 88),
        _Word("R&D", 100, 10, 200, 40, 91),  # '&' is written as &amp;
    ]
    _write_merged_hocr(out, words, 400, 300, hocr_lang="german")

    parsed = _parse_hocr_words(out)

    assert [(w.text, w.x0, w.y0, w.x1, w.y1, w.conf) for w in parsed] == [
        ("hallo", 10, 10, 90, 40, 88),
        ("R&D", 100, 10, 200, 40, 91),
    ]
