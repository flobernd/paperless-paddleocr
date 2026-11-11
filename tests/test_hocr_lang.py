"""Regression tests for Finding 14: consistent, safe hOCR ``lang``.

The single-language path (``paddle_engine``) and the merged multi-language
path (``paddle_engine.merge._merged_page``) must emit an *identical*,
standards-valid ``lang`` attribute for the same input language, and a
hostile language token must never break out of the attribute (Finding 11).
"""

from __future__ import annotations

from lxml import html

from paperless_paddleocr.languages import to_hocr_lang
from paperless_paddleocr.paddle_engine.hocr import Word, write_document
from paperless_paddleocr.paddle_engine.layout import reading_blocks
from paperless_paddleocr.paddle_engine.merge import _merged_page


def test_to_hocr_lang_is_case_insensitive_and_trims():
    assert to_hocr_lang("  German ") == "de"
    assert to_hocr_lang("CH") == "zh"


def test_to_hocr_lang_unknown_and_multilingual_are_undetermined():
    assert to_hocr_lang(None) == "und"
    assert to_hocr_lang("") == "und"
    assert to_hocr_lang("ml") == "und"
    assert to_hocr_lang("latin") == "und"
    assert to_hocr_lang("klingon") == "und"


def test_valid_two_letter_paddle_code_renders_as_itself():
    # Bashkir in PaddleOCR and in ISO 639-1, and no Tesseract code maps to it.
    assert to_hocr_lang("ba") == "ba"


def test_valid_paddle_code_longer_than_two_letters_is_undetermined():
    # PaddleOCR's mah is Magahi. ISO 639-2 mah is Marshallese.
    assert to_hocr_lang("mah") == "und"


def _write_merged(out, words, page_w, page_h, *, hocr_lang):
    blocks = reading_blocks(words, bbox_of=lambda w: w.box)
    page = _merged_page(blocks, page_w, page_h, hocr_lang=hocr_lang)
    write_document(page, out, out.with_suffix(".txt"), sidecar="")


def _pars(hocr_path):
    tree = html.parse(str(hocr_path))
    return tree.findall('.//{*}p[@class="ocr_par"]')


def test_merged_hocr_normalises_paddle_code_to_bcp47(tmp_path):
    out = tmp_path / "merged.hocr"
    words = [Word("hallo", (0, 0, 10, 10), 90), Word("welt", (12, 0, 22, 10), 88)]

    # Callers pass a PaddleOCR code ("german"), and the output must be BCP-47 "de".
    _write_merged(out, words, 100, 100, hocr_lang="german")

    pars = _pars(out)
    assert pars, "expected at least one ocr_par"
    assert {p.get("lang") for p in pars} == {"de"}


def test_merged_hocr_neutralises_attribute_injection(tmp_path):
    out = tmp_path / "evil.hocr"
    words = [Word("x", (0, 0, 5, 5), 50)]

    # A hostile token must not break out of lang="…": normalisation maps the
    # unknown value to "und" and escaping is applied belt-and-suspenders.
    _write_merged(
        out,
        words,
        50,
        50,
        hocr_lang='en" onload="alert(1)',
    )

    raw = out.read_text(encoding="utf-8")
    assert 'onload="alert(1)"' not in raw

    pars = _pars(out)
    assert pars
    assert {p.get("lang") for p in pars} == {"und"}
