"""Regression tests for Finding 14: consistent, safe hOCR ``lang``.

The single-language path (``paddle_engine``) and the merged multi-language
path (``ocrmypdf_plugin._write_merged_hocr``) must emit an *identical*,
standards-valid ``lang`` attribute for the same input language, and a
hostile language token must never break out of the attribute (Finding 11).
"""

from __future__ import annotations

from lxml import html

from paperless_paddleocr.languages import to_hocr_lang
from paperless_paddleocr.ocrmypdf_plugin import _Word, _write_merged_hocr


def test_to_hocr_lang_known_codes():
    assert to_hocr_lang("en") == "en"
    assert to_hocr_lang("german") == "de"
    assert to_hocr_lang("ch") == "zh"
    assert to_hocr_lang("chinese_cht") == "zh"
    assert to_hocr_lang("japan") == "ja"


def test_to_hocr_lang_is_case_insensitive_and_trims():
    assert to_hocr_lang("  German ") == "de"
    assert to_hocr_lang("CH") == "zh"


def test_to_hocr_lang_unknown_and_multilingual_are_undetermined():
    assert to_hocr_lang(None) == "und"
    assert to_hocr_lang("") == "und"
    assert to_hocr_lang("ml") == "und"
    assert to_hocr_lang("latin") == "und"
    assert to_hocr_lang("klingon") == "und"


def _pars(hocr_path):
    tree = html.parse(str(hocr_path))
    return tree.findall('.//{*}p[@class="ocr_par"]')


def test_merged_hocr_normalises_paddle_code_to_bcp47(tmp_path):
    out = tmp_path / "merged.hocr"
    words = [_Word("hallo", 0, 0, 10, 10, 90), _Word("welt", 12, 0, 22, 10, 88)]

    # Callers pass a PaddleOCR code ("german"); output must be BCP-47 "de",
    # exactly what the single-language path (to_hocr_lang) would produce.
    _write_merged_hocr(out, words, 100, 100, hocr_lang="german")

    pars = _pars(out)
    assert pars, "expected at least one ocr_par"
    assert {p.get("lang") for p in pars} == {"de"}
    assert {p.get("lang") for p in pars} == {to_hocr_lang("german")}


def test_merged_hocr_neutralises_attribute_injection(tmp_path):
    out = tmp_path / "evil.hocr"
    words = [_Word("x", 0, 0, 5, 5, 50)]

    # A hostile token must not break out of lang="…": normalisation maps the
    # unknown value to "und" and escaping is applied belt-and-suspenders.
    _write_merged_hocr(
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
