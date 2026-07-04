"""Every code the plugin can hand to PaddleOCR(lang=...) must be valid in 3.x.

PADDLE_VALID_LANGS is vendored from paddleocr 3.7.0; these tests pin the
mapping against it so an invalid target (like the old "spa" -> "spanish")
can never ship again.
"""

from __future__ import annotations

from paperless_paddleocr.languages import (
    PADDLE_VALID_LANGS,
    TESSERACT_TO_PADDLE,
    normalize_paddle_lang,
    resolve_paddle_languages,
)


def test_all_mapped_targets_are_valid_paddle3_codes():
    invalid = {t: p for t, p in TESSERACT_TO_PADDLE.items() if p not in PADDLE_VALID_LANGS}
    assert invalid == {}, f"mapped to codes PaddleOCR 3.x rejects: {invalid}"


def test_spanish_maps_to_es():
    assert TESSERACT_TO_PADDLE["spa"] == "es"
    assert resolve_paddle_languages("spa", None) == ["es"]


def test_hebrew_is_dropped_with_fallback():
    # PaddleOCR 3.x has no Hebrew model; the code must be skipped, not crash later.
    assert "heb" not in TESSERACT_TO_PADDLE
    assert resolve_paddle_languages("heb", None) == ["en"]


def test_common_european_codes_are_mapped():
    expected = {
        "slk": "sk",
        "slv": "sl",
        "hrv": "hr",
        "bul": "bg",
        "cat": "ca",
        "est": "et",
        "lav": "lv",
        "lit": "lt",
        "srp": "rs_cyrillic",
        "srp_latn": "rs_latin",
        "ind": "id",
        "msa": "ms",
        "fas": "fa",
        "urd": "ur",
        "kaz": "kk",
    }
    for tess, paddle in expected.items():
        assert TESSERACT_TO_PADDLE.get(tess) == paddle


def test_script_bundles_normalise_to_valid_codes():
    assert normalize_paddle_lang("latin") == "la"
    assert normalize_paddle_lang("arabic") == "ar"
    assert normalize_paddle_lang("cyrillic") == "rs_cyrillic"
    assert normalize_paddle_lang("devanagari") == "hi"
    for bundle in ("latin", "arabic", "cyrillic", "devanagari"):
        assert normalize_paddle_lang(bundle) in PADDLE_VALID_LANGS


def test_ml_normalises_to_none_for_default_multilingual_model():
    assert normalize_paddle_lang("ml") is None


def test_regular_codes_pass_through_trimmed():
    assert normalize_paddle_lang(" German ") == "german"
    assert normalize_paddle_lang("es") == "es"
