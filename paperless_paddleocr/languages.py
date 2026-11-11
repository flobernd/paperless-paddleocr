"""Language code resolution for PaddleOCR.

Paperless stores OCR language as Tesseract-style ISO 639-2/T codes joined by
``+`` (e.g. ``eng+deu``). PaddleOCR uses a different vocabulary of codes
(e.g. ``en``, ``german``, ``ch``, ``japan``) and runs one language at a time.

`resolve_paddle_languages` translates between the two, with one escape hatch:
the user may set ``PAPERLESS_PADDLEOCR_LANGUAGE`` to force native PaddleOCR
codes (including multilingual codes like ``ml`` and ``latin`` that have no
Tesseract analogue).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Final

logger = logging.getLogger("paperless.paddleocr.languages")

# Valid `lang=` values for paddleocr 3.x, vendored from paddleocr 3.7.0
# (`paddleocr/_utils/langs.py` plus the specials handled in
# `paddleocr/_pipelines/ocr.py:_get_ocr_model_names`). Re-sync when the
# paddleocr dependency floor moves. Kept here so the mapping below can be
# validated in CI without installing paddleocr.
_LATIN_LANGS: Final[frozenset[str]] = frozenset(
    {
        "af",
        "az",
        "bs",
        "ca",
        "cs",
        "cy",
        "da",
        "de",
        "es",
        "et",
        "eu",
        "fi",
        "fr",
        "french",
        "ga",
        "german",
        "gl",
        "hr",
        "hu",
        "id",
        "is",
        "it",
        "ku",
        "la",
        "lb",
        "lt",
        "lv",
        "mi",
        "ms",
        "mt",
        "nl",
        "no",
        "oc",
        "pi",
        "pl",
        "pt",
        "qu",
        "rm",
        "ro",
        "rs_latin",
        "sk",
        "sl",
        "sq",
        "sv",
        "sw",
        "tl",
        "tr",
        "uz",
        "vi",
    },
)
_ARABIC_LANGS: Final[frozenset[str]] = frozenset(
    {"ar", "fa", "ug", "ur", "ps", "ku", "sd", "bal"},
)
_ESLAV_LANGS: Final[frozenset[str]] = frozenset({"ru", "be", "uk"})
_CYRILLIC_LANGS: Final[frozenset[str]] = frozenset(
    {
        "ru",
        "rs_cyrillic",
        "be",
        "bg",
        "uk",
        "mn",
        "abq",
        "ady",
        "kbd",
        "ava",
        "dar",
        "inh",
        "che",
        "lbe",
        "lez",
        "tab",
        "kk",
        "ky",
        "tg",
        "mk",
        "tt",
        "cv",
        "ba",
        "mhr",
        "mo",
        "udm",
        "kv",
        "os",
        "bua",
        "xal",
        "tyv",
        "sah",
        "kaa",
    },
)
_DEVANAGARI_LANGS: Final[frozenset[str]] = frozenset(
    {"hi", "mr", "ne", "bh", "mai", "ang", "bho", "mah", "sck", "new", "gom", "sa", "bgc"},
)
_SPECIAL_LANGS: Final[frozenset[str]] = frozenset(
    {"ch", "chinese_cht", "en", "japan", "korean", "th", "el", "te", "ta", "ka"},
)

PADDLE_VALID_LANGS: Final[frozenset[str]] = (
    _LATIN_LANGS
    | _ARABIC_LANGS
    | _ESLAV_LANGS
    | _CYRILLIC_LANGS
    | _DEVANAGARI_LANGS
    | _SPECIAL_LANGS
)

# Tesseract ISO 639-2/T (tessdata naming) -> PaddleOCR 3.x lang code. Every
# target must be a member of PADDLE_VALID_LANGS; tests enforce this.
# Tesseract codes with no PaddleOCR model (e.g. heb) are deliberately absent:
# resolve_paddle_languages logs and skips unmapped codes.
TESSERACT_TO_PADDLE: Final[dict[str, str]] = {
    "afr": "af",
    "ara": "ar",
    "aze": "az",
    "bel": "be",
    "bos": "bs",
    "bul": "bg",
    "cat": "ca",
    "ces": "cs",
    "chi_sim": "ch",
    "chi_tra": "chinese_cht",
    "cym": "cy",
    "dan": "da",
    "deu": "german",
    "ell": "el",
    "eng": "en",
    "est": "et",
    "eus": "eu",
    "fas": "fa",
    "fil": "tl",
    "fin": "fi",
    "fra": "fr",
    "gle": "ga",
    "glg": "gl",
    "hin": "hi",
    "hrv": "hr",
    "hun": "hu",
    "ind": "id",
    "isl": "is",
    "ita": "it",
    "jpn": "japan",
    "kan": "ka",
    "kaz": "kk",
    "kir": "ky",
    "kmr": "ku",
    "kor": "korean",
    "lat": "la",
    "lav": "lv",
    "lit": "lt",
    "ltz": "lb",
    "mar": "mr",
    "mkd": "mk",
    "mlt": "mt",
    "mon": "mn",
    "mri": "mi",
    "msa": "ms",
    "nep": "ne",
    "nld": "nl",
    "nor": "no",
    "oci": "oc",
    "pol": "pl",
    "por": "pt",
    "pus": "ps",
    "que": "qu",
    "ron": "ro",
    "rus": "ru",
    "san": "sa",
    "slk": "sk",
    "slv": "sl",
    "snd": "sd",
    "spa": "es",
    "sqi": "sq",
    "srp": "rs_cyrillic",
    "srp_latn": "rs_latin",
    "swa": "sw",
    "swe": "sv",
    "tam": "ta",
    "tat": "tt",
    "tel": "te",
    "tgk": "tg",
    "tgl": "tl",
    "tha": "th",
    "tur": "tr",
    "uig": "ug",
    "ukr": "uk",
    "urd": "ur",
    "uzb": "uz",
    "vie": "vi",
}


def first_paddle_lang(languages: Iterable[str] | None) -> str:
    """First requested language as a PaddleOCR code, ``"en"`` when none is requested.

    The paperless parser already translates Tesseract codes; the lookup here
    is a defensive fallback for a code that slipped through untranslated.
    """
    langs = list(languages or [])
    if not langs:
        return "en"
    code = str(langs[0]).strip().lower()
    return TESSERACT_TO_PADDLE.get(code, code)


# PaddleOCR 3.x dropped the 2.x script-bundle codes. Any member language of a
# script group selects that group's recognition model, so each bundle maps to
# a representative member; "cyrillic" deliberately avoids "ru", which 3.x
# routes to the separate East-Slavic model instead of the cyrillic bundle.
# "ml" maps to None: omitting lang= selects the default multilingual
# PP-OCRv6 model, which is 3.x's replacement for the old "ml" bundle.
_SCRIPT_BUNDLES: Final[dict[str, str | None]] = {
    "ml": None,
    "latin": "la",
    "arabic": "ar",
    "cyrillic": "rs_cyrillic",
    "devanagari": "hi",
}

# Native PaddleOCR codes that users can pass via PAPERLESS_PADDLEOCR_LANGUAGE
# without going through the tesseract mapping, including the 2.x-era script
# bundles that normalize_paddle_lang() still accepts. Used only for
# documentation/validation; resolve_paddle_languages passes the override
# through unchanged.
PADDLE_NATIVE: Final[frozenset[str]] = frozenset({*PADDLE_VALID_LANGS, *_SCRIPT_BUNDLES})


def normalize_paddle_lang(code: str) -> str | None:
    """Translate a user-facing PaddleOCR code to a valid 3.x ``lang=`` value.

    Returns ``None`` when the engine should omit the ``lang`` kwarg entirely.
    """
    cleaned = code.strip().lower()
    return _SCRIPT_BUNDLES.get(cleaned, cleaned)


# to_hocr_lang renders a PaddleOCR code as its rename below, else as itself
# when it is a two-letter member of PADDLE_VALID_LANGS, else as BCP-47 "und".
# Longer codes stay "und" because they do not reliably match ISO 639:
# PaddleOCR's mah is Magahi, and ISO 639-2 mah is Marshallese. A two-letter
# code that names another language in BCP-47 needs a rename: PaddleOCR's ka
# is Kannada, and BCP-47 ka is Georgian.
_BCP47_RENAMES: Final[dict[str, str]] = {
    "ch": "zh",
    "chinese_cht": "zh",
    "french": "fr",
    "german": "de",
    "japan": "ja",
    "ka": "kn",
    "korean": "ko",
    "rs_cyrillic": "sr",
    "rs_latin": "sr",
}


def to_hocr_lang(paddle_code: str | None) -> str:
    """Map a PaddleOCR language code to a BCP-47 code for the hOCR ``lang``.

    Unknown or multilingual codes return ``"und"`` (BCP-47 undetermined)
    rather than leaking a PaddleOCR-internal token into the hOCR.
    """
    if not paddle_code:
        return "und"
    code = paddle_code.strip().lower()
    if code in _BCP47_RENAMES:
        return _BCP47_RENAMES[code]
    if len(code) == 2 and code in PADDLE_VALID_LANGS:
        return code
    return "und"


def resolve_paddle_languages(
    paperless_lang: str | None,
    override: str | None,
) -> list[str]:
    """Resolve the list of PaddleOCR lang codes to run on a document.

    Parameters
    ----------
    paperless_lang:
        Value of ``PAPERLESS_OCR_LANGUAGE`` / ``OcrConfig.language`` - Tesseract
        ISO 639-2/T codes joined by ``+`` (e.g. ``"eng+deu"``). May be empty or
        ``None``; we fall back to ``"eng"`` in that case.
    override:
        Value of ``PAPERLESS_PADDLEOCR_LANGUAGE`` if set, else ``None``. When
        set, takes precedence over ``paperless_lang`` and bypasses the
        tesseract→paddle map entirely.

    Returns
    -------
    list[str]
        Ordered, de-duplicated list of PaddleOCR language codes. Always
        contains at least one entry (``["en"]`` as a last resort).
    """
    if override:
        codes = _split_langs(override)
        if not codes:
            logger.warning(
                "PAPERLESS_PADDLEOCR_LANGUAGE was set but empty after parsing; "
                "falling back to PAPERLESS_OCR_LANGUAGE.",
            )
        else:
            logger.debug("PaddleOCR language override active: %s", codes)
            if len(codes) > 1:
                _warn_multi(codes)
            return codes

    if not paperless_lang:
        logger.debug("No language configured; defaulting to 'en'.")
        return ["en"]

    mapped: list[str] = []
    for tess in _split_langs(paperless_lang):
        paddle = TESSERACT_TO_PADDLE.get(tess.lower())
        if paddle is None:
            logger.warning(
                "Tesseract language %r has no PaddleOCR equivalent; skipping. "
                "Set PAPERLESS_PADDLEOCR_LANGUAGE to use native PaddleOCR codes.",
                tess,
            )
            continue
        if paddle not in mapped:
            mapped.append(paddle)

    if not mapped:
        logger.warning(
            "No PaddleOCR-supported languages resolved from %r; falling back to 'en'.",
            paperless_lang,
        )
        return ["en"]

    if len(mapped) > 1:
        _warn_multi(mapped)

    return mapped


def _split_langs(joined: str) -> list[str]:
    """Split a ``+``-separated language string, dropping empty and repeated entries.

    Entries that differ in case only count as repeated, and the first spelling is kept.
    """
    first_spelling: dict[str, str] = {}
    for part in joined.split("+"):
        code = part.strip()
        if code:
            first_spelling.setdefault(code.lower(), code)
    return list(first_spelling.values())


def _warn_multi(codes: list[str]) -> None:
    """Warn that multi-language OCR can run up to one pass per language."""
    logger.warning(
        "Multi-language OCR active for %d languages (%s). PaddleOCR runs up to "
        "one pass per language, so the time of a page can grow with the language "
        "count. If you only need broad script coverage, consider "
        "PAPERLESS_PADDLEOCR_LANGUAGE=ml (or "
        "'latin', 'arabic', 'cyrillic', 'devanagari') for a single-pass "
        "multilingual model.",
        len(codes),
        "+".join(codes),
    )
