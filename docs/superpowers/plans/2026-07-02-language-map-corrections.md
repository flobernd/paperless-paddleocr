# Language Map Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every language code the plugin emits a valid PaddleOCR 3.x `lang=` value, expand Tesseract coverage, and
keep the 2.x script-bundle codes working.

**Architecture:** `languages.py` stays the single source of truth. It gains a vendored copy of PaddleOCR 3.7.0's valid
language sets (with provenance comment), a corrected and expanded `TESSERACT_TO_PADDLE`, and a `normalize_paddle_lang()`
helper that translates 2.x script-bundle codes (`ml`, `latin`, ...) to valid 3.x equivalents. `classic.py` applies the
normalisation right before constructing `PaddleOCR`; `engine.py` derives its ocrmypdf language allowlist from
`languages.py` instead of a hand-maintained frozenset.

**Tech Stack:** Python 3.12, pytest. Tests must run without paddlepaddle/paddleocr installed (CI constraint).

## Global Constraints

- Python `>=3.12` (pyproject `requires-python`).
- `ruff check .` and `ruff format --check .` must pass (line length 100).
- `mypy --ignore-missing-imports paperless_paddleocr` must pass.
- Tests must not import paddlepaddle or paddleocr at module level; CI installs only `ocrmypdf lxml pillow pytest`.
- No em-dashes in prose or comments; comments explain WHY only.

**Background (verified 2026-07-02 against the paddleocr 3.7.0 wheel):** valid `lang=`
values are resolved in `paddleocr/_pipelines/ocr.py:_get_ocr_model_names` from the sets in
`paddleocr/_utils/langs.py` plus the specials `ch`, `chinese_cht`, `en`, `japan`,
`korean`, `th`, `el`, `te`, `ta`, `ka`. `spanish`, `he`, `ml`, `latin`, `arabic`,
`cyrillic`, `devanagari` are NOT valid and make `PaddleOCR.__init__` raise
`ValueError("No models are available for lang=...")`. `lang=None` (omit the kwarg)
selects the default multilingual PP-OCRv6 model.

---

### Task 1: Vendor the valid-language sets and fix the Tesseract map

**Files:**

- Modify: `paperless_paddleocr/languages.py`
- Test: `tests/test_languages_paddle3.py` (create)

**Interfaces:**

- Produces: `languages.PADDLE_VALID_LANGS: frozenset[str]` (every valid 3.x `lang=` value), corrected `TESSERACT_TO_PADDLE`.
- Consumed by Task 2 (`normalize_paddle_lang`) and Task 3 (`engine.languages()`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_languages_paddle3.py`:

```python
"""Every code the plugin can hand to PaddleOCR(lang=...) must be valid in 3.x.

PADDLE_VALID_LANGS is vendored from paddleocr 3.7.0; these tests pin the
mapping against it so an invalid target (like the old "spa" -> "spanish")
can never ship again.
"""

from __future__ import annotations

from paperless_paddleocr.languages import (
    PADDLE_VALID_LANGS,
    TESSERACT_TO_PADDLE,
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_languages_paddle3.py -v`
Expected: FAIL with `ImportError: cannot import name 'PADDLE_VALID_LANGS'`

- [ ] **Step 3: Implement in `languages.py`**

Replace the current `TESSERACT_TO_PADDLE` definition (lines 20-53) with the block below,
and add `PADDLE_VALID_LANGS` directly after it. Keep the module docstring and logger.

```python
# Valid `lang=` values for paddleocr 3.x, vendored from paddleocr 3.7.0
# (`paddleocr/_utils/langs.py` plus the specials handled in
# `paddleocr/_pipelines/ocr.py:_get_ocr_model_names`). Re-sync when the
# paddleocr dependency floor moves. Kept here so the mapping below can be
# validated in CI without installing paddleocr.
_LATIN_LANGS: Final[frozenset[str]] = frozenset(
    {
        "af", "az", "bs", "ca", "cs", "cy", "da", "de", "es", "et", "eu", "fi",
        "fr", "french", "ga", "german", "gl", "hr", "hu", "id", "is", "it", "ku",
        "la", "lb", "lt", "lv", "mi", "ms", "mt", "nl", "no", "oc", "pi", "pl",
        "pt", "qu", "rm", "ro", "rs_latin", "sk", "sl", "sq", "sv", "sw", "tl",
        "tr", "uz", "vi",
    },
)
_ARABIC_LANGS: Final[frozenset[str]] = frozenset(
    {"ar", "fa", "ug", "ur", "ps", "ku", "sd", "bal"},
)
_ESLAV_LANGS: Final[frozenset[str]] = frozenset({"ru", "be", "uk"})
_CYRILLIC_LANGS: Final[frozenset[str]] = frozenset(
    {
        "ru", "rs_cyrillic", "be", "bg", "uk", "mn", "abq", "ady", "kbd", "ava",
        "dar", "inh", "che", "lbe", "lez", "tab", "kk", "ky", "tg", "mk", "tt",
        "cv", "ba", "mhr", "mo", "udm", "kv", "os", "bua", "xal", "tyv", "sah",
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
    _LATIN_LANGS | _ARABIC_LANGS | _ESLAV_LANGS | _CYRILLIC_LANGS | _DEVANAGARI_LANGS
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
    "kat": "ka",
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
```

Also update `_PADDLE_TO_BCP47`: change nothing that exists, but add entries for the newly
reachable codes so the hOCR `lang` stays meaningful (unknown codes already fall back to
`"und"`, so this is best-effort):

```python
    # Newly mapped codes (Task 1); same best-effort ISO 639-1 policy as above.
    "es": "es",
    "af": "af", "az": "az", "be": "be", "bg": "bg", "bs": "bs", "ca": "ca",
    "cy": "cy", "et": "et", "eu": "eu", "fa": "fa", "ga": "ga", "gl": "gl",
    "hr": "hr", "id": "id", "is": "is", "ka": "ka", "kk": "kk", "ky": "ky",
    "ku": "ku", "la": "la", "lb": "lb", "lt": "lt", "lv": "lv", "mi": "mi",
    "mk": "mk", "mn": "mn", "mr": "mr", "ms": "ms", "mt": "mt", "ne": "ne",
    "oc": "oc", "ps": "ps", "qu": "qu", "sa": "sa", "sd": "sd", "sk": "sk",
    "sl": "sl", "sq": "sq", "sw": "sw", "tg": "tg", "tl": "tl", "tt": "tt",
    "ug": "ug", "ur": "ur", "uz": "uz",
    "rs_cyrillic": "sr", "rs_latin": "sr",
```

Remove `"spanish": "es"` and `"he": "he"` from `_PADDLE_TO_BCP47` if present (they map
codes that no longer exist on the paddle side; `"es"` replaces `"spanish"`).

Finally update `PADDLE_NATIVE` (keep it, it documents the override vocabulary):

```python
PADDLE_NATIVE: Final[frozenset[str]] = frozenset(
    {
        *PADDLE_VALID_LANGS,
        # 2.x-era script bundles, still accepted via normalize_paddle_lang().
        "ml", "latin", "arabic", "cyrillic", "devanagari",
    },
)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_languages_paddle3.py tests/test_hocr_lang.py -v`
Expected: PASS (the existing `test_hocr_lang.py` must keep passing;
`to_hocr_lang("german") == "de"` and the injection tests are unaffected).

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/languages.py tests/test_languages_paddle3.py
git commit -m "Fix and expand the Tesseract to PaddleOCR 3.x language map"
```

---

### Task 2: Translate script-bundle codes to valid 3.x values

**Files:**

- Modify: `paperless_paddleocr/languages.py`
- Modify: `paperless_paddleocr/paddle_engine/classic.py:63-89` (`_build_engine`)
- Test: `tests/test_languages_paddle3.py` (extend), `tests/test_classic_device.py` (extend)

**Interfaces:**

- Produces: `languages.normalize_paddle_lang(code: str) -> str | None`. `None` means
  "omit the lang kwarg" (PaddleOCR then loads its default multilingual PP-OCRv6 model).
- Consumes: `PADDLE_VALID_LANGS` from Task 1.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_languages_paddle3.py`:

```python
from paperless_paddleocr.languages import normalize_paddle_lang


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
```

Append to `tests/test_classic_device.py`:

```python
def test_build_engine_ml_omits_lang_kwarg(monkeypatch):
    monkeypatch.setattr(classic, "PaddleOCR", _PaddleOCRStub)
    classic._build_engine(types.SimpleNamespace(languages=["ml"], paddle_engine="classic-cpu"))
    assert "lang" not in _PaddleOCRStub.last_kwargs


def test_build_engine_bundle_code_is_translated(monkeypatch):
    monkeypatch.setattr(classic, "PaddleOCR", _PaddleOCRStub)
    classic._build_engine(
        types.SimpleNamespace(languages=["latin"], paddle_engine="classic-cpu"),
    )
    assert _PaddleOCRStub.last_kwargs["lang"] == "la"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_languages_paddle3.py tests/test_classic_device.py -v`
Expected: FAIL with `ImportError: cannot import name 'normalize_paddle_lang'`

- [ ] **Step 3: Implement**

Add to `languages.py`:

```python
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


def normalize_paddle_lang(code: str) -> str | None:
    """Translate a user-facing PaddleOCR code to a valid 3.x ``lang=`` value.

    Returns ``None`` when the engine should omit the ``lang`` kwarg entirely.
    """
    cleaned = code.strip().lower()
    if cleaned in _SCRIPT_BUNDLES:
        return _SCRIPT_BUNDLES[cleaned]
    return cleaned
```

In `classic.py`, import it and use it in `_build_engine`. Replace:

```python
    kwargs: dict[str, Any] = {
        "lang": _resolve_lang(options),
        "device": device,
```

with:

```python
    kwargs: dict[str, Any] = {
        "device": device,
```

and insert after the `kwargs` literal closes (before the `if device == "cpu":` block):

```python
    lang = normalize_paddle_lang(_resolve_lang(options))
    if lang is not None:
        kwargs["lang"] = lang
```

Import line in `classic.py`:

```python
from paperless_paddleocr.languages import TESSERACT_TO_PADDLE, normalize_paddle_lang
```

Note: `_resolve_lang` keeps returning the user-facing code (`"ml"` etc.) because
`build_page` also uses it for the `Page.lang` metadata, where `to_hocr_lang` already
normalises unknown codes to `"und"`.

- [ ] **Step 4: Run the full suite**

Run: `pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/languages.py paperless_paddleocr/paddle_engine/classic.py tests/test_languages_paddle3.py tests/test_classic_device.py
git commit -m "Translate 2.x script-bundle language codes for PaddleOCR 3.x"
```

---

### Task 3: Derive the ocrmypdf language allowlist from languages.py

**Files:**

- Modify: `paperless_paddleocr/paddle_engine/engine.py:22-82`
- Test: `tests/test_engine_languages.py` (create)

**Interfaces:**

- Consumes: `PADDLE_NATIVE`, `TESSERACT_TO_PADDLE` from Tasks 1-2.
- Produces: `PaddleOCREngine.languages(options)` returning the derived set (ocrmypdf
  17.8.0 `_validation.py:163-166` checks requested codes for membership and aborts on
  a miss, so this set must contain everything `resolve_paddle_languages` can emit plus
  raw Tesseract codes for misconfigured overrides).

- [ ] **Step 1: Write the failing test**

Create `tests/test_engine_languages.py`:

```python
"""ocrmypdf aborts OCR when a requested code is missing from languages().

The allowlist must therefore cover every code resolve_paddle_languages can
emit (mapped targets, bundles, override passthrough) and the raw Tesseract
codes, so a slightly wrong override degrades later with a clear engine error
instead of a spurious preflight abort.
"""

from __future__ import annotations

from paperless_paddleocr.languages import PADDLE_NATIVE, TESSERACT_TO_PADDLE
from paperless_paddleocr.paddle_engine.engine import PaddleOCREngine


def test_languages_covers_all_resolvable_codes():
    supported = PaddleOCREngine.languages(options=None)
    assert set(TESSERACT_TO_PADDLE.values()) <= supported
    assert set(TESSERACT_TO_PADDLE) <= supported
    assert set(PADDLE_NATIVE) <= supported
    assert {"ml", "latin", "es", "sk", "eng"} <= supported
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine_languages.py -v`
Expected: FAIL (current hand-maintained set lacks `sk`, `es`, and others).

- [ ] **Step 3: Implement**

In `engine.py`, delete the whole `_SUPPORTED_LANGUAGES` frozenset literal (lines 22-82)
and replace it with:

```python
from paperless_paddleocr.languages import PADDLE_NATIVE, TESSERACT_TO_PADDLE

#: Reported to ocrmypdf's preflight language check. Derived from the language
#: tables so a code our own mapping can emit never spuriously aborts OCR;
#: raw Tesseract codes are included because a mis-set
#: PAPERLESS_PADDLEOCR_LANGUAGE should fail in the engine with a clear error,
#: not in ocrmypdf preflight.
_SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {
        *PADDLE_NATIVE,
        *TESSERACT_TO_PADDLE,
        *TESSERACT_TO_PADDLE.values(),
    },
)
```

`languages()` itself stays as is (`return set(_SUPPORTED_LANGUAGES)`).

- [ ] **Step 4: Run the full suite**

Run: `pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/paddle_engine/engine.py tests/test_engine_languages.py
git commit -m "Derive ocrmypdf language allowlist from the language tables"
```

---

### Task 4: Update the README language documentation

**Files:**

- Modify: `README.md` ("Language handling" section and the `PAPERLESS_PADDLEOCR_LANGUAGE`
  entry)

- [ ] **Step 1: Edit the `PAPERLESS_PADDLEOCR_LANGUAGE` section**

Replace the bullet list under "Use this to pass native PaddleOCR codes that have no
Tesseract analogue:" with:

```markdown
- Single-script models: `en`, `german`, `fr`, `es`, `ch`, `japan`, `korean`, ...
- Script bundles: `ml`, `latin`, `arabic`, `cyrillic`, `devanagari`. These are 2.x-era
  names that the plugin translates for PaddleOCR 3.x: `ml` selects the default
  multilingual PP-OCRv6 model, the others select the matching script recognition model.
```

- [ ] **Step 2: Edit the "Language handling" section**

In step 1 of that section, replace the mapping examples so they are true:

```markdown
1. **Mapping** - `eng` → `en`, `deu` → `german`, `spa` → `es`, `chi_sim` → `ch`, etc.
   Codes without a PaddleOCR 3.x model (for example `heb`) are logged and dropped. The
   full table is in [`paperless_paddleocr/languages.py`](paperless_paddleocr/languages.py).
```

- [ ] **Step 3: Verify markdown lint**

Run: `npx markdownlint-cli2 "README.md" --config .markdownlint.yaml` (or rely on CI).
Expected: no new violations.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Document corrected PaddleOCR 3.x language handling"
```
