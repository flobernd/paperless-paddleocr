# Multi-Language Merge Tests and VL Single Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the multi-language merge path under test, then stop `vl-remote` from running one identical remote pass per
language, and (optionally) remove the box-count bias from winner scoring.

**Architecture:** All behaviour lives in `paperless_paddleocr/ocrmypdf_plugin.py`. Tests stub
`PaddleOCREngine.generate_hocr` (the per-language pass) exactly the way `tests/test_generate_pdf_dispatch.py` already
stubs engine internals, so no paddle install is needed. The VL short-circuit is a three-line guard in
`MultiLangPaddleEngine.generate_hocr`.

**Tech Stack:** Python 3.12, pytest, Pillow (for the tiny input image `_read_page_size` opens).

## Global Constraints

- Python `>=3.12`; `ruff check .`, `ruff format --check .`, `mypy --ignore-missing-imports paperless_paddleocr` must pass.
- Tests must run without paddlepaddle/paddleocr installed.
- No em-dashes in prose or comments; comments explain WHY only.

---

### Task 1: Test harness and coverage for the multi-language merge

**Files:**

- Test: `tests/test_multilang_merge.py` (create)

**Interfaces:**

- Consumes: `MultiLangPaddleEngine.generate_hocr(input_file, output_hocr, output_text, options)`,
  `_write_merged_hocr`, `_parse_hocr_words`, `_Word` from `paperless_paddleocr.ocrmypdf_plugin`,
  and `PaddleOCREngine` from `paperless_paddleocr.paddle_engine`.
- Produces: `_install_fake_passes(monkeypatch, words_by_lang, fail_langs=())` helper reused
  by Task 2.

- [ ] **Step 1: Write the tests**

Create `tests/test_multilang_merge.py`:

```python
"""Coverage for MultiLangPaddleEngine's multi-language branch.

The per-language pass (PaddleOCREngine.generate_hocr) is stubbed to write
deterministic hOCR per language, so winner selection, NMS merging and the
failure paths are exercised without paddle installed.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
from PIL import Image

from paperless_paddleocr.ocrmypdf_plugin import (
    MultiLangPaddleEngine,
    _parse_hocr_words,
    _Word,
    _write_merged_hocr,
)
from paperless_paddleocr.paddle_engine import PaddleOCREngine


def _install_fake_passes(monkeypatch, words_by_lang, fail_langs=()):
    """Stub the base per-language pass with fixed words per language."""

    def fake(input_file: Path, output_hocr: Path, output_text: Path, options) -> None:
        lang = options.languages[0]
        if lang in fail_langs:
            raise RuntimeError(f"simulated engine failure for {lang}")
        words = words_by_lang[lang]
        _write_merged_hocr(output_hocr, words, 200, 100, hocr_lang=lang)
        output_text.write_text(" ".join(w.text for w in words), encoding="utf-8")

    monkeypatch.setattr(PaddleOCREngine, "generate_hocr", staticmethod(fake))


def _run(tmp_path, monkeypatch, strategy, words_by_lang, fail_langs=()):
    monkeypatch.setenv("PAPERLESS_PADDLEOCR_MULTI_LANG_STRATEGY", strategy)
    _install_fake_passes(monkeypatch, words_by_lang, fail_langs)
    img = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(img)
    out_hocr = tmp_path / "out.hocr"
    out_text = tmp_path / "out.txt"
    options = types.SimpleNamespace(
        languages=list(words_by_lang),
        paddle_engine="classic-cpu",
    )
    MultiLangPaddleEngine.generate_hocr(img, out_hocr, out_text, options)
    return out_hocr, out_text


def test_winner_keeps_the_higher_aggregate_confidence_language(tmp_path, monkeypatch):
    words_by_lang = {
        "en": [_Word("cat", 0, 0, 50, 20, 60)],
        "german": [_Word("Katze", 0, 0, 50, 20, 95)],
    }
    out_hocr, out_text = _run(tmp_path, monkeypatch, "winner", words_by_lang)
    assert [w.text for w in _parse_hocr_words(out_hocr)] == ["Katze"]
    assert out_text.read_text(encoding="utf-8") == "Katze"


def test_merge_keeps_highest_confidence_box_per_location(tmp_path, monkeypatch):
    words_by_lang = {
        # Same location: german wins on confidence. Second en word is
        # elsewhere on the page and must survive the NMS.
        "en": [_Word("cat", 0, 0, 50, 20, 60), _Word("extra", 100, 50, 150, 70, 80)],
        "german": [_Word("Katze", 0, 0, 50, 20, 95)],
    }
    out_hocr, _ = _run(tmp_path, monkeypatch, "merge", words_by_lang)
    assert sorted(w.text for w in _parse_hocr_words(out_hocr)) == ["Katze", "extra"]


def test_partial_language_failure_uses_surviving_passes(tmp_path, monkeypatch):
    words_by_lang = {
        "en": [_Word("hello", 0, 0, 50, 20, 90)],
        "german": [],  # never produced: this pass raises
    }
    out_hocr, out_text = _run(
        tmp_path, monkeypatch, "winner", words_by_lang, fail_langs=("german",)
    )
    assert [w.text for w in _parse_hocr_words(out_hocr)] == ["hello"]
    assert out_text.read_text(encoding="utf-8") == "hello"


def test_all_languages_failing_writes_empty_outputs(tmp_path, monkeypatch):
    words_by_lang = {"en": [], "german": []}
    out_hocr, out_text = _run(
        tmp_path, monkeypatch, "winner", words_by_lang, fail_langs=("en", "german")
    )
    assert _parse_hocr_words(out_hocr) == []
    assert out_text.read_text(encoding="utf-8") == ""


def test_unknown_strategy_falls_back_to_winner(tmp_path, monkeypatch):
    words_by_lang = {
        "en": [_Word("cat", 0, 0, 50, 20, 60)],
        "german": [_Word("Katze", 0, 0, 50, 20, 95)],
    }
    out_hocr, _ = _run(tmp_path, monkeypatch, "bogus-strategy", words_by_lang)
    assert [w.text for w in _parse_hocr_words(out_hocr)] == ["Katze"]
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/test_multilang_merge.py -v`
Expected: all PASS. These pin current behaviour; if any fails, stop and investigate
before continuing (that would be a live bug, not a test problem).

- [ ] **Step 3: Commit**

```bash
git add tests/test_multilang_merge.py
git commit -m "Cover the multi-language merge strategies and failure paths"
```

---

### Task 2: Single pass for vl-remote regardless of language count

**Files:**

- Modify: `paperless_paddleocr/ocrmypdf_plugin.py:122-157`
  (`MultiLangPaddleEngine.generate_hocr`)
- Test: `tests/test_multilang_merge.py` (extend)

**Interfaces:**

- Behaviour change only: with `options.paddle_engine == "vl-remote"` and N > 1 languages,
  exactly one base pass runs (first language). Classic engines keep one pass per language.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multilang_merge.py`:

```python
def _count_passes(tmp_path, monkeypatch, engine: str) -> int:
    calls: list[str] = []

    def fake(input_file, output_hocr, output_text, options) -> None:
        calls.append(options.languages[0])
        _write_merged_hocr(
            output_hocr, [_Word("x", 0, 0, 10, 10, 90)], 200, 100,
            hocr_lang=options.languages[0],
        )
        output_text.write_text("x", encoding="utf-8")

    monkeypatch.setattr(PaddleOCREngine, "generate_hocr", staticmethod(fake))
    img = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(img)
    options = types.SimpleNamespace(languages=["en", "german"], paddle_engine=engine)
    MultiLangPaddleEngine.generate_hocr(
        img, tmp_path / "o.hocr", tmp_path / "o.txt", options
    )
    return len(calls)


def test_vl_remote_runs_a_single_pass_for_multiple_languages(tmp_path, monkeypatch):
    assert _count_passes(tmp_path, monkeypatch, "vl-remote") == 1


def test_classic_still_runs_one_pass_per_language(tmp_path, monkeypatch):
    assert _count_passes(tmp_path, monkeypatch, "classic-cpu") == 2
```

- [ ] **Step 2: Run tests to verify the new one fails**

Run: `pytest tests/test_multilang_merge.py -v`
Expected: `test_vl_remote_runs_a_single_pass_for_multiple_languages` FAILS with `2 == 1`.

- [ ] **Step 3: Implement**

In `MultiLangPaddleEngine.generate_hocr`, insert directly after
`langs = list(getattr(options, "languages", None) or [])`:

```python
        if len(langs) > 1 and getattr(options, "paddle_engine", "") == "vl-remote":
            # The VL recognition model reads every script in one pass; the
            # language code only labels the hOCR. N per-language passes would
            # send the same ~20 s/page remote request N times for identical
            # results.
            log.info(
                "vl-remote is language-agnostic; running one pass instead of %d (%s).",
                len(langs),
                "+".join(langs),
            )
            langs = langs[:1]
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_multilang_merge.py tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/ocrmypdf_plugin.py tests/test_multilang_merge.py
git commit -m "Run a single vl-remote pass for multi-language documents"
```

---

### Task 3 (optional, decision-gated): Remove box-count bias from winner scoring

The current winner score is `sum(w.conf)`, which rewards a pass that emits more boxes: 100
words at confidence 50 (score 5000) beat 50 words at confidence 95 (score 4750), even
though the second pass is plainly better. Weighting by recognised text length measures
"confident text mass" instead of box count. This changes winner selection on some
documents, so treat it as a deliberate behaviour change: implement it, run it against a
real multi-language corpus, and keep it only if results look at least as good.

**Files:**

- Modify: `paperless_paddleocr/ocrmypdf_plugin.py:242-247`
- Test: `tests/test_multilang_merge.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_winner_prefers_confident_text_mass_over_box_count(tmp_path, monkeypatch):
    words_by_lang = {
        # Five low-confidence single-char boxes vs one long high-confidence
        # word. Plain sum(conf) picks the five boxes (300 > 90); the
        # length-weighted score picks the long word (10 * 90 > 5 * 60).
        "en": [_Word(c, i * 20, 0, i * 20 + 10, 20, 60) for i, c in enumerate("abcde")],
        "german": [_Word("Grundstück", 0, 50, 100, 70, 90)],
    }
    out_hocr, _ = _run(tmp_path, monkeypatch, "winner", words_by_lang)
    assert [w.text for w in _parse_hocr_words(out_hocr)] == ["Grundstück"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_multilang_merge.py::test_winner_prefers_confident_text_mass_over_box_count -v`
Expected: FAIL (the five boxes win under the current sum).

- [ ] **Step 3: Implement**

In `generate_hocr`'s winner branch, replace:

```python
            by_lang_conf = {
                lang: sum(w.conf for w in words) for lang, words in per_lang_words.items()
            }
```

with:

```python
            # Confidence weighted by recognised text length: a pure sum of
            # confidences rewards whichever pass fragments the page into more
            # boxes, not the pass that read the page best.
            by_lang_conf = {
                lang: sum(w.conf * max(1, len(w.text)) for w in words)
                for lang, words in per_lang_words.items()
            }
```

- [ ] **Step 4: Run the tests, then validate on a real corpus**

Run: `pytest tests/test_multilang_merge.py -v` - expected: all PASS.

Manual gate: OCR a handful of genuinely mixed-language documents (the `test-run/`
fixtures used during development are a starting point) with `winner` before and after,
and compare sidecar quality. Revert this task if the new score picks worse passes.

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/ocrmypdf_plugin.py tests/test_multilang_merge.py
git commit -m "Weight winner scoring by recognised text length"
```

Also update the `PAPERLESS_PADDLEOCR_MULTI_LANG_STRATEGY` section of `README.md` in this
commit: change "pick the one with the highest aggregate word confidence" to "pick the one
with the highest length-weighted aggregate word confidence".
