# Page-Level Reading Order Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One reading-order pass feeds both the hOCR block order and the sidecar text; the single-language path stops
re-parsing its own freshly written hOCR, and the merged multi-language hOCR regains real line structure instead of
one-word-per-line.

**Architecture:** `engine.py` reorders `Page.blocks` through `layout.reading_blocks` and derives the sidecar from the
ordered blocks' authoritative `Line.text` (preserving CJK spacing, which a word join loses). `hocr.write_document`
accepts a sidecar override. The plugin's single-language rebuild block is deleted. `_write_merged_hocr` keeps its name
and signature but is reimplemented on the typed `Page` model plus `hocr.render_hocr`, so all existing callers and tests
keep working while the merged output gains grouped lines and fitted baselines.

**Tech Stack:** Python 3.12, pytest, lxml.

**Dependency:** requires `layout.reading_blocks` from
`docs/superpowers/plans/2026-07-02-banded-column-detection.md`. Implement that plan first.

## Global Constraints

- Python `>=3.12`; `ruff check .`, `ruff format --check .`, `mypy --ignore-missing-imports paperless_paddleocr` must pass.
- Tests must run without paddlepaddle/paddleocr installed.
- No em-dashes in prose or comments; comments explain WHY only.

---

### Task 1: Sidecar override in write_document

**Files:**

- Modify: `paperless_paddleocr/paddle_engine/hocr.py:144-147`
- Test: `tests/test_hocr_render.py` (extend)

**Interfaces:**

- Produces: `write_document(page: Page, hocr_path: Path, text_path: Path, *, sidecar: str | None = None) -> None`;
  `None` keeps the current `sidecar_text(page)` behaviour.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hocr_render.py`:

```python
def test_write_document_sidecar_override(tmp_path):
    from paperless_paddleocr.paddle_engine.hocr import write_document

    line = Line(box=(10, 10, 90, 40), confidence=88, text="hi",
                words=[Word("hi", (10, 10, 90, 40), 88)])
    hocr_path, text_path = tmp_path / "p.hocr", tmp_path / "p.txt"
    write_document(_page(line), hocr_path, text_path, sidecar="custom order")
    assert text_path.read_text(encoding="utf-8") == "custom order"
    assert "ocrx_word" in hocr_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hocr_render.py -v`
Expected: FAIL with `TypeError: write_document() got an unexpected keyword argument`

- [ ] **Step 3: Implement**

Replace `write_document` in `hocr.py`:

```python
def write_document(
    page: Page,
    hocr_path: Path,
    text_path: Path,
    *,
    sidecar: str | None = None,
) -> None:
    """Write the hOCR document and its plain-text sidecar to disk.

    ``sidecar`` overrides the default per-line rendering when the caller has
    a better (reading-ordered) text for the same page.
    """
    hocr_path.write_text(render_hocr(page), encoding="utf-8")
    text_path.write_text(sidecar if sidecar is not None else sidecar_text(page), encoding="utf-8")
```

- [ ] **Step 4: Run and commit**

Run: `pytest tests/test_hocr_render.py -v` - expected: all PASS.

```bash
git add paperless_paddleocr/paddle_engine/hocr.py tests/test_hocr_render.py
git commit -m "Allow a sidecar override in write_document"
```

---

### Task 2: Order pages and derive the sidecar in the engine

**Files:**

- Modify: `paperless_paddleocr/paddle_engine/engine.py:143-174`
- Test: `tests/test_engine_reading_order.py` (create)

**Interfaces:**

- Produces: `engine._order_page(page: Page) -> str` which reorders `page.blocks` in place
  (banded reading order) and returns the sidecar text: `Line.text` values joined by
  newlines within a reading group and blank lines between groups.

- [ ] **Step 1: Write the failing test**

Create `tests/test_engine_reading_order.py`:

```python
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
        "sender1\nsender2\nsender3"
        "\n\nrecipient1\nrecipient2\nrecipient3"
        "\n\nbody1\nbody2\nbody3"
    )
    hocr = out_hocr.read_text(encoding="utf-8")
    assert hocr.index("sender3") < hocr.index("recipient1") < hocr.index("body1")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine_reading_order.py -v`
Expected: FAIL (sidecar interleaves sender/recipient lines).

- [ ] **Step 3: Implement**

In `engine.py`, add imports:

```python
from paperless_paddleocr.paddle_engine.hocr import Page, write_document
from paperless_paddleocr.paddle_engine.layout import reading_blocks
```

Add the helper and rewire both generate paths:

```python
def _order_page(page: Page) -> str:
    """Reorder page.blocks in banded reading order; return the sidecar text.

    Line.text stays authoritative for the sidecar because engines may space
    a line differently from a naive word join (CJK has no inter-word
    spaces). The hOCR is reordered too so the text layer and the sidecar
    agree on reading order.
    """
    if not page.blocks:
        return ""
    grouped = reading_blocks(page.blocks, bbox_of=lambda b: b.box)
    ordered = [blk for group in grouped for row in group for blk in row]
    page.blocks = ordered
    return "\n\n".join(
        "\n".join(
            line.text for row in group for blk in row for line in blk.lines if line.text
        )
        for group in grouped
    )


class PaddleOCREngine(OcrEngine):
    ...
    @staticmethod
    def _generate_hocr_classic(
        input_file: Path,
        output_hocr: Path,
        output_text: Path,
        options: Any,
    ) -> None:
        page = classic.build_page(input_file, options)
        write_document(page, output_hocr, output_text, sidecar=_order_page(page))

    @staticmethod
    def _generate_hocr_vl(
        input_file: Path,
        output_hocr: Path,
        output_text: Path,
        options: Any,
    ) -> None:
        page = vl.build_page(input_file, options)
        write_document(page, output_hocr, output_text, sidecar=_order_page(page))
```

- [ ] **Step 4: Run and commit**

Run: `pytest tests/test_engine_reading_order.py tests/ -q` - expected: all PASS.

```bash
git add paperless_paddleocr/paddle_engine/engine.py tests/test_engine_reading_order.py
git commit -m "Emit hOCR and sidecar in one banded reading order"
```

---

### Task 3: Drop the plugin's single-language sidecar rebuild

**Files:**

- Modify: `paperless_paddleocr/ocrmypdf_plugin.py:129-156`
- Test: `tests/test_multilang_merge.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multilang_merge.py`:

```python
def test_single_language_keeps_the_engine_sidecar(tmp_path, monkeypatch):
    # The engine now writes a reading-ordered sidecar itself; the plugin
    # must not rebuild (and re-order) it from the hOCR.
    def fake(input_file, output_hocr, output_text, options) -> None:
        _write_merged_hocr(
            output_hocr, [_Word("x", 0, 0, 10, 10, 90)], 200, 100, hocr_lang="en"
        )
        output_text.write_text("ENGINE ORDER", encoding="utf-8")

    monkeypatch.setattr(PaddleOCREngine, "generate_hocr", staticmethod(fake))
    img = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(img)
    out_text = tmp_path / "o.txt"
    options = types.SimpleNamespace(languages=["en"], paddle_engine="classic-cpu")
    MultiLangPaddleEngine.generate_hocr(img, tmp_path / "o.hocr", out_text, options)
    assert out_text.read_text(encoding="utf-8") == "ENGINE ORDER"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_multilang_merge.py::test_single_language_keeps_the_engine_sidecar -v`
Expected: FAIL (the rebuild overwrites the sidecar with "x").

- [ ] **Step 3: Implement**

In `MultiLangPaddleEngine.generate_hocr`, the single-language branch becomes:

```python
        if len(langs) <= 1:
            # Deliberate explicit base-class call (not super()/cls): this is
            # the single-language fast path that must run the base engine
            # exactly once. Dispatching back through this class would
            # re-enter generate_hocr and recurse.
            PaddleOCREngine.generate_hocr(
                input_file,
                output_hocr,
                output_text,
                options,
            )
            return
```

(The whole `try: words = _parse_hocr_words(...)` rebuild block is deleted; the base
engine writes the reading-ordered sidecar since Task 2.)

- [ ] **Step 4: Run and commit**

Run: `pytest tests/test_multilang_merge.py tests/ -q` - expected: all PASS.

```bash
git add paperless_paddleocr/ocrmypdf_plugin.py tests/test_multilang_merge.py
git commit -m "Drop the redundant single-language sidecar rebuild"
```

---

### Task 4: Rebuild the merged hOCR on the Page model

**Files:**

- Modify: `paperless_paddleocr/ocrmypdf_plugin.py:378-438` (`_write_merged_hocr`)
- Test: `tests/test_multilang_merge.py` (extend); existing
  `tests/test_parse_hocr_words.py`, `tests/test_hocr_lang.py`,
  `tests/test_generate_pdf_dispatch.py` must keep passing unchanged.

**Interfaces:**

- `_write_merged_hocr(output_hocr, words, page_w, page_h, *, hocr_lang)` keeps its exact
  signature (three test files author hOCR through it); output becomes standard
  `hocr.render_hocr` markup with real lines.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multilang_merge.py`:

```python
def test_merged_hocr_groups_words_into_lines(tmp_path):
    from lxml import html

    out = tmp_path / "m.hocr"
    words = [
        _Word("hello", 0, 0, 50, 20, 90),
        _Word("world", 60, 0, 110, 20, 90),
        _Word("below", 0, 40, 50, 60, 90),
    ]
    _write_merged_hocr(out, words, 200, 100, hocr_lang="german")
    tree = html.parse(str(out))
    lines = tree.findall('.//{*}span[@class="ocr_line"]')
    assert len(lines) == 2  # not one line per word
    first_line_words = ["".join(s.itertext()) for s in lines[0].findall('{*}span')]
    assert first_line_words == ["hello", "world"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_multilang_merge.py::test_merged_hocr_groups_words_into_lines -v`
Expected: FAIL with `assert 3 == 2`

- [ ] **Step 3: Implement**

In `ocrmypdf_plugin.py`, add imports:

```python
from paperless_paddleocr.paddle_engine.hocr import Block, Line, Page, render_hocr
from paperless_paddleocr.paddle_engine.hocr import Word as HocrWord
from paperless_paddleocr.paddle_engine.layout import fit_baseline, reading_blocks
```

Replace the whole `_write_merged_hocr` function body (delete the manual XHTML string
assembly; the docstring's normalisation note moves to `hocr.render_hocr`, which already
applies it):

```python
def _write_merged_hocr(
    output_hocr: Path,
    words: list[_Word],
    page_w: int,
    page_h: int,
    *,
    hocr_lang: str,
) -> None:
    """Render merged words as a standard Page so the hOCR carries real lines.

    Grouped lines (rather than one line per word) give ocrmypdf's renderer
    line-level baselines and give PDF viewers sane text selection.
    """
    page = Page(
        width=page_w,
        height=page_h,
        lang=hocr_lang,
        ocr_system="paperless-paddleocr multi-lang merge",
    )
    for block in reading_blocks(words, bbox_of=_word_bbox):
        for line_words in block:
            box = (
                min(w.x0 for w in line_words),
                min(w.y0 for w in line_words),
                max(w.x1 for w in line_words),
                max(w.y1 for w in line_words),
            )
            slope, intercept = fit_baseline(
                [((w.x0 + w.x1) / 2.0, float(w.y1)) for w in line_words],
            )
            line = Line(
                box=box,
                confidence=round(sum(w.conf for w in line_words) / len(line_words)),
                text=" ".join(w.text for w in line_words),
                words=[HocrWord(w.text, (w.x0, w.y0, w.x1, w.y1), w.conf) for w in line_words],
                baseline=(slope, slope * box[0] + intercept - box[3]),
            )
            page.blocks.append(Block(box=box, lines=[line]))
    output_hocr.write_text(render_hocr(page), encoding="utf-8")
```

- [ ] **Step 4: Run the full suite**

Run: `pytest tests/ -q`
Expected: all PASS. Pay attention to `tests/test_parse_hocr_words.py` (round-trip through
the new markup), `tests/test_hocr_lang.py` (lang normalisation and injection tests now
flow through `render_hocr`), and `tests/test_generate_pdf_dispatch.py` (HocrParser and
Fpdf2PdfRenderer consume the new markup for real).

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/ocrmypdf_plugin.py tests/test_multilang_merge.py
git commit -m "Render merged hOCR through the typed Page model"
```
