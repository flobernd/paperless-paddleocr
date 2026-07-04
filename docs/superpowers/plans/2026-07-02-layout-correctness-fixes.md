# Layout Correctness Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four reading-order/geometry defects: zero-overlap row joining, the gutter minimum-width formula, the row
ordering key, and degenerate VL line boxes.

**Architecture:** All changes are local to `paddle_engine/layout.py` and `paddle_engine/vl.py`; the public function
signatures do not change. Each fix lands with a regression test derived from the concrete failure construction.

**Tech Stack:** Python 3.12, pytest. No paddle imports needed.

## Global Constraints

- Python `>=3.12`; `ruff check .`, `ruff format --check .`, `mypy --ignore-missing-imports paperless_paddleocr` must pass.
- Tests must run without paddlepaddle/paddleocr installed.
- No em-dashes in prose or comments; comments explain WHY only.

---

### Task 1: A word overlapping no row must start its own row

**Files:**

- Modify: `paperless_paddleocr/paddle_engine/layout.py:156-174` (`_assign_overlap_rows`)
- Test: `tests/test_layout.py` (extend)

**Interfaces:**

- Produces: unchanged signature `group_into_lines(items, bbox_of, baseline_of=None)`.

- [ ] **Step 1: Write the failing test**

Append to the `group_into_lines` section of `tests/test_layout.py`:

```python
def test_group_into_lines_word_with_no_overlap_starts_its_own_row():
    # A tall first row and a small word far below it. The small word overlaps
    # nothing, but its height (20) is under OVERLAP_TOLERANCE * 60, so the
    # buggy zero-overlap path glued it into the first row.
    title = (0, 0, 200, 60)
    page_number = (0, 200, 20, 220)
    assert group_into_lines([title, page_number], bbox_of=_box) == [[title], [page_number]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_layout.py::test_group_into_lines_word_with_no_overlap_starts_its_own_row -v`
Expected: FAIL with `assert [[(0, 0, 200, 60), (0, 200, 20, 220)]] == [[(0, 0, 200, 60)], [(0, 200, 20, 220)]]`

- [ ] **Step 3: Implement the fix**

In `_assign_overlap_rows`, change:

```python
        if best_row is not None:
            non_overlap = span.height - best_ov
```

to:

```python
        # A row candidate needs actual vertical overlap: with best_ov == 0 the
        # tolerance test degenerates to a pure height comparison and glues
        # isolated small words (page numbers, footnote marks) onto whichever
        # row happens to be first in the list.
        if best_row is not None and best_ov > 0.0:
            non_overlap = span.height - best_ov
```

- [ ] **Step 4: Run the layout tests**

Run: `pytest tests/test_layout.py -v`
Expected: all PASS (existing tests rely on merge/reassign passes, not the zero-overlap
path, so they are unaffected).

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/paddle_engine/layout.py tests/test_layout.py
git commit -m "Require vertical overlap before joining a word to a row"
```

---

### Task 2: Minimum gutter width must use the span width

**Files:**

- Modify: `paperless_paddleocr/paddle_engine/layout.py:320` (`detect_columns`)
- Test: `tests/test_layout.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to the `detect_columns` section of `tests/test_layout.py`:

```python
def test_detect_columns_offset_page_measures_gutter_against_span_width():
    # Text span sits at x 2000..2400 (a crop or right-shifted scan). The
    # gutter is 40 px = 10% of the span width, but only 1.7% of the raw
    # right-edge coordinate, which the buggy formula used as "page width".
    items = []
    for i in range(3):
        y0, y1 = i * 30, i * 30 + 20
        items += [(2000, y0, 2160, y1), (2200, y0, 2400, y1)]
    assert len(detect_columns(items, bbox_of=_box)) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_layout.py::test_detect_columns_offset_page_measures_gutter_against_span_width -v`
Expected: FAIL with `assert 1 == 2`

- [ ] **Step 3: Implement the fix**

In `detect_columns`, change:

```python
    min_gutter = max(MIN_GUTTER_PX, int(span_x1 * MIN_GUTTER_RATIO))
```

to:

```python
    min_gutter = max(MIN_GUTTER_PX, int((span_x1 - span_x0) * MIN_GUTTER_RATIO))
```

- [ ] **Step 4: Run the layout tests**

Run: `pytest tests/test_layout.py -v`
Expected: all PASS (the existing column tests have `span_x0 == 0`, where both formulas
agree).

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/paddle_engine/layout.py tests/test_layout.py
git commit -m "Measure minimum column gutter against the text span width"
```

---

### Task 3: Order rows by mean baseline, not extrapolated intercept

**Files:**

- Modify: `paperless_paddleocr/paddle_engine/layout.py:211-214` (`_order`)
- Test: `tests/test_layout.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_skewed_line_reads_after_a_short_line_sitting_above_it():
    # A gently sloped line in a right-hand column plus a short word fully
    # above it. The sloped row's baseline extrapolated to x=0 (about 98) is
    # smaller than the short row's local baseline (110), so intercept
    # ordering read the sloped line first even though it sits below.
    line_a = [(1550, 110, 1650, 130), (1950, 118, 2050, 138), (2350, 126, 2450, 146)]
    word_b = (1550, 90, 1650, 110)
    lines = group_into_lines([*line_a, word_b], bbox_of=_box)
    flat = [w for line in lines for w in line]
    assert flat == [word_b, *line_a]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_layout.py::test_skewed_line_reads_after_a_short_line_sitting_above_it -v`
Expected: FAIL - `word_b` does not come first in the flattened order.

- [ ] **Step 3: Implement the fix**

Replace `_order`:

```python
def _order(rows: list[_Row]) -> list[list[Any]]:
    """Step 6 -- rows top-to-bottom, items left-to-right.

    Sorted by mean member baseline rather than the fitted intercept: the
    intercept extrapolates to x = 0, which on a skewed page shifts long rows
    by slope * x relative to short rows evaluated at their own position.
    The mean baseline compares rows in the same frame the merge pass uses.
    """
    ordered = sorted(rows, key=lambda r: r.mean_baseline)
    return [[span.item for span in sorted(row.spans, key=lambda s: s.bbox[0])] for row in ordered]
```

- [ ] **Step 4: Run the full suite**

Run: `pytest tests/ -q`
Expected: all PASS (`test_words_to_text_columns.py` and `test_layout.py` cover the
consumers).

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/paddle_engine/layout.py tests/test_layout.py
git commit -m "Order reading rows by mean baseline"
```

---

### Task 4: Non-degenerate VL ocr-mode line boxes

**Files:**

- Modify: `paperless_paddleocr/paddle_engine/vl.py:210-256` (`_ocr_page`)
- Test: `tests/test_vl_ocr_page.py` (create)

**Interfaces:**

- `_ocr_page(parsing_res: Any, page: Page) -> None` appends `Block`s to `page`; boxes are
  `(x0, y0, x1, y1)` int tuples with `y1 > y0` for every line.

- [ ] **Step 1: Write the failing test**

Create `tests/test_vl_ocr_page.py`:

```python
"""_ocr_page must never emit zero-height line boxes.

The old integer division (block_height // line_count) collapsed every line
of a block shorter in pixels than its line count.
"""

from __future__ import annotations

from paperless_paddleocr.paddle_engine.hocr import Page
from paperless_paddleocr.paddle_engine.vl import _ocr_page


def _page() -> Page:
    return Page(width=100, height=50, lang="en", ocr_system="test")


def test_ocr_page_line_boxes_are_never_degenerate():
    page = _page()
    block = {
        "block_label": "text",
        "block_content": "one\ntwo\nthree",
        "block_bbox": [0, 10, 100, 12],  # 2 px tall, 3 text lines
    }
    _ocr_page([block], page)
    lines = [line for blk in page.blocks for line in blk.lines]
    assert len(lines) == 3
    for line in lines:
        assert line.box[3] > line.box[1]


def test_ocr_page_line_boxes_partition_the_block_height():
    page = _page()
    block = {
        "block_label": "text",
        "block_content": "one\ntwo\nthree",
        "block_bbox": [0, 0, 100, 30],
    }
    _ocr_page([block], page)
    boxes = [line.box for blk in page.blocks for line in blk.lines]
    assert boxes == [(0, 0, 100, 10), (0, 10, 100, 20), (0, 20, 100, 30)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vl_ocr_page.py -v`
Expected: `test_ocr_page_line_boxes_are_never_degenerate` FAILS (zero-height boxes);
the partition test may pass already.

- [ ] **Step 3: Implement the fix**

In `_ocr_page`, replace:

```python
        line_h = max(by1 - by0, 1) // len(text_lines)

        lines: list[Line] = []
        for i, line_text in enumerate(text_lines):
            tokens = line_text.split()
            if not tokens:
                continue
            ly0 = by0 + i * line_h
            ly1 = min(by0 + (i + 1) * line_h, by1)
```

with:

```python
        block_h = max(by1 - by0, 1)
        n_lines = len(text_lines)

        lines: list[Line] = []
        for i, line_text in enumerate(text_lines):
            tokens = line_text.split()
            if not tokens:
                continue
            # Per-line boundary arithmetic instead of one truncated line
            # height: rounding never accumulates and a block shorter in
            # pixels than its line count still yields non-degenerate boxes.
            ly0 = by0 + (i * block_h) // n_lines
            ly1 = by0 + ((i + 1) * block_h) // n_lines
            if ly1 <= ly0:
                ly1 = ly0 + 1
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_vl_ocr_page.py tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/paddle_engine/vl.py tests/test_vl_ocr_page.py
git commit -m "Prevent degenerate VL ocr-mode line boxes"
```
