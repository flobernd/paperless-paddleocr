# Banded Column Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct reading order for partial-column layouts (the business-letter case: a two-column sender/recipient
header above a full-width body), which the current page-global river detection cannot see.

**Architecture:** `layout.py` gains `reading_blocks(items, bbox_of, baseline_of)`, which groups rows into vertical
*bands*: maximal runs of consecutive rows sharing a persistent wide whitespace gap. Bands with a surviving gap split
into column blocks; all other rows flow full-width in position. `reading_blocks` supersedes `detect_columns`
(page-global coverage voting), which is deleted along with `_whitespace_rivers`; `ocrmypdf_plugin` rewires
`_words_to_text` and `_nms_merge` onto it. `group_into_lines` is untouched.

**Tech Stack:** Python 3.12, pytest.

**Dependency:** land `docs/superpowers/plans/2026-07-02-layout-correctness-fixes.md` first
(it fixes row assignment and ordering that this plan builds on).

## Global Constraints

- Python `>=3.12`; `ruff check .`, `ruff format --check .`, `mypy --ignore-missing-imports paperless_paddleocr` must pass.
- Tests must run without paddlepaddle/paddleocr installed.
- No em-dashes in prose or comments; comments explain WHY only.

---

### Task 1: `reading_blocks` in layout.py

**Files:**

- Modify: `paperless_paddleocr/paddle_engine/layout.py`
- Test: `tests/test_layout.py` (extend)

**Interfaces:**

- Produces: `reading_blocks[T](items: Sequence[T], bbox_of: Callable[[T], BBox], baseline_of: Callable[[T], float] |
  None = None) -> list[list[list[T]]]`
  returning blocks in reading order; each block is a list of lines; each line a list of
  items left-to-right. Constants `MIN_BAND_ROWS = 3`, `BAND_INTRUDER_RATIO = 0.1`.
- Consumes: `group_into_lines`, `_row_whitespace`, `MIN_GUTTER_PX`, `MIN_GUTTER_RATIO`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_layout.py`:

```python
from paperless_paddleocr.paddle_engine.layout import reading_blocks


def _two_col_rows(count, y_start=0):
    left, right, items = [], [], []
    for i in range(count):
        y0, y1 = y_start + i * 30, y_start + i * 30 + 20
        lw, rw = (0, y0, 160, y1), (240, y0, 400, y1)
        items += [lw, rw]
        left.append([lw])
        right.append([rw])
    return items, left, right


def test_reading_blocks_empty_and_single():
    assert reading_blocks([], bbox_of=_box) == []
    assert reading_blocks([(0, 0, 10, 10)], bbox_of=_box) == [[[(0, 0, 10, 10)]]]


def test_reading_blocks_single_column_is_one_flow_block():
    words = [(10, 0, 390, 20), (10, 40, 390, 60), (10, 80, 390, 100)]
    assert reading_blocks(words, bbox_of=_box) == [[[w] for w in words]]


def test_reading_blocks_two_columns():
    items, left, right = _two_col_rows(3)
    assert reading_blocks(items, bbox_of=_box) == [left, right]


def test_reading_blocks_header_columns_above_full_width_body():
    # The business-letter shape the page-global algorithm cannot handle:
    # sender/recipient blocks side by side, then a full-width body.
    items, left, right = _two_col_rows(4)
    body = [(0, 130, 400, 150), (0, 160, 400, 180), (0, 190, 400, 210)]
    blocks = reading_blocks(items + body, bbox_of=_box)
    assert blocks == [left, right, [[b] for b in body]]


def test_reading_blocks_full_width_title_precedes_columns():
    title = (0, -40, 400, -20)
    items, left, right = _two_col_rows(9)
    blocks = reading_blocks([title, *items], bbox_of=_box)
    assert blocks == [[[title]], left, right]


def test_reading_blocks_tolerates_a_gutter_straddling_row_inside_a_band():
    items, _, _ = _two_col_rows(4)
    straddler_row = [(0, 120, 160, 140), (150, 120, 250, 140), (240, 120, 400, 140)]
    more, _, _ = _two_col_rows(4, y_start=150)
    blocks = reading_blocks(items + straddler_row + more, bbox_of=_box)
    # One left block and one right block; the straddling middle word joins a
    # column by its x-centre, same policy as the old page-global algorithm.
    assert len(blocks) == 2
    flat = [w for block in blocks for line in block for w in line]
    assert set(flat) == set(items + straddler_row + more)


def test_reading_blocks_short_side_by_side_run_stays_flow():
    # Two side-by-side rows are below MIN_BAND_ROWS: not enough evidence for
    # columns, so they read line by line.
    items, _, _ = _two_col_rows(2)
    blocks = reading_blocks(items, bbox_of=_box)
    assert blocks == [[[items[0], items[1]], [items[2], items[3]]]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_layout.py -k reading_blocks -v`
Expected: FAIL with `ImportError: cannot import name 'reading_blocks'`

- [ ] **Step 3: Implement**

Add to `layout.py` (below `group_into_lines`, above the `_row_whitespace` helper it
reuses; move `_row_whitespace` up if needed so definitions precede use at import time -
plain function definitions have no ordering constraint, so appending at the end of the
file is also fine):

```python
#: A run of rows must be at least this long before a shared gap is trusted
#: as a column gutter; shorter runs are more likely justified text or tables.
MIN_BAND_ROWS = 3

#: Fraction of a band's rows allowed to intrude into its gutter (full-width
#: titles, gutter-straddling words) without ending the band.
BAND_INTRUDER_RATIO = 0.1


def _wide_gaps(
    row: list[Any],
    bbox_of: Callable[[Any], BBox],
    span_x0: int,
    span_x1: int,
    min_gutter: int,
) -> list[tuple[float, float]]:
    return [
        (a, b)
        for a, b in _row_whitespace(row, bbox_of, span_x0, span_x1)
        if (b - a) >= min_gutter
    ]


def _intersect_gaps(
    a: list[tuple[float, float]],
    b: list[tuple[float, float]],
    min_gutter: int,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a0, a1 in a:
        for b0, b1 in b:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi - lo >= min_gutter:
                out.append((lo, hi))
    return out


def _split_bands(
    rows: list[list[Any]],
    row_gaps: list[list[tuple[float, float]]],
    min_gutter: int,
) -> list[tuple[list[list[Any]], list[tuple[float, float]] | None]]:
    """Partition rows into (rows, gutters) bands, top to bottom.

    A band is a maximal run of consecutive rows whose wide gaps share a
    common interval. Rows outside any band come back with ``None`` gutters
    and read as full-width flow. An occasional intruding row (title,
    straddling word) is absorbed when the gap resumes on the very next row,
    up to BAND_INTRUDER_RATIO of the band.
    """
    bands: list[tuple[list[list[Any]], list[tuple[float, float]] | None]] = []
    flow: list[list[Any]] = []
    i, n = 0, len(rows)
    while i < n:
        if not row_gaps[i]:
            flow.append(rows[i])
            i += 1
            continue
        common = row_gaps[i]
        band = [rows[i]]
        intruders = 0
        j = i + 1
        while j < n:
            nxt = _intersect_gaps(common, row_gaps[j], min_gutter)
            if nxt:
                common = nxt
                band.append(rows[j])
                j += 1
                continue
            allowance = max(1, int(BAND_INTRUDER_RATIO * len(band)))
            if (
                j + 1 < n
                and intruders + 1 <= allowance
                and _intersect_gaps(common, row_gaps[j + 1], min_gutter)
            ):
                band.append(rows[j])
                intruders += 1
                j += 1
                continue
            break
        if len(band) - intruders >= MIN_BAND_ROWS:
            if flow:
                bands.append((flow, None))
                flow = []
            bands.append((band, common))
            i = j
        else:
            # Not enough evidence for columns; retry a band from the next row.
            flow.append(rows[i])
            i += 1
    if flow:
        bands.append((flow, None))
    return bands


def _split_band_columns(
    band_rows: list[list[Any]],
    gutters: list[tuple[float, float]],
    bbox_of: Callable[[Any], BBox],
) -> list[list[list[Any]]]:
    cuts = sorted((a + b) / 2.0 for a, b in gutters)
    columns: list[list[list[Any]]] = [[] for _ in range(len(cuts) + 1)]
    for row in band_rows:
        parts: list[list[Any]] = [[] for _ in columns]
        for item in row:
            x0, _, x1, _ = bbox_of(item)
            center = (x0 + x1) / 2.0
            parts[sum(1 for cut in cuts if center >= cut)].append(item)
        for idx, part in enumerate(parts):
            if part:
                columns[idx].append(part)
    return [col for col in columns if col]


def reading_blocks[T](
    items: Sequence[T],
    bbox_of: Callable[[T], BBox],
    baseline_of: Callable[[T], float] | None = None,
) -> list[list[list[T]]]:
    """Blocks of reading lines in reading order.

    Columns are detected per vertical band rather than page-globally, so a
    two-column header above a full-width body reads header-left,
    header-right, body instead of interleaving the header line by line.
    A single-column page returns one flow block, so plain documents are
    unaffected.
    """
    item_list = list(items)
    if not item_list:
        return []
    rows = group_into_lines(item_list, bbox_of, baseline_of)
    if len(rows) < 2:
        return [rows]

    boxes = [bbox_of(it) for it in item_list]
    span_x0 = min(b[0] for b in boxes)
    span_x1 = max(b[2] for b in boxes)
    if span_x1 <= span_x0:
        return [rows]
    min_gutter = max(MIN_GUTTER_PX, int((span_x1 - span_x0) * MIN_GUTTER_RATIO))

    row_gaps = [_wide_gaps(row, bbox_of, span_x0, span_x1, min_gutter) for row in rows]
    blocks: list[list[list[T]]] = []
    for band_rows, gutters in _split_bands(rows, row_gaps, min_gutter):
        if gutters is None:
            blocks.append(band_rows)
            continue
        columns = _split_band_columns(band_rows, gutters, bbox_of)
        if len(columns) < 2:
            # A gutter that puts everything on one side (a margin artefact)
            # is not a column separator.
            blocks.append(band_rows)
        else:
            blocks.extend(columns)
    return blocks
```

- [ ] **Step 4: Run the new tests**

Run: `pytest tests/test_layout.py -k reading_blocks -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/paddle_engine/layout.py tests/test_layout.py
git commit -m "Add band-based column detection to layout"
```

---

### Task 2: Rewire the plugin onto reading_blocks

**Files:**

- Modify: `paperless_paddleocr/ocrmypdf_plugin.py:42, 335-375`
  (`_words_to_text`, `_nms_merge`, imports)
- Test: `tests/test_words_to_text_columns.py` (extend)

**Interfaces:**

- Consumes: `layout.reading_blocks` from Task 1.
- `_words_to_text(words: list[_Word]) -> str` and
  `_nms_merge(words, iou_threshold) -> list[_Word]` keep their signatures.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_words_to_text_columns.py`:

```python
def test_words_to_text_letter_header_reads_blockwise():
    words = []
    for i in range(3):
        y0, y1 = i * 30, i * 30 + 20
        words += [
            _Word(f"sender{i + 1}", 0, y0, 160, y1, 90),
            _Word(f"recipient{i + 1}", 240, y0, 400, y1, 90),
        ]
    words += [
        _Word("body1", 0, 130, 400, 150, 90),
        _Word("body2", 0, 160, 400, 180, 90),
        _Word("body3", 0, 190, 400, 210, 90),
    ]
    assert _words_to_text(words) == (
        "sender1\nsender2\nsender3"
        "\n\nrecipient1\nrecipient2\nrecipient3"
        "\n\nbody1\nbody2\nbody3"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_words_to_text_columns.py -v`
Expected: the new test FAILS (current output interleaves
`sender1 recipient1` per line); existing tests PASS.

- [ ] **Step 3: Implement**

In `ocrmypdf_plugin.py`, change the layout import to:

```python
from paperless_paddleocr.paddle_engine.layout import reading_blocks
```

Replace `_words_to_text`:

```python
def _words_to_text(words: list[_Word]) -> str:
    """Render OCR words to plain text in banded reading order.

    Blocks come from :func:`layout.reading_blocks` (column bands plus
    full-width flow); lines are space-joined, blocks blank-line-separated so
    downstream consumers (paperless UI, full-text indexer) see paragraphs.
    """
    if not words:
        return ""
    return "\n\n".join(
        "\n".join(" ".join(w.text for w in line) for line in block)
        for block in reading_blocks(words, bbox_of=_word_bbox)
    )
```

Replace the reordering tail of `_nms_merge`:

```python
    ordered: list[_Word] = []
    for block in reading_blocks(kept, bbox_of=_word_bbox):
        for line in block:
            ordered.extend(line)
    return ordered
```

Update both functions' docstrings if they still reference `detect_columns`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_words_to_text_columns.py tests/test_multilang_merge.py tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/ocrmypdf_plugin.py tests/test_words_to_text_columns.py
git commit -m "Rewire sidecar text and NMS ordering onto reading_blocks"
```

---

### Task 3: Delete detect_columns and port its remaining tests

**Files:**

- Modify: `paperless_paddleocr/paddle_engine/layout.py` (delete `detect_columns`,
  `_whitespace_rivers`, `RIVER_COVERAGE`)
- Modify: `tests/test_layout.py` (delete the `detect_columns` test section; the
  behaviours live on as the `reading_blocks` tests from Task 1)

- [ ] **Step 1: Confirm nothing else references the deleted names**

Run: `grep -rn "detect_columns\|_whitespace_rivers\|RIVER_COVERAGE" paperless_paddleocr tests`
Expected: only `layout.py` definitions and `tests/test_layout.py` remain (the plugin was
rewired in Task 2; `vl.py` uses only `group_into_lines` and `fit_baseline`).

If `docs/superpowers/plans/2026-07-02-page-level-reading-order.md` has already been
implemented against `detect_columns`, stop and reconcile first; it is written against
`reading_blocks`, so normally this does not arise.

- [ ] **Step 2: Delete the functions and tests**

Remove from `layout.py`: the `RIVER_COVERAGE` constant, `_whitespace_rivers`, and
`detect_columns` (keep `_row_whitespace`; `_wide_gaps` uses it). Update the module
docstring: replace the mention of column rivers with a sentence about band-based
column detection.

Remove from `tests/test_layout.py`: `test_detect_columns_single_column`,
`test_detect_columns_two_columns`, `test_detect_columns_tolerates_a_full_width_title`,
`test_detect_columns_tolerates_a_gutter_straddling_word`,
`test_detect_columns_offset_page_measures_gutter_against_span_width`, and the
`detect_columns` import.

- [ ] **Step 3: Run the full suite and linters**

Run: `pytest tests/ -q && ruff check . && mypy --ignore-missing-imports paperless_paddleocr`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add paperless_paddleocr/paddle_engine/layout.py tests/test_layout.py
git commit -m "Remove page-global column detection superseded by bands"
```

---

### Task 4: Document the behaviour

**Files:**

- Modify: `README.md` ("Performance notes", sidecar bullet)

- [ ] **Step 1: Update the sidecar bullet**

Replace the "**Sidecar text** is rendered with column-aware reading order ..." bullet
with:

```markdown
- **Sidecar text** is rendered in banded reading order: columns are detected per vertical
  band, so a two-column letter head above a full-width body reads sender block, recipient
  block, body. Detection stays conservative (gutters narrower than 4% of the text span
  are ignored, and a side-by-side run shorter than 3 lines is not treated as columns).
  Rotated text and wrap-around figures can still produce odd ordering - open an issue
  with a fixture if you hit something that should be fixable.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Document banded column reading order"
```
