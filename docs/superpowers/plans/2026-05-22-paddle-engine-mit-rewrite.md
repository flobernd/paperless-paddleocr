# Independent MIT Rewrite of `paddle_engine` - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the MPL-2.0-derived geometry/layout logic in `paperless_paddleocr/paddle_engine/` (and the layout helpers in `ocrmypdf_plugin.py`) with an independently designed, Tesseract-grade implementation so the whole project ships cleanly under MIT.

**Architecture:** A pure-math `geometry.py` (polygon → bounding box) feeds a new Tesseract-grade `layout.py` (baseline fitting, line grouping, whitespace-river column detection). `layout.py` is generic over the caller's word type via two adapter callables, so both the VL adapter and the multi-language merge use it without a shared base class. `hocr.py`'s `Line` gains a fitted `baseline`; the classic and VL adapters fit it; `render_hocr` emits it.

**Tech Stack:** Python 3.12, ocrmypdf 17.x, lxml, fpdf2 (via ocrmypdf), Pillow, pytest. PaddleOCR is an optional runtime dependency - it is **not** installed in the test environment, so all unit tests are written against modules that import cleanly without it.

---

## Design reference

Implements `docs/superpowers/specs/2026-05-22-paddle-engine-mit-rewrite-design.md`. Read it before starting - section numbers below (`§6`, `§7.3`, …) refer to that spec.

## File structure

| File | Action | Responsibility |
|---|---|---|
| `paperless_paddleocr/paddle_engine/geometry.py` | Rewrite | `BBox`, `poly_to_bbox` (exact min/max envelope), `estimate_word_boxes` (length-weighted partition). Pure math only. |
| `paperless_paddleocr/paddle_engine/layout.py` | Create | `fit_baseline`, `group_into_lines`, `detect_columns` - Tesseract-grade reconstruction, generic over the caller's word type. |
| `paperless_paddleocr/paddle_engine/hocr.py` | Modify | `Line` gains `baseline: tuple[float, float]`; `render_hocr` emits it. |
| `paperless_paddleocr/paddle_engine/classic.py` | Modify | Fit each region's baseline via `layout.fit_baseline`. |
| `paperless_paddleocr/paddle_engine/vl.py` | Modify | `_spotting_page` clusters words via `layout.group_into_lines` and carries the fitted baseline. |
| `paperless_paddleocr/ocrmypdf_plugin.py` | Modify | Rewire `_words_to_text` / `_nms_merge` reading order to `layout.py`; delete the old bucket loops. |
| `tests/test_geometry.py` | Create | Unit tests for `geometry.py`. |
| `tests/test_layout.py` | Create | Unit tests for `layout.py`. |
| `tests/test_hocr_render.py` | Create | `render_hocr` emits the `baseline` attribute. |
| `tests/test_classic_baseline.py` | Create | `classic._region_baseline` straight-baseline fit. |
| `tests/test_vl_baseline.py` | Create | `vl._poly_baseline_y` bottom-edge extraction. |
| `tests/test_words_to_text_columns.py` | Create | `_words_to_text` column-aware reading order. |

**Files deliberately NOT changed:** `engine.py`, `pdf.py`, `__init__.py`. Per spec §4 these are API-dictated plumbing (the ocrmypdf `OcrEngine` ABC, the ocrmypdf-17 renderer surface, a package re-export) - there is one correct way to write them and they contain no creative geometry/layout expression, so they need no rewrite.

**Working-tree note:** the repo has unrelated pre-existing uncommitted changes (`parser.py`, `pyproject.toml`, `.github/workflows/ci.yml`, a prior edit to `ocrmypdf_plugin.py`, the `LICENSE.MPL-2.0` deletion, a stray `empty.txt`). Each task below commits **only** the files it touches via an explicit `git add`. Do not `git add -A`. Task 8 handles the licensing-file cleanup.

**Verification commands** (run after every implementation step that changes Python):

```bash
ruff check paperless_paddleocr tests
mypy --ignore-missing-imports paperless_paddleocr
```

`pytest` may not be on PATH locally - if a `pytest` invocation fails with "command not found", run `pip install pytest` first (the CI image installs it fresh). PaddleOCR is intentionally absent; never `pip install paddleocr` to make a test pass.

---

## Task 1: `geometry.py` - exact envelope + weighted partition

**Files:**
- Rewrite: `paperless_paddleocr/paddle_engine/geometry.py`
- Test: `tests/test_geometry.py`

This replaces the corner-averaging `poly_to_bbox` with an exact min/max envelope (§6.2), reworks `estimate_word_boxes` to the length-weighted model (§6.3), and **removes** `group_words_into_lines` and its `_center_y` / `_quad_height` / `_left_x` / `_points` helpers - that logic moves to `layout.py` in Task 2.

- [ ] **Step 1: Write the failing test**

Create `tests/test_geometry.py`:

```python
"""Unit tests for paddle_engine.geometry pure polygon math."""

from __future__ import annotations

from paperless_paddleocr.paddle_engine.geometry import (
    estimate_word_boxes,
    poly_to_bbox,
)


def test_poly_to_bbox_axis_aligned_quad():
    poly = [[10, 20], [110, 20], [110, 60], [10, 60]]
    assert poly_to_bbox(poly) == (10, 20, 110, 60)


def test_poly_to_bbox_skewed_quad_uses_min_max_envelope():
    # both edges tilt; the envelope is the outermost corner on each side
    poly = [[10, 22], [110, 18], [112, 58], [8, 62]]
    assert poly_to_bbox(poly) == (8, 18, 112, 62)


def test_poly_to_bbox_non_quad_polygon():
    poly = [[5, 5], [40, 2], [60, 30], [25, 50], [0, 25]]
    assert poly_to_bbox(poly) == (0, 2, 60, 50)


def test_estimate_word_boxes_single_word_fills_box():
    assert estimate_word_boxes(["hello"], (10, 5, 210, 45)) == [(10, 5, 210, 45)]


def test_estimate_word_boxes_empty_input():
    assert estimate_word_boxes([], (0, 0, 100, 20)) == []


def test_estimate_word_boxes_partitions_and_snaps_last_word():
    boxes = estimate_word_boxes(["aa", "bbbb"], (0, 0, 120, 10))
    # every word inherits the line's y0 / y1
    assert all(b[1] == 0 and b[3] == 10 for b in boxes)
    # words run left-to-right starting at the box left edge
    assert boxes[0][0] == 0
    assert boxes[0][2] <= boxes[1][0]
    # the longer word is wider than the shorter one
    assert (boxes[1][2] - boxes[1][0]) > (boxes[0][2] - boxes[0][0])
    # the last word's right edge is snapped to the box right edge
    assert boxes[-1][2] == 120
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_geometry.py -v`
Expected: FAIL - `test_poly_to_bbox_skewed_quad_uses_min_max_envelope` fails (the old corner-averaging `poly_to_bbox` returns averaged y bounds, not `18`/`62`) and `test_estimate_word_boxes_partitions_and_snaps_last_word` may fail on the new weighting.

- [ ] **Step 3: Rewrite `geometry.py`**

Replace the entire contents of `paperless_paddleocr/paddle_engine/geometry.py` with:

```python
"""Pure polygon geometry for turning PaddleOCR detections into hOCR boxes.

PaddleOCR emits text regions and words as polygons -- usually a 4-point
quad in top-left, top-right, bottom-right, bottom-left order, either as a
plain nested list or a numpy array, so the point accessor stays
duck-typed. These functions are stateless and carry no layout or
reading-order logic; line and column reconstruction lives in
:mod:`paperless_paddleocr.paddle_engine.layout`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: ``(x0, y0, x1, y1)`` axis-aligned box in image pixels.
BBox = tuple[int, int, int, int]

#: Weight of one inter-word gap relative to a single glyph when
#: :func:`estimate_word_boxes` partitions a line -- a space is narrower
#: than an average glyph.
_GAP_WEIGHT = 0.5


def poly_to_bbox(poly: Any) -> BBox:
    """Return the exact axis-aligned envelope of a detection polygon.

    The smallest axis-aligned box that contains every polygon point:
    ``min`` / ``max`` over the xs and ys, cast to ``int``. This is the most
    accurate axis-aligned representation of the detection. Baseline skew is
    handled separately by :mod:`paperless_paddleocr.paddle_engine.layout`,
    which reads the polygon corners directly, so no corner-averaging is
    done here. Duck-typed over nested lists and numpy arrays.
    """
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def estimate_word_boxes(words: Sequence[str], box: BBox) -> list[BBox]:
    """Spread ``words`` across a line ``box`` by a length-weighted partition.

    Fallback only -- used when an engine returns a line transcription but
    no per-word boxes (the VL ``ocr`` response). The invisible text layer
    still needs *a* position per word, so the line box width is divided up:

    * a word's weight is ``max(1, len(word))``;
    * each inter-word gap weighs ``0.5`` (a space is narrower than a glyph);
    * each width is proportional to its weight over the total.

    The last word's right edge is snapped to the ``box`` right edge so
    rounding never leaves a gap. Every word inherits the line ``box``'s
    ``y0`` / ``y1``. A single word fills the whole box; empty input returns
    ``[]``. Rough by design -- it backs the invisible layer, not layout
    analysis.
    """
    if not words:
        return []
    x0, y0, x1, y1 = box
    if len(words) == 1:
        return [(x0, y0, x1, y1)]

    line_width = x1 - x0
    weights = [max(1, len(w)) for w in words]
    total = sum(weights) + _GAP_WEIGHT * (len(words) - 1)
    gap_width = round(line_width * _GAP_WEIGHT / total) if total > 0 else 0

    boxes: list[BBox] = []
    cursor = x0
    last = len(words) - 1
    for i, weight in enumerate(weights):
        width = round(line_width * weight / total) if total > 0 else 0
        right = x1 if i == last else cursor + width
        boxes.append((cursor, y0, right, y1))
        cursor = right + gap_width
    return boxes
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_geometry.py -v`
Expected: PASS - all 6 tests.

- [ ] **Step 5: Lint and type-check**

Run: `ruff check paperless_paddleocr/paddle_engine/geometry.py tests/test_geometry.py && mypy --ignore-missing-imports paperless_paddleocr/paddle_engine/geometry.py`
Expected: no errors. (`group_words_into_lines` is now gone; `vl.py` still imports it and will be fixed in Task 5 - `mypy` on the single file above is clean, a repo-wide `mypy` is expected to flag `vl.py` until Task 5.)

- [ ] **Step 6: Commit**

```bash
git add paperless_paddleocr/paddle_engine/geometry.py tests/test_geometry.py
git commit -m "Rewrite geometry.py with exact envelope and weighted word partition"
```

---

## Task 2: `layout.py` - Tesseract-grade line & column reconstruction

**Files:**
- Create: `paperless_paddleocr/paddle_engine/layout.py`
- Test: `tests/test_layout.py`

The creative core (§7). One module with three public functions, generic over the caller's word type via `bbox_of` / `baseline_of` adapter callables.

- [ ] **Step 1: Write the failing test**

Create `tests/test_layout.py`:

```python
"""Unit tests for paddle_engine.layout line and column reconstruction.

Test items are bare ``BBox`` tuples; the identity ``_box`` adapter keeps
the cases readable. ``baseline_of`` is left to default (the bbox bottom).
"""

from __future__ import annotations

from paperless_paddleocr.paddle_engine.layout import (
    detect_columns,
    fit_baseline,
    group_into_lines,
)


def _box(b):
    return b


# --- fit_baseline ---------------------------------------------------------

def test_fit_baseline_flat_line():
    slope, intercept = fit_baseline([(0.0, 50.0), (100.0, 50.0), (200.0, 50.0)])
    assert abs(slope) < 1e-9
    assert abs(intercept - 50.0) < 1e-9


def test_fit_baseline_sloped_line():
    # points lie exactly on y = 2x + 10
    slope, intercept = fit_baseline([(0.0, 10.0), (10.0, 30.0), (20.0, 50.0)])
    assert abs(slope - 2.0) < 1e-9
    assert abs(intercept - 10.0) < 1e-9


def test_fit_baseline_single_point_is_flat():
    assert fit_baseline([(7.0, 42.0)]) == (0.0, 42.0)


def test_fit_baseline_empty_is_zero():
    assert fit_baseline([]) == (0.0, 0.0)


def test_fit_baseline_all_same_x_is_flat_through_mean():
    slope, intercept = fit_baseline([(5.0, 10.0), (5.0, 20.0), (5.0, 30.0)])
    assert slope == 0.0
    assert abs(intercept - 20.0) < 1e-9


# --- group_into_lines -----------------------------------------------------

def test_group_into_lines_empty():
    assert group_into_lines([], bbox_of=_box) == []


def test_group_into_lines_single_item():
    assert group_into_lines([(0, 0, 10, 10)], bbox_of=_box) == [[(0, 0, 10, 10)]]


def test_group_into_lines_two_separate_lines_ordered():
    a1, a2 = (0, 10, 40, 30), (50, 10, 90, 30)
    b1, b2 = (0, 60, 40, 80), (50, 60, 90, 80)
    # deliberately shuffled input
    lines = group_into_lines([b2, a2, b1, a1], bbox_of=_box)
    assert lines == [[a1, a2], [b1, b2]]


def test_group_into_lines_mixed_font_sizes_share_one_line():
    normal = (0, 10, 40, 30)   # height 20, bottom 30
    tall = (50, 0, 90, 30)     # height 30, bottom 30 -- same baseline
    assert group_into_lines([normal, tall], bbox_of=_box) == [[normal, tall]]


def test_group_into_lines_merges_a_jitter_split_line():
    # the greedy overlap pass splits these; the merge pass recombines them
    left = (0, 10, 40, 30)     # baseline 30
    right = (50, 18, 90, 38)   # baseline 38, overlaps too little to join
    assert group_into_lines([left, right], bbox_of=_box) == [[left, right]]


def test_group_into_lines_handles_a_slightly_rotated_page():
    top = [(0, 10, 40, 30), (60, 14, 100, 34), (120, 18, 160, 38)]
    bottom = [(0, 210, 40, 230), (60, 214, 100, 234), (120, 218, 160, 238)]
    assert group_into_lines(top + bottom, bbox_of=_box) == [top, bottom]


def test_group_into_lines_assigns_boundary_word_by_baseline():
    top = [(0, 0, 50, 20), (60, 0, 110, 20), (120, 0, 170, 20)]
    bottom = [(0, 21, 50, 41), (60, 21, 110, 41)]
    # a tall word straddling both bands; its bottom (41) is the lower baseline
    straddler = (120, 12, 170, 41)
    lines = group_into_lines(top + bottom + [straddler], bbox_of=_box)
    assert lines == [top, [*bottom, straddler]]


# --- detect_columns -------------------------------------------------------

def test_detect_columns_single_column():
    words = [(10, 0, 90, 20), (10, 40, 90, 60)]
    assert detect_columns(words, bbox_of=_box) == [words]


def test_detect_columns_two_columns():
    rows, left, right = [], [], []
    for i in range(3):
        y0, y1 = i * 30, i * 30 + 20
        lw, rw = (0, y0, 160, y1), (240, y0, 400, y1)
        rows += [lw, rw]
        left.append(lw)
        right.append(rw)
    assert detect_columns(rows, bbox_of=_box) == [left, right]


def test_detect_columns_tolerates_a_full_width_title():
    items = [(0, -40, 400, -20)]  # one full-width title row
    for i in range(9):
        y0, y1 = i * 30, i * 30 + 20
        items += [(0, y0, 160, y1), (240, y0, 400, y1)]
    assert len(detect_columns(items, bbox_of=_box)) == 2


def test_detect_columns_tolerates_a_gutter_straddling_word():
    items = []
    for i in range(9):
        y0, y1 = i * 30, i * 30 + 20
        items += [(0, y0, 160, y1), (240, y0, 400, y1)]
    # one extra row whose middle word bridges the gutter
    items += [(0, 270, 160, 290), (150, 270, 250, 290), (240, 270, 400, 290)]
    assert len(detect_columns(items, bbox_of=_box)) == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_layout.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'paperless_paddleocr.paddle_engine.layout'`.

- [ ] **Step 3: Create `layout.py`**

Create `paperless_paddleocr/paddle_engine/layout.py` with:

```python
"""Tesseract-grade line and column reconstruction.

PaddleOCR's word-spotting response and the multi-language merge both hand
over a loose bag of recognised words; this module rebuilds the reading
structure -- which words share a line, which lines share a column, and the
order to read them in.

The algorithm is adapted from Tesseract's ``textord/makerow.cpp``
row-finding, operating at *word* granularity rather than connected-component
blob granularity. The public functions are generic over the caller's word
type via two adapter callables, so neither :mod:`vl` nor
:mod:`ocrmypdf_plugin` needs a shared base class:

* ``bbox_of(item) -> BBox`` -- the item's axis-aligned box.
* ``baseline_of(item) -> float`` -- the baseline y at the item's horizontal
  centre; defaults to the bbox bottom edge (``y1``) when ``None``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from paperless_paddleocr.paddle_engine.geometry import BBox

T = TypeVar("T")

#: A word's vertical band may stick out of its row's band by this fraction
#: of the row height and still join it. Tesseract ``textord_overlap_x``.
OVERLAP_TOLERANCE = 0.375

#: In the reassignment pass a word moves to a better-fitting baseline only
#: when the predicted-vs-actual gap is within this fraction of its height.
REASSIGN_TOLERANCE = 0.5

#: Two rows merge when their mean baselines sit within this fraction of the
#: shorter row's height.
MERGE_TOLERANCE = 0.5

#: A vertical whitespace strip counts as a column gutter when whitespace
#: covers at least this fraction of the rows crossing it.
RIVER_COVERAGE = 0.9

#: Minimum gutter width as a fraction of the page text span; 4% skips the
#: gaps inside justified paragraphs and catches real columns.
MIN_GUTTER_RATIO = 0.04

#: Absolute floor for a gutter width, in pixels.
MIN_GUTTER_PX = 8


def fit_baseline(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Ordinary-least-squares fit of ``y = slope * x + intercept``.

    Minimises vertical residuals -- the maximum-likelihood baseline under
    Gaussian box noise. Degenerate input (fewer than two points, or every
    point sharing one x) returns a flat line through the mean y.
    """
    n = len(points)
    if n == 0:
        return 0.0, 0.0
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    if n == 1:
        return 0.0, float(sy)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, sy / n
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


@dataclass
class _Span:
    """One input item plus the geometry the row algorithm needs."""

    item: Any
    bbox: BBox
    baseline_y: float
    x_center: float
    height: float


@dataclass
class _Row:
    """A growing reading row: its members, vertical band, and baseline fit."""

    spans: list[_Span] = field(default_factory=list)
    y0: float = 0.0
    y1: float = 0.0
    slope: float = 0.0
    intercept: float = 0.0

    @property
    def height(self) -> float:
        return max(1.0, self.y1 - self.y0)

    @property
    def mean_baseline(self) -> float:
        return sum(s.baseline_y for s in self.spans) / len(self.spans)

    def append_extend(self, span: _Span) -> None:
        """Add a span during the greedy pass and grow the vertical band."""
        if not self.spans:
            self.y0, self.y1 = float(span.bbox[1]), float(span.bbox[3])
        else:
            self.y0 = min(self.y0, span.bbox[1])
            self.y1 = max(self.y1, span.bbox[3])
        self.spans.append(span)

    def refresh(self) -> None:
        """Recompute the vertical band and baseline fit from current members."""
        self.y0 = float(min(s.bbox[1] for s in self.spans))
        self.y1 = float(max(s.bbox[3] for s in self.spans))
        if len(self.spans) >= 2:
            self.slope, self.intercept = fit_baseline(
                [(s.x_center, s.baseline_y) for s in self.spans],
            )
        else:
            self.slope, self.intercept = 0.0, self.spans[0].baseline_y

    def predict(self, x: float) -> float:
        """Baseline y predicted at horizontal position ``x``."""
        return self.slope * x + self.intercept


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Length of the overlap of intervals ``[a0, a1]`` and ``[b0, b1]``."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def _make_spans(
    items: Sequence[T],
    bbox_of: Callable[[T], BBox],
    baseline_of: Callable[[T], float] | None,
) -> list[_Span]:
    spans: list[_Span] = []
    for item in items:
        box = bbox_of(item)
        x0, y0, x1, y1 = box
        baseline_y = float(baseline_of(item)) if baseline_of is not None else float(y1)
        spans.append(
            _Span(
                item=item,
                bbox=box,
                baseline_y=baseline_y,
                x_center=(x0 + x1) / 2.0,
                height=max(1.0, float(y1 - y0)),
            ),
        )
    return spans


def _assign_overlap_rows(spans: list[_Span]) -> list[_Row]:
    """Step 2 -- greedy overlap row assignment (Tesseract ``most_overlapping_row``)."""
    rows: list[_Row] = []
    for span in sorted(spans, key=lambda s: s.baseline_y):
        best_row: _Row | None = None
        best_ov = 0.0
        for row in rows:
            ov = _overlap(span.bbox[1], span.bbox[3], row.y0, row.y1)
            if best_row is None or ov > best_ov:
                best_row, best_ov = row, ov
        if best_row is not None:
            non_overlap = span.height - best_ov
            if non_overlap <= OVERLAP_TOLERANCE * best_row.height:
                best_row.append_extend(span)
                continue
        new_row = _Row()
        new_row.append_extend(span)
        rows.append(new_row)
    return rows


def _reassign(rows: list[_Row]) -> list[_Row]:
    """Step 4 -- move each word to its best-fitting baseline (Tesseract ``cleanup_rows``)."""
    buckets: list[list[_Span]] = [[] for _ in rows]
    for cur_idx, row in enumerate(rows):
        for span in row.spans:
            best_idx = cur_idx
            best_dist = abs(span.baseline_y - rows[cur_idx].predict(span.x_center))
            for idx, other in enumerate(rows):
                dist = abs(span.baseline_y - other.predict(span.x_center))
                if dist < best_dist:
                    best_dist, best_idx = dist, idx
            target = best_idx if best_dist <= REASSIGN_TOLERANCE * span.height else cur_idx
            buckets[target].append(span)
    rebuilt = [_Row(spans=bucket) for bucket in buckets if bucket]
    for row in rebuilt:
        row.refresh()
    return rebuilt


def _merge_rows(rows: list[_Row]) -> list[_Row]:
    """Step 5 -- merge rows split by the greedy pass (Tesseract ``expand_rows``)."""
    merged: list[_Row] = []
    for row in sorted(rows, key=lambda r: r.mean_baseline):
        if merged:
            prev = merged[-1]
            limit = MERGE_TOLERANCE * min(prev.height, row.height)
            if abs(prev.mean_baseline - row.mean_baseline) < limit:
                prev.spans.extend(row.spans)
                prev.refresh()
                continue
        merged.append(row)
    return merged


def _order(rows: list[_Row]) -> list[list[Any]]:
    """Step 6 -- rows top-to-bottom by baseline intercept, items left-to-right."""
    ordered = sorted(rows, key=lambda r: r.intercept)
    return [
        [span.item for span in sorted(row.spans, key=lambda s: s.bbox[0])]
        for row in ordered
    ]


def group_into_lines(
    items: Sequence[T],
    bbox_of: Callable[[T], BBox],
    baseline_of: Callable[[T], float] | None = None,
) -> list[list[T]]:
    """Cluster loose words into reading lines, ordered top-to-bottom.

    Each returned line's items are ordered left-to-right. The original
    ``items`` are returned untouched -- this never constructs new objects.
    """
    spans = _make_spans(items, bbox_of, baseline_of)
    if not spans:
        return []
    if len(spans) == 1:
        return [[spans[0].item]]
    rows = _assign_overlap_rows(spans)
    for row in rows:
        row.refresh()
    rows = _reassign(rows)
    rows = _merge_rows(rows)
    return _order(rows)


def _row_whitespace(
    row: list[Any],
    bbox_of: Callable[[Any], BBox],
    span_x0: int,
    span_x1: int,
) -> list[tuple[float, float]]:
    """X-intervals within ``[span_x0, span_x1]`` covered by no word in ``row``."""
    covered = sorted((bbox_of(it)[0], bbox_of(it)[2]) for it in row)
    gaps: list[tuple[float, float]] = []
    cursor: float = span_x0
    for x0, x1 in covered:
        if x0 > cursor:
            gaps.append((cursor, float(x0)))
        cursor = max(cursor, float(x1))
    if cursor < span_x1:
        gaps.append((cursor, float(span_x1)))
    return gaps


def _whitespace_rivers(
    row_gaps: list[list[tuple[float, float]]],
    min_gutter: int,
) -> list[tuple[float, float]]:
    """Maximal x-intervals where whitespace persists across enough rows.

    A river is a run of x where the fraction of rows whose whitespace
    covers x is at least ``RIVER_COVERAGE`` and the run is at least
    ``min_gutter`` wide. Coverage is evaluated on the elementary intervals
    cut by every gap endpoint.
    """
    n_rows = len(row_gaps)
    points: set[float] = set()
    for gaps in row_gaps:
        for a, b in gaps:
            points.add(a)
            points.add(b)
    xs = sorted(points)
    rivers: list[tuple[float, float]] = []
    run_start: float | None = None
    for i in range(len(xs) - 1):
        lo, hi = xs[i], xs[i + 1]
        mid = (lo + hi) / 2.0
        covering = sum(
            1 for gaps in row_gaps if any(a <= mid <= b for a, b in gaps)
        )
        if covering / n_rows >= RIVER_COVERAGE:
            if run_start is None:
                run_start = lo
        elif run_start is not None:
            rivers.append((run_start, lo))
            run_start = None
    if run_start is not None:
        rivers.append((run_start, xs[-1]))
    return [(a, b) for a, b in rivers if (b - a) >= min_gutter]


def detect_columns(
    items: Sequence[T],
    bbox_of: Callable[[T], BBox],
    baseline_of: Callable[[T], float] | None = None,
) -> list[list[T]]:
    """Split words into vertical columns by persistent whitespace rivers.

    Returns columns left-to-right; a single-column page returns one column
    holding every item, so callers behave identically on plain documents.
    Tolerates a few full-width lines (titles, headers) because a river only
    needs ``RIVER_COVERAGE`` of the rows, not all of them.
    """
    item_list = list(items)
    if not item_list:
        return []
    if len(item_list) == 1:
        return [item_list]
    rows = group_into_lines(item_list, bbox_of, baseline_of)
    if len(rows) < 2:
        return [item_list]

    boxes = [bbox_of(it) for it in item_list]
    span_x0 = min(b[0] for b in boxes)
    span_x1 = max(b[2] for b in boxes)
    if span_x1 <= span_x0:
        return [item_list]
    min_gutter = max(MIN_GUTTER_PX, int(span_x1 * MIN_GUTTER_RATIO))

    row_gaps = [_row_whitespace(row, bbox_of, span_x0, span_x1) for row in rows]
    rivers = _whitespace_rivers(row_gaps, min_gutter)
    if not rivers:
        return [item_list]

    cuts = sorted((a + b) / 2.0 for a, b in rivers)
    columns: list[list[Any]] = [[] for _ in range(len(cuts) + 1)]
    for it, box in zip(item_list, boxes, strict=True):
        x_center = (box[0] + box[2]) / 2.0
        col = sum(1 for c in cuts if x_center >= c)
        columns[col].append(it)
    return [c for c in columns if c]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_layout.py -v`
Expected: PASS - all 17 tests.

- [ ] **Step 5: Lint and type-check**

Run: `ruff check paperless_paddleocr/paddle_engine/layout.py tests/test_layout.py && mypy --ignore-missing-imports paperless_paddleocr/paddle_engine/layout.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add paperless_paddleocr/paddle_engine/layout.py tests/test_layout.py
git commit -m "Add layout.py: Tesseract-grade line and column reconstruction"
```

---

## Task 3: `hocr.py` - fitted baseline on `Line`

**Files:**
- Modify: `paperless_paddleocr/paddle_engine/hocr.py`
- Test: `tests/test_hocr_render.py`

`Line` gains `baseline: tuple[float, float]` (the hOCR-relative `(slope, constant)`, §8.2); `render_hocr` emits it instead of the hard-coded `baseline 0 0`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hocr_render.py`:

```python
"""render_hocr must emit each line's fitted baseline into the title attr."""

from __future__ import annotations

from paperless_paddleocr.paddle_engine.hocr import Block, Line, Page, Word, render_hocr


def _page(line: Line) -> Page:
    return Page(
        width=400,
        height=300,
        lang="en",
        ocr_system="test",
        blocks=[Block(box=line.box, lines=[line])],
    )


def test_render_hocr_emits_default_flat_baseline():
    line = Line(
        box=(10, 10, 90, 40),
        confidence=88,
        text="hi",
        words=[Word("hi", (10, 10, 90, 40), 88)],
    )
    assert "baseline 0.000000 0" in render_hocr(_page(line))


def test_render_hocr_emits_fitted_baseline():
    line = Line(
        box=(10, 10, 90, 40),
        confidence=88,
        text="hi",
        words=[Word("hi", (10, 10, 90, 40), 88)],
        baseline=(0.05, -4.0),
    )
    assert "baseline 0.050000 -4" in render_hocr(_page(line))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_hocr_render.py -v`
Expected: FAIL - `Line.__init__` rejects the `baseline=` keyword (`TypeError`); `render_hocr` still emits `baseline 0 0`.

- [ ] **Step 3: Add the `baseline` field to `Line`**

In `paperless_paddleocr/paddle_engine/hocr.py`, replace the `Line` dataclass:

```python
@dataclass
class Line:
    """A reading line: its own box plus the words that make it up.

    ``text`` is the canonical recognised text for the sidecar. It is kept
    separate from ``words`` because some engines return an authoritative
    line transcription whose spacing differs from a naive word join.
    """

    box: BBox
    confidence: int
    text: str
    words: list[Word] = field(default_factory=list)
```

with:

```python
@dataclass
class Line:
    """A reading line: its box, its words, and its fitted baseline.

    ``text`` is the canonical recognised text for the sidecar. It is kept
    separate from ``words`` because some engines return an authoritative
    line transcription whose spacing differs from a naive word join.

    ``baseline`` is ``(slope, constant)`` in hOCR-relative form: ``slope``
    is the baseline gradient and ``constant`` is its y at the line box's
    left edge, measured up from the line box bottom. The default
    ``(0.0, 0.0)`` is a flat baseline on the box bottom.
    """

    box: BBox
    confidence: int
    text: str
    words: list[Word] = field(default_factory=list)
    baseline: tuple[float, float] = (0.0, 0.0)
```

- [ ] **Step 4: Add the `_baseline` helper**

In `hocr.py`, replace:

```python
def _bbox(b: BBox) -> str:
    return f"bbox {b[0]} {b[1]} {b[2]} {b[3]}"
```

with:

```python
def _bbox(b: BBox) -> str:
    return f"bbox {b[0]} {b[1]} {b[2]} {b[3]}"


def _baseline(line: Line) -> str:
    slope, constant = line.baseline
    return f"baseline {slope:.6f} {constant:.0f}"
```

- [ ] **Step 5: Emit the baseline in `render_hocr`**

In `hocr.py`, inside `render_hocr`, replace:

```python
            out.append(
                f'<span class="ocr_line" id="line_{line_no}" '
                f'title="{_bbox(line.box)}; baseline 0 0; '
                f'x_wconf {line.confidence}">',
            )
```

with:

```python
            out.append(
                f'<span class="ocr_line" id="line_{line_no}" '
                f'title="{_bbox(line.box)}; {_baseline(line)}; '
                f'x_wconf {line.confidence}">',
            )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_hocr_render.py tests/test_parse_hocr_words.py tests/test_hocr_lang.py -v`
Expected: PASS - the new render tests, plus the existing hOCR tests still green (`_write_merged_hocr` in `ocrmypdf_plugin.py` is untouched and keeps its own `baseline 0 0`).

- [ ] **Step 7: Lint and type-check**

Run: `ruff check paperless_paddleocr/paddle_engine/hocr.py tests/test_hocr_render.py && mypy --ignore-missing-imports paperless_paddleocr/paddle_engine/hocr.py`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add paperless_paddleocr/paddle_engine/hocr.py tests/test_hocr_render.py
git commit -m "Emit fitted per-line baseline in render_hocr"
```

---

## Task 4: `classic.py` - fit each region's baseline

**Files:**
- Modify: `paperless_paddleocr/paddle_engine/classic.py`
- Test: `tests/test_classic_baseline.py`

Classic PaddleOCR returns one detection region per line, so each region's words share a baseline. A new pure helper `_region_baseline` fits it via `layout.fit_baseline`; `build_page` attaches it to the `Line`. `build_page` itself needs PaddleOCR and is exercised by CI integration, not unit tests - but `_region_baseline` is pure and is unit-tested here (importing `classic.py` works without PaddleOCR: the import is guarded by `try/except ImportError`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_classic_baseline.py`:

```python
"""Unit test for classic._region_baseline straight-baseline fitting."""

from __future__ import annotations

from paperless_paddleocr.paddle_engine.classic import _region_baseline
from paperless_paddleocr.paddle_engine.hocr import Word


def test_region_baseline_flat_line_sits_on_box_bottom():
    words = [Word("a", (0, 10, 40, 30), 90), Word("b", (50, 10, 90, 30), 90)]
    slope, constant = _region_baseline(words, (0, 10, 90, 30))
    assert abs(slope) < 1e-9
    assert abs(constant) < 1e-9


def test_region_baseline_slopes_with_rising_word_bottoms():
    # word bottoms climb toward the right: 30, 26, 22
    words = [
        Word("a", (0, 10, 40, 30), 90),
        Word("b", (50, 6, 90, 26), 90),
        Word("c", (100, 2, 140, 22), 90),
    ]
    slope, _constant = _region_baseline(words, (0, 2, 140, 30))
    assert slope < 0  # baseline y decreases as x increases


def test_region_baseline_no_words_is_flat():
    assert _region_baseline([], (0, 0, 100, 20)) == (0.0, 0.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_classic_baseline.py -v`
Expected: FAIL - `ImportError: cannot import name '_region_baseline'`.

- [ ] **Step 3: Import `fit_baseline` in `classic.py`**

In `paperless_paddleocr/paddle_engine/classic.py`, replace:

```python
from paperless_paddleocr.paddle_engine.geometry import (
    BBox,
    estimate_word_boxes,
    poly_to_bbox,
)
from paperless_paddleocr.paddle_engine.hocr import Block, Line, Page, Word
```

with:

```python
from paperless_paddleocr.paddle_engine.geometry import (
    BBox,
    estimate_word_boxes,
    poly_to_bbox,
)
from paperless_paddleocr.paddle_engine.hocr import Block, Line, Page, Word
from paperless_paddleocr.paddle_engine.layout import fit_baseline
```

- [ ] **Step 4: Add the `_region_baseline` helper**

In `classic.py`, immediately after the `_word_box` function, add:

```python


def _region_baseline(words: list[Word], region_box: BBox) -> tuple[float, float]:
    """Fit a straight baseline for a one-line region from its word boxes.

    Classic PaddleOCR returns one detection region per line, so a region's
    words share a baseline. Returns ``(slope, constant)`` in hOCR-relative
    form: ``constant`` is the fitted baseline's y at the region's left edge
    measured up from the region's bottom. A region with no words gets a
    flat baseline on the box bottom.
    """
    points = [((w.box[0] + w.box[2]) / 2.0, float(w.box[3])) for w in words]
    if not points:
        return 0.0, 0.0
    slope, intercept = fit_baseline(points)
    rx0, _ry0, _rx1, ry1 = region_box
    return slope, slope * rx0 + intercept - ry1
```

- [ ] **Step 5: Attach the baseline in `build_page`**

In `classic.py`, inside `build_page`, replace:

```python
        line = Line(box=region_box, confidence=conf, text=text, words=words)
        page.blocks.append(Block(box=region_box, lines=[line]))
```

with:

```python
        line = Line(
            box=region_box,
            confidence=conf,
            text=text,
            words=words,
            baseline=_region_baseline(words, region_box),
        )
        page.blocks.append(Block(box=region_box, lines=[line]))
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `python -m pytest tests/test_classic_baseline.py -v`
Expected: PASS - all 3 tests.

- [ ] **Step 7: Lint and type-check**

Run: `ruff check paperless_paddleocr/paddle_engine/classic.py tests/test_classic_baseline.py && mypy --ignore-missing-imports paperless_paddleocr/paddle_engine/classic.py`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add paperless_paddleocr/paddle_engine/classic.py tests/test_classic_baseline.py
git commit -m "Fit per-region baseline in the classic adapter"
```

---

## Task 5: `vl.py` - cluster spotting words with `layout`

**Files:**
- Modify: `paperless_paddleocr/paddle_engine/vl.py`
- Test: `tests/test_vl_baseline.py`

`_spotting_page` swaps the removed `group_words_into_lines` for `layout.group_into_lines`, with a skew-aware `baseline_of` reading each detection quad's bottom edge, and fits a baseline per line (§8.1). A new pure helper `_poly_baseline_y` is unit-tested here; `_spotting_page` / `build_page` need PaddleOCR and are covered by CI integration. Importing `vl.py` works without PaddleOCR (guarded `try/except ImportError`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_vl_baseline.py`:

```python
"""Unit test for vl._poly_baseline_y bottom-edge baseline extraction."""

from __future__ import annotations

from paperless_paddleocr.paddle_engine.vl import _poly_baseline_y


def test_poly_baseline_y_quad_uses_bottom_corner_mean():
    # top-left, top-right, bottom-right, bottom-left; bottom ys are 40 and 44
    poly = [[0, 0], [100, 0], [100, 40], [0, 44]]
    assert _poly_baseline_y(poly) == 42.0


def test_poly_baseline_y_non_quad_uses_lowest_point():
    poly = [[0, 0], [50, 5], [25, 60]]
    assert _poly_baseline_y(poly) == 60.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_vl_baseline.py -v`
Expected: FAIL - `ImportError: cannot import name '_poly_baseline_y'`.

- [ ] **Step 3: Fix the `vl.py` imports**

In `paperless_paddleocr/paddle_engine/vl.py`, replace:

```python
from paperless_paddleocr.paddle_engine.geometry import (
    estimate_word_boxes,
    group_words_into_lines,
    poly_to_bbox,
)
from paperless_paddleocr.paddle_engine.hocr import Block, Line, Page, Word
```

with:

```python
from paperless_paddleocr.paddle_engine.geometry import estimate_word_boxes, poly_to_bbox
from paperless_paddleocr.paddle_engine.hocr import Block, Line, Page, Word
from paperless_paddleocr.paddle_engine.layout import fit_baseline, group_into_lines
```

- [ ] **Step 4: Replace `_spotting_page` with the layout-driven version**

In `vl.py`, replace the whole `_spotting_page` function:

```python
def _spotting_page(spotting: Any, page: Page) -> None:
    word_boxes = [
        (txt, poly)
        for txt, poly in zip(spotting["rec_texts"], spotting["rec_polys"], strict=False)
        if txt and txt.strip()
    ]
    lines = group_words_into_lines(word_boxes)
    log.debug("VL spotting: %d words, %d lines", len(word_boxes), len(lines))

    for line_words in lines:
        if not line_words:
            continue
        words = [Word(txt, poly_to_bbox(poly), VL_CONFIDENCE) for txt, poly in line_words]
        x0 = min(w.box[0] for w in words)
        y0 = min(w.box[1] for w in words)
        x1 = max(w.box[2] for w in words)
        y1 = max(w.box[3] for w in words)
        box = (x0, y0, x1, y1)
        text = " ".join(w.text for w in words)
        page.blocks.append(
            Block(box=box, lines=[Line(box=box, confidence=VL_CONFIDENCE, text=text, words=words)]),
        )
```

with:

```python
def _poly_baseline_y(poly: Any) -> float:
    """Baseline y of a detection quad: the mean y of its two bottom corners.

    PaddleOCR quads are ordered top-left, top-right, bottom-right,
    bottom-left, so the bottom corners are points 2 and 3. Using the bottom
    edge keeps the baseline skew-aware on a rotated scan; any other point
    count falls back to the polygon's lowest point.
    """
    pts = [(float(p[0]), float(p[1])) for p in poly]
    if len(pts) == 4:
        return (pts[2][1] + pts[3][1]) / 2.0
    return max(y for _, y in pts)


def _spotting_page(spotting: Any, page: Page) -> None:
    items = [
        (txt, poly)
        for txt, poly in zip(spotting["rec_texts"], spotting["rec_polys"], strict=False)
        if txt and txt.strip()
    ]
    lines = group_into_lines(
        items,
        bbox_of=lambda it: poly_to_bbox(it[1]),
        baseline_of=lambda it: _poly_baseline_y(it[1]),
    )
    log.debug("VL spotting: %d words, %d lines", len(items), len(lines))

    for line_items in lines:
        words = [Word(txt, poly_to_bbox(poly), VL_CONFIDENCE) for txt, poly in line_items]
        if not words:
            continue
        x0 = min(w.box[0] for w in words)
        y0 = min(w.box[1] for w in words)
        x1 = max(w.box[2] for w in words)
        y1 = max(w.box[3] for w in words)
        box = (x0, y0, x1, y1)
        text = " ".join(w.text for w in words)
        fit_points = [
            (
                (poly_to_bbox(poly)[0] + poly_to_bbox(poly)[2]) / 2.0,
                _poly_baseline_y(poly),
            )
            for _txt, poly in line_items
        ]
        slope, intercept = fit_baseline(fit_points)
        page.blocks.append(
            Block(
                box=box,
                lines=[
                    Line(
                        box=box,
                        confidence=VL_CONFIDENCE,
                        text=text,
                        words=words,
                        baseline=(slope, slope * x0 + intercept - y1),
                    ),
                ],
            ),
        )
```

`_ocr_page` is left unchanged: its lines keep the default flat `Line.baseline` because its word boxes are only proportional estimates.

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_vl_baseline.py -v`
Expected: PASS - both tests.

- [ ] **Step 6: Lint and type-check**

Run: `ruff check paperless_paddleocr/paddle_engine/vl.py tests/test_vl_baseline.py && mypy --ignore-missing-imports paperless_paddleocr/paddle_engine`
Expected: no errors - this is now a clean type-check of the whole `paddle_engine` package (`group_words_into_lines` is fully removed and no longer referenced).

- [ ] **Step 7: Commit**

```bash
git add paperless_paddleocr/paddle_engine/vl.py tests/test_vl_baseline.py
git commit -m "Cluster VL spotting words with layout.group_into_lines"
```

---

## Task 6: `ocrmypdf_plugin.py` - rewire reading order to `layout`

**Files:**
- Modify: `paperless_paddleocr/ocrmypdf_plugin.py`
- Test: `tests/test_words_to_text_columns.py`

Deletes the three duplicated "bucket words by y-centre" loops (`_detect_columns`, `_column_to_text`, and the sort in `_nms_merge`) and routes `_words_to_text` / `_nms_merge` reading order through `layout.py` (§8.3). `_Word` stays the plugin's type; a tiny `_word_bbox` adapter bridges it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_words_to_text_columns.py`:

```python
"""_words_to_text must read columns left-to-right, lines top-to-bottom."""

from __future__ import annotations

from paperless_paddleocr.ocrmypdf_plugin import _Word, _words_to_text


def test_words_to_text_single_column_top_to_bottom():
    words = [
        _Word("beta", 10, 40, 90, 60, 90),
        _Word("alpha", 10, 0, 90, 20, 90),
    ]
    assert _words_to_text(words) == "alpha\nbeta"


def test_words_to_text_reads_left_column_fully_before_right():
    words = [
        _Word("left1", 0, 0, 160, 20, 90),
        _Word("right1", 240, 0, 400, 20, 90),
        _Word("left2", 0, 30, 160, 50, 90),
        _Word("right2", 240, 30, 400, 50, 90),
        _Word("left3", 0, 60, 160, 80, 90),
        _Word("right3", 240, 60, 400, 80, 90),
    ]
    assert _words_to_text(words) == (
        "left1\nleft2\nleft3\n\nright1\nright2\nright3"
    )


def test_words_to_text_empty_input():
    assert _words_to_text([]) == ""
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_words_to_text_columns.py -v`
Expected: the column test fails - today's `_column_to_text` bucket loop does not guarantee the full-column-before-next-column ordering this asserts. (`test_words_to_text_empty_input` already passes; that is fine.)

- [ ] **Step 3: Add the `layout` imports**

In `paperless_paddleocr/ocrmypdf_plugin.py`, replace:

```python
from paperless_paddleocr.languages import to_hocr_lang
from paperless_paddleocr.paddle_engine import PaddleOCREngine
```

with:

```python
from paperless_paddleocr.languages import to_hocr_lang
from paperless_paddleocr.paddle_engine import PaddleOCREngine
from paperless_paddleocr.paddle_engine.geometry import BBox
from paperless_paddleocr.paddle_engine.layout import detect_columns, group_into_lines
```

- [ ] **Step 4: Replace `_MIN_GUTTER_RATIO` and `_detect_columns` with the `_word_bbox` adapter**

In `ocrmypdf_plugin.py`, replace this whole block:

```python
#: Minimum vertical gap (as a fraction of page width) treated as a column
#: separator in :func:`_detect_columns`. 4% is wide enough to skip the gutters
#: within justified paragraphs and narrow enough to catch real columns.
_MIN_GUTTER_RATIO: float = 0.04


def _detect_columns(
    words: list[_Word],
    min_gutter_ratio: float = _MIN_GUTTER_RATIO,
) -> list[list[_Word]]:
    """Cluster words into vertical columns by x-axis gap analysis.

    Sort word x-intervals, merge overlapping/abutting ones into "bands", and
    treat any gap between adjacent bands that's wider than
    ``min_gutter_ratio * page_width`` as a column separator. Single-column
    pages return a single column → the rest of the pipeline behaves exactly
    as before. Columns are returned left-to-right.
    """
    if len(words) <= 1:
        return [list(words)]
    page_w = max(w.x1 for w in words)
    if page_w <= 0:
        return [list(words)]
    min_gutter_px = max(8, int(page_w * min_gutter_ratio))

    intervals = sorted((w.x0, w.x1) for w in words)
    bands: list[list[int]] = []
    for x0, x1 in intervals:
        if bands and x0 <= bands[-1][1]:
            bands[-1][1] = max(bands[-1][1], x1)
        else:
            bands.append([x0, x1])

    if len(bands) <= 1:
        return [list(words)]

    splits: list[int] = []
    for i in range(len(bands) - 1):
        gap = bands[i + 1][0] - bands[i][1]
        if gap >= min_gutter_px:
            splits.append(bands[i][1] + gap // 2)

    if not splits:
        return [list(words)]

    columns: list[list[_Word]] = [[] for _ in range(len(splits) + 1)]
    for w in words:
        x_center = (w.x0 + w.x1) // 2
        col = 0
        for s in splits:
            if x_center >= s:
                col += 1
        columns[col].append(w)

    return [c for c in columns if c]
```

with:

```python
def _word_bbox(w: _Word) -> BBox:
    """Adapter: expose a plugin :class:`_Word` as a layout ``BBox``.

    The ``layout`` reconstruction functions are generic over the caller's
    word type; this bridges the multi-language merge's axis-aligned
    :class:`_Word` to them. ``baseline_of`` is left to default (the bbox
    bottom edge), which is exact for these already axis-aligned boxes.
    """
    return (w.x0, w.y0, w.x1, w.y1)
```

- [ ] **Step 5: Replace `_column_to_text` and `_words_to_text` with the layout-driven `_words_to_text`**

In `ocrmypdf_plugin.py`, replace this whole block (the `_column_to_text` function followed by the old `_words_to_text`):

```python
def _column_to_text(words: list[_Word]) -> str:
    """Group a column's words into reading lines (space-joined).

    Words within a column are bucketed by y-center: words whose centers fall
    in the same bucket form one line. The bucket size is ~60% of the column's
    median word height - small enough to keep adjacent lines apart, large
    enough to absorb minor baseline jitter from OCR.
    """
    if not words:
        return ""
    ordered = sorted(words, key=lambda w: ((w.y0 + w.y1) // 2, w.x0))
    median_height = sorted(max(1, w.y1 - w.y0) for w in ordered)[len(ordered) // 2]
    bucket = max(8, int(median_height * 0.6))

    lines: list[str] = []
    current: list[str] = []
    current_bucket: int | None = None
    for w in ordered:
        y_center = (w.y0 + w.y1) // 2
        b = y_center // bucket
        if current_bucket is not None and b != current_bucket and current:
            lines.append(" ".join(current))
            current = []
        current.append(w.text)
        current_bucket = b
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def _words_to_text(words: list[_Word]) -> str:
    """Render OCR words to plain text with column-aware reading order.

    Detects columns by x-axis gap analysis (:func:`_detect_columns`), then
    formats each column independently with :func:`_column_to_text`. Columns
    are joined with a blank line so downstream consumers (paperless's UI,
    full-text indexer) see them as distinct paragraphs.
    """
    if not words:
        return ""
    columns = _detect_columns(words)
    return "\n\n".join(_column_to_text(c) for c in columns)
```

with:

```python
def _words_to_text(words: list[_Word]) -> str:
    """Render OCR words to plain text with column-aware reading order.

    Columns are found with :func:`layout.detect_columns` (vertical
    whitespace rivers); each column's words are grouped into reading lines
    with :func:`layout.group_into_lines` (Tesseract-grade baseline
    clustering), space-joined per line and newline-joined per column.
    Columns are separated by a blank line so downstream consumers
    (paperless's UI, full-text indexer) see them as distinct paragraphs.
    """
    if not words:
        return ""
    blocks: list[str] = []
    for column in detect_columns(words, bbox_of=_word_bbox):
        lines = group_into_lines(column, bbox_of=_word_bbox)
        blocks.append("\n".join(" ".join(w.text for w in line) for line in lines))
    return "\n\n".join(blocks)
```

- [ ] **Step 6: Rewire `_nms_merge` reading order**

In `ocrmypdf_plugin.py`, replace the whole `_nms_merge` function:

```python
def _nms_merge(words: list[_Word], iou_threshold: float) -> list[_Word]:
    """Greedy NMS by descending confidence, then re-sort in reading order."""
    if not words:
        return []

    by_conf = sorted(words, key=lambda w: w.conf, reverse=True)
    kept: list[_Word] = []
    for candidate in by_conf:
        if any(candidate.iou(k) >= iou_threshold for k in kept):
            continue
        kept.append(candidate)

    if not kept:
        return kept
    median_height = sorted(max(1, w.y1 - w.y0) for w in kept)[len(kept) // 2]
    bucket = max(8, int(median_height * 0.6))

    def reading_key(w: _Word) -> tuple[int, int]:
        y_center = (w.y0 + w.y1) // 2
        return (y_center // bucket, w.x0)

    return sorted(kept, key=reading_key)
```

with:

```python
def _nms_merge(words: list[_Word], iou_threshold: float) -> list[_Word]:
    """Greedy NMS by descending confidence, then re-order for reading.

    Surviving words are re-sequenced through :func:`layout.detect_columns`
    and :func:`layout.group_into_lines` so the merged hOCR and sidecar read
    column-by-column, top-to-bottom, left-to-right.
    """
    if not words:
        return []

    by_conf = sorted(words, key=lambda w: w.conf, reverse=True)
    kept: list[_Word] = []
    for candidate in by_conf:
        if any(candidate.iou(k) >= iou_threshold for k in kept):
            continue
        kept.append(candidate)

    ordered: list[_Word] = []
    for column in detect_columns(kept, bbox_of=_word_bbox):
        for line in group_into_lines(column, bbox_of=_word_bbox):
            ordered.extend(line)
    return ordered
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python -m pytest tests/test_words_to_text_columns.py tests/test_parse_hocr_words.py tests/test_generate_pdf_dispatch.py -v`
Expected: PASS - the new column tests, plus both existing regression suites still green (`_Word`, `_parse_hocr_words`, `_write_merged_hocr` are untouched).

- [ ] **Step 8: Lint and type-check**

Run: `ruff check paperless_paddleocr/ocrmypdf_plugin.py tests/test_words_to_text_columns.py && mypy --ignore-missing-imports paperless_paddleocr`
Expected: no errors - a clean repo-wide type-check.

- [ ] **Step 9: Commit**

```bash
git add paperless_paddleocr/ocrmypdf_plugin.py tests/test_words_to_text_columns.py
git commit -m "Rewire multi-language reading order through layout.py"
```

---

## Task 7: Full verification pass

**Files:** none - verification only.

- [ ] **Step 1: Run the whole test suite**

Run: `python -m pytest tests/ -q`
Expected: PASS - `test_geometry`, `test_layout`, `test_hocr_render`, `test_classic_baseline`, `test_vl_baseline`, `test_words_to_text_columns`, plus the pre-existing `test_hocr_lang`, `test_parse_hocr_words`, `test_generate_pdf_dispatch`.

- [ ] **Step 2: Repo-wide lint and type-check**

Run: `ruff check paperless_paddleocr tests && mypy --ignore-missing-imports paperless_paddleocr`
Expected: no errors.

- [ ] **Step 3: Confirm no MPL-derived helpers remain**

Run: `grep -rn "group_words_into_lines\|_detect_columns\|_column_to_text\|corner-averag" paperless_paddleocr/`
Expected: no matches. (If any appear, a Task 1/5/6 edit was incomplete - fix before continuing.)

- [ ] **Step 4: Commit (only if Steps 1-3 surfaced fixes)**

If Steps 1-3 were clean, skip this step. If a fix was needed:

```bash
git add -- paperless_paddleocr tests
git commit -m "Fix verification findings in paddle_engine rewrite"
```

---

## Task 8: Licensing-file cleanup

**Files:**
- Delete: `empty.txt`
- Stage deletion: `LICENSE.MPL-2.0`

The `paddle_engine/` package files are already tracked, so no `git add` of new package modules is needed beyond the per-task commits above. This task finalises the licensing state (spec §10): the MPL-2.0 license file is removed and `LICENSE` (MIT) becomes the project's sole license.

- [ ] **Step 1: Verify and remove the stray `empty.txt`**

Run: `wc -c empty.txt`
Expected: `0 empty.txt` - confirm it is a 0-byte stray, then:

```bash
git rm --ignore-unmatch empty.txt
rm -f empty.txt
```

- [ ] **Step 2: Stage the `LICENSE.MPL-2.0` deletion**

The file is already removed from the working tree (the rewrite removed every MPL-2.0-derived source). Stage the deletion:

```bash
git rm --ignore-unmatch LICENSE.MPL-2.0
```

- [ ] **Step 3: Confirm the MIT license is the only one**

Run: `ls LICENSE* && grep -n 'license' pyproject.toml`
Expected: only `LICENSE` exists; `pyproject.toml` carries `license = "MIT"` and `license-files = ["LICENSE"]` (already in place - no edit needed).

- [ ] **Step 4: Commit**

```bash
git add LICENSE.MPL-2.0 empty.txt
git commit -m "Remove MPL-2.0 license file after independent rewrite"
```

(`git add` of a deleted path stages its removal; `empty.txt` is included so its deletion is recorded if it was ever tracked.)

---

## Self-review

Checked against the design spec:

- **§6 geometry** - Task 1: `poly_to_bbox` exact min/max envelope; `estimate_word_boxes` length-weighted partition with `_GAP_WEIGHT = 0.5`, last-word snap, single-word and empty cases. ✓
- **§7.2 fit_baseline** - Task 2: OLS with the spec's exact formula; degenerate cases (0/1 point, zero denominator) return flat-through-mean. ✓
- **§7.3 group_into_lines** - Task 2: six passes - spans, overlap assignment (`OVERLAP_TOLERANCE`), baseline fit, reassignment (`REASSIGN_TOLERANCE`), row merge (`MERGE_TOLERANCE`), ordering. ✓
- **§7.4 detect_columns** - Task 2: whitespace-river detection with `RIVER_COVERAGE` and `MIN_GUTTER`; full-width-title and gutter-straddle tolerance covered by tests. ✓
- **§7.5 constants** - Task 2: all five constants present with the spec's values. ✓
- **§8.1 vl.py** - Task 5: `_spotting_page` uses `group_into_lines`, `baseline_of` reads the quad bottom edge, per-line baseline fitted. ✓
- **§8.2 hocr.py** - Task 3: `Line.baseline` added; `render_hocr` emits `baseline {slope} {constant}`. Task 4: `classic.py` fits region baselines. ✓
- **§8.3 ocrmypdf_plugin.py** - Task 6: `_detect_columns` / `_column_to_text` deleted; `_words_to_text` and `_nms_merge` rewired to `layout`; `_write_merged_hocr` left untouched. ✓
- **§10 licensing** - Task 8: `LICENSE.MPL-2.0` removed, `empty.txt` cleaned up. ✓
- **§11 testing** - geometry, layout, hocr-render, classic-baseline, vl-baseline, words-to-text tests created; existing `test_parse_hocr_words` / `test_generate_pdf_dispatch` re-run in Tasks 3, 6, 7. ✓

Type consistency: `BBox` is `tuple[int, int, int, int]` throughout; `baseline` is `tuple[float, float]` (slope, constant) at every producer (`classic._region_baseline`, `vl._spotting_page`) and the consumer (`hocr._baseline`); `group_into_lines` / `detect_columns` share the identical `(items, bbox_of, baseline_of)` signature. No placeholders.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-22-paddle-engine-mit-rewrite.md`. Two execution options:

1. **Subagent-Driven (recommended)** - a fresh subagent per task, two-stage review between tasks, fast iteration.
2. **Inline Execution** - execute the tasks in this session with checkpoints for review.

Which approach?
