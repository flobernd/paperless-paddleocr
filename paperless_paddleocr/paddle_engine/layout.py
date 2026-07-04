"""Tesseract-grade line and column reconstruction.

PaddleOCR's word-spotting response and the multi-language merge both hand
over a loose bag of recognised words; this module rebuilds the reading
structure: which words share a line, which lines form a column within a
vertical band, and the order to read them in.

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
from typing import Any

from paperless_paddleocr.paddle_engine.geometry import BBox

#: A word's vertical band may stick out of its row's band by this fraction
#: of the row height and still join it. Tesseract ``textord_overlap_x``.
OVERLAP_TOLERANCE = 0.375

#: In the reassignment pass a word moves to a better-fitting baseline only
#: when the predicted-vs-actual gap is within this fraction of its height.
REASSIGN_TOLERANCE = 0.5

#: Two rows merge when their mean baselines sit within this fraction of the
#: shorter row's height.
MERGE_TOLERANCE = 0.5

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


def _make_spans[T](
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
        # A row candidate needs actual vertical overlap: with best_ov == 0 the
        # tolerance test degenerates to a pure height comparison and glues
        # isolated small words (page numbers, footnote marks) onto whichever
        # row happens to be first in the list.
        if best_row is not None and best_ov > 0.0:
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
    """Step 6 -- rows top-to-bottom, items left-to-right.

    Sorted by mean member baseline rather than the fitted intercept: the
    intercept extrapolates to x = 0, which on a skewed page shifts long rows
    by slope * x relative to short rows evaluated at their own position.
    The mean baseline compares rows in the same frame the merge pass uses.
    """
    ordered = sorted(rows, key=lambda r: r.mean_baseline)
    return [[span.item for span in sorted(row.spans, key=lambda s: s.bbox[0])] for row in ordered]


def group_into_lines[T](
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
        (a, b) for a, b in _row_whitespace(row, bbox_of, span_x0, span_x1) if (b - a) >= min_gutter
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
