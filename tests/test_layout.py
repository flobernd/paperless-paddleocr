"""Unit tests for paddle_engine.layout line and column reconstruction.

Test items are bare ``BBox`` tuples; the identity ``_box`` adapter keeps
the cases readable. ``baseline_of`` is left to default (the bbox bottom).
"""

from __future__ import annotations

from paperless_paddleocr.paddle_engine.layout import (
    fit_baseline,
    group_into_lines,
    reading_blocks,
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
    normal = (0, 10, 40, 30)  # height 20, bottom 30
    tall = (50, 0, 90, 30)  # height 30, bottom 30 -- same baseline
    assert group_into_lines([normal, tall], bbox_of=_box) == [[normal, tall]]


def test_group_into_lines_merges_a_jitter_split_line():
    # the greedy overlap pass splits these; the merge pass recombines them
    left = (0, 10, 40, 30)  # baseline 30
    right = (50, 18, 90, 38)  # baseline 38, overlaps too little to join
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


def test_group_into_lines_word_with_no_overlap_starts_its_own_row():
    # A tall first row and a small word far below it. The small word overlaps
    # nothing, but its height (20) is under OVERLAP_TOLERANCE * 60, so the
    # buggy zero-overlap path glued it into the first row.
    title = (0, 0, 200, 60)
    page_number = (0, 200, 20, 220)
    assert group_into_lines([title, page_number], bbox_of=_box) == [[title], [page_number]]


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


# --- reading_blocks -------------------------------------------------------


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
