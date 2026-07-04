"""_spotting_page must split two-column rows at the gutter so reading_blocks
recovers column-major order.

PaddleOCR-VL spots words across the full page width without per-column
grouping, so a two-column row arrives as one reading line spanning the
gutter. Splitting that line at wide inter-word gaps exposes the columns to
the banded reading-order pass; a single-column line (no wide gap) stays one
segment, so plain pages are unaffected.
"""

from __future__ import annotations

from paperless_paddleocr.paddle_engine.hocr import Page
from paperless_paddleocr.paddle_engine.layout import reading_blocks
from paperless_paddleocr.paddle_engine.vl import _split_by_gutter, _spotting_page


def _poly(x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _two_column_spotting() -> dict:
    # 4 rows, 2 columns, a wide gutter between x~270 and x=500.
    texts: list[str] = []
    polys: list[list[list[int]]] = []
    rows = [
        ("LEFT ALPHA ONE", "RIGHT AARDVARK ONE"),
        ("LEFT BETA TWO", "RIGHT BADGER TWO"),
        ("LEFT GAMMA THREE", "RIGHT CHEETAH THREE"),
        ("LEFT DELTA FOUR", "RIGHT DEER FOUR"),
    ]
    for i, (left, right) in enumerate(rows):
        y0, y1 = 100 + i * 80, 124 + i * 80
        x = 50
        for w in left.split():
            texts.append(w)
            polys.append(_poly(x, y0, x + 70, y1))
            x += 80
        x = 500
        for w in right.split():
            texts.append(w)
            polys.append(_poly(x, y0, x + 80, y1))
            x += 90
    return {"rec_texts": texts, "rec_polys": polys}


def _sidecar(page: Page) -> str:
    grouped = reading_blocks(page.blocks, bbox_of=lambda b: b.box)
    return "\n\n".join(
        "\n".join(line.text for row in group for blk in row for line in blk.lines if line.text)
        for group in grouped
    )


def test_split_by_gutter_splits_at_wide_gap():
    items = [
        ("A", _poly(50, 0, 120, 20)),
        ("B", _poly(130, 0, 200, 20)),
        ("C", _poly(500, 0, 580, 20)),
        ("D", _poly(590, 0, 660, 20)),
    ]
    segs = _split_by_gutter(items, min_gutter=50)
    assert len(segs) == 2
    assert [t for t, _ in segs[0]] == ["A", "B"]
    assert [t for t, _ in segs[1]] == ["C", "D"]


def test_split_by_gutter_keeps_single_column():
    items = [
        ("A", _poly(50, 0, 120, 20)),
        ("B", _poly(130, 0, 200, 20)),
        ("C", _poly(210, 0, 280, 20)),
    ]
    assert len(_split_by_gutter(items, min_gutter=50)) == 1


def test_spotting_two_column_reads_column_major():
    page = Page(width=800, height=500, lang="en", ocr_system="test")
    _spotting_page(_two_column_spotting(), page)
    # 4 rows x 2 columns -> 8 column-segment blocks, not 4 full-width rows.
    assert len(page.blocks) == 8
    sidecar = _sidecar(page)
    lines = sidecar.splitlines()
    last_left = max(i for i, ln in enumerate(lines) if ln.startswith("LEFT"))
    first_right = min(i for i, ln in enumerate(lines) if ln.startswith("RIGHT"))
    assert last_left < first_right, sidecar


def test_spotting_single_column_unaffected():
    texts: list[str] = []
    polys: list[list[list[int]]] = []
    for i, line in enumerate(["ALPHA ONE", "BETA TWO", "GAMMA THREE", "DELTA FOUR"]):
        y0, y1 = 100 + i * 80, 124 + i * 80
        x = 50
        for w in line.split():
            texts.append(w)
            polys.append(_poly(x, y0, x + 80, y1))
            x += 85
    page = Page(width=800, height=500, lang="en", ocr_system="test")
    _spotting_page({"rec_texts": texts, "rec_polys": polys}, page)
    assert len(page.blocks) == 4  # one per row, no gutter split
    assert _sidecar(page).splitlines() == [
        "ALPHA ONE",
        "BETA TWO",
        "GAMMA THREE",
        "DELTA FOUR",
    ]


def test_split_by_gutter_ignores_gap_covered_by_wide_box():
    # A wide box can horizontally contain later boxes; the gap to the next
    # segment must be measured from the widest right edge seen so far, not
    # from the immediately preceding (nested) box.
    items = [
        ("A", _poly(0, 0, 195, 20)),
        ("B", _poly(10, 0, 20, 20)),
        ("C", _poly(200, 0, 280, 20)),
    ]
    assert len(_split_by_gutter(items, min_gutter=50)) == 1
