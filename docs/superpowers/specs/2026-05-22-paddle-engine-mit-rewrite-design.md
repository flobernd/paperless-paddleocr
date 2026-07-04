# Independent rewrite of `paddle_engine/` for MIT licensing

**Date:** 2026-05-22
**Status:** Design - approved, pending spec review
**Scope:** `paperless_paddleocr/paddle_engine/` (full rewrite) + the layout
helpers in `paperless_paddleocr/ocrmypdf_plugin.py`

## 1. Context & motivation

The original `paperless_paddleocr/paddle_engine.py` was vendored from Clemens
Fruhwirth's `ocrmypdf-paddleocr`, which is licensed under **MPL-2.0**. Its own
header stated the `PaddleOCREngine` class and the `_poly_to_bbox` /
`_group_words_into_lines` helpers were "vendored verbatim with only cosmetic
changes."

The working tree has since split that file into a `paddle_engine/` package,
but the split is a *refactor of derived code* - it does not make the result
independent. The project cannot ship under a single MIT license while any
MPL-2.0-derived expression remains.

**Decision (already taken):** fully and independently re-implement the
`paddle_engine/` package so the whole project is cleanly **MIT**-licensed.
`LICENSE.MPL-2.0` is removed; `pyproject.toml` carries `license = "MIT"` /
`license-files = ["LICENSE"]`.

## 2. Goals

- No MPL-2.0-derived code remains anywhere in the project.
- Every file in `paddle_engine/` is implemented from the *external API
  surfaces* it targets - the ocrmypdf `OcrEngine` ABC, the PaddleOCR
  `PaddleOCR` / `PaddleOCRVL` Python APIs, and the W3C hOCR spec - not from
  clefru's source.
- The layout-reconstruction logic (line grouping, column detection, reading
  order) is consolidated into **one** Tesseract-grade module and is the most
  accurate option practical at word granularity.
- No change to external behaviour: CLI flags, `PAPERLESS_PADDLEOCR_*` env
  vars, engine variants, and hOCR/PDF output contracts are unchanged.

## 3. Non-goals

- Spline / curved baselines (Tesseract's `make_baseline_spline`). A straight
  per-line baseline is fitted; curvature is approximated, not tracked.
- Improving recognition accuracy - that is PaddleOCR's responsibility.
- Hard-pinning `paddleocr >= 3.4.0` (see §9).
- Re-implementing PaddleOCR's classic detection: classic PaddleOCR already
  returns line-level regions, so classic mode keeps one region = one line and
  does **not** run line grouping.

## 4. Why most of the package is not copyrightable expression

Copyright protects expression, not algorithms, interfaces, or facts. Every
file except the geometry/layout algorithms is plumbing dictated by an external
API - there is essentially one correct way to write it, so an independent
re-implementation naturally lands on equivalent code:

| File | What dictates its shape |
|---|---|
| `__init__.py`, `engine.py` | ocrmypdf's `OcrEngine` ABC - method names and signatures |
| `classic.py` | `paddleocr.PaddleOCR.predict()` output keys (`rec_texts`, `rec_scores`, `rec_polys`, `text_word`, `text_word_region`, `doc_preprocessor_res`) |
| `vl.py` | `paddleocr.PaddleOCRVL` constructor kwargs and `predict()` output shape |
| `hocr.py` | W3C hOCR / XHTML wire format |
| `pdf.py` | ocrmypdf 17's `HocrParser` / `Fpdf2PdfRenderer` / `MultiFontManager` API |

The genuinely creative content is the **geometry and layout algorithms**.
Those are designed fresh here (§6–§7). The module decomposition itself - a
typed `Page`/`Block`/`Line`/`Word` model fed by narrow adapters - is original
to this project; clefru's original is a single `plugin.py`.

The new layout algorithms take inspiration from **Tesseract** (`textord/
makerow.cpp`). Tesseract is Apache-2.0, which is MIT-compatible; and only the
*algorithm and numeric constants* are reused, which are not copyrightable in
the first place.

## 5. Module layout

```
paperless_paddleocr/paddle_engine/
  __init__.py    re-export PaddleOCREngine
  engine.py      PaddleOCREngine(OcrEngine) - variant dispatch, identity
  geometry.py    pure polygon math: BBox, poly_to_bbox, estimate_word_boxes
  layout.py      NEW - Tesseract-grade line/column reconstruction
  classic.py     paddleocr.PaddleOCR adapter -> Page
  vl.py          paddleocr.PaddleOCRVL adapter -> Page
  hocr.py        typed Page model + hOCR / sidecar serialisation
  pdf.py         ocrmypdf-17 text-only PDF renderer wiring
```

`ocrmypdf_plugin.py` keeps its `MultiLangPaddleEngine` and hookimpls, but its
layout helpers (`_detect_columns`, `_column_to_text`, `_words_to_text`, and
the reading-order sort inside `_nms_merge`) are rewired to call `layout.py`
(§8).

## 6. `geometry.py` - pure polygon math

Stateless functions over PaddleOCR detection polygons. No layout logic.

### 6.1 `BBox`

`BBox = tuple[int, int, int, int]` - `(x0, y0, x1, y1)` axis-aligned, in image
pixels.

### 6.2 `poly_to_bbox(poly) -> BBox`

The **exact min/max envelope** of the polygon's points: `x0 = min(xs)`,
`y0 = min(ys)`, `x1 = max(xs)`, `y1 = max(ys)`, cast to `int`. Duck-typed over
nested lists and numpy arrays.

Rationale: the exact envelope is the unique smallest axis-aligned box that
contains the detection - the most accurate axis-aligned representation. (The
previous corner-averaging heuristic deliberately *shrank* the box; baseline
skew is instead handled by `layout.py`, which reads the polygon's bottom
corners directly.)

### 6.3 `estimate_word_boxes(words, box) -> list[BBox]`

Fallback only - used when an engine returns a line transcription but no
per-word boxes (VL `ocr` mode, §7.3). Distributes the line `box` width across
`words` by a weight model:

- weight of a word = `max(1, len(word))`
- weight of each inter-word gap = `0.5` (a space is narrower than an average
  glyph)
- `total = sum(word weights) + 0.5 * (n - 1)`
- word `i` width = `round(line_width * weight_i / total)`; the cursor advances
  by the word width then a gap width.
- the last word's right edge is snapped to `box.x1` so rounding never leaves a
  gap.
- all words inherit the line `box`'s `y0` / `y1`.

Single word → the whole box; empty input → `[]`.

## 7. `layout.py` - Tesseract-grade line & column reconstruction

One module, used by `vl.py` and by `ocrmypdf_plugin.py`. Adapted from
Tesseract's `textord/makerow.cpp` row-finding, operating at **word**
granularity rather than connected-component blob granularity.

### 7.1 Generic item interface

The functions are generic over the caller's word type. Each takes:

- `items: Sequence[T]`
- `bbox_of: Callable[[T], BBox]`
- `baseline_of: Callable[[T], float] | None` - the baseline y at the item's
  horizontal centre; defaults to the bbox bottom (`y1`) when `None`.

This lets `vl.py` pass `(text, polygon)` tuples (baseline = mean y of the
polygon's two bottom corners - skew-aware) and `ocrmypdf_plugin.py` pass its
axis-aligned `_Word` (baseline = `y1`), with no shared base class.

### 7.2 `fit_baseline(points) -> tuple[float, float]`

Ordinary least-squares fit of `y = slope * x + intercept` minimising vertical
residuals over `points = [(x, y), ...]`:

```
slope     = (n*Sxy - Sx*Sy) / (n*Sxx - Sx*Sx)
intercept = (Sy - slope*Sx) / n
```

Degenerate cases (one point, or all points share an x → zero denominator):
`slope = 0`, `intercept = mean(y)`.

OLS is the maximum-likelihood baseline under Gaussian box-noise, i.e. the most
accurate fit for a correctly grouped line; outliers are handled by the
grouping/reassignment passes (§7.3), not by the fit.

### 7.3 `group_into_lines(items, bbox_of, baseline_of=None) -> list[list[T]]`

Returns reading lines, ordered top-to-bottom; each line's items ordered
left-to-right. The original `T` objects are returned (the function never
constructs new ones).

**Step 1 - spans.** For each item compute its `bbox` and `baseline_y`.
0 items → `[]`; 1 item → `[[item]]`.

**Step 2 - overlap row assignment** (Tesseract `most_overlapping_row` +
`textord_overlap_x`). Sort items by `baseline_y`. Maintain rows, each tracking
the running vertical band `[y0, y1]` (min top / max bottom of its members).
For each item:

- compute the vertical interval-overlap of the item's `[y0, y1]` with every
  existing row's band; let `best` be the row of greatest overlap `ov`, and
  `best_height` the height of that row's band.
- `non_overlap = item_height - ov`.
- if `best` exists and `non_overlap <= OVERLAP_TOLERANCE * best_height`, append
  to `best` and extend its band; otherwise open a new row.

This adapts per-row (the tolerance scales with `best_height`), so mixed font
sizes on one line stay together while distinct lines stay apart.

**Step 3 - baseline fit.** For each row with ≥2 members, `fit_baseline` over
its members' `(x_center, baseline_y)` points. Rows of 1 member: `slope = 0`,
`intercept = baseline_y`.

**Step 4 - reassignment** (Tesseract `cleanup_rows`). For each item, evaluate
`predicted = slope*x_center + intercept` for every row; move the item to the
row of smallest `|baseline_y - predicted|`, but only when that distance is
`<= REASSIGN_TOLERANCE * item_height` (otherwise leave it put - a word with no
good baseline match is not yanked). Drop emptied rows, then re-fit baselines
once.

**Step 5 - row merge** (Tesseract `expand_rows`). Sort rows by mean
`baseline_y`; merge two adjacent rows when
`|mean_baseline_a - mean_baseline_b| < MERGE_TOLERANCE * min(row_height)`.
Re-fit merged rows. This recombines a single line accidentally split by the
greedy pass.

**Step 6 - ordering.** Rows sorted top-to-bottom by baseline intercept; items
within a row sorted left-to-right by `bbox.x0`.

### 7.4 `detect_columns(items, bbox_of, baseline_of=None) -> list[list[T]]`

Replaces the global x-interval merge in today's `_detect_columns`, which a
single gutter-straddling word or a full-width title collapses into one column.
This uses Tesseract's idea of a whitespace **river** - whitespace that
*persists vertically*:

1. `rows = group_into_lines(items, ...)`. Fewer than 2 rows → return
   `[list(items)]` (single column).
2. For each row compute its whitespace gaps - the x-intervals inside the page
   text span `[min x0, max x1]` not covered by any word in that row.
3. Over the text span, compute per-x **row-coverage of whitespace**: the
   fraction of rows whose whitespace covers that x.
4. A column separator is a maximal x-interval where whitespace coverage
   `>= RIVER_COVERAGE` and the interval width `>= MIN_GUTTER`.
5. Cut at each separator's centre; assign every item to a column by its
   x-centre. Return columns left-to-right, dropping empties.

`RIVER_COVERAGE < 1.0` is what tolerates a few full-width lines (titles,
headers) that would otherwise break the river.

### 7.5 Constants

| Name | Value | Origin |
|---|---|---|
| `OVERLAP_TOLERANCE` | `0.375` | Tesseract `textord_overlap_x` |
| `REASSIGN_TOLERANCE` | `0.5` | this design |
| `MERGE_TOLERANCE` | `0.5` | this design |
| `RIVER_COVERAGE` | `0.9` | this design |
| `MIN_GUTTER` | `max(8 px, 0.04 * page_width)` | existing `_MIN_GUTTER_RATIO` |

## 8. Integration

### 8.1 `vl.py`

`_spotting_page` calls `layout.group_into_lines` on its `(text, polygon)`
items, with `baseline_of` reading the polygon's bottom-corner mean. `_ocr_page`
is unchanged in shape - it still splits a block into lines and calls
`geometry.estimate_word_boxes`.

### 8.2 `hocr.py`

`Line` gains `baseline: tuple[float, float]` - the fitted `(slope, constant)`
in hOCR-relative form: `constant = baseline_y(line.x0) - line.y1`. `render_hocr`
emits `baseline {slope} {constant}` instead of the hard-coded `baseline 0 0`,
so ocrmypdf's renderer places the invisible text on the true (possibly sloped)
baseline. `classic.py` fits each region's baseline from its word boxes via
`layout.fit_baseline`; VL spotting lines carry the fit from
`group_into_lines`; VL `ocr`-mode lines use `slope = 0`.

### 8.3 `ocrmypdf_plugin.py`

The three duplicated "bucket words by y-centre, bucket = 0.6×median height"
loops are deleted and rewired to `layout.py`:

- `_detect_columns` → `layout.detect_columns`.
- `_column_to_text` → `layout.group_into_lines` on the column's `_Word`s,
  words space-joined per line, lines newline-joined.
- `_words_to_text` → `detect_columns`, then `group_into_lines` per column;
  columns joined by a blank line.
- `_nms_merge` reading-order → `detect_columns` then `group_into_lines` per
  column, flattened.

`_Word` stays the plugin's type; the generic `bbox_of` adapter bridges it.
`_write_merged_hocr` keeps its one-word-per-line structure (`baseline 0 0` is
correct for a single-word line).

## 9. Known limitations

- **VL word boxes require `paddleocr >= 3.4.0`.** Text spotting (per-word
  polygons) shipped with the PaddleOCR-VL-1.5 pipeline in `paddleocr` v3.4.0
  (2026-01-29). `vl.py` feature-detects the `pipeline_version` constructor
  parameter; on older `paddleocr` it falls back to `ocr` mode, which returns
  block-level boxes only and relies on `estimate_word_boxes`.
- **`pyproject.toml` pins `paddleocr[doc-parser]>=3.0`** - looser than the
  3.4.0 needed for spotting. This is intentional: classic-only users do not
  need 3.4.0. The 3.4.0 requirement for spotting is **documented** in the
  README rather than hard-pinned.
- **The VL server must serve PaddleOCR-VL-1.5.** The example compose uses the
  rolling `paddleocr-genai-vllm-server:latest-nvidia-gpu[-sm120]` tags with
  `--model_name=PaddleOCR-VL-1.5-0.9B`; there is no version-pinned server tag.
- **Straight baselines approximate, but do not track, curved handwriting.**
  Spline baselines are an explicit non-goal (§3).
- **Heavy line interleaving** (handwritten descenders/ascenders crossing into
  neighbouring lines) can still mis-assign a word. The reassignment pass
  mitigates this; it does not eliminate it.

## 10. Licensing outcome

After the rewrite: no MPL-2.0-derived code remains; `LICENSE` (MIT) is the
project's sole license; `pyproject.toml` already carries `license = "MIT"` and
`license-files = ["LICENSE"]`. The staged deletion of `LICENSE.MPL-2.0` is then
correct. Tesseract-derived algorithm constants carry no licensing obligation
(algorithms and facts are uncopyrightable; Apache-2.0 is MIT-compatible
regardless).

## 11. Testing strategy

- **`geometry`** - `poly_to_bbox` on axis-aligned, skewed, and non-quad
  polygons; `estimate_word_boxes` partitioning, last-word snap, single-word and
  empty input.
- **`layout`** - `fit_baseline` flat / sloped / degenerate; `group_into_lines`
  for single line, multiple lines, mixed font sizes on one line, a slightly
  rotated page, boundary-word reassignment, and a split-line merge;
  `detect_columns` for one column, two columns, a tolerated full-width title,
  and a tolerated gutter-straddling word.
- **`hocr`** - the existing `test_parse_hocr_words.py` round-trip must still
  pass; a new assertion checks the `baseline` attribute is emitted.
- **Regression** - `test_parse_hocr_words.py` and
  `test_generate_pdf_dispatch.py` must pass unchanged.
- `paddleocr` and `pytest` are not installed in the available `.venv-test`
  environment; tests are authored against the contracts above and their green
  status is verified by CI, not locally.

## 12. Implementation sequence (for the plan)

1. `geometry.py` - `BBox`, `poly_to_bbox`, `estimate_word_boxes`.
2. `layout.py` - `fit_baseline`, `group_into_lines`, `detect_columns`.
3. `hocr.py` - `Line.baseline`, `render_hocr` baseline emission.
4. `classic.py`, `vl.py` - adapters, wired to `geometry` + `layout`.
5. `engine.py`, `pdf.py`, `__init__.py`.
6. `ocrmypdf_plugin.py` - rewire layout helpers to `layout.py`.
7. Cleanup - remove the stray `empty.txt`; finalise the `LICENSE.MPL-2.0`
   deletion; `git add` the new package files.
