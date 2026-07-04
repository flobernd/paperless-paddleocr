# Full project review: findings and execution plans

**Date:** 2026-07-02
**Scope:** whole repository on branch `paddle-engine-mit-rewrite` - design, architecture,
correctness, accuracy, performance, packaging, CI, docs.
**Verification:** every claim about third-party behaviour below was checked against real
sources, not documentation from memory: the `ocrmypdf` 17.8.0 wheel and the `paddleocr`
3.7.0 wheel were downloaded and read. File and line references to those packages cite the
extracted wheel sources.

Each finding links to a detailed execution plan under `docs/superpowers/plans/`. The plans
are written to be implementable by an engineer (or model) with no prior context: exact
files, complete code, failing-test-first steps, and verification commands.

## Suggested implementation order

1. `2026-07-02-language-map-corrections.md` (Finding 1 - critical, small)
2. `2026-07-02-layout-correctness-fixes.md` (Findings 5, 6, 7, 8 - small)
3. `2026-07-02-multilang-merge-tests-and-vl-single-pass.md` (Findings 4, 12)
4. `2026-07-02-engine-instance-caching.md` (Finding 3)
5. `2026-07-02-rotate-and-deskew-support.md` (Finding 2)
6. `2026-07-02-vl-remote-hardening.md` (Finding 10)
7. `2026-07-02-banded-column-detection.md` (Finding 9 - depends on plan 2)
8. `2026-07-02-page-level-reading-order.md` (Finding 13 - depends on plan 7)
9. `2026-07-02-mkldnn-reverification.md` (Finding 11 - investigation)
10. `2026-07-02-docs-and-infra-cleanup.md` (Finding 14)

---

## Finding 1 (Critical): several advertised language codes are invalid in PaddleOCR 3.x

**Where:** `paperless_paddleocr/languages.py:29` (`"spa": "spanish"`),
`languages.py:53` (`"heb": "he"`), `languages.py:59-68` (`PADDLE_NATIVE` includes `ml`,
`latin`, `arabic`, `cyrillic`, `devanagari`), `paddle_engine/engine.py:25-82`
(`_SUPPORTED_LANGUAGES`), README "Language handling" section.

**Evidence:** paddleocr 3.7.0 resolves `lang=` through
`paddleocr/_pipelines/ocr.py:_get_ocr_model_names` against the sets in
`paddleocr/_utils/langs.py`. Spanish is `es` (member of `LATIN_LANGS`); `spanish` is not a
valid value. Hebrew appears in no set - there is no Hebrew model. The 2.x-era script
bundle codes `ml`, `latin`, `arabic`, `cyrillic`, `devanagari` are not valid `lang=`
values in 3.x either. Any of these reaches `PaddleOCR.__init__`, which raises
`ValueError: No models are available for lang=...` (`ocr.py:114-118`).

**Impact:** `PAPERLESS_OCR_LANGUAGE=spa` (every Spanish deployment) fails OCR on every
document. The README's recommended fix for mixed-script documents
(`PAPERLESS_PADDLEOCR_LANGUAGE=ml`) crashes too. Many languages PaddleOCR does support
(sk, sl, hr, bg, ca, id, ms, lv, lt, et, af, az, kk, ky, fa, ur, ...) are missing from the
map and get dropped with a warning.

**Fix:** correct `spa`, drop `heb`, expand the map from the verified 3.7 language sets,
translate the script-bundle codes to valid 3.x equivalents (`ml` selects the default
multilingual PP-OCRv6 model via `lang=None`), and derive the ocrmypdf language allowlist
from one source of truth. **Plan:** `2026-07-02-language-map-corrections.md`.

## Finding 2 (High): rotate-pages and deskew are silently dead

**Where:** `paddle_engine/engine.py:134-141` (`get_orientation` returns
`OrientationConfidence(angle=0, confidence=0.0)`, `get_deskew` returns `0.0`), README
"Honoured paperless-ngx settings" table.

**Evidence:** ocrmypdf 17.8.0 `_pipeline.py:485` rotates a page only when
`ocr_engine.get_orientation(...).confidence >= options.rotate_pages_threshold`, and
`_pipeline.py:651` rotates the raster by exactly `ocr_engine.get_deskew(...)` degrees.
With constant zeros, `PAPERLESS_OCR_ROTATE_PAGES` and `PAPERLESS_OCR_DESKEW` are accepted
and do nothing. The classic pipeline also runs with `use_doc_orientation_classify=False`
and `use_textline_orientation=False` (`classic.py:68-70`), so a 90/180/270-degree rotated
scan is recognised as garbage with no correction anywhere in the chain.

**Impact:** silent accuracy loss on rotated and skewed scans; README contradicts actual
behaviour.

**Fix:** implement `get_orientation` with PaddleOCR's document orientation classifier
(`paddleocr.DocImgOrientationClassification`, verified exported in 3.7.0) and
`get_deskew` with a pure PIL+numpy projection-profile estimator, both behind defensive
fallbacks to the current zeros. **Plan:** `2026-07-02-rotate-and-deskew-support.md`.

## Finding 3 (High): OCR engines are rebuilt for every page, with no thread safety

**Where:** `paddle_engine/classic.py:174` (`_build_engine(options)` inside `build_page`),
`paddle_engine/vl.py:262` (`_build_pipeline(options)` inside `build_page`),
`parser.py:374-375` (`use_threads: True, jobs: settings.THREADS_PER_WORKER`).

**Evidence:** ocrmypdf calls `generate_hocr` once per page from a thread pool sized by
`jobs`. Every call constructs a fresh `PaddleOCR(...)` / `PaddleOCRVL(...)`, which reloads
model weights from disk (seconds of CPU and hundreds of MB of RAM per instance). With
`jobs > 1`, several instances are constructed concurrently, and Paddle inference
predictors are not thread-safe.

**Impact:** on classic-cpu the per-page model reload can exceed the actual inference time;
multi-page documents multiply the waste; concurrent construction risks OOM and crashes.

**Fix:** module-level engine cache keyed by the options that influence construction, plus
a predict lock so concurrent ocrmypdf worker threads serialise inference. Worker recycling
(paperless sets `CELERY_WORKER_MAX_TASKS_PER_CHILD=1`) bounds cache lifetime to one
document. **Plan:** `2026-07-02-engine-instance-caching.md`.

## Finding 4 (Medium-High): multi-language mode runs N identical passes on vl-remote

**Where:** `ocrmypdf_plugin.py:167-199` (per-language loop),
`paddle_engine/vl.py` (language is never sent to the VL pipeline; `_resolve_lang` only
labels the hOCR).

**Evidence:** the VL recognition model is script-agnostic; the language code influences
nothing in `vl.build_page` except metadata. With `PAPERLESS_OCR_LANGUAGE=eng+deu` and
`vl-remote`, the same ~20 s/page remote recognition runs twice and the results are merged
with themselves.

**Impact:** 2x-Nx wasted wall-clock and GPU time for zero accuracy gain.

**Fix:** collapse to a single pass when `paddle_engine == "vl-remote"`. Bundled with the
missing multi-language merge test coverage (Finding 12).
**Plan:** `2026-07-02-multilang-merge-tests-and-vl-single-pass.md`.

## Finding 5 (Medium): words overlapping no row can join the first row

**Where:** `paddle_engine/layout.py:156-174` (`_assign_overlap_rows`).

**Evidence:** when a span overlaps no existing row, `best_row` is still the first row with
`best_ov == 0.0`. The join test `non_overlap <= OVERLAP_TOLERANCE * best_row.height`
degenerates to `span.height <= 0.375 * first_row.height`, so any small word (page number,
footnote mark) that overlaps nothing joins the top row of the page if that row is tall
enough - corrupting the row's band, baseline fit, and the reading order.

**Fix:** require `best_ov > 0` before joining. **Plan:**
`2026-07-02-layout-correctness-fixes.md`.

## Finding 6 (Medium): column gutter minimum uses the wrong variable

**Where:** `paddle_engine/layout.py:320`
(`min_gutter = max(MIN_GUTTER_PX, int(span_x1 * MIN_GUTTER_RATIO))`).

**Evidence:** the design spec (`docs/superpowers/specs/2026-05-22-...-design.md` section
7.5) defines the minimum gutter as 4% of the page text span width. The code uses the
right-edge coordinate `span_x1` instead of the width `span_x1 - span_x0`, overestimating
the minimum whenever the text span does not start near x = 0, which misses real columns.

**Fix:** use the span width. **Plan:** `2026-07-02-layout-correctness-fixes.md`.

## Finding 7 (Medium): rows are ordered by baseline extrapolated to x = 0

**Where:** `paddle_engine/layout.py:211-214` (`_order` sorts by `row.intercept`).

**Evidence:** a fitted row's `intercept` is its baseline extrapolated to x = 0, while a
single-member row's intercept is its baseline at its own x position. On a skewed page
these two reference frames disagree by `slope * x`, so a short line that is visually above
a sloped line can sort after it (a concrete failing construction is in the plan).
`_merge_rows` already uses `mean_baseline`; `_order` should too.

**Fix:** sort rows by `mean_baseline`. **Plan:** `2026-07-02-layout-correctness-fixes.md`.

## Finding 8 (Low): VL ocr-mode produces degenerate line boxes for short blocks

**Where:** `paddle_engine/vl.py:238` (`line_h = max(by1 - by0, 1) // len(text_lines)`).

**Evidence:** integer division yields `line_h == 0` whenever a block has more text lines
than pixel rows, collapsing every line box to zero height.

**Fix:** distribute line boundaries with per-line arithmetic and guarantee a minimum
height of 1. **Plan:** `2026-07-02-layout-correctness-fixes.md`.

## Finding 9 (Medium): column detection is page-global, so partial-column layouts interleave

**Where:** `paddle_engine/layout.py:294-333` (`detect_columns`),
`ocrmypdf_plugin.py:_words_to_text`.

**Evidence:** a whitespace river must persist across `RIVER_COVERAGE = 90%` of all rows on
the page. The most common business-letter layout - a two-column header (sender/recipient
address blocks) above a full-width body - never reaches 90%, so header words are grouped
line-by-line across the gutter: `sender-line-1 recipient-line-1 / sender-line-2 ...`.
Paperless ingests letters more than any other document class, so this hits the primary use
case.

**Fix:** band-based segmentation: find runs of consecutive rows that share a persistent
wide gap, emit those runs as column blocks and the remaining rows as full-width flow.
**Plan:** `2026-07-02-banded-column-detection.md` (land the layout fixes plan first).

## Finding 10 (Medium): vl-remote operational gaps

**Where:** `ocrmypdf_plugin.py:536-551` (`check_options` vl-remote branch),
`paddle_engine/vl.py:104-146` (`_build_pipeline`), `pyproject.toml:39-47`.

Four related gaps, verified against paddleocr 3.7.0:

- **No connectivity preflight.** A wrong URL or API key surfaces only at first predict,
  deep inside a celery task, after the local pipeline is fully constructed.
- **Pipeline version is hard-pinned to v1.5.** paddleocr 3.7 supports `v1`, `v1.5`,
  `v1.6` and defaults to `v1.6` (`_pipelines/paddleocr_vl.py:25-26`). Users running a
  newer server model cannot match it.
- **`paddleocr[doc-parser]` is a mandatory dependency** (`pyproject.toml:46`), pulling
  `paddlex[genai-client,ocr]` (verified in the paddleocr wheel metadata) into every
  classic-cpu image that never uses VL.
- **Missing doc-parser dependencies fail with an opaque paddlex error** at
  `PaddleOCRVL(...)` init instead of an actionable install hint.

**Fix:** preflight probe of `<server>/v1/models` in `check_options`, a
`PAPERLESS_PADDLEOCR_VL_PIPELINE_VERSION` setting, a `[vl]` packaging extra, and friendly
init-error mapping. **Plan:** `2026-07-02-vl-remote-hardening.md`.

## Finding 11 (Medium): the OneDNN-off workaround is an unverified CPU performance lever

**Where:** `paddle_engine/classic.py:73-79` (`enable_mkldnn = False`, comment says
"retained ... pending re-verification").

**Evidence:** disabling OneDNN typically costs a large factor on CPU convolution
inference. The crash it works around was observed on an earlier paddlepaddle build and has
not been re-tested against current wheels.

**Fix:** a reproduction-and-benchmark runbook plus a version-gated removal of the
workaround if the crash is fixed upstream. **Plan:** `2026-07-02-mkldnn-reverification.md`.

## Finding 12 (Medium): the multi-language merge path has no test coverage

**Where:** `ocrmypdf_plugin.py:122-271` - `generate_hocr`'s multi-language branch,
`_read_strategy`, `_nms_merge`, winner scoring, partial and total per-language failure
handling are all untested (the existing tests cover only parsing, rendering and dispatch).

**Impact:** the package's headline feature relies on manual testing only; regressions in
strategy selection or failure handling would ship silently. The winner score
(`sum(w.conf)`) also biases toward passes that emit more boxes; a
confidence-times-text-length score is proposed as an optional, tested change.

**Fix:** a dedicated test module with stubbed per-language passes.
**Plan:** `2026-07-02-multilang-merge-tests-and-vl-single-pass.md`.

## Finding 13 (Low): three writers, two orderings, one redundant re-parse

**Where:** `ocrmypdf_plugin.py:141-156` (single-language sidecar rebuilt by re-parsing the
hOCR file that was written milliseconds earlier), `ocrmypdf_plugin.py:378-438`
(`_write_merged_hocr` duplicates `hocr.render_hocr` with a degraded one-word-per-line
structure), `paddle_engine/hocr.py:139-141` (`sidecar_text` order differs from the rebuilt
sidecar).

**Impact:** wasted lxml parse per page, hOCR text-layer order differing from sidecar
order, and merged-path hOCR that loses line grouping (worse text selection in viewers).

**Fix:** derive hOCR block order and sidecar text from one reading-order pass over the
`Page` model and delete `_write_merged_hocr`.
**Plan:** `2026-07-02-page-level-reading-order.md` (after the banded-columns plan).

## Finding 14 (Low): docs and infra debt

- `.github/workflows/release.yml` is a retired stub whose own header says to delete it.
- README performance figures ("1-3 pages/sec", "~20 s/page on an A100") read as measured
  facts but are unverified; they should be labelled as rough guidance.
- CJK limitation is undocumented: `_merge_subtokens` (`classic.py:112-136`) joins
  sub-tokens on whitespace, and CJK text has none, so a whole line becomes one hOCR word
  (line-level search highlighting).
- The CI `static` job runs mypy with no dependencies installed, weakening the check.

**Plan:** `2026-07-02-docs-and-infra-cleanup.md`.

---

## What is in good shape (no action)

- **Layered architecture** matches the design spec: adapters build a typed `Page`, one
  module owns the hOCR wire format, layout logic is consolidated and generic. The
  clean-room rewrite discipline (design doc, provenance notes) is exemplary.
- **Security posture:** hOCR `lang` normalisation plus attribute escaping with injection
  tests, `no_network` lxml parser, masked API key in logs, compose example exposes the
  inference server only on the internal network and requires an API key.
- **Fail-fast checks:** classic-gpu CUDA probing in `check_options` with actionable
  messages, argparse-bypass validation of `paddle_engine`.
- **Ops ergonomics:** idempotent bootstrap scripts with CPU/GPU wheel conflict guards,
  model-cache volumes in every compose example, the paddle signal-handler silencing for
  celery worker recycling.
- **CI:** branch filters are correct for this repo (remote default branch is `master`;
  verified with `git remote show origin`). The smoke test against the real paperless-ngx
  image covers exactly the drift the unit suite cannot.
