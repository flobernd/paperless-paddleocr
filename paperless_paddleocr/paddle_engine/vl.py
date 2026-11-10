"""PaddleOCR-VL remote adapter.

The local ``PaddleOCRVL`` pipeline runs on CPU with document orientation
classification, unwarping and layout detection switched off, so no layout
model runs on the paperless side. It sends the whole page image to a
remote vLLM server for recognition, and this module places the words the
server returns. Two response shapes are handled:

* **spotting** - native per-word polygons (available when the installed
  ``paddleocr`` exposes ``pipeline_version``). The words are clustered into
  reading lines geometrically.
* **ocr** - blocks with a multi-line text transcription but no word
  boxes. Each block is split into lines and the words are positioned by the
  proportional estimator.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from paperless_paddleocr import DEFAULT_VL_MODEL_NAME, DEFAULT_VL_PIPELINE_VERSION
from paperless_paddleocr.languages import first_paddle_lang
from paperless_paddleocr.paddle_engine.geometry import BBox, poly_to_bbox, union_boxes
from paperless_paddleocr.paddle_engine.hocr import Block, Line, Page, Word, words_from_text
from paperless_paddleocr.paddle_engine.layout import (
    fit_hocr_baseline,
    group_into_lines,
    gutter_width,
)

try:
    from paddleocr import PaddleOCRVL
except ImportError:  # pragma: no cover - resolved at install time
    PaddleOCRVL = None

log = logging.getLogger("paperless.paddleocr.vl")

OCR_SYSTEM = "PaddleOCR-VL via paperless-paddleocr"

#: VL recognition returns no per-word confidence; the invisible layer still
#: needs an ``x_wconf``, so a fixed high value is used.
VL_CONFIDENCE = 95

#: Layout labels whose content is body text worth indexing. Headers,
#: captions and structural titles are kept; figures/tables/seals are not.
_TEXT_LABELS: frozenset[str] = frozenset(
    {
        "text",
        "content",
        "paragraph_title",
        "doc_title",
        "abstract_title",
        "reference_title",
        "content_title",
        "table_title",
        "figure_title",
        "chart_title",
        "abstract",
        "reference",
        "reference_content",
        "algorithm",
        "number",
        "footnote",
        "header",
        "footer",
        "aside_text",
        "vertical_text",
        "vision_footnote",
        # Whole-block labels emitted by the VL prompt modes themselves:
        # "ocr" (ocr prompt) and "spotting" (spotting prompt) both carry the
        # full-page transcription as a single block.
        "ocr",
        "spotting",
    },
)

#: A spotted word as text, envelope and baseline y, each derived from the
#: polygon once so the row grouping, the gutter split and the baseline fit
#: share them.
_Spot = tuple[str, BBox, float]

# Same rationale as classic.py: construction is expensive, Paddle's local
# preprocessing predictors are not thread-safe, and ocrmypdf may call
# generate_hocr from several worker threads. Serialising predict also keeps
# at most one in-flight request per paperless worker on the remote server.
_CACHE_LOCK = threading.Lock()
_PIPELINE_CACHE: dict[tuple[Any, ...], Any] = {}
_PREDICT_LOCK = threading.Lock()


def _vl_option(options: Any, name: str, default: str = "") -> str:
    return (getattr(options, name, "") or "").strip() or default


def resolve_vl_api_key(options: Any) -> str:
    """The ``paddle_vl_api_key`` option when set, else ``PAPERLESS_PADDLEOCR_VL_API_KEY``.

    The paperless parser never passes the key as an option, because the
    inherited parse method logs the ocrmypdf arguments at debug level. The
    option serves a hand-run ocrmypdf.
    """
    return (
        _vl_option(options, "paddle_vl_api_key")
        or os.environ.get("PAPERLESS_PADDLEOCR_VL_API_KEY", "").strip()
    )


def _pipeline_key(options: Any) -> tuple[Any, ...]:
    return (
        _vl_option(options, "paddle_vl_server_url"),
        _vl_option(options, "paddle_vl_model_name"),
        resolve_vl_api_key(options),
        _vl_option(options, "paddle_vl_pipeline_version", DEFAULT_VL_PIPELINE_VERSION),
    )


def _get_pipeline(options: Any) -> Any:
    key = _pipeline_key(options)
    with _CACHE_LOCK:
        pipeline = _PIPELINE_CACHE.get(key)
        if pipeline is None:
            pipeline = _build_pipeline(options)
            _PIPELINE_CACHE[key] = pipeline
    return pipeline


def normalize_server_url(raw: str) -> str:
    """Normalised inference-server URL.

    PaddleOCR-VL forwards the URL verbatim to an OpenAI-style client that
    appends ``/chat/completions``; the genai-vllm-server only serves under
    the standard ``/v1`` prefix. Accept the URL with or without it.
    """
    cleaned = (raw or "").strip()
    if not cleaned:
        raise RuntimeError(
            "vl-remote engine requires PAPERLESS_PADDLEOCR_VL_SERVER_URL "
            "(or --paddle-vl-server-url) to point at a PaddleOCR-VL "
            "inference server.",
        )
    url = cleaned.rstrip("/")
    if not re.search(r"/v\d+$", url):
        url = f"{url}/v1"
    return url


def _spotting_capable(pipeline: Any) -> bool:
    """Word spotting shipped with the v1.5 pipeline; v1 predates it.

    Anything newer than v1 is assumed capable; build_page already falls back
    to ocr mode when a response carries no spotting result.
    """
    version = getattr(pipeline, "pipeline_version", None)
    return version is not None and version != "v1"


def _build_pipeline(options: Any) -> Any:
    import paddle

    url = normalize_server_url(_vl_option(options, "paddle_vl_server_url"))
    model_name = _vl_option(options, "paddle_vl_model_name", DEFAULT_VL_MODEL_NAME)
    api_key = resolve_vl_api_key(options) or None

    # PaddleOCRVL.__init__ probes paddle.amp.is_bfloat16_supported() against
    # the active device before the device= kwarg is applied; with no device
    # set that probe raises TypeError on Place(undefined:0). Pinning CPU
    # first is harmless (recognition runs on the remote server regardless)
    # and avoids the crash. Retained pending re-verification.
    paddle.set_device("cpu")

    kwargs: dict[str, Any] = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_layout_detection": False,
        "device": "cpu",
        "vl_rec_backend": "vllm-server",
        "vl_rec_server_url": url,
        "vl_rec_api_model_name": model_name,
    }
    if api_key:
        kwargs["vl_rec_api_key"] = api_key

    version = _vl_option(options, "paddle_vl_pipeline_version", DEFAULT_VL_PIPELINE_VERSION)
    # pipeline_version is a newer constructor parameter; older paddleocr
    # falls back to its only pipeline (ocr mode, no word spotting).
    if "pipeline_version" in inspect.signature(PaddleOCRVL.__init__).parameters:
        kwargs["pipeline_version"] = version
        log.debug("PaddleOCR-VL: pipeline_version=%s", version)
    else:
        log.debug("PaddleOCR-VL: pipeline_version unsupported (ocr mode only)")

    log.debug(
        "PaddleOCRVL url=%s model=%s api_key=%s",
        url,
        model_name,
        "<set>" if api_key else "<unset>",
    )
    try:
        return PaddleOCRVL(**kwargs)
    except Exception as e:
        # paddlex raises its own DependencyError when the doc-parser extra is
        # missing; matching by name avoids importing paddlex here.
        if type(e).__name__ == "DependencyError":
            raise RuntimeError(
                "PaddleOCR-VL dependencies are missing. Install the plugin "
                "with the vl extra: pip install 'paperless-paddleocr[vl]' "
                "(or pip install 'paddleocr[doc-parser]').",
            ) from e
        raise


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


def _split_by_gutter(items: list[_Spot], min_gutter: int) -> list[list[_Spot]]:
    """Split a spotting line into column segments at wide inter-word gaps.

    PaddleOCR-VL spots words across the full page width without per-column
    grouping, so a two-column row arrives as one line spanning the gutter.
    Splitting at gaps wider than ``min_gutter`` exposes the columns to the
    banded reading-order pass; a single-column line (no wide gap) stays one
    segment, so plain pages are unaffected.
    """
    if not items:
        return []
    ordered = sorted(items, key=lambda it: it[1][0])
    segments: list[list[_Spot]] = [[ordered[0]]]
    # Gap is measured from the widest right edge seen so far (as in
    # _row_whitespace): a wide box containing later boxes must not fake
    # a gutter.
    cursor = ordered[0][1][2]
    for cur in ordered[1:]:
        box = cur[1]
        if box[0] - cursor >= min_gutter:
            segments.append([cur])
        else:
            segments[-1].append(cur)
        cursor = max(cursor, box[2])
    return segments


def _spotting_page(spotting: Any, page: Page) -> None:
    items: list[_Spot] = [
        (txt, poly_to_bbox(poly), _poly_baseline_y(poly))
        for txt, poly in zip(spotting["rec_texts"], spotting["rec_polys"], strict=False)
        if txt and txt.strip()
    ]
    lines = group_into_lines(items, bbox_of=lambda it: it[1], baseline_of=lambda it: it[2])
    log.debug("VL spotting: %d words, %d lines", len(items), len(lines))
    if not items:
        return

    # The VL pipeline returns no column structure; a row spanning the gutter
    # must be split here so reading_blocks can reconstruct column-major order.
    span_x0, _, span_x1, _ = union_boxes(box for _, box, _ in items)
    min_gutter = gutter_width(span_x0, span_x1)

    for line_items in lines:
        for segment in _split_by_gutter(line_items, min_gutter):
            words = [Word(txt, box, VL_CONFIDENCE) for txt, box, _ in segment]
            box = union_boxes(w.box for w in words)
            text = " ".join(w.text for w in words)
            fit_points = [((b[0] + b[2]) / 2.0, baseline_y) for _, b, baseline_y in segment]
            page.blocks.append(
                Block(
                    box=box,
                    lines=[
                        Line(
                            box=box,
                            confidence=VL_CONFIDENCE,
                            text=text,
                            words=words,
                            baseline=fit_hocr_baseline(fit_points, box),
                        ),
                    ],
                ),
            )


def _ocr_page(parsing_res: Any, page: Page) -> None:
    log.debug("VL ocr: %d layout blocks", len(parsing_res))
    for block in parsing_res:
        # Layout blocks are PaddleOCRVLBlock objects on current paddlex, but
        # older paddleocr builds returned dicts with block_-prefixed keys.
        # Read by type: a falsy attribute (notably content="", the
        # PaddleOCRVLBlock default) must not fall through to dict .get(),
        # which an object does not have.
        if isinstance(block, dict):
            label = block.get("block_label", "")
            content = block.get("block_content", "")
            bbox = block.get("block_bbox")
        else:
            label = getattr(block, "label", "")
            content = getattr(block, "content", "")
            bbox = getattr(block, "bbox", None)

        if not content or not content.strip():
            continue
        if label not in _TEXT_LABELS:
            continue
        if bbox is None or len(bbox) < 4:
            continue

        bx0, by0, bx1, by1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        text_lines = [ln for ln in content.split("\n") if ln.strip()]
        if not text_lines:
            continue
        block_h = max(by1 - by0, 1)
        n_lines = len(text_lines)

        lines: list[Line] = []
        for i, line_text in enumerate(text_lines):
            # Per-line boundary arithmetic instead of one truncated line
            # height: rounding never accumulates and a block shorter in
            # pixels than its line count still yields non-degenerate boxes.
            ly0 = by0 + (i * block_h) // n_lines
            ly1 = by0 + ((i + 1) * block_h) // n_lines
            if ly1 <= ly0:
                ly1 = ly0 + 1
            line_box = (bx0, ly0, bx1, ly1)
            lines.append(
                Line(
                    box=line_box,
                    confidence=VL_CONFIDENCE,
                    text=line_text,
                    words=words_from_text(line_text, line_box, VL_CONFIDENCE),
                ),
            )
        page.blocks.append(Block(box=(bx0, by0, bx1, by1), lines=lines))


def _warn_out_of_bounds(page: Page) -> None:
    """Flag boxes outside the page: a symptom of mis-scaled VL coordinates.

    The VL pipeline is trusted to report original-image coordinates (the
    classic adapter rescales, this one cannot); if that contract breaks in a
    paddleocr update, the text layer silently lands in the wrong place.
    2% slack allows detector overshoot at the margins.
    """
    limit_x, limit_y = page.width * 1.02, page.height * 1.02
    bad = sum(
        1
        for block in page.blocks
        for line in block.lines
        for w in line.words
        if w.box[2] > limit_x or w.box[3] > limit_y
    )
    if bad:
        log.warning(
            "%d VL word boxes fall outside the %dx%d page; "
            "coordinates may be mis-scaled by the pipeline.",
            bad,
            page.width,
            page.height,
        )


def build_page(input_file: Path, options: Any) -> Page:
    """OCR ``input_file`` via the remote VL pipeline into a :class:`Page`."""
    log.debug("Running PaddleOCR-VL on %s", input_file)
    pipeline = _get_pipeline(options)

    with Image.open(input_file) as img:
        width, height = img.size

    has_spotting = _spotting_capable(pipeline)
    with _PREDICT_LOCK:
        result = pipeline.predict(
            str(input_file),
            use_layout_detection=False,
            use_queues=False,
            prompt_label="spotting" if has_spotting else "ocr",
        )

    lang = first_paddle_lang(getattr(options, "languages", None))
    page = Page(width=width, height=height, lang=lang, ocr_system=OCR_SYSTEM)
    if not result:
        return page

    page_result = result[0]
    spotting = page_result.get("spotting_res") if hasattr(page_result, "get") else None
    if spotting and spotting.get("rec_polys") and spotting.get("rec_texts"):
        _spotting_page(spotting, page)
    else:
        parsing_res = (
            page_result.get("parsing_res_list") if hasattr(page_result, "get") else None
        ) or []
        _ocr_page(parsing_res, page)

    _warn_out_of_bounds(page)
    return page
