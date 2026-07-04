"""Classic PaddleOCR CNN pipeline adapter.

Runs ``paddleocr.PaddleOCR`` locally on CPU and turns its detection /
recognition output into the typed :class:`~paperless_paddleocr.paddle_engine.hocr.Page`
model. PaddleOCR may preprocess (resize) the image before detection; word
boxes are reported in that preprocessed space, so coordinates are rescaled
back to the original pixel dimensions before they reach the hOCR.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from paperless_paddleocr.languages import TESSERACT_TO_PADDLE, normalize_paddle_lang
from paperless_paddleocr.paddle_engine.geometry import (
    BBox,
    estimate_word_boxes,
    poly_to_bbox,
)
from paperless_paddleocr.paddle_engine.hocr import Block, Line, Page, Word
from paperless_paddleocr.paddle_engine.layout import fit_baseline

try:
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover - resolved at install time
    PaddleOCR = None

log = logging.getLogger("paperless.paddleocr.classic")

OCR_SYSTEM = "PaddleOCR via paperless-paddleocr"

# One engine per configuration, one prediction at a time. Constructing
# PaddleOCR reloads model weights from disk, and Paddle inference predictors
# are not thread-safe; ocrmypdf calls generate_hocr from several worker
# threads when the caller passes jobs > 1. The cache lives until the celery
# worker is recycled (paperless recycles after every document).
_CACHE_LOCK = threading.Lock()
_ENGINE_CACHE: dict[tuple[Any, ...], Any] = {}
_PREDICT_LOCK = threading.Lock()


def _engine_key(options: Any) -> tuple[Any, ...]:
    return (
        _resolve_lang(options),
        _resolve_device(options),
        getattr(options, "paddle_det_model_dir", None),
        getattr(options, "paddle_rec_model_dir", None),
        getattr(options, "paddle_cls_model_dir", None),
    )


def _get_engine(options: Any) -> Any:
    key = _engine_key(options)
    with _CACHE_LOCK:
        engine = _ENGINE_CACHE.get(key)
        if engine is None:
            engine = _build_engine(options)
            _ENGINE_CACHE[key] = engine
    return engine


def _resolve_lang(options: Any) -> str:
    """First requested language as a PaddleOCR code.

    The paperless parser already translates Tesseract codes; the
    ``TESSERACT_TO_PADDLE`` lookup here is a defensive fallback for a code
    that slipped through untranslated.
    """
    langs = list(getattr(options, "languages", None) or [])
    if not langs:
        return "en"
    code = str(langs[0]).strip().lower()
    return TESSERACT_TO_PADDLE.get(code, code)


def _resolve_device(options: Any) -> str:
    """PaddleOCR ``device=`` value for the requested classic variant.

    ``classic-gpu`` runs the local pipeline on the first CUDA device;
    everything else (``classic-cpu`` default, unknown values) stays on CPU.
    The runtime image determines whether GPU is actually available - see
    ``examples/Dockerfile.classic-gpu``. ``check_options`` in the ocrmypdf
    plugin verifies a CUDA device is present before dispatch reaches here.
    """
    return "gpu" if getattr(options, "paddle_engine", "") == "classic-gpu" else "cpu"


def _build_engine(options: Any) -> Any:
    device = _resolve_device(options)
    kwargs: dict[str, Any] = {
        "device": device,
        "use_textline_orientation": False,
        "use_doc_unwarping": False,
        "use_doc_orientation_classify": False,
    }
    lang = normalize_paddle_lang(_resolve_lang(options))
    if lang is not None:
        kwargs["lang"] = lang
    if device == "cpu":
        # PaddlePaddle 3.x + OneDNN + the PIR executor crash at predict time
        # on many CPU builds ("ConvertPirAttribute2RuntimeAttribute not
        # support [pir::ArrayAttribute<pir::DoubleAttribute>]"). Disabling
        # OneDNN sidesteps it for a small, fixed per-page cost. Still
        # reproduces on paddle 3.3.1 / paddleocr 3.7.0 (re-verified
        # 2026-07-03), so the workaround stays. The GPU path doesn't touch
        # OneDNN, so it is CPU-only.
        kwargs["enable_mkldnn"] = False
    for opt_attr, paddle_kw in (
        ("paddle_det_model_dir", "text_detection_model_dir"),
        ("paddle_rec_model_dir", "text_recognition_model_dir"),
        ("paddle_cls_model_dir", "textline_orientation_model_dir"),
    ):
        value = getattr(options, opt_attr, None)
        if value:
            kwargs[paddle_kw] = value
    log.debug("PaddleOCR(classic) kwargs: %s", kwargs)
    return PaddleOCR(**kwargs)


def _preprocess_scale(
    page_result: Any,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Scale factors from PaddleOCR's preprocessed space back to pixels."""
    prep = page_result.get("doc_preprocessor_res") if hasattr(page_result, "get") else None
    img = prep.get("output_img") if prep and hasattr(prep, "get") else None
    shape = getattr(img, "shape", None)
    if shape is not None and len(shape) >= 2:
        prep_h, prep_w = int(shape[0]), int(shape[1])
        if prep_w and prep_h:
            return width / prep_w, height / prep_h
    return 1.0, 1.0


def _scaled(poly: Any, sx: float, sy: float) -> list[tuple[float, float]]:
    return [(float(p[0]) * sx, float(p[1]) * sy) for p in poly]


def _merge_subtokens(
    tokens: Any,
    boxes: Any,
) -> list[tuple[str, list[Any]]]:
    """Join PaddleOCR's sub-token pieces back into whole words.

    ``return_word_box=True`` yields recognition at sub-word granularity with
    whitespace tokens as separators. Accumulate non-space pieces and their
    boxes, flushing a word on each whitespace token.
    """
    words: list[tuple[str, list[Any]]] = []
    buf: list[str] = []
    buf_boxes: list[Any] = []
    for token, box in zip(tokens, boxes, strict=False):
        piece = str(token).strip()
        if not piece:
            if buf:
                words.append(("".join(buf), buf_boxes))
                buf, buf_boxes = [], []
        else:
            buf.append(piece)
            buf_boxes.append(box)
    if buf:
        words.append(("".join(buf), buf_boxes))
    return words


def _word_box(boxes: list[Any], sx: float, sy: float) -> BBox:
    xs: list[int] = []
    tops: list[int] = []
    bottoms: list[int] = []
    for box in boxes:
        bx0, by0, bx1, by1 = poly_to_bbox(_scaled(box, sx, sy))
        xs += [bx0, bx1]
        tops.append(by0)
        bottoms.append(by1)
    return min(xs), min(tops), max(xs), max(bottoms)


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


def build_page(input_file: Path, options: Any) -> Page:
    """OCR ``input_file`` with the classic pipeline into a :class:`Page`."""
    with Image.open(input_file) as img:
        width, height = img.size

    log.debug("Running classic PaddleOCR on %s (%dx%d)", input_file, width, height)
    engine = _get_engine(options)
    with _PREDICT_LOCK:
        result = engine.predict(str(input_file), return_word_box=True)

    page = Page(width=width, height=height, lang=_resolve_lang(options), ocr_system=OCR_SYSTEM)
    if not result:
        return page

    page_result = result[0]
    sx, sy = _preprocess_scale(page_result, width, height)

    texts = page_result.get("rec_texts", []) or []
    scores = page_result.get("rec_scores", []) or []
    polys = page_result.get("rec_polys", []) or []
    token_lines = page_result.get("text_word", []) or []
    token_boxes = page_result.get("text_word_region", []) or []
    has_word_boxes = bool(token_lines and token_boxes)
    log.debug("classic: %d regions, word boxes=%s", len(texts), has_word_boxes)

    for idx, (text, score, poly) in enumerate(zip(texts, scores, polys, strict=False)):
        if not text:
            continue
        region_box = poly_to_bbox(_scaled(poly, sx, sy))
        conf = int(float(score) * 100)

        words: list[Word] = []
        if (
            has_word_boxes
            and idx < len(token_lines)
            and idx < len(token_boxes)
            and token_lines[idx]
            and token_boxes[idx]
        ):
            for w_text, w_boxes in _merge_subtokens(token_lines[idx], token_boxes[idx]):
                if w_text:
                    words.append(Word(w_text, _word_box(w_boxes, sx, sy), conf))
        else:
            tokens = text.split()
            for w_text, box in zip(tokens, estimate_word_boxes(tokens, region_box), strict=False):
                words.append(Word(w_text, box, conf))

        line = Line(
            box=region_box,
            confidence=conf,
            text=text,
            words=words,
            baseline=_region_baseline(words, region_box),
        )
        page.blocks.append(Block(box=region_box, lines=[line]))

    return page
