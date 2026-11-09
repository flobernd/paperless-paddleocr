"""ocrmypdf plugin: paperless-paddleocr engine + multi-language merge.

This module is the one loaded by ``ocrmypdf.ocr(plugins=[…])`` when the
paperless parser in :mod:`paperless_paddleocr.parser` invokes ocrmypdf. It
provides two pieces:

* :class:`MultiLangPaddleEngine` - subclass of
  :class:`paperless_paddleocr.paddle_engine.PaddleOCREngine` that overrides
  ``generate_hocr`` to run PaddleOCR once per language and NMS-merge the
  word boxes by confidence. Single-language input falls through to the parent
  unchanged.
* :func:`add_options`, :func:`check_options`, :func:`get_ocr_engine`
  hookimpls that register the ``--paddle-engine`` /
  ``--paddle-multi-lang-strategy`` / ``--paddle-vl-*`` /
  ``--paddle-*-model-dir`` CLI args (so they're acceptable as kwargs to
  ``ocrmypdf.ocr()``) and bind the engine.

Why this plugin is only invoked when paperless explicitly asks for it: the
engine lives inside this package (see :mod:`paperless_paddleocr.paddle_engine`)
rather than being published as a standalone ocrmypdf plugin, so ocrmypdf's
entry-point auto-discovery never registers it. PaddleOCR is only
used when our paperless parser passes
``plugins=["paperless_paddleocr.ocrmypdf_plugin"]`` to ``ocrmypdf.ocr()`` -
calls from paperless's built-in Tesseract parser are unaffected.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request
from urllib.request import urlopen as _urlopen

import ocrmypdf
from ocrmypdf.exceptions import MissingDependencyError

from paperless_paddleocr import (
    DEFAULT_ENGINE,
    DEFAULT_STRATEGY,
    DEFAULT_VL_MODEL_NAME,
    DEFAULT_VL_PIPELINE_VERSION,
    ENGINE_CHOICES,
    STRATEGY_CHOICES,
)
from paperless_paddleocr.paddle_engine import PaddleOCREngine
from paperless_paddleocr.paddle_engine.geometry import BBox, union_boxes
from paperless_paddleocr.paddle_engine.hocr import (
    Block,
    Line,
    Page,
    Word,
    clamp_box,
    clean_text,
    write_document,
)
from paperless_paddleocr.paddle_engine.layout import fit_hocr_baseline, reading_blocks
from paperless_paddleocr.paddle_engine.vl import normalize_server_url

log = logging.getLogger("paperless.paddleocr.plugin")

#: Successful probes per (url, api_key); check_options runs once per
#: document, and one probe per worker process is enough.
_PROBED_SERVERS: set[tuple[str, str]] = set()


def _probe_vl_server(server_url: str, api_key: str) -> None:
    """Fail fast when the VL server is unreachable or rejects the API key.

    Without this, a wrong URL or key surfaces only at first predict, deep
    inside a celery task, after the local pipeline is fully constructed.
    """
    base = normalize_server_url(server_url)
    cache_key = (base, api_key)
    if cache_key in _PROBED_SERVERS:
        return

    request = Request(f"{base}/models")  # noqa: S310 - operator-configured URL
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        # A vLLM server under load answers /models slowly. Only a successful
        # probe is cached per worker process, so a host that never answers
        # delays each document by this timeout.
        with _urlopen(request, timeout=30):  # noqa: S310
            pass
    except HTTPError as e:
        if e.code in (401, 403):
            raise MissingDependencyError(
                f"The PaddleOCR-VL server at {base} rejected the API key "
                f"(HTTP {e.code}). Check PAPERLESS_PADDLEOCR_VL_API_KEY against "
                "the api-key configured in the server's vllm_config.yaml.",
            ) from e
        log.warning(
            "PaddleOCR-VL server preflight got HTTP %d from %s/models; continuing.",
            e.code,
            base,
        )
    except OSError as e:
        raise MissingDependencyError(
            f"The PaddleOCR-VL server at {base} is not reachable ({getattr(e, 'reason', e)}). "
            "Check PAPERLESS_PADDLEOCR_VL_SERVER_URL and that the "
            "paddleocr-genai-vllm-server container is running.",
        ) from e
    _PROBED_SERVERS.add(cache_key)


# ----------------------------------------------------------------------
# Multi-language merge: NMS over word boxes from per-language passes
# ----------------------------------------------------------------------

_NMS_IOU_THRESHOLD = 0.5

#: Band height for the NMS index. Any value gives the same result, since a
#: kept word is indexed into every band it touches; one near a word height
#: keeps each band's list short.
_NMS_BAND_PX = 32


class MultiLangPaddleEngine(PaddleOCREngine):
    """PaddleOCR engine that handles ``options.languages`` with > 1 entry.

    Single-language input falls straight through to the base implementation.

    Multi-language input builds one page per language with the base engine
    and combines the words of those pages based on the
    ``paddle_multi_lang_strategy`` option, which the paperless parser sets
    from ``PAPERLESS_PADDLEOCR_MULTI_LANG_STRATEGY``:

    * ``winner`` (default) - keep only the words of the language with the
      highest aggregate confidence (the sum of its word confidences).
      Faster and more consistent on documents whose pages are single-script.
    * ``merge`` - NMS-merge word boxes from all languages by confidence.
      Right choice for genuinely mixed-script pages.

    The first pass that raises fails the document.
    """

    @staticmethod
    def generate_hocr(
        input_file: Path,
        output_hocr: Path,
        output_text: Path,
        options: Any,
    ) -> None:
        langs = list(getattr(options, "languages", None) or [])
        if len(langs) > 1 and getattr(options, "paddle_engine", "") == "vl-remote":
            # The VL recognition model reads every script in one pass; the
            # language code only labels the hOCR. N per-language passes would
            # send the same slow remote request (tens of seconds per page) N
            # times for identical results.
            log.info(
                "vl-remote is language-agnostic; running one pass instead of %d (%s).",
                len(langs),
                "+".join(langs),
            )
            langs = langs[:1]
        if len(langs) <= 1:
            # Deliberate explicit base-class call (not super()/cls): this is
            # the single-language fast path that must run the base engine
            # exactly once. Dispatching back through this class would
            # re-enter generate_hocr and recurse.
            PaddleOCREngine.generate_hocr(
                input_file,
                output_hocr,
                output_text,
                options,
            )
            return

        strategy = getattr(options, "paddle_multi_lang_strategy", DEFAULT_STRATEGY)
        log.info(
            "Multi-language PaddleOCR pass on %s: %d languages (%s), strategy=%s",
            input_file,
            len(langs),
            "+".join(langs),
            strategy,
        )

        pages: dict[str, Page] = {}
        for lang in langs:
            sub_options = copy.copy(options)
            sub_options.languages = [lang]
            try:
                pages[lang] = PaddleOCREngine.build_page(input_file, sub_options)
            except Exception as e:
                raise RuntimeError(
                    f"PaddleOCR pass for language {lang!r} failed on {input_file}: {e}",
                ) from e

        per_lang_words = {lang: page_words(page) for lang, page in pages.items()}
        merged, primary = _select(strategy, per_lang_words, langs)
        blocks = reading_blocks(merged, bbox_of=lambda w: w.box)
        # The adapters size each page from the input image, so any pass gives the page size.
        first = pages[langs[0]]
        write_document(
            _merged_page(blocks, first.width, first.height, hocr_lang=primary),
            output_hocr,
            output_text,
            sidecar=_blocks_to_text(blocks),
        )


def page_words(page: Page) -> list[Word]:
    """Every word of ``page`` that has text, in the block order of the single-language hOCR.

    The merge sorts stably, so the word order decides its ties. Boxes are
    clamped to the page and texts cleaned and stripped here, ahead of the
    merge, because the merged lines derive their boxes, baselines and word
    spacing from these words.
    """
    grouped = reading_blocks(page.blocks, bbox_of=lambda b: b.box)
    words: list[Word] = []
    for block in (blk for group in grouped for row in group for blk in row):
        for line in block.lines:
            for word in line.words:
                text = clean_text(word.text).strip()
                if text:
                    box = clamp_box(word.box, page.width, page.height)
                    words.append(Word(text, box, word.confidence))
    return words


def _select(
    strategy: str,
    per_lang_words: dict[str, list[Word]],
    langs: list[str],
) -> tuple[list[Word], str]:
    """Words to emit, and the language the merged hOCR is labelled with."""
    if strategy == "winner":
        by_lang_conf = {
            lang: sum(w.confidence for w in words) for lang, words in per_lang_words.items()
        }
        winner_lang = max(by_lang_conf, key=lambda k: by_lang_conf[k])
        merged = per_lang_words[winner_lang]
        log.info(
            "multi-lang winner: %s (score=%d, %d words). Other scores: %s",
            winner_lang,
            by_lang_conf[winner_lang],
            len(merged),
            {lang: s for lang, s in by_lang_conf.items() if lang != winner_lang},
        )
        return merged, winner_lang

    flat = [w for words in per_lang_words.values() for w in words]
    merged = _nms_merge(flat, _NMS_IOU_THRESHOLD)
    log.debug(
        "merge: %d total words from %d languages → %d after NMS",
        len(flat),
        len(per_lang_words),
        len(merged),
    )
    return merged, langs[0]


# ----------------------------------------------------------------------
# hOCR helpers
# ----------------------------------------------------------------------


def _iou(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _blocks_to_text(blocks: list[list[list[Word]]]) -> str:
    """Render reading-ordered blocks to plain text.

    Blocks come from :func:`layout.reading_blocks` (column bands plus
    full-width flow); lines are space-joined, blocks blank-line-separated so
    downstream consumers (paperless UI, full-text indexer) see paragraphs.
    """
    return "\n\n".join(
        "\n".join(" ".join(w.text for w in line) for line in block) for block in blocks
    )


def _nms_merge(words: list[Word], iou_threshold: float) -> list[Word]:
    """Greedy NMS by descending confidence, then re-order for reading.

    Surviving words are re-sequenced through :func:`layout.reading_blocks`
    so the merged hOCR and sidecar read in banded reading order:
    column bands left-to-right, lines top-to-bottom within each band.
    """
    # Kept words are indexed by the horizontal bands they touch. IoU is zero
    # for vertically disjoint boxes, so a candidate only needs the kept words
    # of its own bands and the outcome matches a scan over all of them.
    bands: dict[int, list[Word]] = {}
    kept: list[Word] = []
    for candidate in sorted(words, key=lambda w: w.confidence, reverse=True):
        _x0, y0, _x1, y1 = candidate.box
        touched = range(min(y0, y1) // _NMS_BAND_PX, max(y0, y1) // _NMS_BAND_PX + 1)
        if any(
            _iou(candidate.box, k.box) >= iou_threshold for b in touched for k in bands.get(b, ())
        ):
            continue
        kept.append(candidate)
        for b in touched:
            bands.setdefault(b, []).append(candidate)

    return [
        w for block in reading_blocks(kept, bbox_of=lambda w: w.box) for line in block for w in line
    ]


def _merged_page(
    blocks: list[list[list[Word]]],
    page_w: int,
    page_h: int,
    *,
    hocr_lang: str,
) -> Page:
    """Page from reading-ordered blocks, so the merged hOCR carries real lines.

    Grouped lines (rather than one line per word) give ocrmypdf's renderer
    line-level baselines and give PDF viewers sane text selection.
    """
    page = Page(
        width=page_w,
        height=page_h,
        lang=hocr_lang,
        ocr_system="paperless-paddleocr multi-lang merge",
    )
    for block in blocks:
        for line_words in block:
            box = union_boxes(w.box for w in line_words)
            line = Line(
                box=box,
                confidence=round(sum(w.confidence for w in line_words) / len(line_words)),
                text=" ".join(w.text for w in line_words),
                words=list(line_words),
                baseline=fit_hocr_baseline(
                    [((w.box[0] + w.box[2]) / 2.0, float(w.box[3])) for w in line_words],
                    box,
                ),
            )
            page.blocks.append(Block(box=box, lines=[line]))
    return page


# ----------------------------------------------------------------------
# Plugin hookimpls
# ----------------------------------------------------------------------


@ocrmypdf.hookimpl
def add_options(parser: Any) -> None:
    """Register the ``--paddle-*`` CLI args so they're accepted as kwargs.

    ocrmypdf validates kwargs against its argparse parser; we need these
    options registered before ``ocrmypdf.ocr(paddle_engine=…, …)`` is called
    from the paperless parser.
    """
    paddle = parser.add_argument_group(
        "PaddleOCR",
        "Options for PaddleOCR engine",
    )
    paddle.add_argument(
        "--paddle-engine",
        choices=ENGINE_CHOICES,
        default=DEFAULT_ENGINE,
        dest="paddle_engine",
        help=(
            "OCR engine variant: 'classic-cpu' (default, fast CNN pipeline on "
            "CPU), 'classic-gpu' (same CNN pipeline on a local NVIDIA GPU; "
            "requires paddlepaddle-gpu), or 'vl-remote' (PaddleOCR-VL-1.5 "
            "served by a remote paddleocr-genai-vllm-server; local layout "
            "analysis only)."
        ),
    )
    paddle.add_argument(
        "--paddle-multi-lang-strategy",
        choices=STRATEGY_CHOICES,
        default=DEFAULT_STRATEGY,
        dest="paddle_multi_lang_strategy",
        help=(
            "How to combine the per-language passes when more than one "
            "language is given: 'winner' (default, keep the pass with the "
            "highest aggregate confidence) or 'merge' (NMS-merge the word "
            "boxes of all passes)."
        ),
    )
    paddle.add_argument(
        "--paddle-vl-server-url",
        default="",
        dest="paddle_vl_server_url",
        metavar="URL",
        help=("vl-remote: URL of the PaddleOCR-VL inference server (e.g. http://gpu-box:8118)."),
    )
    paddle.add_argument(
        "--paddle-vl-model-name",
        default=DEFAULT_VL_MODEL_NAME,
        dest="paddle_vl_model_name",
        metavar="NAME",
        help=(
            "vl-remote: model name advertised by the inference server "
            f"(default: {DEFAULT_VL_MODEL_NAME})."
        ),
    )
    paddle.add_argument(
        "--paddle-vl-api-key",
        default="",
        dest="paddle_vl_api_key",
        metavar="KEY",
        help=(
            "vl-remote: Bearer token for the inference server's /v1/* "
            "endpoints. Leave blank if the server doesn't require auth."
        ),
    )
    paddle.add_argument(
        "--paddle-vl-pipeline-version",
        default=DEFAULT_VL_PIPELINE_VERSION,
        dest="paddle_vl_pipeline_version",
        metavar="VERSION",
        help=(
            "vl-remote: PaddleOCR-VL pipeline version (v1, v1.5, v1.6). Must "
            "match the model family the inference server serves "
            f"(default: {DEFAULT_VL_PIPELINE_VERSION})."
        ),
    )
    paddle.add_argument(
        "--paddle-det-model-name",
        metavar="NAME",
        dest="paddle_det_model_name",
        help=(
            "Name of the text-detection model (e.g. PP-OCRv6_small_det). "
            "Official models are downloaded on first use. When set, the "
            "language setting no longer influences model selection."
        ),
    )
    paddle.add_argument(
        "--paddle-det-model-dir",
        metavar="DIR",
        dest="paddle_det_model_dir",
        help="Path to a custom text-detection model directory.",
    )
    paddle.add_argument(
        "--paddle-rec-model-name",
        metavar="NAME",
        dest="paddle_rec_model_name",
        help=(
            "Name of the text-recognition model (e.g. PP-OCRv6_small_rec). "
            "Official models are downloaded on first use. When set, the "
            "language setting no longer influences model selection."
        ),
    )
    paddle.add_argument(
        "--paddle-rec-model-dir",
        metavar="DIR",
        dest="paddle_rec_model_dir",
        help="Path to a custom text-recognition model directory.",
    )
    paddle.add_argument(
        "--paddle-cls-model-name",
        metavar="NAME",
        dest="paddle_cls_model_name",
        help=("Name of the textline-orientation model (e.g. PP-LCNet_x0_25_textline_ori)."),
    )
    paddle.add_argument(
        "--paddle-cls-model-dir",
        metavar="DIR",
        dest="paddle_cls_model_dir",
        help="Path to a custom textline-orientation model directory.",
    )


@ocrmypdf.hookimpl
def check_options(options: Any) -> None:
    """Validate the PaddleOCR options and that the requested engine variant is installed."""
    engine = getattr(options, "paddle_engine", DEFAULT_ENGINE)
    if engine not in ENGINE_CHOICES:
        # argparse validates --paddle-engine via `choices=` on CLI usage, but
        # the Python API (ocrmypdf.ocr(paddle_engine=…)) skips that check.
        # Validate explicitly so an unknown engine fails fast rather than
        # silently falling through to classic-cpu.
        raise ValueError(
            f"Unknown paddle_engine={engine!r}. Valid choices: {list(ENGINE_CHOICES)}.",
        )
    strategy = getattr(options, "paddle_multi_lang_strategy", DEFAULT_STRATEGY)
    if strategy not in STRATEGY_CHOICES:
        raise ValueError(
            f"Unknown paddle_multi_lang_strategy={strategy!r}. "
            f"Valid choices: {list(STRATEGY_CHOICES)}.",
        )

    if engine == "vl-remote":
        try:
            from paddleocr import PaddleOCRVL  # noqa: F401
        except ImportError as e:
            raise MissingDependencyError(
                "PaddleOCR-VL is not available. Install it with: "
                "pip install 'paddleocr[doc-parser]'",
            ) from e
        server_url = (getattr(options, "paddle_vl_server_url", "") or "").strip()
        if not server_url:
            raise MissingDependencyError(
                "vl-remote engine requires --paddle-vl-server-url (or the "
                "PAPERLESS_PADDLEOCR_VL_SERVER_URL env var) to point at a "
                "paddleocr-genai-vllm-server endpoint.",
            )
        _probe_vl_server(server_url, (getattr(options, "paddle_vl_api_key", "") or "").strip())
        return

    # classic-cpu / classic-gpu both need PaddleOCR importable locally.
    try:
        from paddleocr import PaddleOCR  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            "PaddleOCR is not installed. Install with: pip install paddlepaddle paddleocr "
            "(or paddlepaddle-gpu for the classic-gpu engine).",
        ) from e

    if engine == "classic-gpu":
        try:
            import paddle
        except ImportError as e:
            raise MissingDependencyError(
                "classic-gpu engine requires paddlepaddle-gpu. Build from "
                "examples/Dockerfile.classic-gpu or install with: "
                "pip install --index-url "
                "https://www.paddlepaddle.org.cn/packages/stable/cu126/ paddlepaddle-gpu",
            ) from e
        # paddlepaddle-gpu reports 0 devices when the host has no NVIDIA
        # driver or when nvidia-container-toolkit isn't wired up - fail
        # fast rather than crashing inside predict().
        try:
            visible = int(paddle.device.cuda.device_count())
        except Exception as e:  # pragma: no cover - defensive: paddle API drift
            raise MissingDependencyError(
                f"classic-gpu engine could not query CUDA devices: {e!s}. "
                "Verify paddlepaddle-gpu is installed and the host NVIDIA "
                "driver is reachable from inside the container.",
            ) from e
        if visible <= 0:
            raise MissingDependencyError(
                "classic-gpu engine requested but no CUDA device is visible. "
                "Check that the host has an NVIDIA driver, that "
                "nvidia-container-toolkit is configured, and that the "
                "container has GPU access (e.g. compose `deploy.resources."
                "reservations.devices` or `docker run --gpus all`).",
            )


@ocrmypdf.hookimpl(tryfirst=True)
def get_ocr_engine() -> Any:
    """Return our multi-language engine.

    ``tryfirst=True`` makes this hookimpl run before the built-in Tesseract
    plugin's ``get_ocr_engine``; ocrmypdf's ``firstresult=True`` policy then
    short-circuits, so Tesseract never gets a chance to claim the engine.
    """
    return MultiLangPaddleEngine()
