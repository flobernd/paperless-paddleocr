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
  hookimpls that register the ``--paddle-engine`` / ``--paddle-vl-*`` /
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
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request
from urllib.request import urlopen as _urlopen

import ocrmypdf

from paperless_paddleocr.paddle_engine import PaddleOCREngine
from paperless_paddleocr.paddle_engine.geometry import BBox
from paperless_paddleocr.paddle_engine.hocr import Block, Line, Page, render_hocr
from paperless_paddleocr.paddle_engine.hocr import Word as HocrWord
from paperless_paddleocr.paddle_engine.layout import fit_baseline, reading_blocks

log = logging.getLogger("paperless.paddleocr.plugin")

_BBOX_RE = re.compile(r"bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")
_CONF_RE = re.compile(r"x_wconf\s+(\d+)")

#: Successful probes per (url, api_key); check_options runs once per
#: document, and one probe per worker process is enough.
_PROBED_SERVERS: set[tuple[str, str]] = set()


def _probe_vl_server(server_url: str, api_key: str) -> None:
    """Fail fast when the VL server is unreachable or rejects the API key.

    Without this, a wrong URL or key surfaces only at first predict, deep
    inside a celery task, after the local pipeline is fully constructed.
    """
    from urllib.error import HTTPError, URLError

    from ocrmypdf.exceptions import MissingDependencyError

    from paperless_paddleocr.paddle_engine.vl import normalize_server_url

    base = normalize_server_url(server_url)
    cache_key = (base, api_key)
    if cache_key in _PROBED_SERVERS:
        return

    request = Request(f"{base}/models")  # noqa: S310 - operator-configured URL
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with _urlopen(request, timeout=5):  # noqa: S310
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
    except URLError as e:
        raise MissingDependencyError(
            f"The PaddleOCR-VL server at {base} is not reachable ({e.reason}). "
            "Check PAPERLESS_PADDLEOCR_VL_SERVER_URL and that the "
            "paddleocr-genai-vllm-server container is running.",
        ) from e
    _PROBED_SERVERS.add(cache_key)


# ----------------------------------------------------------------------
# Multi-language merge: NMS over word boxes from per-language passes
# ----------------------------------------------------------------------


@dataclass
class _Word:
    """A single OCR word extracted from a per-language hOCR file."""

    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    conf: int  # 0–100

    @property
    def area(self) -> int:
        return max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)

    def iou(self, other: _Word) -> float:
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        union = self.area + other.area - inter
        if union <= 0:
            return 0.0
        return inter / union


class MultiLangPaddleEngine(PaddleOCREngine):
    """PaddleOCR engine that handles ``options.languages`` with > 1 entry.

    Single-language input falls straight through to the base implementation.

    Multi-language input runs the base implementation once per language, parses
    each per-language hOCR, and combines the results based on the
    ``PAPERLESS_PADDLEOCR_MULTI_LANG_STRATEGY`` env var:

    * ``winner`` (default) - keep only the hOCR from the language with the
      highest aggregate confidence (sum of ``x_wconf`` across all words).
      Faster and more consistent on documents whose pages are single-script.
    * ``merge`` - NMS-merge word boxes from all languages by confidence.
      Right choice for genuinely mixed-script pages.
    """

    NMS_IOU_THRESHOLD: float = 0.5
    DEFAULT_STRATEGY: str = "winner"
    VALID_STRATEGIES: frozenset[str] = frozenset({"winner", "merge"})

    @staticmethod
    def _read_strategy() -> str:
        raw = (os.environ.get("PAPERLESS_PADDLEOCR_MULTI_LANG_STRATEGY", "") or "").strip().lower()
        if not raw:
            return MultiLangPaddleEngine.DEFAULT_STRATEGY
        if raw not in MultiLangPaddleEngine.VALID_STRATEGIES:
            log.warning(
                "Unknown PAPERLESS_PADDLEOCR_MULTI_LANG_STRATEGY=%r; "
                "falling back to %r. Valid values: %s",
                raw,
                MultiLangPaddleEngine.DEFAULT_STRATEGY,
                sorted(MultiLangPaddleEngine.VALID_STRATEGIES),
            )
            return MultiLangPaddleEngine.DEFAULT_STRATEGY
        return raw

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

        strategy = MultiLangPaddleEngine._read_strategy()
        log.info(
            "Multi-language PaddleOCR pass on %s: %d languages (%s), strategy=%s",
            input_file,
            len(langs),
            "+".join(langs),
            strategy,
        )

        page_w, page_h = _read_page_size(input_file)

        per_lang_words: dict[str, list[_Word]] = {}
        failed_langs: list[str] = []
        tmp_files: list[Path] = []
        try:
            for lang in langs:
                sub_options = copy.copy(options)
                sub_options.languages = [lang]

                tmp_hocr = output_hocr.with_name(
                    f"{output_hocr.stem}.{lang}.hocr",
                )
                tmp_text = output_text.with_name(
                    f"{output_text.stem}.{lang}.txt",
                )
                tmp_files.extend([tmp_hocr, tmp_text])

                try:
                    PaddleOCREngine.generate_hocr(
                        input_file,
                        tmp_hocr,
                        tmp_text,
                        sub_options,
                    )
                except Exception:
                    failed_langs.append(lang)
                    log.exception(
                        "PaddleOCR pass for lang=%s on %s failed; continuing.",
                        lang,
                        input_file,
                    )
                    continue

                try:
                    lang_words = _parse_hocr_words(tmp_hocr)
                except Exception:
                    failed_langs.append(lang)
                    log.exception(
                        "Failed to parse temp hOCR %s; skipping.",
                        tmp_hocr,
                    )
                    continue

                log.debug("lang=%s: %d words", lang, len(lang_words))
                per_lang_words[lang] = lang_words

            if failed_langs:
                log.info(
                    "%d/%d per-language passes succeeded (failed: %s).",
                    len(per_lang_words),
                    len(langs),
                    ", ".join(failed_langs),
                )

            if not per_lang_words:
                log.warning(
                    "All %d per-language PaddleOCR passes failed for %s; writing empty sidecar.",
                    len(langs),
                    input_file,
                )
                _write_merged_hocr(
                    output_hocr,
                    [],
                    page_w,
                    page_h,
                    hocr_lang=langs[0],
                )
                output_text.write_text("", encoding="utf-8")
                return
        finally:
            for tmp in tmp_files:
                if tmp.exists():
                    tmp.unlink()

        if strategy == "winner":
            by_lang_conf = {
                lang: sum(w.conf for w in words) for lang, words in per_lang_words.items()
            }
            winner_lang = max(by_lang_conf, key=lambda k: by_lang_conf[k])
            merged = per_lang_words[winner_lang]
            primary = winner_lang
            log.info(
                "multi-lang winner: %s (score=%d, %d words). Other scores: %s",
                winner_lang,
                by_lang_conf[winner_lang],
                len(merged),
                {lang: s for lang, s in by_lang_conf.items() if lang != winner_lang},
            )
        else:  # merge
            flat = [w for words in per_lang_words.values() for w in words]
            merged = _nms_merge(flat, MultiLangPaddleEngine.NMS_IOU_THRESHOLD)
            primary = langs[0]
            log.debug(
                "merge: %d total words from %d languages → %d after NMS",
                len(flat),
                len(per_lang_words),
                len(merged),
            )

        _write_merged_hocr(output_hocr, merged, page_w, page_h, hocr_lang=primary)
        output_text.write_text(
            _words_to_text(merged),
            encoding="utf-8",
        )


# ----------------------------------------------------------------------
# hOCR helpers
# ----------------------------------------------------------------------


def _read_page_size(image_path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` of the image in pixels."""
    from PIL import Image

    with Image.open(image_path) as img:
        return img.size


def _parse_hocr_words(hocr_path: Path) -> list[_Word]:
    """Extract every ``ocrx_word`` span from an hOCR file."""
    from lxml import html

    # Defence-in-depth: the hOCR is engine-generated and trusted. libxml2's
    # HTML parser does not expand XML/DTD entities (so XXE is not a vector
    # on this path), and no_network=True blocks any external fetch.
    parser = html.HTMLParser(no_network=True)

    words: list[_Word] = []
    tree = html.parse(str(hocr_path), parser=parser)
    for span in tree.iter("span"):
        cls = (span.get("class") or "").split()
        if "ocrx_word" not in cls:
            continue
        title = span.get("title") or ""
        bbox_m = _BBOX_RE.search(title)
        if bbox_m is None:
            continue
        conf_m = _CONF_RE.search(title)
        conf = int(conf_m.group(1)) if conf_m else 0
        text = "".join(span.itertext()).strip()
        if not text:
            continue
        words.append(
            _Word(
                text=text,
                x0=int(bbox_m.group(1)),
                y0=int(bbox_m.group(2)),
                x1=int(bbox_m.group(3)),
                y1=int(bbox_m.group(4)),
                conf=conf,
            ),
        )
    return words


def _word_bbox(w: _Word) -> BBox:
    """Adapter: expose a plugin :class:`_Word` as a layout ``BBox``.

    The ``layout`` reconstruction functions are generic over the caller's
    word type; this bridges the multi-language merge's axis-aligned
    :class:`_Word` to them. ``baseline_of`` is left to default (the bbox
    bottom edge), which is exact for these already axis-aligned boxes.
    """
    return (w.x0, w.y0, w.x1, w.y1)


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


def _nms_merge(words: list[_Word], iou_threshold: float) -> list[_Word]:
    """Greedy NMS by descending confidence, then re-order for reading.

    Surviving words are re-sequenced through :func:`layout.reading_blocks`
    so the merged hOCR and sidecar read in banded reading order:
    column bands left-to-right, lines top-to-bottom within each band.
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
    for block in reading_blocks(kept, bbox_of=_word_bbox):
        for line in block:
            ordered.extend(line)
    return ordered


def _write_merged_hocr(
    output_hocr: Path,
    words: list[_Word],
    page_w: int,
    page_h: int,
    *,
    hocr_lang: str,
) -> None:
    """Render merged words as a standard Page so the hOCR carries real lines.

    Grouped lines (rather than one line per word) give ocrmypdf's renderer
    line-level baselines and give PDF viewers sane text selection.
    """
    page = Page(
        width=page_w,
        height=page_h,
        lang=hocr_lang,
        ocr_system="paperless-paddleocr multi-lang merge",
    )
    for block in reading_blocks(words, bbox_of=_word_bbox):
        for line_words in block:
            box = (
                min(w.x0 for w in line_words),
                min(w.y0 for w in line_words),
                max(w.x1 for w in line_words),
                max(w.y1 for w in line_words),
            )
            slope, intercept = fit_baseline(
                [((w.x0 + w.x1) / 2.0, float(w.y1)) for w in line_words],
            )
            line = Line(
                box=box,
                confidence=round(sum(w.conf for w in line_words) / len(line_words)),
                text=" ".join(w.text for w in line_words),
                words=[HocrWord(w.text, (w.x0, w.y0, w.x1, w.y1), w.conf) for w in line_words],
                baseline=(slope, slope * box[0] + intercept - box[3]),
            )
            page.blocks.append(Block(box=box, lines=[line]))
    output_hocr.write_text(render_hocr(page), encoding="utf-8")


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
        choices=["classic-cpu", "classic-gpu", "vl-remote"],
        default="classic-cpu",
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
        "--paddle-vl-server-url",
        default="",
        dest="paddle_vl_server_url",
        metavar="URL",
        help=("vl-remote: URL of the PaddleOCR-VL inference server (e.g. http://gpu-box:8118)."),
    )
    paddle.add_argument(
        "--paddle-vl-model-name",
        default="PaddleOCR-VL-1.5-0.9B",
        dest="paddle_vl_model_name",
        metavar="NAME",
        help=(
            "vl-remote: model name advertised by the inference server "
            "(default: PaddleOCR-VL-1.5-0.9B)."
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
        default="v1.5",
        dest="paddle_vl_pipeline_version",
        metavar="VERSION",
        help=(
            "vl-remote: PaddleOCR-VL pipeline version (v1, v1.5, v1.6). Must "
            "match the model family the inference server serves "
            "(default: v1.5)."
        ),
    )
    paddle.add_argument(
        "--paddle-det-model-dir",
        metavar="DIR",
        dest="paddle_det_model_dir",
        help="Path to a custom text-detection model directory.",
    )
    paddle.add_argument(
        "--paddle-rec-model-dir",
        metavar="DIR",
        dest="paddle_rec_model_dir",
        help="Path to a custom text-recognition model directory.",
    )
    paddle.add_argument(
        "--paddle-cls-model-dir",
        metavar="DIR",
        dest="paddle_cls_model_dir",
        help="Path to a custom textline-orientation model directory.",
    )


_VALID_ENGINES: frozenset[str] = frozenset({"classic-cpu", "classic-gpu", "vl-remote"})


@ocrmypdf.hookimpl
def check_options(options: Any) -> None:
    """Validate that the requested PaddleOCR engine variant is installed."""
    from ocrmypdf.exceptions import MissingDependencyError

    engine = getattr(options, "paddle_engine", "classic-cpu")
    if engine not in _VALID_ENGINES:
        # argparse validates --paddle-engine via `choices=` on CLI usage, but
        # the Python API (ocrmypdf.ocr(paddle_engine=…)) skips that check.
        # Validate explicitly so an unknown engine fails fast rather than
        # silently falling through to classic-cpu.
        raise ValueError(
            f"Unknown paddle_engine={engine!r}. Valid choices: {sorted(_VALID_ENGINES)}.",
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
