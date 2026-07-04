"""The ocrmypdf ``OcrEngine`` backed by PaddleOCR.

This is the public surface of the package. ocrmypdf drives OCR through the
:class:`PaddleOCREngine` ABC implementation; the actual recognition lives in
the ``classic`` / ``vl`` adapters and the wire format in ``hocr`` / ``pdf``.
``generate_hocr`` dispatches on the requested engine variant;
``_generate_hocr_classic`` / ``_generate_hocr_vl`` stay as overridable
staticmethods so the multi-language subclass and tests can intercept a
single pass.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence

from paperless_paddleocr.languages import PADDLE_NATIVE, TESSERACT_TO_PADDLE
from paperless_paddleocr.paddle_engine import classic, pdf, vl
from paperless_paddleocr.paddle_engine.hocr import Page, write_document
from paperless_paddleocr.paddle_engine.layout import reading_blocks

log = logging.getLogger("paperless.paddleocr.engine")

#: Reported to ocrmypdf's preflight language check. Derived from the language
#: tables so a code our own mapping can emit never spuriously aborts OCR;
#: raw Tesseract codes are included because a mis-set
#: PAPERLESS_PADDLEOCR_LANGUAGE should fail in the engine with a clear error,
#: not in ocrmypdf preflight.
_SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {
        *PADDLE_NATIVE,
        *TESSERACT_TO_PADDLE,
        *TESSERACT_TO_PADDLE.values(),
    },
)


def _variant(options: Any) -> str:
    return getattr(options, "paddle_engine", "classic-cpu")


def _silence_paddle_signal_handler() -> None:
    """Drop paddle's C++ signal handler so worker recycling stays quiet.

    paddlepaddle installs a SIGTERM/SIGSEGV handler that prints an alarming
    C++ traceback - for a plain SIGTERM, a fake ``FatalError: Termination
    signal`` with an empty stack. paperless sets
    ``CELERY_WORKER_MAX_TASKS_PER_CHILD = 1``, so every document ends with its
    celery worker being SIGTERM-recycled; without this the log gets that fake
    error on every page. Called per task because the recycled worker is a
    fresh process that re-imports (and re-arms) paddle. Best-effort: a missing
    API or any failure must never block OCR.
    """
    try:
        import paddle

        paddle.disable_signal_handler()
    except Exception:  # pragma: no cover - cosmetic only
        pass


def _order_page(page: Page) -> str:
    """Reorder page.blocks in banded reading order; return the sidecar text.

    Line.text stays authoritative for the sidecar because engines may space
    a line differently from a naive word join (CJK has no inter-word
    spaces). The hOCR is reordered too so the text layer and the sidecar
    agree on reading order.
    """
    if not page.blocks:
        return ""
    grouped = reading_blocks(page.blocks, bbox_of=lambda b: b.box)
    ordered = [blk for group in grouped for row in group for blk in row]
    page.blocks = ordered
    return "\n\n".join(
        "\n".join(line.text for row in group for blk in row for line in blk.lines if line.text)
        for group in grouped
    )


class PaddleOCREngine(OcrEngine):
    """PaddleOCR implementation of ocrmypdf's ``OcrEngine``."""

    @staticmethod
    def version() -> str:
        try:
            import paddleocr

            return paddleocr.__version__
        except (ImportError, AttributeError):
            return "unknown"

    @staticmethod
    def creator_tag(options: Any) -> str:
        if _variant(options) == "vl-remote":
            return f"PaddleOCR-VL-1.5 {PaddleOCREngine.version()}"
        return f"PaddleOCR {PaddleOCREngine.version()}"

    def __str__(self) -> str:
        return f"PaddleOCR {PaddleOCREngine.version()}"

    @staticmethod
    def languages(options: Any) -> set[str]:
        return set(_SUPPORTED_LANGUAGES)

    @staticmethod
    def get_orientation(input_file: Path, options: Any) -> OrientationConfidence:
        from paperless_paddleocr.paddle_engine import orientation

        return orientation.get_orientation(input_file)

    @staticmethod
    def get_deskew(input_file: Path, options: Any) -> float:
        from paperless_paddleocr.paddle_engine import deskew

        try:
            return deskew.estimate_skew(input_file)
        except Exception:
            log.exception("Deskew estimation failed for %s; skipping deskew.", input_file)
            return 0.0

    @staticmethod
    def _generate_hocr_classic(
        input_file: Path,
        output_hocr: Path,
        output_text: Path,
        options: Any,
    ) -> None:
        page = classic.build_page(input_file, options)
        write_document(page, output_hocr, output_text, sidecar=_order_page(page))

    @staticmethod
    def _generate_hocr_vl(
        input_file: Path,
        output_hocr: Path,
        output_text: Path,
        options: Any,
    ) -> None:
        page = vl.build_page(input_file, options)
        write_document(page, output_hocr, output_text, sidecar=_order_page(page))

    @staticmethod
    def generate_hocr(
        input_file: Path,
        output_hocr: Path,
        output_text: Path,
        options: Any,
    ) -> None:
        _silence_paddle_signal_handler()
        if _variant(options) == "vl-remote":
            PaddleOCREngine._generate_hocr_vl(input_file, output_hocr, output_text, options)
        else:
            PaddleOCREngine._generate_hocr_classic(input_file, output_hocr, output_text, options)

    @classmethod
    def generate_pdf(
        cls,
        input_file: Path,
        output_pdf: Path,
        output_text: Path,
        options: Any,
    ) -> None:
        """Render OCR output as a text-only PDF.

        ``cls.generate_hocr`` (not the base method) so a subclass - the
        multi-language engine ocrmypdf actually instantiates - still runs if
        a user forces the ``generate_pdf`` renderer path.
        """
        output_hocr = output_pdf.with_suffix(".hocr")
        cls.generate_hocr(input_file, output_hocr, output_text, options)
        pdf.render_textonly(input_file, output_hocr, output_pdf)
