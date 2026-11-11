"""The ocrmypdf ``OcrEngine`` backed by PaddleOCR.

This is the public surface of the package. ocrmypdf drives OCR through the
:class:`PaddleOCREngine` ABC implementation; the actual recognition lives in
the ``classic`` / ``vl`` adapters and the wire format in ``hocr`` / ``pdf``.
``build_page`` picks the adapter by the requested engine variant.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence

from paperless_paddleocr import DEFAULT_ENGINE, DEFAULT_STRATEGY
from paperless_paddleocr.languages import PADDLE_NATIVE, TESSERACT_TO_PADDLE
from paperless_paddleocr.paddle_engine import classic, pdf, vl, worker
from paperless_paddleocr.paddle_engine.hocr import Page, Word, order_page, write_document
from paperless_paddleocr.paddle_engine.layout import reading_blocks
from paperless_paddleocr.paddle_engine.merge import (
    _blocks_to_text,
    _merged_page,
    _select,
    page_words,
)

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
    return getattr(options, "paddle_engine", DEFAULT_ENGINE)


class PaddleOCREngine(OcrEngine):
    """PaddleOCR implementation of ocrmypdf's ``OcrEngine``."""

    @staticmethod
    def version() -> str:
        # ocrmypdf can ask for the version at a point the plugin does not
        # control, so it comes from the ping of check_options and never from a
        # worker started here.
        ping = worker.cached_ping()
        return "unknown" if ping is None else str(ping["paddleocr"])

    @staticmethod
    def creator_tag(options: Any) -> str:
        if _variant(options) == "vl-remote":
            # The name follows the version string, because the option admits a
            # release newer than this code knows. The first release has no
            # number in its name.
            version = vl.pipeline_version(options)
            family = (
                "PaddleOCR-VL" if version == "v1" else f"PaddleOCR-VL-{version.removeprefix('v')}"
            )
            return f"{family} {PaddleOCREngine.version()}"
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
    def build_page(input_file: Path, options: Any) -> Page:
        """OCR ``input_file`` with the adapter of the requested engine variant."""
        if _variant(options) == "vl-remote":
            return vl.build_page(input_file, options)
        return classic.build_page(input_file, options)

    @staticmethod
    def generate_hocr(
        input_file: Path,
        output_hocr: Path,
        output_text: Path,
        options: Any,
    ) -> None:
        """Write the hOCR and the sidecar text of ``input_file``.

        Single-language input runs one pass. Multi-language input builds one
        page per language and combines the words of those pages based on the
        ``paddle_multi_lang_strategy`` option, which the paperless parser sets
        from ``PAPERLESS_PADDLEOCR_MULTI_LANG_STRATEGY``:

        * ``winner`` (default) - keep only the words of the language with the
          highest aggregate confidence (the sum of its word confidences).
          Faster and more consistent on documents whose pages are single-script.
        * ``merge`` - NMS-merge word boxes from all languages by confidence.
          Right choice for genuinely mixed-script pages.

        The first pass that raises fails the document.
        """
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
            page = PaddleOCREngine.build_page(input_file, options)
            write_document(page, output_hocr, output_text, sidecar=order_page(page))
            return

        strategy = getattr(options, "paddle_multi_lang_strategy", DEFAULT_STRATEGY)
        log.info(
            "Multi-language PaddleOCR pass on %s: %d languages (%s), strategy=%s",
            input_file,
            len(langs),
            "+".join(langs),
            strategy,
        )

        per_lang_words: dict[str, list[Word]] = {}
        for lang in langs:
            sub_options = copy.copy(options)
            sub_options.languages = [lang]
            try:
                page = PaddleOCREngine.build_page(input_file, sub_options)
            except Exception as e:
                raise RuntimeError(
                    f"PaddleOCR pass for language {lang!r} failed on {input_file}: {e}",
                ) from e
            per_lang_words[lang] = page_words(page)

        merged, primary = _select(strategy, per_lang_words)
        blocks = reading_blocks(merged, bbox_of=lambda w: w.box)
        # The adapters size each page from the input image, so any pass gives the page size.
        write_document(
            _merged_page(blocks, page.width, page.height, hocr_lang=primary),
            output_hocr,
            output_text,
            sidecar=_blocks_to_text(blocks),
        )

    @staticmethod
    def generate_pdf(
        input_file: Path,
        output_pdf: Path,
        output_text: Path,
        options: Any,
    ) -> None:
        """Render OCR output as a text-only PDF."""
        output_hocr = output_pdf.with_suffix(".hocr")
        PaddleOCREngine.generate_hocr(input_file, output_hocr, output_text, options)
        pdf.render_textonly(input_file, output_hocr, output_pdf)
