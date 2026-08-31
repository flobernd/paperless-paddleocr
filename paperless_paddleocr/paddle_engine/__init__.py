"""PaddleOCR-backed ``OcrEngine`` for ocrmypdf.

The engine is a first-party part of this package rather than an installed
ocrmypdf plugin on purpose: ocrmypdf auto-registers any engine published in
its plugin entry-point group for *every* ``ocrmypdf.ocr()`` call on the
host, which would displace Tesseract for unrelated callers. Keeping the
engine here means PaddleOCR is only used when
:mod:`paperless_paddleocr.ocrmypdf_plugin` is passed explicitly via
``plugins=[…]``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PaddleOCREngine"]


def __getattr__(name: str) -> Any:
    """Lazily import :class:`PaddleOCREngine` on first attribute access.

    Importing the engine pulls in PaddleOCR/PaddleX, which initializes
    PaddleX. Deferring the import keeps parser-registry discovery from
    initializing PaddleX until ``ocrmypdf`` actually needs the engine.
    """
    if name == "PaddleOCREngine":
        from paperless_paddleocr.paddle_engine.engine import PaddleOCREngine

        return PaddleOCREngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
