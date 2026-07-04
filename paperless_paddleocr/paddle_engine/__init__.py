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

from paperless_paddleocr.paddle_engine.engine import PaddleOCREngine

__all__ = ["PaddleOCREngine"]
