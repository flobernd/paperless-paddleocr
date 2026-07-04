"""ocrmypdf aborts OCR when a requested code is missing from languages().

The allowlist must therefore cover every code resolve_paddle_languages can
emit (mapped targets, bundles, override passthrough) and the raw Tesseract
codes, so a slightly wrong override degrades later with a clear engine error
instead of a spurious preflight abort.
"""

from __future__ import annotations

from paperless_paddleocr.languages import PADDLE_NATIVE, TESSERACT_TO_PADDLE
from paperless_paddleocr.paddle_engine.engine import PaddleOCREngine


def test_languages_covers_all_resolvable_codes():
    supported = PaddleOCREngine.languages(options=None)
    assert set(TESSERACT_TO_PADDLE.values()) <= supported
    assert set(TESSERACT_TO_PADDLE) <= supported
    assert set(PADDLE_NATIVE) <= supported
    assert {"ml", "latin", "es", "sk", "eng"} <= supported
