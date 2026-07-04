"""Smoke test executed inside the paperless-ngx Docker image.

This is *not* collected by pytest (filename is not ``test_*``); it is run
directly by ``.github/workflows/smoke.yml`` with the real paperless-ngx
runtime on ``PYTHONPATH``. Its job is to catch the failure modes the
lightweight unit suite cannot:

* the ``paperless_ngx.parsers`` entry point is actually discoverable, and
  resolves to :class:`PaperlessPaddleParser`;
* ``paperless_paddleocr.parser`` imports cleanly against the *real*
  paperless-ngx API surface (``documents.parsers``, ``paperless.config``,
  ``paperless.parsers.*``) - i.e. paperless has not renamed/moved anything
  the parser depends on;
* ``paperless_paddleocr.ocrmypdf_plugin`` imports and the ocrmypdf hookimpl
  wiring binds our engine;
* the Tesseract→PaddleOCR language mapping behaves.

Exit code is non-zero on the first failed check so the workflow fails loudly.
"""

from __future__ import annotations

import importlib.metadata
import sys


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}", flush=True)
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"  PASS  {msg}", flush=True)


def check_entry_point() -> None:
    eps = importlib.metadata.entry_points(group="paperless_ngx.parsers")
    names = {ep.name: ep for ep in eps}
    if "paddleocr" not in names:
        _fail(f"entry point 'paddleocr' not found in paperless_ngx.parsers ({sorted(names)})")
    target = names["paddleocr"].value
    if "PaperlessPaddleParser" not in target:
        _fail(f"entry point 'paddleocr' resolves to unexpected target: {target!r}")
    _ok(f"entry point discoverable: paddleocr -> {target}")


def check_parser_imports() -> None:
    # Importing this module exercises every `from documents...` /
    # `from paperless...` import against the real paperless-ngx install.
    import django

    django.setup()

    from paperless_paddleocr.parser import PaperlessPaddleParser

    mimes = PaperlessPaddleParser.supported_mime_types()
    if "application/pdf" not in mimes:
        _fail(f"application/pdf missing from supported_mime_types(): {mimes}")
    score = PaperlessPaddleParser.score("application/pdf", "doc.pdf")
    if score is None or score <= 10:
        _fail(f"score() should beat Tesseract's 10, got {score!r}")
    _ok(f"parser imports against real paperless-ngx; score={score}")


def check_plugin_and_languages() -> None:
    from paperless_paddleocr.languages import resolve_paddle_languages
    from paperless_paddleocr.ocrmypdf_plugin import MultiLangPaddleEngine, get_ocr_engine

    resolved = resolve_paddle_languages("eng+deu", None)
    if resolved != ["en", "german"]:
        _fail(f"resolve_paddle_languages('eng+deu') = {resolved!r}, expected ['en','german']")

    engine = get_ocr_engine()
    if not isinstance(engine, MultiLangPaddleEngine):
        _fail(f"get_ocr_engine() returned {type(engine).__name__}, expected MultiLangPaddleEngine")
    _ok("ocrmypdf plugin wiring + language mapping OK")


def main() -> None:
    print("== paperless-paddleocr smoke test ==", flush=True)
    check_entry_point()
    check_plugin_and_languages()
    check_parser_imports()
    print("== all smoke checks passed ==", flush=True)


if __name__ == "__main__":
    main()
