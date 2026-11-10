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
* the Tesseract→PaddleOCR language mapping behaves;
* the parser subclasses the upstream Tesseract parser, and the arguments its
  builder returns carry the engine wiring, keep the user args' precedence and
  never carry the API key.

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


def _expect(args: dict, key: str, expected: object) -> None:
    if args.get(key) != expected:
        _fail(f"args[{key!r}] = {args.get(key)!r}, expected {expected!r}")


def _expect_no_api_key(args: dict) -> None:
    if "paddle_vl_api_key" in args:
        _fail("the ocrmypdf arguments carry paddle_vl_api_key")


def check_parser_contract() -> None:
    # Runs after django.setup() in check_parser_imports.
    import tempfile
    from pathlib import Path

    from django.test import override_settings
    from paperless.parsers.tesseract import RasterisedDocumentParser
    from PIL import Image

    from paperless_paddleocr.parser import PaperlessPaddleParser

    if not issubclass(PaperlessPaddleParser, RasterisedDocumentParser):
        _fail("PaperlessPaddleParser does not subclass RasterisedDocumentParser")
    if "construct_ocrmypdf_parameters" not in PaperlessPaddleParser.__dict__:
        _fail("PaperlessPaddleParser does not override construct_ocrmypdf_parameters")
    if PaperlessPaddleParser.uses_remote_service is not False:
        _fail("uses_remote_service is not False for the classic-cpu engine")

    module = "paperless_paddleocr.ocrmypdf_plugin"
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        png = tmp / "page.png"
        Image.new("RGB", (300, 300), "white").save(png, dpi=(72, 72))

        def build() -> dict:
            with PaperlessPaddleParser() as parser:
                return parser.construct_ocrmypdf_parameters(
                    png,
                    "image/png",
                    tmp / "out.pdf",
                    tmp / "sidecar.txt",
                )

        args = build()
        _expect(args, "plugins", [module])
        _expect(args, "language", "en")
        _expect_no_api_key(args)
        _expect(args, "paddle_engine", "classic-cpu")
        _ok("builder adds the plugin, the PaddleOCR language and the engine")

        # OcrConfig reads settings.OCR_USER_ARGS, which paperless fills from
        # PAPERLESS_OCR_USER_ARGS once at import, so the environment cannot
        # change it here.
        user_args = (
            '{"language": "eng+deu", "paddle_vl_model_name": "custom", '
            '"paddle_vl_api_key": "leak", "paddle_engine": "vl-remote", '
            '"plugins": "paperless_paddleocr.ocrmypdf_plugin"}'
        )
        with override_settings(OCR_USER_ARGS=user_args):
            args = build()
        _expect(args, "language", "eng+deu")
        _expect(args, "paddle_vl_model_name", "custom")
        _expect_no_api_key(args)
        _expect(args, "paddle_engine", "classic-cpu")
        _expect(args, "plugins", [module])
        _ok("user args keep precedence, except for the API key and the engine")

        user_args = '{"plugins": ["paperless_paddleocr.ocrmypdf_plugin", "other_plugin"]}'
        with override_settings(OCR_USER_ARGS=user_args):
            args = build()
        _expect(args, "plugins", [module, "other_plugin"])
        _ok("user plugin list is kept, with this plugin listed once")


def main() -> None:
    print("== paperless-paddleocr smoke test ==", flush=True)
    check_entry_point()
    check_plugin_and_languages()
    check_parser_imports()
    check_parser_contract()
    print("== all smoke checks passed ==", flush=True)


if __name__ == "__main__":
    main()
