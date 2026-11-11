"""Smoke test executed inside the paperless-ngx Docker image.

This is *not* collected by pytest (filename is not ``test_*``). It is run
directly by ``.github/workflows/smoke.yml`` with the real paperless-ngx
runtime on ``PYTHONPATH``. Its job is to catch the failure modes the
lightweight unit suite cannot:

* the ``paperless_ngx.parsers`` entry point is actually discoverable, and
  resolves to :class:`PaperlessPaddleParser`
* ``paperless_paddleocr.parser`` imports cleanly against the *real*
  paperless-ngx API surface (``paperless.parsers.tesseract``,
  ``paperless.version``) - i.e. paperless has not renamed/moved anything
  the parser depends on
* ``paperless_paddleocr.ocrmypdf_plugin`` imports and the ocrmypdf hookimpl
  wiring binds our engine
* the Tesseract→PaddleOCR language mapping behaves
* the parser subclasses the upstream Tesseract parser, and the arguments its
  builder returns carry the engine wiring and the options from the
  environment, keep the user args' precedence and never carry the API key
* one page goes through the paddle worker: the classic-cpu engine reads its
  text and tags the archive, and the vl-remote engine fails the document
  when its server does not answer
* with the vl-remote engine the parser declares itself a remote service, and
  the registry gives a PDF to it only while remote parsers are allowed.

The document run needs a worker venv, as the recipes and ``setup.sh``
install it.

Exit code is non-zero on the first failed check so the workflow fails loudly.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


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
    # Importing the parser exercises each of its `from paperless...` imports
    # against the real paperless-ngx install.
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
    import os
    import tempfile
    from unittest import mock

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

        def build(**kwargs: bool) -> dict:
            with PaperlessPaddleParser() as parser:
                return parser.construct_ocrmypdf_parameters(
                    png,
                    "image/png",
                    tmp / "out.pdf",
                    tmp / "sidecar.txt",
                    **kwargs,
                )

        args = build()
        _expect(args, "plugins", [module])
        _expect(args, "language", "en")
        _expect_no_api_key(args)
        _expect(args, "paddle_engine", "classic-cpu")
        _ok("builder adds the plugin, the PaddleOCR language and the engine")

        # Values other than the defaults, so a misspelled key fails. The
        # parser reads the environment when it is constructed.
        environment = {
            "PAPERLESS_PADDLEOCR_MULTI_LANG_STRATEGY": "merge",
            "PAPERLESS_PADDLEOCR_VL_MODEL_NAME": "smoke-model",
            "PAPERLESS_PADDLEOCR_VL_PIPELINE_VERSION": "v1",
            "PAPERLESS_PADDLEOCR_VL_SERVER_URL": "http://vl.invalid:8118",
        }
        with mock.patch.dict(os.environ, environment):
            args = build()
        _expect(args, "paddle_multi_lang_strategy", "merge")
        _expect(args, "paddle_vl_model_name", "smoke-model")
        _expect(args, "paddle_vl_pipeline_version", "v1")
        _expect(args, "paddle_vl_server_url", "http://vl.invalid:8118")
        _ok("builder passes the strategy and the VL options from the environment")

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

        args = build(skip_text=True)
        _expect(args, "skip_text", True)
        _ok("builder hands its keywords to the upstream builder")

        user_args = '{"plugins": ["paperless_paddleocr.ocrmypdf_plugin", "other_plugin"]}'
        with override_settings(OCR_USER_ARGS=user_args):
            args = build()
        _expect(args, "plugins", [module, "other_plugin"])
        _ok("user plugin list is kept, with this plugin listed once")


def _make_test_page() -> ModuleType:
    # By path, because neither tests/ nor tests/e2e/ is a package.
    path = Path(__file__).resolve().parent / "e2e" / "make_test_page.py"
    spec = importlib.util.spec_from_file_location("make_test_page", path)
    if spec is None or spec.loader is None:
        _fail(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_document_run() -> None:
    # Runs after django.setup() in check_parser_imports.
    import inspect
    import os
    import tempfile
    from unittest import mock

    import pikepdf
    from documents.parsers import ParseError
    from paperless.parsers.registry import get_parser_registry, reset_parser_registry
    from paperless.version import __version__ as host_version

    import paperless_paddleocr.parser as parser_module

    with tempfile.TemporaryDirectory() as tmp_name:
        png = Path(tmp_name) / "page.png"
        _make_test_page().write_page(png)

        # patch.dict undoes each change to the environment, the one of the
        # parse method (OMP_THREAD_LIMIT) included, also when a check exits.
        with mock.patch.dict(os.environ):
            os.environ.pop("PAPERLESS_PADDLEOCR_ENGINE", None)
            with parser_module.PaperlessPaddleParser() as parser:
                parser.parse(png, "image/png")
                text = parser.get_text()
                if "Paperless" not in text:
                    _fail(f"the classic-cpu text lacks 'Paperless': {text!r}")
                with pikepdf.open(parser.get_archive_path()) as pdf:
                    creator = str(pdf.docinfo.get("/Creator", ""))
                if "PaddleOCR" not in creator:
                    _fail(f"the archive's /Creator {creator!r} does not name PaddleOCR")
        _ok(f"classic-cpu reads one page through the worker, /Creator {creator!r}")

        remote = {
            "PAPERLESS_PADDLEOCR_ENGINE": "vl-remote",
            # Nothing listens on the discard port 9, so the connection is refused.
            "PAPERLESS_PADDLEOCR_VL_SERVER_URL": "http://127.0.0.1:9",
        }
        try:
            with mock.patch.dict(os.environ, remote):
                # The parser reads the remote flag at import, and the class
                # of the old module keeps the old value.
                parser_class = importlib.reload(parser_module).PaperlessPaddleParser
                if parser_class.uses_remote_service is not True:
                    _fail("uses_remote_service is not True for the vl-remote engine")
                # A registry built before the reload holds the class of the old module.
                reset_parser_registry()
                registry = get_parser_registry()
                if registry.get_parser_for_file("application/pdf", "doc.pdf") is not parser_class:
                    _fail("the registry does not give a PDF to the vl-remote parser")
                # paperless-ngx 3.0 predates the allow_remote keyword, so only 3.0 skips
                # the second query. A later release without it fails, or the run against
                # :latest would stop checking the flag once upstream drops or renames it.
                if "allow_remote" in inspect.signature(registry.get_parser_for_file).parameters:
                    local = registry.get_parser_for_file(
                        "application/pdf", "doc.pdf", allow_remote=False
                    )
                    if local is parser_class:
                        _fail(
                            "the registry gives a PDF to the vl-remote parser "
                            "without remote parsers"
                        )
                elif host_version[:2] != (3, 0):
                    _fail(
                        f"the registry of paperless-ngx {'.'.join(map(str, host_version))} has no "
                        "allow_remote keyword, so the check cannot ask it to leave out "
                        "remote parsers"
                    )
                with parser_class() as parser:
                    try:
                        parser.parse(png, "image/png")
                    except ParseError as e:
                        error = str(e)
                    else:
                        _fail("parse succeeded although the VL server does not answer")
                if "127.0.0.1:9" not in error:
                    _fail(f"the ParseError does not name the VL server: {error!r}")
        finally:
            # Later code sees the module of the restored environment.
            importlib.reload(parser_module)
            reset_parser_registry()
        _ok(f"vl-remote fails the document without a server: {error.splitlines()[0]}")


def main() -> None:
    print("== paperless-paddleocr smoke test ==", flush=True)
    check_entry_point()
    check_plugin_and_languages()
    check_parser_imports()
    check_parser_contract()
    check_document_run()
    print("== all smoke checks passed ==", flush=True)


if __name__ == "__main__":
    main()
