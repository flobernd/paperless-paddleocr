"""The parser module on top of fake paperless-ngx modules.

The unit environment has no paperless-ngx, so each case imports the parser
anew against fakes that give it the parent class and the host version.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

import pytest

import paperless_paddleocr

_PARSER_MODULE = "paperless_paddleocr.parser"


class _ParentWithoutBuilder:
    def __init__(self, logging_group: object | None = None) -> None:
        pass


class _Parent(_ParentWithoutBuilder):
    def construct_ocrmypdf_parameters(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}


@pytest.fixture
def load_parser(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., ModuleType]]:
    def load(
        *,
        parent: type = _Parent,
        version: tuple[int, ...] = (3, 4, 2),
        engine: str | None = None,
    ) -> ModuleType:
        if engine is None:
            monkeypatch.delenv("PAPERLESS_PADDLEOCR_ENGINE", raising=False)
        else:
            monkeypatch.setenv("PAPERLESS_PADDLEOCR_ENGINE", engine)
        tesseract = ModuleType("paperless.parsers.tesseract")
        tesseract.RasterisedDocumentParser = parent
        host = ModuleType("paperless.version")
        host.__version__ = version
        monkeypatch.setitem(sys.modules, "paperless", ModuleType("paperless"))
        monkeypatch.setitem(sys.modules, "paperless.parsers", ModuleType("paperless.parsers"))
        monkeypatch.setitem(sys.modules, "paperless.parsers.tesseract", tesseract)
        monkeypatch.setitem(sys.modules, "paperless.version", host)
        sys.modules.pop(_PARSER_MODULE, None)
        return importlib.import_module(_PARSER_MODULE)

    yield load
    sys.modules.pop(_PARSER_MODULE, None)
    # The import binds the module to the package as well, and
    # `from paperless_paddleocr import parser` would still find it there.
    vars(paperless_paddleocr).pop("parser", None)


def test_parser_without_the_builder_hook_fails_at_construction(load_parser):
    load_parser().PaperlessPaddleParser()
    broken = load_parser(parent=_ParentWithoutBuilder)
    with pytest.raises(RuntimeError, match="construct_ocrmypdf_parameters"):
        broken.PaperlessPaddleParser()


@pytest.mark.parametrize(("version", "levels"), [((3, 4, 2), []), ((4, 0, 0), ["WARNING"])])
def test_host_major_above_the_tested_one_logs_one_warning(load_parser, caplog, version, levels):
    load_parser(version=version)
    logged = [r.levelname for r in caplog.records if r.name == "paperless.parsing.paddleocr"]
    assert logged == levels
