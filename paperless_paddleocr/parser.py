"""Paperless-ngx parser plugin that runs ocrmypdf with the PaddleOCR engine.

:class:`PaperlessPaddleParser` subclasses paperless-ngx's Tesseract parser and
overrides only ``construct_ocrmypdf_parameters``, so each paperless-ngx
version keeps its own parse flow, image handling, PDF/A conversion and
``OCR_MODE`` semantics. The override adds this package's ocrmypdf plugin,
translates paperless's Tesseract language codes to PaddleOCR codes and
passes the ``paddle_engine``, ``paddle_multi_lang_strategy``,
``paddle_vl_model_name``, ``paddle_vl_pipeline_version`` and
``paddle_vl_server_url`` options.

The plugin relies on three contracts of the parent's parse method:

* It calls the builder for each ocrmypdf run, the safe-fallback run
  included, so each run loads the PaddleOCR engine.
* It takes the document text from the sidecar file when there is one, and
  ocrmypdf assembles that file from the text the engine writes per page.
* Its exception handling decides which errors fail the consumption and
  which ones start the fallback run.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Final

from paperless.parsers.tesseract import RasterisedDocumentParser
from paperless.version import __version__ as _host_version

from paperless_paddleocr import (
    DEFAULT_STRATEGY,
    DEFAULT_VL_MODEL_NAME,
    DEFAULT_VL_PIPELINE_VERSION,
    STRATEGY_CHOICES,
    __version__,
)
from paperless_paddleocr.env import engine_from_env, env_choice, env_int
from paperless_paddleocr.languages import resolve_paddle_languages

logger = logging.getLogger("paperless.parsing.paddleocr")

_BUILDER_HOOK_PRESENT: Final[bool] = hasattr(
    RasterisedDocumentParser, "construct_ocrmypdf_parameters"
)
_BUILDER_HOOK_MISSING: Final[str] = (
    "paperless-ngx's RasterisedDocumentParser no longer defines "
    "construct_ocrmypdf_parameters, so paperless-paddleocr cannot attach its engine"
)
# The module stays importable so that a broken install cannot pass unnoticed.
# The registry logs a failed import once and then gives each document to
# Tesseract, whereas a parser that fails at construction fails each
# consumption with this message.
if not _BUILDER_HOOK_PRESENT:
    logger.error(_BUILDER_HOOK_MISSING)

_TESTED_MAJOR: Final[int] = 3
if _host_version[0] > _TESTED_MAJOR:
    logger.warning(
        "paperless-ngx %s is newer than the versions paperless-paddleocr was tested with "
        "(3.0 to 3.4).",
        ".".join(map(str, _host_version)),
    )

_OCRMYPDF_PLUGIN_MODULE: Final[str] = "paperless_paddleocr.ocrmypdf_plugin"


def _user_plugins(value: Any) -> list[Any]:
    """The ``plugins`` value of the user args as a list, without this package's plugin.

    ocrmypdf accepts a single string or path as ``plugins``, so a bare one must
    not expand per character. The parser lists its own plugin first, and
    pluggy rejects a module registered twice.
    """
    if value is None:
        return []
    items = [value] if isinstance(value, str | Path) else list(value)
    return [item for item in items if str(item) != _OCRMYPDF_PLUGIN_MODULE]


class PaperlessPaddleParser(RasterisedDocumentParser):
    """OCR parser using PaddleOCR via ocrmypdf (drop-in for the Tesseract parser)."""

    name: str = "Paperless-ngx PaddleOCR Parser"
    version: str = __version__
    author: str = "Florian Bernd"
    url: str = "https://github.com/flobernd/paperless-paddleocr"
    # The registry reads this flag on the class, so the engine variable sets it
    # at import. The builder forces the same engine onto each ocrmypdf run.
    uses_remote_service: bool = engine_from_env() == "vl-remote"

    @classmethod
    def score(
        cls,
        mime_type: str,
        filename: str,
        path: Path | None = None,
    ) -> int | None:
        if mime_type not in cls.supported_mime_types():
            return None
        # Default 15 beats Tesseract's 10 - once installed, PaddleOCR wins.
        return env_int("PAPERLESS_PADDLEOCR_SCORE", default=15)

    def __init__(self, logging_group: object | None = None) -> None:
        # Raised before the parent creates its scratch directory, which no
        # __exit__ would remove after a failed construction.
        if not _BUILDER_HOOK_PRESENT:
            raise RuntimeError(_BUILDER_HOOK_MISSING)
        super().__init__(logging_group)
        self.log = logger

        # Plugin-specific env vars, read once per parser instance.
        self._engine: str = engine_from_env()
        self._strategy: str = env_choice(
            "PAPERLESS_PADDLEOCR_MULTI_LANG_STRATEGY",
            STRATEGY_CHOICES,
            DEFAULT_STRATEGY,
        )
        self._vl_server_url: str = os.environ.get("PAPERLESS_PADDLEOCR_VL_SERVER_URL", "").strip()
        self._vl_model_name: str = (
            os.environ.get("PAPERLESS_PADDLEOCR_VL_MODEL_NAME", "").strip() or DEFAULT_VL_MODEL_NAME
        )
        self._vl_pipeline_version: str = (
            os.environ.get("PAPERLESS_PADDLEOCR_VL_PIPELINE_VERSION", "").strip()
            or DEFAULT_VL_PIPELINE_VERSION
        )
        self._lang_override: str | None = os.environ.get("PAPERLESS_PADDLEOCR_LANGUAGE") or None

    def construct_ocrmypdf_parameters(
        self,
        input_file: Path,
        mime_type: str,
        output_file: Path,
        sidecar_file: Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # A later paperless-ngx release can add a keyword to its call of the
        # builder. The keyword reaches the parent's builder unchanged, and a
        # list of named keywords here would raise TypeError for each document.
        args: dict[str, Any] = super().construct_ocrmypdf_parameters(
            input_file,
            mime_type,
            output_file,
            sidecar_file,
            **kwargs,
        )

        args["plugins"] = [_OCRMYPDF_PLUGIN_MODULE, *_user_plugins(args.get("plugins"))]

        # The inherited parse method logs these arguments at debug level, and
        # paperless writes debug messages to its log file by default.
        if "paddle_vl_api_key" in args:
            del args["paddle_vl_api_key"]
            self.log.warning(
                "Ignoring paddle_vl_api_key from the OCR user args. "
                "Set PAPERLESS_PADDLEOCR_VL_API_KEY instead.",
            )

        # Upstream merges the user args last, and a language the user sets
        # there keeps that precedence.
        user_args = self.settings.user_args
        if not (isinstance(user_args, dict) and "language" in user_args):
            args["language"] = "+".join(
                resolve_paddle_languages(self.settings.language, self._lang_override),
            )

        # uses_remote_service follows PAPERLESS_PADDLEOCR_ENGINE. A user arg
        # that switched a classic engine to vl-remote would send documents to
        # a server while the registry treats the parser as local.
        user_engine = args.get("paddle_engine")
        if user_engine is not None and user_engine != self._engine:
            self.log.warning(
                "Ignoring paddle_engine=%r from the OCR user args. The engine comes from "
                "PAPERLESS_PADDLEOCR_ENGINE only (%s).",
                user_engine,
                self._engine,
            )
        args["paddle_engine"] = self._engine

        # A user arg keeps precedence over these environment values.
        args.setdefault("paddle_multi_lang_strategy", self._strategy)
        args.setdefault("paddle_vl_model_name", self._vl_model_name)
        args.setdefault("paddle_vl_pipeline_version", self._vl_pipeline_version)
        args.setdefault("paddle_vl_server_url", self._vl_server_url)
        return args
