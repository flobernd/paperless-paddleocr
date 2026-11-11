"""PaddleOCR provider for paperless-ngx."""

from __future__ import annotations

from typing import Final

__version__ = "0.2.0"

# One home for the option defaults the parser, the ocrmypdf plugin and the
# engine all read. This module stays free of package imports on purpose:
# paperless imports the parser at startup, where loading the plugin would
# load ocrmypdf, which the upstream parser imports only when it parses.
ENGINE_CHOICES: Final[tuple[str, ...]] = ("classic-cpu", "classic-gpu", "vl-remote")
DEFAULT_ENGINE: Final[str] = "classic-cpu"
STRATEGY_CHOICES: Final[tuple[str, ...]] = ("merge", "winner")
DEFAULT_STRATEGY: Final[str] = "winner"
DEFAULT_VL_MODEL_NAME: Final[str] = "PaddleOCR-VL-1.5-0.9B"
DEFAULT_VL_PIPELINE_VERSION: Final[str] = "v1.5"
DEFAULT_WORKER_PYTHON: Final[str] = "/opt/paperless-paddleocr/venv/bin/python"
