"""Readers for the ``PAPERLESS_PADDLEOCR_*`` environment variables.

This module imports neither Django nor paddle, so code that runs under a
hand-run ocrmypdf can use it as well as the parser.
"""

from __future__ import annotations

import logging
import os

from paperless_paddleocr import DEFAULT_ENGINE, ENGINE_CHOICES

log = logging.getLogger("paperless.paddleocr.env")


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("Invalid integer value for %s=%r; using default %d.", name, raw, default)
        return default


def env_choice(name: str, choices: tuple[str, ...], default: str) -> str:
    """Read ``name`` as one of ``choices``, falling back to ``default``.

    The value is stripped and lower-cased before the comparison, so a
    mixed-case entry in ``choices`` never matches.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    value = raw.lower()
    if value not in choices:
        log.warning(
            "Invalid value for %s=%r; valid choices are %s. Using default %r.",
            name,
            raw,
            choices,
            default,
        )
        return default
    return value


def engine_from_env() -> str:
    return env_choice("PAPERLESS_PADDLEOCR_ENGINE", ENGINE_CHOICES, DEFAULT_ENGINE)
