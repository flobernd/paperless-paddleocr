"""Readers for the ``PAPERLESS_PADDLEOCR_*`` environment variables.

This module imports no Django, so code that runs under a hand-run ocrmypdf
can use it as well as the parser.
"""

from __future__ import annotations

import logging
import os

from paperless_paddleocr import DEFAULT_ENGINE, ENGINE_CHOICES

log = logging.getLogger("paperless.paddleocr.env")


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def env_int(name: str, default: int) -> int:
    raw = env_str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("Invalid integer value for %s=%r. Using the default %d.", name, raw, default)
        return default


def env_choice(name: str, choices: tuple[str, ...], default: str, *, strict: bool = False) -> str:
    """Read ``name`` as one of ``choices``, falling back to ``default``.

    The value is stripped and lower-cased before the comparison, so a
    mixed-case entry in ``choices`` never matches. With ``strict``, a value
    outside ``choices`` raises ``ValueError`` instead of falling back.
    """
    raw = env_str(name)
    if not raw:
        return default
    value = raw.lower()
    if value not in choices:
        if strict:
            raise ValueError(
                f"Invalid value for {name}={raw!r}. Valid values are "
                f"{', '.join(choices[:-1])} and {choices[-1]}.",
            )
        log.warning(
            "Invalid value for %s=%r. Valid choices are %s. Using the default %r.",
            name,
            raw,
            choices,
            default,
        )
        return default
    return value


def engine_from_env() -> str:
    """Read ``PAPERLESS_PADDLEOCR_ENGINE``, raising ``ValueError`` for an unknown value.

    A fallback to the default would run local OCR for a mistyped
    ``vl-remote``. The ocrmypdf plugin rejects the same value among its
    options.
    """
    return env_choice("PAPERLESS_PADDLEOCR_ENGINE", ENGINE_CHOICES, DEFAULT_ENGINE, strict=True)
