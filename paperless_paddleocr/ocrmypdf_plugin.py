"""ocrmypdf plugin that binds the paperless-paddleocr engine.

This module is the one loaded by ``ocrmypdf.ocr(plugins=[…])`` when the
paperless parser in :mod:`paperless_paddleocr.parser` invokes ocrmypdf. Its
:func:`add_options`, :func:`check_options` and :func:`get_ocr_engine`
hookimpls register the ``--paddle-engine`` /
``--paddle-multi-lang-strategy`` / ``--paddle-vl-*`` /
``--paddle-*-model-dir`` options of the ocrmypdf command line and bind the
engine, :class:`paperless_paddleocr.paddle_engine.PaddleOCREngine`. That one
class also runs the passes of a document in several languages, and
:mod:`paperless_paddleocr.paddle_engine.merge` chooses or merges their pages.

Why this plugin is only invoked when paperless explicitly asks for it: the
engine lives inside this package (see :mod:`paperless_paddleocr.paddle_engine`)
rather than being published as a standalone ocrmypdf plugin, so ocrmypdf's
entry-point auto-discovery never registers it. PaddleOCR is only
used when our paperless parser passes
``plugins=["paperless_paddleocr.ocrmypdf_plugin"]`` to ``ocrmypdf.ocr()`` -
calls from paperless's built-in Tesseract parser are unaffected.
"""

from __future__ import annotations

import http.client
import logging
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request
from urllib.request import urlopen as _urlopen

import ocrmypdf
from ocrmypdf.exceptions import MissingDependencyError

from paperless_paddleocr import (
    DEFAULT_ENGINE,
    DEFAULT_STRATEGY,
    DEFAULT_VL_MODEL_NAME,
    DEFAULT_VL_PIPELINE_VERSION,
    DEFAULT_WORKER_PYTHON,
    ENGINE_CHOICES,
    STRATEGY_CHOICES,
)
from paperless_paddleocr.paddle_engine import PaddleOCREngine, worker
from paperless_paddleocr.paddle_engine.vl import (
    VL_REQUIREMENT,
    normalize_server_url,
    resolve_vl_api_key,
)

log = logging.getLogger("paperless.paddleocr.plugin")

#: Successful probes per (url, api_key). check_options runs once per
#: document, and one probe per Celery worker process is enough.
_PROBED_SERVERS: set[tuple[str, str]] = set()


def _probe_vl_server(server_url: str, api_key: str) -> None:
    """Fail fast when the VL server is unreachable or rejects the API key.

    Without this, a wrong URL or key surfaces only at first predict, deep
    inside a celery task, after the local pipeline is fully constructed.
    """
    base = normalize_server_url(server_url)
    cache_key = (base, api_key)
    if cache_key in _PROBED_SERVERS:
        return

    try:
        # A URL without a scheme makes Request raise ValueError.
        request = Request(f"{base}/models")  # noqa: S310 - operator-configured URL
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        # A vLLM server under load answers /models slowly. Only a successful
        # probe is cached per Celery worker process, so for a host that never
        # answers, each document waits this long and then fails.
        with _urlopen(request, timeout=30):  # noqa: S310
            pass
    except HTTPError as e:
        if e.code in (401, 403):
            raise MissingDependencyError(
                f"The PaddleOCR-VL server at {base} rejected the API key "
                f"(HTTP {e.code}). Check PAPERLESS_PADDLEOCR_VL_API_KEY against "
                "the api-key configured in the server's vllm_config.yaml.",
            ) from e
        log.warning(
            "PaddleOCR-VL server preflight got HTTP %d from %s/models; continuing.",
            e.code,
            base,
        )
    except (OSError, http.client.HTTPException, ValueError) as e:
        # urllib lets http.client errors such as BadStatusLine through
        # unwrapped, and they are no OSError. The text of a BadStatusLine is
        # the raw line of the peer, line ends included, and repr escapes them.
        cause = repr(e) if isinstance(e, http.client.BadStatusLine) else getattr(e, "reason", e)
        raise MissingDependencyError(
            f"The PaddleOCR-VL server at {base} is not reachable ({cause}). "
            "Check PAPERLESS_PADDLEOCR_VL_SERVER_URL and that the "
            "paddleocr-genai-vllm-server container is running.",
        ) from e
    _PROBED_SERVERS.add(cache_key)


# ----------------------------------------------------------------------
# Plugin hookimpls
# ----------------------------------------------------------------------


@ocrmypdf.hookimpl
def add_options(parser: Any) -> None:
    """Register the ``--paddle-*`` options of the ocrmypdf command line.

    From ocrmypdf 17.4.0 on, the Python API stores an unknown keyword such as
    ``paddle_engine`` without a check against this parser.
    """
    paddle = parser.add_argument_group(
        "PaddleOCR",
        "Options for PaddleOCR engine",
    )
    paddle.add_argument(
        "--paddle-engine",
        choices=ENGINE_CHOICES,
        default=DEFAULT_ENGINE,
        dest="paddle_engine",
        help=(
            "OCR engine variant: 'classic-cpu' (default, fast CNN pipeline on "
            "CPU), 'classic-gpu' (same CNN pipeline on a local NVIDIA GPU; "
            "requires paddlepaddle-gpu), or 'vl-remote' (the page image goes "
            "to a remote paddleocr-genai-vllm-server running PaddleOCR-VL-1.5, "
            "and no layout model runs locally)."
        ),
    )
    paddle.add_argument(
        "--paddle-multi-lang-strategy",
        choices=STRATEGY_CHOICES,
        default=DEFAULT_STRATEGY,
        dest="paddle_multi_lang_strategy",
        help=(
            "How to combine the per-language passes when more than one "
            "language is given: 'winner' (default, keep the pass with the "
            "highest aggregate confidence) or 'merge' (NMS-merge the word "
            "boxes of all passes)."
        ),
    )
    paddle.add_argument(
        "--paddle-vl-server-url",
        default="",
        dest="paddle_vl_server_url",
        metavar="URL",
        help=("vl-remote: URL of the PaddleOCR-VL inference server (e.g. http://gpu-box:8118)."),
    )
    paddle.add_argument(
        "--paddle-vl-model-name",
        default=DEFAULT_VL_MODEL_NAME,
        dest="paddle_vl_model_name",
        metavar="NAME",
        help=(
            "vl-remote: model name advertised by the inference server "
            f"(default: {DEFAULT_VL_MODEL_NAME})."
        ),
    )
    paddle.add_argument(
        "--paddle-vl-api-key",
        default="",
        dest="paddle_vl_api_key",
        metavar="KEY",
        help=(
            "vl-remote: Bearer token for the inference server's /v1/* "
            "endpoints. Leave blank if the server doesn't require auth."
        ),
    )
    paddle.add_argument(
        "--paddle-vl-pipeline-version",
        default=DEFAULT_VL_PIPELINE_VERSION,
        dest="paddle_vl_pipeline_version",
        metavar="VERSION",
        help=(
            "vl-remote: PaddleOCR-VL pipeline version (v1, v1.5, v1.6). Must "
            "match the model family the inference server serves "
            f"(default: {DEFAULT_VL_PIPELINE_VERSION})."
        ),
    )
    paddle.add_argument(
        "--paddle-det-model-name",
        metavar="NAME",
        dest="paddle_det_model_name",
        help=(
            "Name of the text-detection model (e.g. PP-OCRv6_small_det). "
            "Official models are downloaded on first use. When set, the "
            "language setting no longer influences model selection."
        ),
    )
    paddle.add_argument(
        "--paddle-det-model-dir",
        metavar="DIR",
        dest="paddle_det_model_dir",
        help="Path to a custom text-detection model directory.",
    )
    paddle.add_argument(
        "--paddle-rec-model-name",
        metavar="NAME",
        dest="paddle_rec_model_name",
        help=(
            "Name of the text-recognition model (e.g. PP-OCRv6_small_rec). "
            "Official models are downloaded on first use. When set, the "
            "language setting no longer influences model selection."
        ),
    )
    paddle.add_argument(
        "--paddle-rec-model-dir",
        metavar="DIR",
        dest="paddle_rec_model_dir",
        help="Path to a custom text-recognition model directory.",
    )


def _checked_choice(options: Any, name: str, choices: tuple[str, ...], default: str) -> str:
    # argparse validates `choices=` on CLI usage, but the Python API
    # (ocrmypdf.ocr(paddle_engine=…)) skips that check. An unknown value fails
    # here, where it would otherwise fall through to another variant
    # (classic-cpu for the engine).
    value = getattr(options, name, default)
    if value not in choices:
        raise ValueError(f"Unknown {name}={value!r}. Valid choices: {list(choices)}.")
    return value


@ocrmypdf.hookimpl
def check_options(options: Any) -> None:
    """Validate the PaddleOCR options and that the worker interpreter can run the engine variant."""
    engine = _checked_choice(options, "paddle_engine", ENGINE_CHOICES, DEFAULT_ENGINE)
    _checked_choice(options, "paddle_multi_lang_strategy", STRATEGY_CHOICES, DEFAULT_STRATEGY)

    # The ping runs here and never at import: a process that forks after its
    # worker started leaves the child a worker the child cannot reap, and the
    # child's first call warns about a dead worker or hangs on a lock that was
    # held at the fork.
    ping = _ping_worker()
    # The worker's reason tells a missing package from an installed one that
    # fails on import, for example on a cache directory it cannot write.
    if ping["paddleocr"] is None:
        paddle_line = (
            f"PaddlePaddle does not import either: {ping['paddle_error']}\n"
            if ping["paddle"] is None
            else ""
        )
        raise MissingDependencyError(
            "PaddleOCR does not import in the worker interpreter "
            f"{worker.worker_interpreter()}: {ping['paddleocr_error']}\n"
            + paddle_line
            + _install_hint(
                "If it is missing, install paddlepaddle and paddleocr there "
                "(see README, Installation).",
            ),
        )
    if ping["paddle"] is None:
        raise MissingDependencyError(
            "PaddlePaddle does not import in the worker interpreter "
            f"{worker.worker_interpreter()}: {ping['paddle_error']}\n"
            + _install_hint(
                "If it is missing, install paddlepaddle there, or paddlepaddle-gpu "
                "for the classic-gpu engine (see README, Installation).",
            ),
        )

    if engine == "vl-remote":
        if not ping["paddleocr_vl"]:
            raise MissingDependencyError(
                f"PaddleOCR-VL is not available: paddleocr {ping['paddleocr']} in the worker "
                f"interpreter {worker.worker_interpreter()} is older than the vl-remote engine "
                f'supports. Upgrade it there with "{VL_REQUIREMENT}" (see README, '
                "Troubleshooting).",
            )
        server_url = (getattr(options, "paddle_vl_server_url", "") or "").strip()
        if not server_url:
            raise MissingDependencyError(
                "vl-remote engine requires --paddle-vl-server-url (or the "
                "PAPERLESS_PADDLEOCR_VL_SERVER_URL env var) to point at a "
                "paddleocr-genai-vllm-server endpoint.",
            )
        # paperless-ngx runs ocrmypdf with skip_text for a PDF that already
        # holds text, and such a run sends a page to the server only when that
        # page has no text.
        if not getattr(options, "skip_text", False):
            _probe_vl_server(server_url, resolve_vl_api_key(options))
        return

    if engine != "classic-gpu":
        return
    if ping["cuda_error"] is not None:
        raise MissingDependencyError(
            f"classic-gpu engine could not query CUDA devices: {ping['cuda_error']}. "
            "Verify paddlepaddle-gpu is installed and the host NVIDIA driver is reachable "
            "from inside the container.",
        )
    # paddlepaddle-gpu reports 0 devices when the host has no NVIDIA
    # driver or when nvidia-container-toolkit isn't wired up - fail
    # fast rather than crashing inside predict().
    if ping["cuda_devices"] <= 0:
        raise MissingDependencyError(
            "classic-gpu engine requested but no CUDA device is visible. "
            "Check that the host has an NVIDIA driver, that "
            "nvidia-container-toolkit is configured, and that the "
            "container has GPU access (e.g. compose `deploy.resources."
            "reservations.devices` or `docker run --gpus all`).",
        )


def _install_hint(hint: str) -> str:
    # The fallback is the interpreter of paperless-ngx, and its Python 3.14 in
    # the 3.1 and later images has no PaddlePaddle wheel to install there.
    if worker.uses_fallback_interpreter():
        return (
            f"No worker venv was found at {DEFAULT_WORKER_PYTHON}. Install the worker venv "
            "as the README describes (Installation)."
        )
    return hint


def _ping_worker() -> dict[str, Any]:
    try:
        return worker.ping()
    except worker.PaddleWorkerError as e:
        raise MissingDependencyError(
            "The paddle worker cannot run under the worker interpreter "
            f"{worker.worker_interpreter()}: {e}\n"
            + _install_hint(
                "If that interpreter lacks paddlepaddle or paddleocr, install the worker venv "
                "as the README describes (Installation), or set PAPERLESS_PADDLEOCR_PYTHON to "
                "an interpreter that has both.",
            ),
        ) from e


@ocrmypdf.hookimpl(tryfirst=True)
def get_ocr_engine() -> Any:
    """Return the PaddleOCR engine.

    ``tryfirst=True`` makes this hookimpl run before the built-in Tesseract
    plugin's ``get_ocr_engine``; ocrmypdf's ``firstresult=True`` policy then
    short-circuits, so Tesseract never gets a chance to claim the engine.
    """
    return PaddleOCREngine()
