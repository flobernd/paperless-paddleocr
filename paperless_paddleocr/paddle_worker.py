"""PaddlePaddle worker process.

The paperless-ngx 3.1 and later images run Python 3.14, which PaddlePaddle
does not support, so the plugin runs each paddle and paddleocr call in this
script under a second interpreter. That interpreter has ``paddleocr`` and
not ``paperless_paddleocr``, so the script imports nothing from the package
and nothing beyond the standard library at module level.

Protocol: the parent writes one JSON object per line to stdin, and the
worker answers each line with one JSON object per line on stdout, in order.

* Request: ``{"id": int, "op": str, ...}`` with the fields of the op.
* Success: ``{"id": int, "ok": true, "result": {...}}``.
* Failure: ``{"id": int, "ok": false, "error": {"type": str, "message": str,
  "traceback": str}}``, where ``type`` is the exception class name, so the
  parent can tell a paddlex ``DependencyError`` from other failures.
* A line that is not a JSON object with an ``id`` gets the failure shape
  with ``"id": null``, and the worker goes on with the next line.
* At stdin EOF the worker exits with status 0.

Ops:

* ``classic`` with ``image`` and ``kwargs`` (the ``PaddleOCR`` constructor
  arguments): the first page result as ``rec_polys``, ``rec_scores``,
  ``rec_texts``, ``text_word``, ``text_word_region`` (each a list, empty
  when the result lacks it) and ``preprocessed_shape`` (``[h, w]`` of the
  image the pipeline detected on, or null).
* ``orientation`` with ``image``: the top ``label`` and its ``score``.
* ``ping``: ``cuda_devices``, ``paddle`` and ``paddleocr`` (versions, null
  when the import fails), ``paddle_error`` and ``paddleocr_error`` (null
  when the package imports, otherwise the exception class name and message
  of its first failed import in this worker), ``paddleocr_vl`` (whether
  ``PaddleOCRVL`` imports) and ``python``. Each ping retries the import of
  a package that failed before, so a package installed while the worker
  runs is found.
* ``vl`` with ``image``, ``kwargs`` (the ``PaddleOCRVL`` constructor
  arguments without ``pipeline_version``) and ``pipeline_version``:
  ``blocks`` (``label``, ``content`` and ``bbox`` of each layout block),
  ``spotting`` (``rec_texts`` and ``rec_polys``, or null) and
  ``spotting_capable``.

Results are plain JSON: numpy arrays and scalars become lists and Python
scalars at any depth.

A paddle library that prints to stdout, from Python or from native code,
would put a stray line into the protocol. Before paddle is imported, the
worker keeps a private duplicate of the real stdout for its protocol lines
and points file descriptor 1 and ``sys.stdout`` at stderr.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import json
import os
import platform
import sys
import traceback
from collections.abc import Callable
from typing import Any, TextIO

_ORIENTATION_MODEL_NAME = "PP-LCNet_x1_0_doc_ori"

#: Classic result fields passed to the parent, each made a plain list.
_CLASSIC_FIELDS = ("rec_polys", "rec_scores", "rec_texts", "text_word", "text_word_region")

# Building a pipeline loads model weights from disk, so each one lives as
# long as the worker. A cache key is the JSON text of the build arguments.
_CLASSIC_ENGINES: dict[str, Any] = {}
_ORIENTATION_MODELS: dict[str, Any] = {}
_VL_PIPELINES: dict[str, Any] = {}

# The first failure is kept because a package that fails half-way through
# its import leaves its loaded submodules in sys.modules, and the next import
# fails on them with an error that hides the cause, such as paddle's
# "partially initialized module" after a PermissionError on ~/.cache/paddle.
_IMPORT_ERRORS: dict[str, str] = {}


def _cached(cache: dict[str, Any], key: str, build: Callable[[], Any]) -> Any:
    model = cache.get(key)
    if model is None:
        model = cache[key] = build()
    return model


def _plain(value: Any) -> Any:
    """``value`` with each nested numpy array or numpy scalar made a JSON type.

    The pipelines nest arrays inside Python lists and put numpy scalars into
    score lists, so converting the top level alone leaves values that
    ``json.dumps`` rejects.
    """
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        # numpy arrays and numpy scalars both have tolist(), which spares the
        # numpy import. An object array keeps its elements as they are, so
        # the output is walked again.
        return _plain(value.tolist())
    return value


def _get(container: Any, key: str) -> Any:
    """``container.get(key)``, or None for a container without ``get``, such as None."""
    getter = getattr(container, "get", None)
    return None if getter is None else getter(key)


def _list_field(result: Any, key: str) -> Any:
    # A field can be a numpy array, whose truth value raises for more than
    # one element, so a missing field is told apart by identity. A result
    # without get() raises, so an unexpected type becomes an error response
    # instead of an empty page.
    value = result.get(key)
    return [] if value is None else _plain(value)


def _first(results: Any) -> Any:
    return results[0] if len(results) > 0 else None


def _version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _import(name: str) -> Any:
    """The module ``name``, or None when its import raises.

    Any exception counts, because an installed package can fail its import
    with an error other than ``ImportError``, such as a ``PermissionError``.
    The traceback goes to stderr, which the parent logs.
    """
    try:
        return importlib.import_module(name)
    except Exception as exc:
        traceback.print_exc()
        _IMPORT_ERRORS.setdefault(name, f"{type(exc).__name__}: {exc}")
        return None


def _ping(request: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cuda_devices": 0,
        "paddle": None,
        "paddle_error": None,
        "paddleocr": None,
        "paddleocr_error": None,
        "paddleocr_vl": False,
        "python": platform.python_version(),
    }
    # An import failure is a result, so the parent can phrase the install
    # hint and name the cause.
    paddle = _import("paddle")
    if paddle is None:
        result["paddle_error"] = _IMPORT_ERRORS["paddle"]
    else:
        result["paddle"] = _version(paddle)
        # The plugin pings for each engine and only classic-gpu needs a
        # device, so a failed query counts as no device.
        try:
            result["cuda_devices"] = int(paddle.device.cuda.device_count())
        except Exception:
            traceback.print_exc()
    paddleocr = _import("paddleocr")
    if paddleocr is None:
        result["paddleocr_error"] = _IMPORT_ERRORS["paddleocr"]
    else:
        result["paddleocr"] = _version(paddleocr)
        try:
            from paddleocr import PaddleOCRVL  # noqa: F401
        except ImportError:
            pass
        else:
            result["paddleocr_vl"] = True
    return result


def _classic(request: dict[str, Any]) -> dict[str, Any]:
    kwargs = request["kwargs"]

    def build() -> Any:
        from paddleocr import PaddleOCR

        return PaddleOCR(**kwargs)

    engine = _cached(_CLASSIC_ENGINES, json.dumps(kwargs, sort_keys=True), build)
    # Sub-word tokens and their boxes come only with return_word_box. The
    # parent joins them back into words with exact boxes.
    results = engine.predict(request["image"], return_word_box=True)
    if len(results) == 0:
        return {**{key: [] for key in _CLASSIC_FIELDS}, "preprocessed_shape": None}
    page = results[0]

    # Word boxes are in the space of the preprocessed image, which the parent
    # scales back to the page.
    image = _get(page.get("doc_preprocessor_res"), "output_img")
    shape = getattr(image, "shape", None)
    preprocessed_shape = (
        [int(shape[0]), int(shape[1])] if shape is not None and len(shape) >= 2 else None
    )
    fields = {key: _list_field(page, key) for key in _CLASSIC_FIELDS}
    return {**fields, "preprocessed_shape": preprocessed_shape}


def _spotting_capable(pipeline: Any) -> bool:
    """Word spotting shipped with the v1.5 pipeline, and v1 predates it.

    A newer version counts as capable, because the parent falls back to the
    blocks when a response carries no spotting result.
    """
    version = getattr(pipeline, "pipeline_version", None)
    return version is not None and version != "v1"


def _vl_block(block: Any) -> dict[str, Any]:
    # Current paddlex returns block objects, and older paddleocr releases
    # returned dicts with block_-prefixed keys. The type decides, because a
    # falsy attribute (content="" is the object default) must not fall
    # through to a dict lookup that an object does not have.
    if isinstance(block, dict):
        label = block.get("block_label", "")
        content = block.get("block_content", "")
        bbox = block.get("block_bbox")
    else:
        label = getattr(block, "label", "")
        content = getattr(block, "content", "")
        bbox = getattr(block, "bbox", None)
    return {"label": _plain(label), "content": _plain(content), "bbox": _plain(bbox)}


def _vl(request: dict[str, Any]) -> dict[str, Any]:
    kwargs = request["kwargs"]
    version = request["pipeline_version"]

    def build() -> Any:
        import paddle
        from paddleocr import PaddleOCRVL

        # PaddleOCRVL.__init__ probes bfloat16 support on the active device
        # before it applies device=, and with no device set the probe raises
        # TypeError on Place(undefined:0). Recognition runs on the remote
        # server, so pinning CPU first costs nothing.
        paddle.set_device("cpu")
        pipeline_kwargs = dict(kwargs)
        # Older paddleocr releases lack the parameter and run their only
        # pipeline, which has no word spotting.
        if "pipeline_version" in inspect.signature(PaddleOCRVL.__init__).parameters:
            pipeline_kwargs["pipeline_version"] = version
        return PaddleOCRVL(**pipeline_kwargs)

    key = json.dumps({"kwargs": kwargs, "pipeline_version": version}, sort_keys=True)
    pipeline = _cached(_VL_PIPELINES, key, build)
    spotting_capable = _spotting_capable(pipeline)
    page = _first(
        pipeline.predict(
            request["image"],
            use_layout_detection=False,
            use_queues=False,
            prompt_label="spotting" if spotting_capable else "ocr",
        ),
    )

    spotting = _get(page, "spotting_res")
    if spotting is not None:
        spotting = {
            "rec_texts": _list_field(spotting, "rec_texts"),
            "rec_polys": _list_field(spotting, "rec_polys"),
        }
    blocks = _get(page, "parsing_res_list")
    return {
        "spotting_capable": spotting_capable,
        "spotting": spotting,
        "blocks": [] if blocks is None else [_vl_block(block) for block in blocks],
    }


def _orientation(request: dict[str, Any]) -> dict[str, Any]:
    def build() -> Any:
        from paddleocr import DocImgOrientationClassification

        return DocImgOrientationClassification(model_name=_ORIENTATION_MODEL_NAME)

    model = _cached(_ORIENTATION_MODELS, _ORIENTATION_MODEL_NAME, build)
    data = model.predict(request["image"])[0]
    labels = _list_field(data, "label_names")
    scores = _list_field(data, "scores")
    return {
        "label": str(labels[0]) if labels else "0",
        "score": float(scores[0]) if scores else 0.0,
    }


_OPS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "classic": _classic,
    "orientation": _orientation,
    "ping": _ping,
    "vl": _vl,
}


def _respond(line: str) -> str:
    """The response line for one request line."""
    request_id = None
    try:
        request = json.loads(line)
        if not isinstance(request, dict) or "id" not in request:
            raise ValueError("a request is a JSON object with an id")
        request_id = request["id"]
        name = request.get("op")
        op = _OPS.get(name) if isinstance(name, str) else None
        if op is None:
            raise ValueError(f"unknown op {name!r}")
        # Serialized inside the try, so a value json cannot encode becomes
        # an error response instead of ending the worker.
        return json.dumps({"id": request_id, "ok": True, "result": op(request)})
    except Exception as exc:
        # An exception's __str__ can raise too, and unguarded that would end
        # the worker in the middle of a request.
        try:
            message = str(exc)
        except Exception:
            message = "<str() of the exception failed>"
        error = {
            "type": type(exc).__name__,
            "message": message,
            "traceback": traceback.format_exc(),
        }
        return json.dumps({"id": request_id, "ok": False, "error": error})


def _claim_stdout() -> TextIO:
    """A private stream on the real stdout, with fd 1 and ``sys.stdout`` on stderr.

    ``os.dup`` sets close-on-exec on the copy, so a program that paddle
    executes does not inherit the protocol pipe and cannot keep it open
    after the worker has died.
    """
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    # The old sys.stdout now writes to stderr too, but through a block
    # buffer, which holds Python prints back from the log.
    sys.stdout = sys.stderr
    return protocol


def _disable_paddle_signal_handler() -> None:
    """Drop the C++ signal handler that paddle installs on import.

    For a plain SIGTERM the handler prints a fake ``FatalError: Termination
    signal`` with an empty stack, and the parent logs this process's
    stderr. A paddle that does not import is left to ``ping`` to report.
    """
    paddle = _import("paddle")
    if paddle is not None and hasattr(paddle, "disable_signal_handler"):
        with contextlib.suppress(Exception):
            paddle.disable_signal_handler()


def main() -> int:
    protocol = _claim_stdout()
    _disable_paddle_signal_handler()
    for line in sys.stdin:
        protocol.write(_respond(line) + "\n")
        protocol.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
