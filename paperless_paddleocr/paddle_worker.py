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
* Failure: ``{"id": int, "ok": false, "error": {"chain": [str],
  "message": str, "traceback": str}}``. ``chain`` holds the class names of
  the exception and of each exception before it, the one it was raised
  from (``__cause__``) or else the one it was raised while handling
  (``__context__``, unless suppressed), at most 16. paddlex raises a
  ``RuntimeError`` from its ``DependencyError`` when it creates a
  pipeline, so the parent looks for that class in ``chain``.
* A line that is not a JSON object with an ``id`` gets the failure shape
  with ``"id": null``, and the worker goes on with the next line.
* At stdin EOF the worker exits with status 0.

On Linux the worker asks the kernel to kill it with SIGKILL when the thread
that started it ends, and with ``--parent-pid PID`` it exits at once with
status 1 when its parent is not ``PID``. A worker started without the
option, as by hand, skips that check.

Ops:

* ``classic`` with ``image`` and ``kwargs`` (the ``PaddleOCR`` constructor
  arguments): the first page result as ``rec_polys``, ``rec_scores`` and
  ``rec_texts`` (lists of one length), ``text_word`` and
  ``text_word_region`` (lists, empty when the result lacks them) and
  ``preprocessed_shape`` (``[h, w]`` of the image the pipeline detected on,
  or null).
* ``orientation`` with ``image``: the top ``label`` and its ``score``.
* ``ping``: ``cuda_devices``, ``cuda_error`` (null when the query succeeds
  or paddle does not import, otherwise the exception class name and
  message of the failed query), ``paddle`` and ``paddleocr`` (versions,
  null when the import fails), ``paddle_error`` and ``paddleocr_error``
  (null when the package imports, otherwise the exception class name and
  message of its first failed import in this worker) and ``paddleocr_vl``
  (whether ``PaddleOCRVL`` imports and its constructor takes
  ``pipeline_version``). Each ping retries the import of a package that
  failed before, so a package installed while the worker runs is found.
* ``vl`` with ``image``, ``kwargs`` (the ``PaddleOCRVL`` constructor
  arguments without ``pipeline_version``) and ``pipeline_version``:
  ``blocks`` (``label``, ``content`` and ``bbox`` of each layout block),
  ``spotting`` (``rec_texts`` and ``rec_polys``, lists of one length, each
  empty when the spotting result lacks it, or null when the page result has
  no spotting result) and ``spotting_capable``.

A page without text still gives one page result with the fields the parent
needs, so a result of another shape is an error response that names what is
missing and the paddleocr version: a prediction without a page result, a
classic page result without ``rec_polys``, ``rec_scores`` or ``rec_texts``
or with lists of different lengths, an orientation result without
``label_names`` or ``scores``, a VL page result without
``parsing_res_list``, a spotting result whose two lists differ in length,
or a layout block of an unknown shape on a page whose spotting result is
missing or lacks ``rec_texts`` or ``rec_polys``.

Results are plain JSON: numpy arrays and scalars become lists and Python
scalars at any depth.

A paddle library that prints to stdout, from Python or from native code,
would put a stray line into the protocol. Before paddle is imported, the
worker keeps a private duplicate of the real stdout for its protocol lines
and points file descriptor 1 and ``sys.stdout`` at stderr.

A crash in native code leaves Python's "Fatal Python error" block with the
stack of each thread on stderr, which the parent logs and puts into its
error.
"""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import functools
import importlib
import inspect
import json
import os
import signal
import sys
import traceback
from collections.abc import Callable
from typing import Any, TextIO

_ORIENTATION_MODEL_NAME = "PP-LCNet_x1_0_doc_ori"

#: The option of prctl(2) from <linux/prctl.h>.
_PR_SET_PDEATHSIG = 1

#: Classic result fields with one item per text region, which the parent needs.
_CLASSIC_REGION_FIELDS = ("rec_polys", "rec_scores", "rec_texts")
#: Classic result fields of the sub-word tokens, which the parent can do without.
_CLASSIC_WORD_FIELDS = ("text_word", "text_word_region")
#: Members of the layout block objects of paddlex.
_VL_BLOCK_MEMBERS = ("bbox", "content", "label")

# The first failure is kept because a package that fails half-way through
# its import leaves its loaded submodules in sys.modules, and the next import
# fails on them with an error that hides the cause, such as paddle's
# "partially initialized module" after a PermissionError on ~/.cache/paddle.
_IMPORT_ERRORS: dict[str, str] = {}


# Building a pipeline loads model weights from disk, so each one lives as
# long as the worker. A cache key is the JSON text of the build arguments.
@functools.cache
def _classic_engine(kwargs_json: str) -> Any:
    from paddleocr import PaddleOCR

    return PaddleOCR(**json.loads(kwargs_json))


@functools.cache
def _orientation_model() -> Any:
    from paddleocr import DocImgOrientationClassification

    return DocImgOrientationClassification(model_name=_ORIENTATION_MODEL_NAME)


@functools.cache
def _vl_pipeline(config_json: str) -> Any:
    import paddle
    from paddleocr import PaddleOCRVL

    # PaddleOCRVL.__init__ probes bfloat16 support on the active device
    # before it applies device=, and with no device set the probe raises
    # TypeError on Place(undefined:0). Recognition runs on the remote
    # server, so pinning CPU first costs nothing.
    paddle.set_device("cpu")
    config = json.loads(config_json)
    pipeline_kwargs = dict(config["kwargs"])
    pipeline_kwargs["pipeline_version"] = config["pipeline_version"]
    return PaddleOCRVL(**pipeline_kwargs)


def _json_default(value: Any) -> Any:
    """The JSON form of a numpy array or numpy scalar, for ``json.dumps``.

    The pipelines nest arrays inside Python lists and put numpy scalars into
    score lists, and ``json.dumps`` calls this for each of them at any depth.
    """
    # numpy arrays and numpy scalars both have tolist(), which spares the
    # numpy import.
    if hasattr(value, "tolist"):
        return value.tolist()
    # json expects a TypeError for a value it cannot encode, and this is the
    # text it gives.
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _get(container: Any, key: str) -> Any:
    """``container.get(key)``, or None for a container without ``get``, such as None."""
    getter = getattr(container, "get", None)
    return None if getter is None else getter(key)


def _unexpected_result(subject: str, problem: str) -> RuntimeError:
    # The message is all the operator sees of a result this script cannot read.
    import paddleocr

    return RuntimeError(
        f"The {subject} of paddleocr {_version(paddleocr)} {problem}. This paddleocr release "
        "may be newer than the plugin supports."
    )


def _page_result(results: Any, subject: str) -> Any:
    if len(results) == 0:
        raise _unexpected_result(subject, "returned no page result")
    return results[0]


def _required_field(result: Any, key: str, subject: str) -> Any:
    # A field can be a numpy array, whose truth value raises for more than
    # one element, so a missing field is told apart by identity.
    value = _get(result, key)
    if value is None:
        raise _unexpected_result(subject, f"has no {key}")
    return value


def _list_field(result: Any, key: str) -> Any:
    # For a field the parent can do without, so a missing one counts as
    # empty, told apart by identity as in _required_field. A result without
    # get() raises, so an unexpected type becomes an error response instead
    # of an empty page.
    value = result.get(key)
    return [] if value is None else value


def _version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _error_text(exc: BaseException) -> str:
    # An exception's __str__ can raise too, and unguarded that would end the
    # worker at its start-up or in the middle of a request.
    try:
        return str(exc)
    except Exception:
        return "<str() of the exception failed>"


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
        _IMPORT_ERRORS.setdefault(name, f"{type(exc).__name__}: {_error_text(exc)}")
        return None


def _ping(request: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cuda_devices": 0,
        "cuda_error": None,
        "paddle": None,
        "paddle_error": None,
        "paddleocr": None,
        "paddleocr_error": None,
        "paddleocr_vl": False,
    }
    # An import failure is a result, so the parent can phrase the install
    # hint and name the cause.
    paddle = _import("paddle")
    if paddle is None:
        result["paddle_error"] = _IMPORT_ERRORS["paddle"]
    else:
        result["paddle"] = _version(paddle)
        # The plugin pings for each engine and only classic-gpu needs a
        # device, so a failed query is a result and not an error response.
        try:
            result["cuda_devices"] = int(paddle.device.cuda.device_count())
        except Exception as exc:
            traceback.print_exc()
            result["cuda_error"] = f"{type(exc).__name__}: {_error_text(exc)}"
    paddleocr = _import("paddleocr")
    if paddleocr is None:
        result["paddleocr_error"] = _IMPORT_ERRORS["paddleocr"]
    else:
        result["paddleocr"] = _version(paddleocr)
        try:
            from paddleocr import PaddleOCRVL
        except ImportError:
            pass
        else:
            result["paddleocr_vl"] = (
                "pipeline_version" in inspect.signature(PaddleOCRVL.__init__).parameters
            )
    return result


def _classic(request: dict[str, Any]) -> dict[str, Any]:
    engine = _classic_engine(json.dumps(request["kwargs"], sort_keys=True))
    # Sub-word tokens and their boxes come only with return_word_box. The
    # parent joins them back into words with exact boxes.
    page = _page_result(engine.predict(request["image"], return_word_box=True), "classic pipeline")
    regions = {
        key: _required_field(page, key, "classic page result") for key in _CLASSIC_REGION_FIELDS
    }
    lengths = [len(value) for value in regions.values()]
    if len(set(lengths)) > 1:
        raise _unexpected_result(
            "classic page result",
            f"has {', '.join(_CLASSIC_REGION_FIELDS)} of different lengths {lengths}",
        )

    # Word boxes are in the space of the preprocessed image, which the parent
    # scales back to the page.
    image = _get(page.get("doc_preprocessor_res"), "output_img")
    shape = getattr(image, "shape", None)
    preprocessed_shape = (
        [int(shape[0]), int(shape[1])] if shape is not None and len(shape) >= 2 else None
    )
    words = {key: _list_field(page, key) for key in _CLASSIC_WORD_FIELDS}
    return {**regions, **words, "preprocessed_shape": preprocessed_shape}


def _spotting_capable(pipeline: Any) -> bool:
    """Word spotting shipped with the v1.5 pipeline, and v1 predates it.

    A newer version counts as capable, because the parent falls back to the
    blocks when a response carries no spotting result.
    """
    return pipeline.pipeline_version != "v1"


def _vl_block(block: Any, *, check_shape: bool) -> dict[str, Any]:
    missing = [name for name in _VL_BLOCK_MEMBERS if not hasattr(block, name)]
    label = getattr(block, "label", "")
    content = getattr(block, "content", "")
    bbox = getattr(block, "bbox", None)
    # The empty defaults would turn a block of an unknown shape into a page
    # without text and without an error. Only an absent field counts as
    # missing, so the empty content of a blank page passes.
    if check_shape and missing:
        raise _unexpected_result(
            "PaddleOCR-VL layout block",
            f"is a {type(block).__name__} without {', '.join(missing)}",
        )
    return {"label": label, "content": content, "bbox": bbox}


def _vl(request: dict[str, Any]) -> dict[str, Any]:
    config = {"kwargs": request["kwargs"], "pipeline_version": request["pipeline_version"]}
    pipeline = _vl_pipeline(json.dumps(config, sort_keys=True))
    spotting_capable = _spotting_capable(pipeline)
    page = _page_result(
        pipeline.predict(
            request["image"],
            use_layout_detection=False,
            use_queues=False,
            prompt_label="spotting" if spotting_capable else "ocr",
        ),
        "PaddleOCR-VL pipeline",
    )

    blocks = _required_field(page, "parsing_res_list", "PaddleOCR-VL page result")
    # The v1 pipeline answers {} and the parent then reads the blocks, so the
    # spotting result and its fields are optional. A spotting result that
    # has both fields holds the page's text, with words or blank, because
    # paddlex builds the content of the spotting block from its texts. The
    # blocks give the text only on a page without one, so only there does a
    # block of an unknown shape fail.
    spotting = _get(page, "spotting_res")
    spotting_complete = all(_get(spotting, key) is not None for key in ("rec_polys", "rec_texts"))
    if spotting is not None:
        spotting = {
            "rec_texts": _list_field(spotting, "rec_texts"),
            "rec_polys": _list_field(spotting, "rec_polys"),
        }
        # A release that renames one of the two fields leaves its list empty.
        if len(spotting["rec_texts"]) != len(spotting["rec_polys"]):
            raise _unexpected_result(
                "PaddleOCR-VL spotting result",
                f"has {len(spotting['rec_texts'])} rec_texts and "
                f"{len(spotting['rec_polys'])} rec_polys",
            )
    return {
        "spotting_capable": spotting_capable,
        "spotting": spotting,
        "blocks": [_vl_block(block, check_shape=not spotting_complete) for block in blocks],
    }


def _orientation(request: dict[str, Any]) -> dict[str, Any]:
    data = _page_result(_orientation_model().predict(request["image"]), "orientation model")
    labels = _required_field(data, "label_names", "orientation result")
    scores = _required_field(data, "scores", "orientation result")
    return {"label": str(labels[0]), "score": float(scores[0])}


_OPS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "classic": _classic,
    "orientation": _orientation,
    "ping": _ping,
    "vl": _vl,
}

#: A real chain is a few links long. The bound ends the walk of a chain whose
#: exceptions cause each other.
_CHAIN_LIMIT = 16


def _chain(exc: BaseException) -> list[str]:
    """Class names of ``exc`` and of the exceptions before it, ``exc`` first."""
    names: list[str] = []
    current: BaseException | None = exc
    while current is not None and len(names) < _CHAIN_LIMIT:
        names.append(type(current).__name__)
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__suppress_context__:
            current = None
        else:
            current = current.__context__
    return names


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
        return json.dumps(
            {"id": request_id, "ok": True, "result": op(request)}, default=_json_default
        )
    except Exception as exc:
        error = {
            "chain": _chain(exc),
            "message": _error_text(exc),
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
    if paddle is not None:
        with contextlib.suppress(Exception):
            paddle.disable_signal_handler()


def _die_with_parent() -> bool:
    """Ask the kernel to send SIGKILL when the thread that started this process ends.

    Returns whether the call succeeded. Only Linux has it.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        # A Python built without libffi has no ctypes, and such a worker runs
        # on without the signal.
        import ctypes

        libc = ctypes.CDLL(None)
        # prctl reads the arguments after the option as unsigned longs.
        return libc.prctl(_PR_SET_PDEATHSIG, ctypes.c_ulong(signal.SIGKILL)) == 0
    except Exception:
        return False


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer the paddle requests of paperless-paddleocr, one JSON line each."
    )
    parser.add_argument(
        "--parent-pid",
        type=int,
        help="pid of the process that starts the worker. On Linux the worker exits at once "
        "when that process is not its parent.",
    )
    return parser.parse_args()


def main() -> int:
    # Enabled before the paddle import below, so a crash inside that import
    # leaves a stack too.
    faulthandler.enable()
    # The signal comes before the slow paddle import below, so a parent that
    # dies during that import still takes the worker with it.
    pdeathsig = _die_with_parent()
    parent_pid = _parse_arguments().parent_pid
    # The kernel sends no signal for a parent that died before the prctl
    # call, so the worker compares its parent with the process that started it.
    actual_parent = os.getppid()
    if pdeathsig and parent_pid is not None and actual_parent != parent_pid:
        message = (
            f"The paddle worker exits because its parent is process {actual_parent}, not "
            f"process {parent_pid}. If PAPERLESS_PADDLEOCR_PYTHON names a script, the script "
            "must start Python with exec.\n"
        )
        # After the death of the parent, stderr can be a pipe that nobody
        # reads. The write bypasses the buffer of sys.stderr, because text
        # left there would fail the flush at exit and turn status 1 into 120.
        with contextlib.suppress(OSError):
            os.write(2, message.encode())
        return 1
    protocol = _claim_stdout()
    _disable_paddle_signal_handler()
    # The paddle import replaces the handler of the first enable() with
    # paddle's own for each signal both cover, and dropping paddle's handler
    # leaves those signals at their default action. enable() alone installs
    # nothing while faulthandler counts as enabled, so disable() comes first.
    faulthandler.disable()
    faulthandler.enable()
    for line in sys.stdin:
        protocol.write(_respond(line) + "\n")
        protocol.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
