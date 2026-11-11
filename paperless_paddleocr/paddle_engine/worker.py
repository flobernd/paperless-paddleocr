"""Client for the paddle worker process.

The engine adapters reach PaddlePaddle through :func:`call`, which runs each
request in the script :mod:`paperless_paddleocr.paddle_worker` under a second
interpreter. The script's docstring gives the reason and the line protocol.
The first call spawns the worker, and the worker serves each later call of
this process, so the models it loads stay loaded.

``call`` holds one lock from writing a request to reading its response,
because the worker answers one line at a time and ocrmypdf calls the engine
from a thread pool.

A worker that fails in one of these ways is killed and reaped, ``call``
raises :class:`PaddleWorkerError` with the last lines of its stderr, and the
next call spawns a fresh worker:

* it exits, or closes its stdout, before it answers,
* it sends a line that is not the response to the pending request, such as a
  line that is not JSON or a response with another id or a null id,
* it sends a response off protocol: an ``ok`` that is not a bool, an ``ok``
  response without a ``result`` dict, or an error response without an
  ``error`` dict that holds a string ``message``,
* it sends nothing within ``PAPERLESS_PADDLEOCR_WORKER_TIMEOUT`` seconds
  (600 by default).

The worker answers in order, so a line that is not the pending response means
the two sides disagree about which line answers which request. No later line
can be trusted, so the call fails at once instead of waiting for the deadline.
``call`` also raises ``PaddleWorkerError`` when the worker cannot start. A
timeout below 1 is invalid, and the call logs a warning and uses the default.

A worker found dead at the start of a call holds no request, so the call
replaces it before it writes the request: it logs a warning with the exit
status and the last stderr lines, spawns a fresh worker and sends the request
there. A worker that dies with a request in flight fails that call as above,
and the request is not retried.

``ping`` also stops a worker whose answer reports paddle or paddleocr
missing, so the next ping runs in a fresh interpreter.

A request can carry a secret, such as the VL API key among the kwargs, so the
log and the error messages show at most the op and the id of a request, never
its content.
"""

from __future__ import annotations

import collections
import contextlib
import itertools
import json
import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import IO, Any

import paperless_paddleocr
from paperless_paddleocr import DEFAULT_WORKER_PYTHON
from paperless_paddleocr.env import env_int

log = logging.getLogger("paperless.paddleocr.worker")

WORKER_SCRIPT = Path(paperless_paddleocr.__file__).with_name("paddle_worker.py")

_TIMEOUT_VARIABLE = "PAPERLESS_PADDLEOCR_WORKER_TIMEOUT"
_DEFAULT_TIMEOUT = 600
_STDERR_LINES = 50
# A child process of the worker that inherited its stderr can hold the pipe
# open after the worker died, so the wait for the last stderr lines is bounded.
_STDERR_DRAIN_SECONDS = 5


class PaddleWorkerError(RuntimeError):
    """An error response of the worker, or a worker that failed.

    ``error_chain`` holds the class names of the exception the worker
    reported and of each exception before it, as the worker reported them,
    and is empty when the worker itself failed or its error response carried
    no chain.
    """

    def __init__(self, message: str, *, error_chain: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.error_chain = error_chain


def _interpreter() -> tuple[str, bool]:
    """The interpreter that runs the worker, and whether it is the fallback.

    ``PAPERLESS_PADDLEOCR_PYTHON`` when set, else the default worker venv when
    it exists, else this interpreter as the fallback, which serves an install
    that put paddle beside the plugin.
    """
    configured = os.environ.get("PAPERLESS_PADDLEOCR_PYTHON")
    if configured:
        return configured, False
    if Path(DEFAULT_WORKER_PYTHON).exists():
        return DEFAULT_WORKER_PYTHON, False
    return sys.executable, True


def worker_interpreter() -> str:
    """The interpreter that runs the worker."""
    return _interpreter()[0]


def uses_fallback_interpreter() -> bool:
    """Whether :func:`worker_interpreter` falls back to this interpreter."""
    return _interpreter()[1]


def worker_environment() -> dict[str, str]:
    # The upstream parse method sets OMP_THREAD_LIMIT=1 for Tesseract, which
    # would hold PaddlePaddle to one thread.
    env = {key: value for key, value in os.environ.items() if key != "OMP_THREAD_LIMIT"}
    # Unbuffered, what the worker prints from Python reaches the log as it is
    # written, and a killed worker loses none of it in a buffer.
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _follows_protocol(response: dict[str, Any]) -> bool:
    """Whether ``response`` carries a dict result or an error with a message."""
    ok = response.get("ok")
    if ok is True:
        return isinstance(response.get("result"), dict)
    if ok is False:
        error = response.get("error")
        return isinstance(error, dict) and isinstance(error.get("message"), str)
    return False


class _Worker:
    """One worker process, with a thread on each of its output pipes.

    A third thread starts the process and ends at :meth:`stop`, which kills
    and reaps the process first. When the constructor fails after the start,
    that thread kills and reaps the process itself.
    """

    def __init__(self) -> None:
        interpreter = worker_interpreter()
        self._stopped = threading.Event()
        # A reader thread closes its own pipe when the stream ends, so the
        # parent thread closes only the pipes not listed here.
        self._reader_pipes: list[IO[str] | None] = []
        started: queue.SimpleQueue[subprocess.Popen[str] | BaseException] = queue.SimpleQueue()
        # On Linux the worker dies when the thread that started it ends.
        # ocrmypdf calls from page threads that end with their document, so
        # the worker starts from a thread of its own, which puts the process
        # or the exception of the start on the queue.
        self._parent_thread = threading.Thread(
            target=self._parent,
            args=(interpreter, started),
            name="paddle-worker-parent",
            daemon=True,
        )
        try:
            # Inside the try, so that an exception that interrupts the start
            # after the thread runs still sets the stop event.
            self._parent_thread.start()
            outcome = started.get()
            if isinstance(outcome, OSError):
                raise PaddleWorkerError(
                    f"Cannot start the paddle worker {interpreter}: {outcome}"
                ) from outcome
            if isinstance(outcome, BaseException):
                raise outcome
            self.process = outcome
            log.debug("Started the paddle worker %s (pid %d)", interpreter, self.process.pid)
            self._lines: queue.Queue[str | None] = queue.Queue()
            self._tail: collections.deque[str] = collections.deque(maxlen=_STDERR_LINES)
            self._tail_lock = threading.Lock()
            threading.Thread(
                target=self._read_stdout,
                args=(self.process.stdout,),
                name="paddle-worker-stdout",
                daemon=True,
            ).start()
            self._reader_pipes.append(self.process.stdout)
            # The worker sends what paddle prints to stdout to stderr as well, so
            # a stderr pipe that nobody reads during a request fills up and stalls
            # the worker.
            self._stderr_thread = threading.Thread(
                target=self._forward_stderr,
                args=(self.process.stderr,),
                name="paddle-worker-stderr",
                daemon=True,
            )
            self._stderr_thread.start()
            self._reader_pipes.append(self.process.stderr)
        except BaseException:
            # The parent thread then kills the process, even one whose start
            # returns only after this exception.
            self._stopped.set()
            raise

    def _parent(
        self,
        interpreter: str,
        started: queue.SimpleQueue[subprocess.Popen[str] | BaseException],
    ) -> None:
        try:
            # On Linux the worker asks the kernel to kill it when this thread
            # ends. stop() kills the worker before this thread ends, the thread
            # does so itself after a constructor that failed, and the signal
            # covers the death of this process, so a worker in the middle of a
            # request ends with this process too. Elsewhere the worker exits at
            # the EOF of stdin, which it reads between requests only, so a busy
            # worker ends once its request ends. Either way the client needs no
            # signal handler or exit hook to end it.
            process = subprocess.Popen(
                [interpreter, "-P", str(WORKER_SCRIPT), "--parent-pid", str(os.getpid())],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # A stderr thread that died on an undecodable byte would stop
                # draining the pipe and stall the worker.
                encoding="utf-8",
                errors="replace",
                env=worker_environment(),
            )
        except BaseException as exc:
            # The constructor waits on the queue for the outcome, so any
            # exception goes there.
            started.put(exc)
            return
        started.put(process)
        self._stopped.wait()
        # Only a constructor that failed leaves the process running here,
        # because stop() kills and reaps it first.
        process.kill()
        process.wait()
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe is not None and pipe not in self._reader_pipes:
                # The process is gone, so a request left in the write buffer
                # of stdin has nowhere to go.
                with contextlib.suppress(OSError):
                    pipe.close()

    def _read_stdout(self, stream: IO[str]) -> None:
        try:
            for line in stream:
                self._lines.put(line)
        finally:
            # None marks the end of the stream, so a waiting call fails at once
            # instead of at the deadline.
            self._lines.put(None)
            stream.close()

    def _forward_stderr(self, stream: IO[str]) -> None:
        with stream:
            for line in stream:
                text = line.rstrip("\n")
                with self._tail_lock:
                    self._tail.append(text)
                log.debug("paddle worker: %s", text)

    def exchange(self, request_id: int, op: Any, line: str, timeout: int) -> dict[str, Any]:
        """Write ``line`` and return its response, or stop a failed worker and raise."""
        stdin = self.process.stdin
        assert stdin is not None
        try:
            stdin.write(line + "\n")
            stdin.flush()
            answer = self._lines.get(timeout=timeout)
        except OSError:
            # A write fails once the worker has exited.
            answer = None
        except queue.Empty:
            raise self._fail(
                f"The paddle worker did not answer the {op} request within {timeout} s and was "
                f"killed. {_TIMEOUT_VARIABLE} sets the limit"
            ) from None
        if answer is None:
            raise self._fail(f"The paddle worker exited before it answered the {op} request")
        try:
            response = json.loads(answer)
        except ValueError:
            response = None
        if not isinstance(response, dict) or response.get("id") != request_id:
            raise self._fail(
                f"The paddle worker answered the {op} request with a line that is not its "
                f"response: {answer[:200]!r}"
            )
        if not _follows_protocol(response):
            raise self._fail(
                f"The paddle worker answered the {op} request with a response that does not "
                f"follow the protocol: {answer[:200]!r}"
            )
        return response

    def _fail(self, reason: str) -> PaddleWorkerError:
        tail = self.stop()
        return PaddleWorkerError(
            f"{reason}.\nExit status: {self.process.returncode}\nLast lines of its stderr:\n{tail}"
        )

    def stop(self) -> str:
        """Kill and reap the process, and return the last lines of its stderr.

        The thread that started the process ends too.
        """
        self.process.kill()
        self.process.wait()
        self._stopped.set()
        # In a child forked after the start the thread does not exist, and
        # join returns at once.
        self._parent_thread.join()
        self._stderr_thread.join(_STDERR_DRAIN_SECONDS)
        with self._tail_lock:
            return "\n".join(self._tail)


_lock = threading.Lock()
_ids = itertools.count(1)
_worker: _Worker | None = None


def _timeout() -> int:
    timeout = env_int(_TIMEOUT_VARIABLE, _DEFAULT_TIMEOUT)
    if timeout < 1:
        log.warning(
            "Invalid value for %s=%d, which must be 1 or more. Using the default %d.",
            _TIMEOUT_VARIABLE,
            timeout,
            _DEFAULT_TIMEOUT,
        )
        return _DEFAULT_TIMEOUT
    return timeout


def call(request: dict[str, Any]) -> dict[str, Any]:
    """Run ``request`` in the worker and return the result of its response.

    Raises ``PaddleWorkerError`` for an error response, with the chain the
    worker reported, and for a worker that failed.
    """
    global _worker
    op = request.get("op")
    timeout = _timeout()
    with _lock:
        request_id = next(_ids)
        line = json.dumps({**request, "id": request_id})
        if _worker is not None and _worker.process.poll() is not None:
            # No request is in flight, so this one can go to a fresh worker
            # without running twice.
            dead = _worker
            _worker = None
            tail = dead.stop()
            log.warning(
                "The paddle worker exited with status %s between requests and is replaced. "
                "Last lines of its stderr:\n%s",
                dead.process.returncode,
                tail,
            )
        if _worker is None:
            _worker = _Worker()
        current = _worker
        log.debug("Paddle worker request %d: %s", request_id, op)
        try:
            response = current.exchange(request_id, op, line, timeout)
        except BaseException:
            # The worker failed, or it holds a request whose late answer would
            # reach the next call, so it goes either way.
            _worker = None
            current.stop()
            raise
    if response["ok"]:
        return response["result"]
    error = response["error"]
    log.debug("Paddle worker request %d failed:\n%s", request_id, error.get("traceback", ""))
    chain = error.get("chain")
    if not isinstance(chain, list):
        chain = []
    raise PaddleWorkerError(error["message"], error_chain=tuple(chain))


_ping_result: dict[str, Any] | None = None


def ping() -> dict[str, Any]:
    """The worker's answer to a ``ping`` request, asked once per process.

    An answer that reports paddle or paddleocr missing is not kept, and
    neither is a failed call. ``ping`` stops the worker after such an
    answer, because a package that failed half-way through its import fails
    again in the same interpreter with an error that hides the cause.
    Raises ``PaddleWorkerError`` as :func:`call` does.
    """
    global _ping_result, _worker
    if _ping_result is not None:
        return _ping_result
    result = call({"op": "ping"})
    if result["paddle"] is not None and result["paddleocr"] is not None:
        _ping_result = result
    else:
        with _lock:
            stale = _worker
            _worker = None
            if stale is not None:
                stale.stop()
    return result


def cached_ping() -> dict[str, Any] | None:
    """The answer :func:`ping` kept, or None before one. It never starts the worker."""
    return _ping_result
