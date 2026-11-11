"""Stand-in for paddlepaddle in the worker tests.

Importable only when ``tests/fakes`` is on ``PYTHONPATH``, which the tests
set for the worker process alone.
"""

import os
import sys
import time
from types import ModuleType, SimpleNamespace

# Writes the worker's pid to the named file and sleeps, as a slow import of
# the real package, so a test knows the worker is blocked in the import. The
# rename means a reader never finds the file empty.
_hang_pid_file = os.environ.get("FAKE_PADDLEPADDLE_HANG_IMPORT")
if _hang_pid_file:
    with open(_hang_pid_file + ".tmp", "w", encoding="utf-8") as _stream:
        _stream.write(str(os.getpid()))
    os.replace(_hang_pid_file + ".tmp", _hang_pid_file)
    time.sleep(30)

# Fails the paddle import. FAKE_PADDLE_FAIL_IMPORT fails the paddleocr
# import.
if os.environ.get("FAKE_PADDLEPADDLE_FAIL_IMPORT") == "1":
    raise ImportError("No module named 'paddle'")

# The real package fails half-way through its import when it cannot create
# ~/.cache/paddle. The submodules it loaded before stay in sys.modules, and a
# second import fails on them with another error.
if os.environ.get("FAKE_PADDLEPADDLE_FAIL_IMPORT") == "permission":
    if "paddle.base" in sys.modules:
        raise AttributeError(
            "partially initialized module 'paddle' has no attribute 'tensor' "
            "(most likely due to a circular import)",
        )
    sys.modules["paddle.base"] = ModuleType("paddle.base")
    raise PermissionError(13, "Permission denied", "/home/paperless/.cache/paddle")

# One write from Python and one from native code at import. The native write
# fails the first response of a worker that imports paddle before it claims
# stdout. The print does so only under PYTHONUNBUFFERED, which the worker
# client sets: otherwise it waits in the block buffer of sys.stdout until
# fd 1 already points at stderr.
print("noise from the paddle import")
os.write(1, b"native noise from the paddle import\n")

__version__ = "3.2.1"

_device = None


def set_device(device):
    global _device
    _device = device


def disable_signal_handler():
    pass


def _device_count():
    if "FAKE_CUDA_QUERY_ERROR" in os.environ:
        raise RuntimeError(os.environ["FAKE_CUDA_QUERY_ERROR"])
    return int(os.environ.get("FAKE_CUDA_DEVICES", "0"))


device = SimpleNamespace(cuda=SimpleNamespace(device_count=_device_count))
