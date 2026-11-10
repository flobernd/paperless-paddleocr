"""Stand-in for paddlepaddle in the worker tests.

Importable only when ``tests/fakes`` is on ``PYTHONPATH``, which the tests
set for the worker process alone.
"""

import os
from types import SimpleNamespace

if os.environ.get("FAKE_PADDLEPADDLE_FAIL_IMPORT") == "1":
    raise ImportError("No module named 'paddle'")

# One write from Python and one from native code at import, so each spawned
# worker fails its first response when it imports paddle before it claims
# stdout.
print("noise from the paddle import")
os.write(1, b"native noise from the paddle import\n")

__version__ = "3.2.1"

_device = None


def set_device(device):
    global _device
    _device = device


def disable_signal_handler():
    pass


device = SimpleNamespace(
    cuda=SimpleNamespace(device_count=lambda: int(os.environ.get("FAKE_CUDA_DEVICES", "0"))),
)
