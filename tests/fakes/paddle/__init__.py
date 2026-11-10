"""Stand-in for paddlepaddle in the worker tests.

Importable only when ``tests/fakes`` is on ``PYTHONPATH``, which the tests
set for the worker process alone.
"""

import os
from types import SimpleNamespace

# Fails the paddle import. FAKE_PADDLE_FAIL_IMPORT fails the paddleocr
# import.
if os.environ.get("FAKE_PADDLEPADDLE_FAIL_IMPORT") == "1":
    raise ImportError("No module named 'paddle'")

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


device = SimpleNamespace(
    cuda=SimpleNamespace(device_count=lambda: int(os.environ.get("FAKE_CUDA_DEVICES", "0"))),
)
