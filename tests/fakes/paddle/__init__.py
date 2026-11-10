"""Stand-in for paddlepaddle in the worker tests.

Importable only when ``tests/fakes`` is on ``PYTHONPATH``, which the tests
set for the worker process alone.
"""

import os
from types import SimpleNamespace

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
