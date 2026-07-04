# OneDNN Workaround Re-Verification Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> This is an investigation plan: Task 3 is decision-gated on the results of Tasks 1-2.

**Goal:** Determine whether `enable_mkldnn=False` (a crash workaround in `classic.py:73-79`) is still needed on current
paddlepaddle, measure what it costs, and gate or remove it accordingly.

**Architecture:** A benchmark script under `scripts/` reproduces the crash condition and measures throughput with OneDNN
on and off. The outcome feeds one of two prepared code changes: keep the workaround with an updated comment (crash still
present) or gate it by paddle version (crash fixed in a known release).

**Tech Stack:** Python 3.12 venv with `paddlepaddle` and `paddleocr` installed (this cannot run in CI; it needs the real
wheels and a x86_64 Linux or Windows host).

## Global Constraints

- Python `>=3.12`; `ruff check .`, `ruff format --check .`, `mypy --ignore-missing-imports paperless_paddleocr` must
  pass for any committed code.
- The benchmark script is a dev tool: it lives in `scripts/`, is excluded from the wheel
  (`pyproject.toml` packages filter already only includes `paperless_paddleocr*`), and
  must not be imported by the package.
- No em-dashes in prose or comments; comments explain WHY only.

---

### Task 1: Benchmark and reproduction script

**Files:**

- Create: `scripts/bench_classic_cpu.py`

- [ ] **Step 1: Write the script**

```python
"""Reproduce the OneDNN/PIR crash and benchmark OneDNN on vs off.

Usage (in a venv with paddlepaddle + paddleocr installed):

    python scripts/bench_classic_cpu.py page1.png [page2.png ...]

Runs the classic pipeline twice over the given page images, once with
enable_mkldnn=True and once with False, and prints per-page timings. A crash
with OneDNN on reproduces the bug the classic.py workaround guards against
("ConvertPirAttribute2RuntimeAttribute not support ...").
"""

from __future__ import annotations

import sys
import time

import paddle
from paddleocr import PaddleOCR


def bench(enable_mkldnn: bool, pages: list[str]) -> None:
    print(f"\n=== enable_mkldnn={enable_mkldnn} (paddle {paddle.__version__}) ===")
    start = time.perf_counter()
    engine = PaddleOCR(
        lang="en",
        device="cpu",
        use_textline_orientation=False,
        use_doc_unwarping=False,
        use_doc_orientation_classify=False,
        enable_mkldnn=enable_mkldnn,
    )
    print(f"init: {time.perf_counter() - start:.2f}s")
    for page in pages:
        start = time.perf_counter()
        try:
            result = engine.predict(page, return_word_box=True)
        except Exception as e:  # noqa: BLE001 - the crash IS the data point
            print(f"{page}: CRASH {type(e).__name__}: {e}")
            continue
        regions = len(result[0].get("rec_texts", [])) if result else 0
        print(f"{page}: {time.perf_counter() - start:.2f}s ({regions} regions)")


def main() -> None:
    pages = sys.argv[1:]
    if not pages:
        sys.exit("usage: bench_classic_cpu.py <page.png> [...]")
    bench(False, pages)
    bench(True, pages)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/bench_classic_cpu.py
git commit -m "Add classic-cpu OneDNN benchmark script"
```

---

### Task 2: Run the investigation

- [ ] **Step 1: Prepare the environment**

```bash
python3.12 -m venv .venv-bench
.venv-bench/bin/pip install paddlepaddle paddleocr   # latest stable wheels
```

Also prepare 3-5 representative page images (300 DPI scans; the fixtures under
`test-run/out/` from earlier development runs are a starting point, or rasterise any PDF
with `pdftoppm -r 300 -png in.pdf page`).

- [ ] **Step 2: Run and record**

```bash
.venv-bench/bin/python scripts/bench_classic_cpu.py page-1.png page-2.png page-3.png
```

Record in the decision table below (append the results to this plan file):

| paddlepaddle | paddleocr | OneDNN on | OneDNN off | Crash with OneDNN? |
|--------------|-----------|-----------|------------|--------------------|
| (fill in)    | (fill in) | (s/page)  | (s/page)   | yes / no           |

If OneDNN on crashes: also test the newest paddlepaddle pre-release, and search the
PaddlePaddle issue tracker for "ConvertPirAttribute2RuntimeAttribute"; file an upstream
issue with the reproduction if none exists, and link it here.

---

### Task 3 (decision-gated): Apply the outcome

**Case A - crash still reproduces on current paddlepaddle.** Update the comment in
`classic.py` so it carries the evidence instead of "pending re-verification". Replace the
existing comment block above `kwargs["enable_mkldnn"] = False` with:

```python
        # PaddlePaddle <version tested> + OneDNN + the PIR executor still
        # crash at predict time on CPU builds
        # ("ConvertPirAttribute2RuntimeAttribute not support ..."); verified
        # <date> with scripts/bench_classic_cpu.py (upstream: <issue link>).
        # Costs about <measured factor>x per page versus OneDNN on.
```

Commit: `git commit -m "Document verified OneDNN crash workaround"`

**Case B - fixed in paddlepaddle >= X.** Gate the workaround by version so fixed
installs get OneDNN's speed back. In `classic.py`, replace:

```python
    if device == "cpu":
        kwargs["enable_mkldnn"] = False
```

with:

```python
    if device == "cpu" and _paddle_older_than(_MKLDNN_FIXED_VERSION):
        # Older paddlepaddle crashes at predict time with OneDNN + PIR
        # ("ConvertPirAttribute2RuntimeAttribute not support ..."); fixed in
        # <X> (verified <date> with scripts/bench_classic_cpu.py).
        kwargs["enable_mkldnn"] = False
```

and add at module level:

```python
_MKLDNN_FIXED_VERSION = (3, 2)  # replace with the verified fixed release


def _paddle_older_than(minimum: tuple[int, int]) -> bool:
    try:
        import paddle

        parts = str(paddle.__version__).split(".")
        return (int(parts[0]), int(parts[1])) < minimum
    except Exception:
        # No paddle or unparseable version: keep the safe workaround.
        return True
```

Test (append to `tests/test_classic_device.py`; stub `paddle` in `sys.modules` the way
`tests/test_check_options_gpu.py` does):

```python
def test_mkldnn_workaround_dropped_on_fixed_paddle(monkeypatch):
    import sys
    import types

    fake_paddle = types.ModuleType("paddle")
    fake_paddle.__version__ = "9.9.9"
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    monkeypatch.setattr(classic, "PaddleOCR", _PaddleOCRStub)
    classic._build_engine(_opts("classic-cpu"))
    assert "enable_mkldnn" not in _PaddleOCRStub.last_kwargs


def test_mkldnn_workaround_kept_on_old_paddle(monkeypatch):
    import sys
    import types

    fake_paddle = types.ModuleType("paddle")
    fake_paddle.__version__ = "3.0.0"
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    monkeypatch.setattr(classic, "PaddleOCR", _PaddleOCRStub)
    classic._build_engine(_opts("classic-cpu"))
    assert _PaddleOCRStub.last_kwargs.get("enable_mkldnn") is False
```

Note: the existing
`test_build_engine_classic_cpu_uses_cpu_and_mkldnn_workaround` test must then stub an old
paddle version too, otherwise it depends on whether paddle is importable on the test
host.

Run: `pytest tests/test_classic_device.py -v` - expected: all PASS.

Commit: `git commit -m "Gate the OneDNN workaround by paddlepaddle version"`

Update `README.md` "Performance notes" in the same commit with the measured
classic-cpu throughput from Task 2, replacing the unverified "roughly 1-3 pages/sec"
claim with the measured figure and the hardware it was measured on.
