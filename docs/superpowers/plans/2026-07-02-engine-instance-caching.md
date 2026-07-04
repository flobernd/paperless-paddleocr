# Engine Instance Caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop rebuilding `PaddleOCR` / `PaddleOCRVL` for every page and make prediction safe under ocrmypdf's worker threads.

**Architecture:** Module-level caches in `classic.py` and `vl.py`, keyed by every option that influences construction,
guarded by a cache lock. A separate predict lock serialises inference because Paddle predictors are not thread-safe (the
paperless parser passes `use_threads=True, jobs=THREADS_PER_WORKER` to ocrmypdf, which calls `generate_hocr`
concurrently). Cache lifetime is one celery task: paperless recycles the worker process after each document
(`CELERY_WORKER_MAX_TASKS_PER_CHILD=1`), so a multi-page or multi-language document reuses instances and the process
exit frees them.

**Tech Stack:** Python 3.12 `threading`, pytest with import stubs (pattern from `tests/test_classic_device.py` and
`tests/test_check_options_gpu.py`).

## Global Constraints

- Python `>=3.12`; `ruff check .`, `ruff format --check .`, `mypy --ignore-missing-imports paperless_paddleocr` must pass.
- Tests must run without paddlepaddle/paddleocr installed.
- No em-dashes in prose or comments; comments explain WHY only.
- If the language-map plan (`2026-07-02-language-map-corrections.md`) has landed,
  `_build_engine` may omit the `lang` kwarg for `ml`; the cache key below uses
  `_resolve_lang(options)`, which is deterministic either way, so no adjustment is needed.

---

### Task 1: Cache the classic engine per configuration

**Files:**

- Modify: `paperless_paddleocr/paddle_engine/classic.py`
- Test: `tests/test_classic_engine_cache.py` (create)

**Interfaces:**

- Produces: `classic._get_engine(options) -> Any` (cached), `classic._ENGINE_CACHE`
  (dict, cleared by tests), `classic._PREDICT_LOCK` (used by Task 2's vl code too via its
  own module copy). `build_page` behaviour is unchanged apart from reuse.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_classic_engine_cache.py`:

```python
"""build_page must reuse one PaddleOCR instance per configuration.

Constructing PaddleOCR reloads model weights (seconds, hundreds of MB), and
ocrmypdf calls generate_hocr once per page from a thread pool, so without a
cache an N-page document pays N model loads and races construction.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
from PIL import Image

from paperless_paddleocr.paddle_engine import classic


class _CountingPaddleOCR:
    instances = 0

    def __init__(self, **kwargs: Any) -> None:
        _CountingPaddleOCR.instances += 1
        self.kwargs = kwargs

    def predict(self, path: str, **kwargs: Any) -> list:
        return []


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(classic, "PaddleOCR", _CountingPaddleOCR)
    _CountingPaddleOCR.instances = 0
    classic._ENGINE_CACHE.clear()
    yield
    classic._ENGINE_CACHE.clear()


def _opts(lang: str = "en") -> types.SimpleNamespace:
    return types.SimpleNamespace(languages=[lang], paddle_engine="classic-cpu")


def _png(tmp_path):
    img = tmp_path / "page.png"
    Image.new("RGB", (60, 40), "white").save(img)
    return img


def test_same_options_reuse_one_engine(tmp_path):
    img = _png(tmp_path)
    classic.build_page(img, _opts())
    classic.build_page(img, _opts())
    assert _CountingPaddleOCR.instances == 1


def test_each_language_gets_its_own_engine(tmp_path):
    img = _png(tmp_path)
    classic.build_page(img, _opts("en"))
    classic.build_page(img, _opts("german"))
    assert _CountingPaddleOCR.instances == 2


def test_custom_model_dirs_are_part_of_the_key(tmp_path):
    img = _png(tmp_path)
    classic.build_page(img, _opts())
    with_dir = _opts()
    with_dir.paddle_det_model_dir = "/models/det"
    classic.build_page(img, with_dir)
    assert _CountingPaddleOCR.instances == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_classic_engine_cache.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_ENGINE_CACHE'`

- [ ] **Step 3: Implement**

In `classic.py`, add `import threading` to the imports, then add below `OCR_SYSTEM`:

```python
# One engine per configuration, one prediction at a time. Constructing
# PaddleOCR reloads model weights from disk, and Paddle inference predictors
# are not thread-safe; ocrmypdf calls generate_hocr from several worker
# threads when the caller passes jobs > 1. The cache lives until the celery
# worker is recycled (paperless recycles after every document).
_CACHE_LOCK = threading.Lock()
_ENGINE_CACHE: dict[tuple[Any, ...], Any] = {}
_PREDICT_LOCK = threading.Lock()


def _engine_key(options: Any) -> tuple[Any, ...]:
    return (
        _resolve_lang(options),
        _resolve_device(options),
        getattr(options, "paddle_det_model_dir", None),
        getattr(options, "paddle_rec_model_dir", None),
        getattr(options, "paddle_cls_model_dir", None),
    )


def _get_engine(options: Any) -> Any:
    key = _engine_key(options)
    with _CACHE_LOCK:
        engine = _ENGINE_CACHE.get(key)
        if engine is None:
            engine = _build_engine(options)
            _ENGINE_CACHE[key] = engine
    return engine
```

In `build_page`, replace:

```python
    engine = _build_engine(options)
    result = engine.predict(str(input_file), return_word_box=True)
```

with:

```python
    engine = _get_engine(options)
    with _PREDICT_LOCK:
        result = engine.predict(str(input_file), return_word_box=True)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_classic_engine_cache.py tests/test_classic_device.py -v`
Expected: all PASS (`test_classic_device.py` calls `_build_engine` directly, bypassing the
cache, so it is unaffected).

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/paddle_engine/classic.py tests/test_classic_engine_cache.py
git commit -m "Cache classic PaddleOCR engines and serialise predictions"
```

---

### Task 2: Cache the VL pipeline per configuration

**Files:**

- Modify: `paperless_paddleocr/paddle_engine/vl.py`
- Test: `tests/test_vl_pipeline_cache.py` (create)

**Interfaces:**

- Produces: `vl._get_pipeline(options) -> Any` (cached), `vl._PIPELINE_CACHE` (dict,
  cleared by tests). `build_page` unchanged apart from reuse.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vl_pipeline_cache.py`:

```python
"""build_page must reuse one PaddleOCRVL pipeline per configuration.

PaddleOCRVL construction loads local preprocessing models and builds the
remote client; per-page construction wastes seconds and RAM on every page.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from PIL import Image

from paperless_paddleocr.paddle_engine import vl


class _CountingPipeline:
    instances = 0

    def __init__(self, **kwargs: Any) -> None:
        _CountingPipeline.instances += 1
        self.kwargs = kwargs
        self.pipeline_version = kwargs.get("pipeline_version")

    def predict(self, path: str, **kwargs: Any) -> list:
        return []


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    # vl._build_pipeline does `import paddle` and calls set_device; provide a
    # stub so the tests run without paddlepaddle installed.
    fake_paddle = types.ModuleType("paddle")
    fake_paddle.set_device = lambda device: None
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    monkeypatch.setattr(vl, "PaddleOCRVL", _CountingPipeline)
    _CountingPipeline.instances = 0
    vl._PIPELINE_CACHE.clear()
    yield
    vl._PIPELINE_CACHE.clear()


def _opts(url: str = "http://gpu:8118") -> types.SimpleNamespace:
    return types.SimpleNamespace(
        languages=["en"],
        paddle_engine="vl-remote",
        paddle_vl_server_url=url,
        paddle_vl_model_name="PaddleOCR-VL-1.5-0.9B",
        paddle_vl_api_key="",
    )


def _png(tmp_path):
    img = tmp_path / "page.png"
    Image.new("RGB", (60, 40), "white").save(img)
    return img


def test_same_options_reuse_one_pipeline(tmp_path):
    img = _png(tmp_path)
    vl.build_page(img, _opts())
    vl.build_page(img, _opts())
    assert _CountingPipeline.instances == 1


def test_different_server_gets_its_own_pipeline(tmp_path):
    img = _png(tmp_path)
    vl.build_page(img, _opts("http://gpu-a:8118"))
    vl.build_page(img, _opts("http://gpu-b:8118"))
    assert _CountingPipeline.instances == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vl_pipeline_cache.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_PIPELINE_CACHE'`

- [ ] **Step 3: Implement**

In `vl.py`, add `import threading` to the imports, then add below `_TEXT_LABELS`:

```python
# Same rationale as classic.py: construction is expensive, Paddle's local
# preprocessing predictors are not thread-safe, and ocrmypdf may call
# generate_hocr from several worker threads. Serialising predict also keeps
# at most one in-flight request per paperless worker on the remote server.
_CACHE_LOCK = threading.Lock()
_PIPELINE_CACHE: dict[tuple[Any, ...], Any] = {}
_PREDICT_LOCK = threading.Lock()


def _pipeline_key(options: Any) -> tuple[Any, ...]:
    return (
        (getattr(options, "paddle_vl_server_url", "") or "").strip(),
        (getattr(options, "paddle_vl_model_name", "") or "").strip(),
        (getattr(options, "paddle_vl_api_key", "") or "").strip(),
    )


def _get_pipeline(options: Any) -> Any:
    key = _pipeline_key(options)
    with _CACHE_LOCK:
        pipeline = _PIPELINE_CACHE.get(key)
        if pipeline is None:
            pipeline = _build_pipeline(options)
            _PIPELINE_CACHE[key] = pipeline
    return pipeline
```

In `build_page`, replace:

```python
    pipeline = _build_pipeline(options)
```

with:

```python
    pipeline = _get_pipeline(options)
```

and wrap the predict call:

```python
    with _PREDICT_LOCK:
        result = pipeline.predict(
            str(input_file),
            use_layout_detection=False,
            use_queues=False,
            prompt_label="spotting" if has_spotting else "ocr",
        )
```

- [ ] **Step 4: Run the full suite**

Run: `pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add paperless_paddleocr/paddle_engine/vl.py tests/test_vl_pipeline_cache.py
git commit -m "Cache the PaddleOCR-VL pipeline and serialise predictions"
```

---

### Task 3: Document the concurrency model

**Files:**

- Modify: `README.md` ("Performance notes" section)

- [ ] **Step 1: Add a bullet to "Performance notes"**

```markdown
- **Concurrency:** within one paperless worker, OCR inference runs one page at a time
  (Paddle predictors are not thread-safe); ocrmypdf still parallelises rasterisation and
  post-processing across `PAPERLESS_THREADS_PER_WORKER` threads. Scale page throughput
  with `PAPERLESS_TASK_WORKERS` (each worker process gets its own engine instance).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Document the OCR concurrency model"
```

---

### Manual verification (after Tasks 1-2)

On a machine with paddlepaddle + paddleocr installed, OCR a 10-page PDF through the
plugin (`ocrmypdf --plugin paperless_paddleocr.ocrmypdf_plugin --paddle-engine classic-cpu
in.pdf out.pdf`) and confirm in the debug log that `PaddleOCR(classic) kwargs:` is printed
once, not ten times, and that wall-clock time drops accordingly.
