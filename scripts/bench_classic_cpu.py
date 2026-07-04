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
