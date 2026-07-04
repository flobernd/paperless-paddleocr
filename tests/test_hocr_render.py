"""render_hocr must emit each line's fitted baseline into the title attr."""

from __future__ import annotations

from paperless_paddleocr.paddle_engine.hocr import Block, Line, Page, Word, render_hocr


def _page(line: Line) -> Page:
    return Page(
        width=400,
        height=300,
        lang="en",
        ocr_system="test",
        blocks=[Block(box=line.box, lines=[line])],
    )


def test_render_hocr_emits_default_flat_baseline():
    line = Line(
        box=(10, 10, 90, 40),
        confidence=88,
        text="hi",
        words=[Word("hi", (10, 10, 90, 40), 88)],
    )
    assert "baseline 0.000000 0" in render_hocr(_page(line))


def test_render_hocr_emits_fitted_baseline():
    line = Line(
        box=(10, 10, 90, 40),
        confidence=88,
        text="hi",
        words=[Word("hi", (10, 10, 90, 40), 88)],
        baseline=(0.05, -4.0),
    )
    assert "baseline 0.050000 -4" in render_hocr(_page(line))


def test_write_document_sidecar_override(tmp_path):
    from paperless_paddleocr.paddle_engine.hocr import write_document

    line = Line(
        box=(10, 10, 90, 40), confidence=88, text="hi", words=[Word("hi", (10, 10, 90, 40), 88)]
    )
    hocr_path, text_path = tmp_path / "p.hocr", tmp_path / "p.txt"
    write_document(_page(line), hocr_path, text_path, sidecar="custom order")
    assert text_path.read_text(encoding="utf-8") == "custom order"
    assert "ocrx_word" in hocr_path.read_text(encoding="utf-8")
