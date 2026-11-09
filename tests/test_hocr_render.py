"""render_hocr output: fitted baselines, boxes clamped to the page and XML-valid text."""

from __future__ import annotations

from xml.etree import ElementTree

from paperless_paddleocr.paddle_engine.hocr import (
    Block,
    Line,
    Page,
    Word,
    clean_text,
    render_hocr,
    write_document,
)


def _page(line: Line, *, width: int = 400, height: int = 300) -> Page:
    return Page(
        width=width,
        height=height,
        lang="en",
        ocr_system="test",
        blocks=[Block(box=line.box, lines=[line])],
    )


def _line(word: Word) -> Line:
    return Line(box=word.box, confidence=word.confidence, text=word.text, words=[word])


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
    line = Line(
        box=(10, 10, 90, 40), confidence=88, text="hi", words=[Word("hi", (10, 10, 90, 40), 88)]
    )
    hocr_path, text_path = tmp_path / "p.hocr", tmp_path / "p.txt"
    write_document(_page(line), hocr_path, text_path, sidecar="custom order")
    assert text_path.read_text(encoding="utf-8") == "custom order"
    assert "ocrx_word" in hocr_path.read_text(encoding="utf-8")


def test_render_drops_xml_invalid_characters_from_words():
    # ocrmypdf parses the hOCR with ElementTree, which rejects a NUL, and a
    # lone surrogate cannot be encoded as UTF-8.
    hocr = render_hocr(_page(_line(Word("in\x00voice\ud800", (10, 10, 60, 30), 90))))
    ElementTree.fromstring(hocr.encode("utf-8"))
    assert "invoice" in hocr and "\x00" not in hocr


def test_clean_text_keeps_xml_valid_characters():
    # write_document cleans the sidecar, which separates its lines with newlines.
    text = "a\tb\r\nc \u65e5 \ufffd \U00020bb7"
    assert clean_text(text) == text


def test_write_document_drops_xml_invalid_characters_from_the_sidecar(tmp_path):
    page = _page(_line(Word("hi", (10, 10, 90, 40), 88)))
    write_document(page, tmp_path / "p.hocr", tmp_path / "p.txt", sidecar="a\x00b")
    assert (tmp_path / "p.txt").read_text(encoding="utf-8") == "ab"


def test_render_clamps_a_box_outside_the_page():
    page = _page(_line(Word("x", (-5, -5, 1300, 40), 90)), width=1200, height=1600)
    hocr = render_hocr(page)
    assert 'title="bbox 0 0 1200 40' in hocr
    assert "1300" not in hocr


def test_render_gives_a_zero_width_box_one_pixel():
    page = _page(_line(Word("x", (50, 10, 50, 30), 90)))
    assert 'title="bbox 50 10 51 30' in render_hocr(page)


def test_render_keeps_one_pixel_for_a_box_past_the_right_edge():
    # estimate_word_boxes can place the last of six one-character words past
    # a line box that ends at the page edge.
    page = _page(_line(Word("x", (1201, 10, 1202, 30), 90)), width=1200, height=1600)
    assert 'title="bbox 1199 10 1200 30' in render_hocr(page)


def test_render_keeps_one_pixel_for_a_box_below_the_bottom_edge():
    page = _page(_line(Word("x", (10, 1601, 20, 1700), 90)), width=1200, height=1600)
    assert 'title="bbox 10 1599 20 1600' in render_hocr(page)
