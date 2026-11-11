"""The choice and the merge of the pages that the language passes give.

When :class:`~paperless_paddleocr.paddle_engine.PaddleOCREngine` runs one pass
per language, the ``winner`` strategy picks the pass with the highest aggregate
confidence, whose page the engine writes as it stands, and ``merge`` keeps the
word boxes of all passes that survive an NMS by confidence. :func:`_merged_page`
builds the page of the hOCR from the words that remain.
"""

from __future__ import annotations

import copy
import logging

from paperless_paddleocr.paddle_engine.geometry import BBox, union_boxes
from paperless_paddleocr.paddle_engine.hocr import (
    Block,
    Line,
    Page,
    Word,
    clamp_box,
    clean_text,
    order_page,
)
from paperless_paddleocr.paddle_engine.layout import fit_hocr_baseline, reading_blocks

log = logging.getLogger("paperless.paddleocr.merge")

_NMS_IOU_THRESHOLD = 0.5

#: Band height for the NMS index. Any value gives the same result, since a
#: kept word is indexed into every band it touches; one near a word height
#: keeps each band's list short.
_NMS_BAND_PX = 32


def page_words(page: Page) -> list[Word]:
    """Every word of ``page`` that has text, in the block order of the single-language hOCR.

    The words come from an ordered copy, so ``page`` keeps the order of its
    regions: the winning page is ordered at its write, and a second ordering
    can build other rows from regions that share a bottom edge. The merge
    sorts stably, so the word order decides its ties. Boxes are clamped to
    the page and texts cleaned and stripped here, ahead of the merge,
    because the merged lines derive their boxes, baselines and word spacing
    from these words.
    """
    ordered = copy.copy(page)
    order_page(ordered)
    words: list[Word] = []
    for block in ordered.blocks:
        for line in block.lines:
            for word in line.words:
                text = clean_text(word.text).strip()
                if text:
                    box = clamp_box(word.box, page.width, page.height)
                    words.append(Word(text, box, word.confidence))
    return words


def winner(per_lang_words: dict[str, list[Word]]) -> str:
    """The language whose words have the highest sum of confidences."""
    by_lang_conf = {
        lang: sum(w.confidence for w in words) for lang, words in per_lang_words.items()
    }
    winner_lang = max(by_lang_conf, key=lambda k: by_lang_conf[k])
    log.info(
        "multi-lang winner: %s (score=%d, %d words). Other scores: %s",
        winner_lang,
        by_lang_conf[winner_lang],
        len(per_lang_words[winner_lang]),
        {lang: s for lang, s in by_lang_conf.items() if lang != winner_lang},
    )
    return winner_lang


def merge_page(
    per_lang_words: dict[str, list[Word]],
    width: int,
    height: int,
) -> tuple[Page, str]:
    """The page and the sidecar of the NMS merge, labelled with the first language."""
    flat = [w for words in per_lang_words.values() for w in words]
    merged = _nms_merge(flat, _NMS_IOU_THRESHOLD)
    log.debug(
        "merge: %d total words from %d languages → %d after NMS",
        len(flat),
        len(per_lang_words),
        len(merged),
    )
    blocks = reading_blocks(merged, bbox_of=lambda w: w.box)
    page = _merged_page(blocks, width, height, hocr_lang=next(iter(per_lang_words)))
    return page, _blocks_to_text(blocks)


def _iou(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _blocks_to_text(blocks: list[list[list[Word]]]) -> str:
    """Render reading-ordered blocks to plain text.

    Blocks come from :func:`layout.reading_blocks` (column bands plus
    full-width flow); lines are space-joined, blocks blank-line-separated so
    downstream consumers (paperless UI, full-text indexer) see paragraphs.
    """
    return "\n\n".join(
        "\n".join(" ".join(w.text for w in line) for line in block) for block in blocks
    )


def _nms_merge(words: list[Word], iou_threshold: float) -> list[Word]:
    """Greedy NMS by descending confidence, then re-order for reading.

    Surviving words are re-sequenced through :func:`layout.reading_blocks`
    so the merged hOCR and sidecar read in banded reading order:
    column bands left-to-right, lines top-to-bottom within each band.
    """
    # Kept words are indexed by the horizontal bands they touch. IoU is zero
    # for vertically disjoint boxes, so a candidate only needs the kept words
    # of its own bands and the outcome matches a scan over all of them.
    bands: dict[int, list[Word]] = {}
    kept: list[Word] = []
    for candidate in sorted(words, key=lambda w: w.confidence, reverse=True):
        _x0, y0, _x1, y1 = candidate.box
        touched = range(min(y0, y1) // _NMS_BAND_PX, max(y0, y1) // _NMS_BAND_PX + 1)
        if any(
            _iou(candidate.box, k.box) >= iou_threshold for b in touched for k in bands.get(b, ())
        ):
            continue
        kept.append(candidate)
        for b in touched:
            bands.setdefault(b, []).append(candidate)

    return [
        w for block in reading_blocks(kept, bbox_of=lambda w: w.box) for line in block for w in line
    ]


def _merged_page(
    blocks: list[list[list[Word]]],
    page_w: int,
    page_h: int,
    *,
    hocr_lang: str,
) -> Page:
    """Page from reading-ordered blocks, so the merged hOCR carries real lines.

    Grouped lines (rather than one line per word) give ocrmypdf's renderer
    line-level baselines and give PDF viewers sane text selection.
    """
    page = Page(
        width=page_w,
        height=page_h,
        lang=hocr_lang,
        ocr_system="paperless-paddleocr multi-lang merge",
    )
    for block in blocks:
        for line_words in block:
            box = union_boxes(w.box for w in line_words)
            line = Line(
                box=box,
                confidence=round(sum(w.confidence for w in line_words) / len(line_words)),
                text=" ".join(w.text for w in line_words),
                words=list(line_words),
                baseline=fit_hocr_baseline(
                    [((w.box[0] + w.box[2]) / 2.0, float(w.box[3])) for w in line_words],
                    box,
                ),
            )
            page.blocks.append(Block(box=box, lines=[line]))
    return page
