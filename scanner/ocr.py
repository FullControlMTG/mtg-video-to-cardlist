"""EasyOCR initialisation (lazy + thread-safe) and the name-strip reader.

EasyOCR is heavy (CRAFT detector + recognizer + models). We instantiate it
lazily on the first call, and `prewarm_ocr()` kicks that off in a background
thread at startup so the first user-facing OCR doesn't pay the cost.
"""

from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from config import NAME_COL_FRACTION, NAME_ROW_FRACTION, OCR_ENGINE, OCR_MIN_CONFIDENCE

log = logging.getLogger(__name__)

_ocr_reader = None
_ocr_lock = threading.Lock()


def _get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        with _ocr_lock:
            if _ocr_reader is None:
                import easyocr  # noqa: PLC0415
                log.info("Initialising EasyOCR (first run downloads models)…")
                _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                log.info("EasyOCR ready.")
    return _ocr_reader


def prewarm_ocr() -> None:
    """Trigger EasyOCR initialisation off the request path."""
    t = threading.Thread(target=_get_ocr, daemon=True, name="OCR-prewarm")
    t.start()


def ocr_name_strip(card_img: np.ndarray) -> str:
    """OCR the card's name strip and return the raw text.

    The raw text is then fuzzy-matched to a card name (the injected matcher),
    and the Confirmer votes on the resolved names — not on this noisy raw
    string.
    """
    ch, cw = card_img.shape[:2]
    sh = max(1, int(ch * NAME_ROW_FRACTION))
    sw = max(1, int(cw * NAME_COL_FRACTION))
    strip = card_img[:sh, :sw]

    strip_up = cv2.resize(strip, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(strip_up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    proc = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8,
    )

    # We already cropped to the name strip, so EasyOCR's text *detection* stage
    # (the heavy CRAFT model) is redundant for the fast path: calling the
    # recognizer directly on the crop is ~20-25x faster (~7 ms vs ~150-190 ms).
    # OCR_ENGINE="readtext" re-enables full detection for crop-robustness.
    reader = _get_ocr()
    if OCR_ENGINE == "readtext":
        results = reader.readtext(proc, detail=1, paragraph=False)
    else:
        results = reader.recognize(proc, detail=1, paragraph=False)
    raw_parts = [
        text for (_, text, conf) in results
        if conf >= OCR_MIN_CONFIDENCE and text.strip()
    ]
    return " ".join(raw_parts).strip()
