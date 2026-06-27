"""CardScanner — orchestrator that ties everything together.

Owns three threads:
  * `CameraStream`        — grabs the latest frame off the device (in cameras.py).
  * `_stream_loop`        — pulls the latest frame, runs `annotate()`, encodes
                            JPEG, publishes to `frame_queue` for the MJPEG endpoint.
  * `_detect_loop`        — pulls the latest frame, runs detection → OCR → matcher
                            → Confirmer; emits confirmed names on `detection_queue`.

`matcher` is injected (provided by the API layer): raw OCR text → (card name | None,
score). Keeping it injected means this module doesn't import Scryfall directly.

This file also re-exports `list_cameras` and `prewarm_ocr` so existing callers
that `from scanner.detector import X` keep working.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Generator, Optional

import cv2
import numpy as np

from config import (
    CARD_BORDER_MARGIN,
    CARD_TOP_EXTRA,
    JPEG_QUALITY,
    VIDEO_SOURCE,
)

from .annotate import annotate
from .cameras import CameraStream, list_cameras
from .confirmer import Confirmer, Detection
from .detection import DetectedCard, detect_card_candidates
from .geometry import expand_quad, four_point_transform
from .ocr import ocr_name_strip, prewarm_ocr

# Re-exports so `from scanner.detector import list_cameras, prewarm_ocr` keeps working.
__all__ = ["CardScanner", "Matcher", "list_cameras", "prewarm_ocr"]

# Injected per-read matcher: raw OCR text → (card name | None, score).
Matcher = Callable[[str], "tuple[Optional[str], float]"]

log = logging.getLogger(__name__)

# Placeholder card_image for DetectedCard — annotation only reads contour/text.
_EMPTY_IMG = np.zeros((1, 1, 3), dtype=np.uint8)


class CardScanner:
    """Camera grabber + detection/confirmation + annotated MJPEG stream."""

    def __init__(
        self,
        card_names: Optional[list[str]] = None,
        matcher: Optional[Matcher] = None,
    ) -> None:
        # `card_names` accepted for API compatibility but unused.
        self.frame_queue:     queue.Queue[bytes]     = queue.Queue(maxsize=2)
        self.detection_queue: queue.Queue[Detection] = queue.Queue(maxsize=64)

        self._camera = CameraStream(VIDEO_SOURCE)
        self._matcher = matcher
        self._confirmer = Confirmer()
        self._results_lock = threading.Lock()
        self._overlay: tuple[list, list, list] = ([], [], [])

        self._running = False
        self._stream_thread: Optional[threading.Thread] = None
        self._detect_thread: Optional[threading.Thread] = None

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._camera.start()
        self._stream_thread = threading.Thread(
            target=self._stream_loop, daemon=True, name="Scanner-stream")
        self._detect_thread = threading.Thread(
            target=self._detect_loop, daemon=True, name="Scanner-detect")
        self._stream_thread.start()
        self._detect_thread.start()
        log.info("CardScanner started.")

    def stop(self) -> None:
        self._running = False
        self._camera.stop()
        for t in (self._stream_thread, self._detect_thread):
            if t:
                t.join(timeout=3)
        log.info("CardScanner stopped.")

    # ── API-compatible surface used by app.py ────────────────────────────
    def update_card_names(self, names: list[str]) -> None:
        pass  # matching is via the injected matcher; nothing to do here

    def switch_source(self, source: int | str) -> None:
        self._camera.switch(source)

    def set_rotation(self, degrees: int) -> None:
        self._camera.set_rotation(degrees)

    @property
    def current_source(self) -> int | str:
        return self._camera.source

    @property
    def rotation(self) -> int:
        return self._camera.rotation

    def camera_status(self) -> dict:
        return self._camera.status()

    # ── threads ──────────────────────────────────────────────────────────
    def _stream_loop(self) -> None:
        """Produce the annotated MJPEG at camera frame-rate (cheap work only)."""
        last_id = -1
        while self._running:
            frame, fid = self._camera.latest()
            if frame is None:
                # No frames yet — show the camera state so the page isn't blank.
                self._publish(self._encode_jpeg(self._status_frame()))
                time.sleep(0.15)
                continue
            if fid == last_id:
                time.sleep(0.005)
                continue
            last_id = fid
            with self._results_lock:
                lines, quads, detected = self._overlay
            self._publish(self._encode_jpeg(annotate(frame, lines, quads, detected)))

    def _detect_loop(self) -> None:
        """Detect → OCR the single most prominent card → match → vote → emit.

        Only the LARGEST quad is OCR'd (the card being presented on top of the
        stack), so we run OCR once per frame instead of once per detected box —
        the big speed win. The matched name feeds one global vote (jitter-proof).
        The only wait is a tiny yield when the camera has no new frame yet.
        """
        last_id = -1
        while self._running:
            frame, fid = self._camera.latest()
            if frame is None or fid == last_id:
                time.sleep(0.002)  # wait briefly for the next camera frame
                continue
            last_id = fid

            raw_lines, quads = detect_card_candidates(frame)
            overlay: list[DetectedCard] = []
            confirmed_name: Optional[str] = None

            if quads:
                # The presented (top) card is the largest valid quad.
                largest = max(quads, key=lambda q: cv2.contourArea(q.astype(np.float32)))
                ocr_quad = expand_quad(largest, CARD_BORDER_MARGIN, CARD_TOP_EXTRA)

                raw = ""
                try:
                    raw = ocr_name_strip(four_point_transform(frame, ocr_quad))
                except cv2.error as exc:
                    log.debug("Perspective unwarp failed: %s", exc)
                except Exception as exc:  # OCR is third-party — never kill the loop
                    log.warning("OCR failed: %s", exc)

                name = None
                if raw and self._matcher is not None:
                    try:
                        name, _score = self._matcher(raw)
                    except Exception as exc:
                        log.warning("Card match failed for %r: %s", raw, exc)

                confirmed_name = self._confirmer.add(name)
                overlay.append(DetectedCard(
                    raw_ocr_text=self._confirmer.candidate or raw,
                    matched_name=self._confirmer.confirmed,
                    confidence=self._confirmer.confidence,
                    contour=ocr_quad.astype(int),
                    card_image=_EMPTY_IMG,
                ))
            else:
                self._confirmer.add(None)  # no card this frame — ages the window

            with self._results_lock:
                self._overlay = (raw_lines, quads, overlay)

            if confirmed_name:
                try:
                    self.detection_queue.put_nowait(
                        Detection(confirmed_name, self._confirmer.confidence))
                except queue.Full:
                    log.warning("Detection queue full; dropping confirmed %r", confirmed_name)

    # ── helpers ──────────────────────────────────────────────────────────
    def _publish(self, jpeg: bytes) -> None:
        try:
            self.frame_queue.put_nowait(jpeg)
        except queue.Full:  # consumer behind — keep only the freshest
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(jpeg)
            except queue.Empty:
                pass

    def _status_frame(self) -> np.ndarray:
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        st = self._camera.status()
        msg = f"Camera: {st['state']} (source {st['source']})"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(img, msg, ((640 - tw) // 2, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        return img

    @staticmethod
    def _encode_jpeg(frame: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        return bytes(buf) if ok else b""

    def latest_jpeg(self) -> Generator[bytes, None, None]:
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            try:
                jpeg = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            yield boundary + jpeg + b"\r\n"
