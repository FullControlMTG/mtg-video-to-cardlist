"""Camera enumeration + threaded latest-frame grabber with a lifecycle
state machine.

This module owns everything related to the physical camera device:
listing what's available (`list_cameras`), opening one (`CameraStream`),
and the connect → warm-up → stream → reconnect cycle.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from config import (
    CAMERA_PROBE_MAX,
    CAMERA_READ_FAIL_LIMIT,
    CAMERA_RECONNECT_DELAY,
    CAMERA_WARMUP_TIMEOUT,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
)

log = logging.getLogger(__name__)


def _os_camera_names() -> list[str]:
    # macOS: system_profiler enumerates cameras in the same order as OpenCV's
    # AVFoundation backend, so indices match directly.
    if platform.system() != "Darwin":
        return []
    try:
        import json as _json
        import subprocess as _sp
        out = _sp.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True, text=True, timeout=5,
        )
        data = _json.loads(out.stdout)
        return [c.get("_name", "") for c in data.get("SPCameraDataType", [])]
    except Exception:
        return []


def list_cameras(max_probe: int = 8) -> list[dict]:
    os_names = _os_camera_names()
    cameras: list[dict] = []

    for idx in range(max_probe):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            break

        ret, _ = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if not ret:
            continue

        name = os_names[idx] if idx < len(os_names) and os_names[idx] else f"Camera {idx}"
        cameras.append({"index": idx, "name": name, "resolution": f"{w}×{h}"})

    return cameras


# ── Lifecycle ───────────────────────────────────────────────────────────────

class CameraState(Enum):
    CONNECTING   = "connecting"
    WARMING_UP   = "warming_up"
    STREAMING    = "streaming"
    RECONNECTING = "reconnecting"
    FAILED       = "failed"


class CameraStream:
    """Threaded grabber that always exposes the most recent frame.

    This is the standard low-latency capture pattern: a dedicated thread reads
    continuously (draining the driver's buffer) so consumers never get stale
    frames, and capture is decoupled from detection/streaming.

    IMPORTANT: capture uses ``cv2.VideoCapture(source)`` with OpenCV's DEFAULT
    backend on purpose — that is what reliably opens the iPhone Continuity
    Camera on this Mac. Do not switch it to an explicit backend (e.g.
    CAP_AVFOUNDATION) or an ffmpeg shim; that regressed Continuity before.
    """

    def __init__(self, source: int | str) -> None:
        self._source = source
        self._pending_source: Optional[int | str] = None
        self._rotation = 0
        self._cap: Optional[cv2.VideoCapture] = None
        self._state = CameraState.CONNECTING
        self._latest: Optional[np.ndarray] = None
        self._frame_id = 0
        self._last_frame_time = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="CameraStream")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def switch(self, source: int | str) -> None:
        self._pending_source = source
        log.info("Camera switch requested → %s", source)

    def set_rotation(self, degrees: int) -> None:
        self._rotation = degrees % 360
        log.info("Camera rotation set to %d°", self._rotation)

    # ── accessors ────────────────────────────────────────────────────────
    @property
    def source(self) -> int | str:
        return self._source

    @property
    def rotation(self) -> int:
        return self._rotation

    @property
    def state(self) -> CameraState:
        return self._state

    @property
    def is_streaming(self) -> bool:
        return self._state is CameraState.STREAMING

    def status(self) -> dict:
        with self._lock:
            age = (time.monotonic() - self._last_frame_time) if self._last_frame_time else None
        return {
            "state": self._state.value,
            "source": self._source,
            "streaming": self._state is CameraState.STREAMING,
            "seconds_since_frame": round(age, 2) if age is not None else None,
        }

    def latest(self) -> tuple[Optional[np.ndarray], int]:
        with self._lock:
            return self._latest, self._frame_id

    # ── internals ────────────────────────────────────────────────────────
    def _set_state(self, state: CameraState) -> None:
        if state is not self._state:
            self._state = state
            log.info("Camera source %s → %s", self._source, state.value)

    def _open_capture(self, source: int | str) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(source)  # DEFAULT backend — see class docstring
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        return cap

    def _await_first_frame(self, cap: cv2.VideoCapture) -> bool:
        """Read until the device delivers a real frame or warmup times out."""
        self._set_state(CameraState.WARMING_UP)
        deadline = time.monotonic() + CAMERA_WARMUP_TIMEOUT
        while self._running and time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                return True
            time.sleep(0.1)
        return False

    def _auto_select(self) -> tuple[Optional[cv2.VideoCapture], int | str]:
        """Startup auto-pick: configured source if it streams, else probe."""
        cap = self._open_capture(self._source)
        if cap is not None and self._await_first_frame(cap):
            return cap, self._source
        if cap is not None:
            cap.release()
        log.warning("Configured source %s did not deliver frames; probing 0-%d…",
                    self._source, CAMERA_PROBE_MAX - 1)
        for idx in range(CAMERA_PROBE_MAX):
            if idx == self._source:
                continue
            cap = self._open_capture(idx)
            if cap is not None and self._await_first_frame(cap):
                log.info("Auto-selected working camera %d", idx)
                return cap, idx
            if cap is not None:
                cap.release()
        return None, self._source

    def _rotate(self, frame: np.ndarray) -> np.ndarray:
        if self._rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self._rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self._rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _loop(self) -> None:
        cap, src = self._auto_select()
        attempts = 0
        while cap is None and self._running:
            attempts += 1
            self._set_state(CameraState.FAILED if attempts == 1 else CameraState.RECONNECTING)
            log.error("No camera is delivering frames (attempt %d); retrying in %.1fs…",
                      attempts, CAMERA_RECONNECT_DELAY)
            time.sleep(CAMERA_RECONNECT_DELAY)
            cap, src = self._auto_select()
        if not self._running:
            if cap is not None:
                cap.release()
            return
        self._cap = cap
        self._source = src

        fail = 0
        while self._running:
            # User-requested switch: commit to the new device if it OPENS, and
            # keep its session alive while it warms up. We never silently revert
            # to the old camera — the read-failure path below rebuilds the SAME
            # (selected) source.
            if self._pending_source is not None:
                new_src = self._pending_source
                self._pending_source = None
                self._set_state(CameraState.CONNECTING)
                log.info("Switching camera %s → %s", self._source, new_src)
                new_cap = self._open_capture(new_src)
                if new_cap is not None:
                    old = self._cap
                    self._cap = new_cap
                    self._source = new_src
                    if old is not None:
                        old.release()
                    with self._lock:
                        self._latest = None  # drop the previous camera's last frame
                    self._set_state(CameraState.WARMING_UP)
                    fail = 0
                else:
                    log.warning("Could not open source %s; staying on %s.",
                                new_src, self._source)

            cap = self._cap
            ok, frame = cap.read() if cap is not None else (False, None)
            if not ok or frame is None:
                fail += 1
                if fail >= CAMERA_READ_FAIL_LIMIT:
                    self._set_state(CameraState.RECONNECTING)
                    log.warning("No frames from source %s for %d reads; rebuilding session…",
                                self._source, fail)
                    if self._cap is not None:
                        self._cap.release()
                    time.sleep(CAMERA_RECONNECT_DELAY)
                    self._cap = self._open_capture(self._source)  # same source — never revert
                    if self._cap is None:
                        self._set_state(CameraState.FAILED)
                    fail = 0
                else:
                    time.sleep(0.05)
                continue

            fail = 0
            if self._rotation:
                frame = self._rotate(frame)
            with self._lock:
                self._latest = frame
                self._frame_id += 1
                self._last_frame_time = time.monotonic()
            if self._state is not CameraState.STREAMING:
                self._set_state(CameraState.STREAMING)

        if self._cap is not None:
            self._cap.release()
