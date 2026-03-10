"""
Card detection pipeline (v2 — Hough-line based).

Detection steps, each with its own visual indicator:

  [1] LINES    (grey, thin)   — every HoughLinesP segment in the frame
  [2] QUAD     (orange)       — a perpendicular pair of line groups forms a
                                card-shaped quadrilateral
  [3] STRIP    (yellow box)   — the name-strip region back-projected onto the
                                original frame (shows exactly what we OCR)
  [4] NO_MATCH (red)          — OCR ran but fuzzy-match score too low
  [5] MATCHED  (green)        — card name confirmed; label drawn above card

MTG card anatomy (consistent across all frame variants):

  ┌──────────────────────────────┐
  │  Name (left)   Mana (right)  │  ← top ~17 % — we OCR left 72 % of this
  ├──────────────────────────────┤
  │            Art               │
  ├──────────────────────────────┤
  │  Type line                   │
  ├──────────────────────────────┤
  │  Text / rules / flavour      │
  └─────────────────────── P/T ──┘

"Showcase", borderless, retro and other alternate frames change the visual
style but preserve this structural layout, so the name-strip approach works
across virtually all printed Magic cards.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Generator, Optional

import cv2
import numpy as np

from config import (
    CARD_ASPECT_MAX,
    CARD_ASPECT_MIN,
    FRAME_SKIP,
    FUZZY_MATCH_THRESHOLD,
    JPEG_QUALITY,
    MIN_CARD_AREA,
    NAME_COL_FRACTION,
    NAME_ROW_FRACTION,
    OCR_MIN_CONFIDENCE,
    VIDEO_SOURCE,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera enumeration
# ---------------------------------------------------------------------------

def list_cameras(max_probe: int = 8) -> list[dict]:
    """
    Probe video-capture device indices 0 … max_probe and return those that
    open and produce frames.

    We use generic "Camera N" labels because the OS-level device order
    (e.g. system_profiler on macOS) does not reliably match the indices that
    OpenCV assigns, making native names misleading.  The native resolution is
    included so users can distinguish devices (e.g. iPhone cameras typically
    report different resolutions from built-in webcams).

    Returns a list of dicts:
        {"index": int, "name": str, "resolution": "WxH"}
    """
    cameras: list[dict] = []

    for idx in range(max_probe):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            break          # assume no further indices exist after first gap

        ret, _ = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if not ret:
            continue       # index exists but delivers no frames — skip

        cameras.append({"index": idx, "name": f"Camera {idx}",
                        "resolution": f"{w}×{h}"})

    return cameras


# ---------------------------------------------------------------------------
# Lazy OCR reader
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Detection step enum and visual styles
# ---------------------------------------------------------------------------

class DetectionStep(Enum):
    LINES    = "lines"     # step 1 — raw Hough segments
    QUAD     = "quad"      # step 2 — card-shaped quad found
    STRIP    = "strip"     # step 3 — name strip isolated
    NO_MATCH = "no_match"  # step 4 — OCR ran, score below threshold
    MATCHED  = "matched"   # step 5 — fuzzy match confirmed


# (BGR colour, line thickness)
_STYLE: dict[DetectionStep, tuple[tuple[int, int, int], int]] = {
    DetectionStep.LINES:    ((90,  90,  90),  1),   # dim grey
    DetectionStep.QUAD:     ((0,  165, 255),  2),   # orange
    DetectionStep.STRIP:    ((0,  255, 255),  2),   # yellow
    DetectionStep.NO_MATCH: ((0,   60, 220),  2),   # red
    DetectionStep.MATCHED:  ((0,  210,   0),  3),   # green
}


# ---------------------------------------------------------------------------
# Public data types (unchanged interface)
# ---------------------------------------------------------------------------

@dataclass
class DetectedCard:
    raw_ocr_text: str
    matched_name: Optional[str]
    confidence: float
    contour: np.ndarray    # shape (4, 2), int, frame coordinates
    card_image: np.ndarray  # perspective-corrected card crop


@dataclass
class ScanFrame:
    jpeg_bytes: bytes
    detected: list[DetectedCard] = field(default_factory=list)
    timestamp: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return (TL, TR, BR, BL) ordering for a 4-point array."""
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # TL: min(x+y)
    rect[2] = pts[np.argmax(s)]    # BR: max(x+y)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR: min(y−x)
    rect[3] = pts[np.argmax(diff)] # BL: max(y−x)
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    tl, tr, br, bl = rect
    width  = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 2 or height < 2:
        raise cv2.error("Degenerate quad")
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))


def _name_strip_frame_quad(
    quad: np.ndarray, card_h: int, card_w: int
) -> np.ndarray:
    """
    Back-project the name-strip bounding box from card-image space into
    frame space.  Returns a (4, 2) float32 array for overlay drawing.
    """
    rect = _order_points(quad.astype("float32"))
    dst = np.array(
        [[0, 0], [card_w - 1, 0], [card_w - 1, card_h - 1], [0, card_h - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return quad.astype("float32")

    sh = int(card_h * NAME_ROW_FRACTION)
    sw = int(card_w * NAME_COL_FRACTION)
    strip_corners = np.array(
        [[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype="float32"
    ).reshape(-1, 1, 2)
    frame_corners = cv2.perspectiveTransform(strip_corners, M_inv)
    return frame_corners.reshape(4, 2)


# ---------------------------------------------------------------------------
# Hough-line based card detection
# ---------------------------------------------------------------------------

def _segment_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    """Line angle in [0, 180) degrees (direction-agnostic)."""
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180


def _cluster_by_angle(
    segments: list[tuple],
    tolerance: float = 15.0,
) -> list[list[tuple]]:
    """
    Group (x1, y1, x2, y2, angle) tuples by angle similarity.

    Uses a running-average cluster centroid so later segments are compared
    against the true mean of the cluster, not just its first member.
    Greedy single-pass; fast for the typical <100-segment case after length
    filtering.
    """
    clusters: list[list[tuple]] = []
    centroids: list[float] = []          # running average angle per cluster

    for seg in segments:
        angle = seg[4]
        best_idx   = -1
        best_diff  = tolerance
        for k, centroid in enumerate(centroids):
            diff = abs(angle - centroid) % 180
            diff = min(diff, 180.0 - diff)
            if diff < best_diff:
                best_diff = diff
                best_idx  = k

        if best_idx >= 0:
            clusters[best_idx].append(seg)
            # Update running average (avoid 0/180 wrap-around issues by
            # anchoring to the existing centroid direction)
            n = len(clusters[best_idx])
            centroids[best_idx] = (centroids[best_idx] * (n - 1) + angle) / n
        else:
            clusters.append([seg])
            centroids.append(angle)

    return clusters


def _boundary_lines(cluster: list[tuple]) -> tuple[tuple, tuple]:
    """
    Return the two segments in a cluster that are furthest apart
    (the two parallel edges of the card on that axis).
    Measures separation by projecting midpoints onto the perpendicular direction.
    """
    avg_angle = sum(s[4] for s in cluster) / len(cluster)
    perp = math.radians(avg_angle + 90.0)
    px, py = math.cos(perp), math.sin(perp)

    projections = [
        (((s[0] + s[2]) / 2.0) * px + ((s[1] + s[3]) / 2.0) * py, s)
        for s in cluster
    ]
    projections.sort(key=lambda x: x[0])
    return projections[0][1], projections[-1][1]


def _to_line_eq(seg: tuple) -> tuple[float, float, float]:
    """Segment (x1,y1,x2,y2,...) → (a, b, c) where ax + by = c."""
    x1, y1, x2, y2 = seg[0], seg[1], seg[2], seg[3]
    a = float(y2 - y1)
    b = float(x1 - x2)
    c = a * x1 + b * y1
    return a, b, c


def _intersect(s1: tuple, s2: tuple) -> Optional[tuple[float, float]]:
    a1, b1, c1 = _to_line_eq(s1)
    a2, b2, c2 = _to_line_eq(s2)
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    return (c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det


def _detect_lines_and_quads(
    frame: np.ndarray,
) -> tuple[list[tuple], list[np.ndarray]]:
    """
    Steps 1 & 2: detect Hough line segments, cluster by angle, find quads.

    Why bilateral filter instead of Gaussian:
      A card placed on a table has a strong, continuous edge at its border.
      Inside the card there are many short parallel features: text rows, the
      type-line separator, mana-cost symbols, art detail lines.  GaussianBlur
      smooths everything equally, so all those interior features survive into
      Canny and produce dozens of short line segments.  bilateralFilter
      preserves strong edges (the card border) while blurring regions of
      similar intensity (interior text areas), so far fewer interior lines
      reach Canny.

    Why no dilation:
      Dilating the edge image widens every detected edge, which (a) makes
      the card interior features even more prominent and (b) can merge
      nearby unrelated edges into a single apparent long line.

    Why longer minLineLength:
      A standard card border at typical scanning distance produces edges
      ~150–500 px long.  Text rows, separator lines, and art details produce
      edges 10–80 px long.  Setting minLineLength to ~10 % of the frame's
      shorter dimension (~72 px for 720 p) eliminates almost all card-interior
      noise while keeping the four outer border segments.

    Returns:
      raw_lines — (x1, y1, x2, y2, angle) for every accepted long segment
      quads     — (4, 2) float32 in TL/TR/BR/BL order, shape-validated
    """
    h, w = frame.shape[:2]

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Bilateral filter: smooths flat-ish interior regions of the card while
    # preserving the sharp luminance jump at the card's outer border.
    smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    edges  = cv2.Canny(smooth, 50, 130)
    # No dilation — see note above.

    # Minimum line length: card borders span at least 10 % of the frame.
    min_len = max(80, int(min(h, w) * 0.10))

    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_len,
        maxLineGap=25,
    )
    if raw is None:
        return [], []

    segments: list[tuple] = [
        (float(x1), float(y1), float(x2), float(y2),
         _segment_angle(x1, y1, x2, y2))
        for x1, y1, x2, y2 in raw[:, 0]
    ]

    clusters = [c for c in _cluster_by_angle(segments) if len(c) >= 2]

    def _cluster_angle(segs: list[tuple]) -> float:
        return sum(s[4] for s in segs) / len(segs)

    def _perp_sep(segs: list[tuple], ba: tuple, bb: tuple) -> float:
        """Perpendicular separation between the two boundary line midpoints."""
        pr = math.radians(_cluster_angle(segs) + 90.0)
        ppx, ppy = math.cos(pr), math.sin(pr)
        def proj(s: tuple) -> float:
            return ((s[0]+s[2])/2)*ppx + ((s[1]+s[3])/2)*ppy
        return abs(proj(ba) - proj(bb))

    quads: list[np.ndarray] = []
    checked: set[frozenset] = set()
    min_sep = max(80.0, math.sqrt(MIN_CARD_AREA) * 0.5)

    for i, ci in enumerate(clusters):
        for j, cj in enumerate(clusters):
            if j <= i:
                continue
            key = frozenset([i, j])
            if key in checked:
                continue
            checked.add(key)

            # Require near-perpendicular groups (65°–115°)
            ai = _cluster_angle(ci)
            aj = _cluster_angle(cj)
            diff = min(abs(ai - aj) % 180, 180.0 - abs(ai - aj) % 180)
            if not (65.0 <= diff <= 115.0):
                continue

            li_a, li_b = _boundary_lines(ci)
            lj_a, lj_b = _boundary_lines(cj)

            # Both axes must span at least min_sep pixels
            if _perp_sep(ci, li_a, li_b) < min_sep or _perp_sep(cj, lj_a, lj_b) < min_sep:
                continue

            # Compute 4 intersection corners
            corners: list[tuple[float, float]] = []
            ok = True
            for la in (li_a, li_b):
                for lb in (lj_a, lj_b):
                    pt = _intersect(la, lb)
                    if pt is None:
                        ok = False
                        break
                    corners.append(pt)
                if not ok:
                    break
            if not ok or len(corners) != 4:
                continue

            # Corners must be within a generous frame margin
            margin = max(w, h) * 0.3
            if any(
                cx < -margin or cx > w + margin or cy < -margin or cy > h + margin
                for cx, cy in corners
            ):
                continue

            pts = np.array(corners, dtype="float32")
            ordered = _order_points(pts)

            if cv2.contourArea(ordered) < MIN_CARD_AREA:
                continue

            tl, tr, br, bl = ordered
            width_px  = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
            height_px = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
            if height_px < 1:
                continue
            aspect = width_px / height_px
            if not (CARD_ASPECT_MIN <= aspect <= CARD_ASPECT_MAX):
                continue

            quads.append(ordered)

    return segments, _deduplicate_quads(quads)


def _deduplicate_quads(
    quads: list[np.ndarray], iou_threshold: float = 0.4
) -> list[np.ndarray]:
    if not quads:
        return []

    def bbox(q: np.ndarray) -> tuple[float, float, float, float]:
        return q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()

    def iou(qa: np.ndarray, qb: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = bbox(qa)
        bx1, by1, bx2, by2 = bbox(qb)
        inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * \
                max(0.0, min(ay2, by2) - max(ay1, by1))
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

    kept: list[np.ndarray] = []
    for q in quads:
        if not any(iou(q, k) > iou_threshold for k in kept):
            kept.append(q)
    return kept


# ---------------------------------------------------------------------------
# OCR — improved name strip extraction
# ---------------------------------------------------------------------------

def _ocr_card_name(
    card_img: np.ndarray,
    card_names_set: set[str],
    fuzzy_threshold: int,
) -> tuple[str, Optional[str], float, np.ndarray]:
    """
    Step 3 → 4/5: OCR the upper-left name strip of a corrected card image.

    Strip geometry:
      height = top NAME_ROW_FRACTION  of card  (catches full name bar)
      width  = left NAME_COL_FRACTION of strip  (excludes mana-cost symbols)

    Preprocessing:
      - 3× upscale
      - CLAHE contrast normalisation (handles gold / dark name bars)
      - Adaptive Gaussian threshold (robust to colour variation)

    Returns (raw_text, matched_name, confidence 0–1, processed strip image).
    """
    from rapidfuzz import fuzz, process  # noqa: PLC0415

    ch, cw = card_img.shape[:2]
    sh = max(1, int(ch * NAME_ROW_FRACTION))
    sw = max(1, int(cw * NAME_COL_FRACTION))
    strip = card_img[:sh, :sw]

    # Upscale for glyph resolution
    strip_up = cv2.resize(strip, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # CLAHE — normalises contrast on coloured name bars
    gray = cv2.cvtColor(strip_up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # Adaptive threshold — more robust than Otsu for varied backgrounds
    proc = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8,
    )

    results = _get_ocr().readtext(proc, detail=1, paragraph=False)
    raw_parts = [
        text for (_, text, conf) in results
        if conf >= OCR_MIN_CONFIDENCE and text.strip()
    ]
    raw_text = " ".join(raw_parts).strip()

    if not raw_text or not card_names_set:
        return raw_text, None, 0.0, proc

    match = process.extractOne(
        raw_text,
        card_names_set,
        scorer=fuzz.WRatio,
        score_cutoff=fuzzy_threshold,
    )
    if match:
        matched_name, score, _ = match
        return raw_text, matched_name, score / 100.0, proc

    return raw_text, None, 0.0, proc


# ---------------------------------------------------------------------------
# Annotation — one visual layer per detection step
# ---------------------------------------------------------------------------

def _annotate(
    frame: np.ndarray,
    raw_lines: list[tuple],
    quads: list[np.ndarray],
    detected: list[DetectedCard],
) -> np.ndarray:
    out = frame.copy()
    fh, fw = out.shape[:2]

    # ── Step 1: grey Hough lines ──────────────────────────────────────
    col_l, th_l = _STYLE[DetectionStep.LINES]
    for seg in raw_lines:
        cv2.line(out, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])),
                 col_l, th_l)

    # ── Step 2: orange card-shaped quads ─────────────────────────────
    col_q, th_q = _STYLE[DetectionStep.QUAD]
    for quad in quads:
        pts = quad.reshape(-1, 1, 2).astype(int)
        cv2.polylines(out, [pts], isClosed=True, color=col_q, thickness=th_q)
        for pt in quad.astype(int):
            cv2.circle(out, tuple(pt), 5, col_q, -1)

    # ── Steps 3 / 4 / 5: detected cards ──────────────────────────────
    for card in detected:
        cnt   = card.contour.reshape(-1, 1, 2).astype(int)
        quad_f = card.contour.astype("float32")
        cimg_h, cimg_w = card.card_image.shape[:2]

        # Always draw the name-strip overlay (step 3)
        col_s, _ = _STYLE[DetectionStep.STRIP]
        strip_poly = _name_strip_frame_quad(quad_f, cimg_h, cimg_w)
        sp = strip_poly.reshape(-1, 1, 2).astype(int)
        cv2.polylines(out, [sp], isClosed=True, color=col_s, thickness=2)

        if card.matched_name:
            # Step 5 — green outline + name label
            col, thick = _STYLE[DetectionStep.MATCHED]
            cv2.polylines(out, [cnt], isClosed=True, color=col, thickness=thick)
            for pt in card.contour.astype(int):
                cv2.circle(out, tuple(pt), 6, col, -1)

            label = card.matched_name[:34]
            x = int(card.contour[:, 0].min())
            y = int(card.contour[:, 1].min())
            (tw, th2), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            pad = 4
            cv2.rectangle(out,
                          (x, max(y - th2 - pad * 2, 0)),
                          (x + tw + pad * 2, y),
                          (0, 0, 0), -1)
            cv2.putText(out, label, (x + pad, max(y - pad, th2 + pad)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        else:
            # Step 4 — red outline + raw OCR snippet
            col, thick = _STYLE[DetectionStep.NO_MATCH]
            cv2.polylines(out, [cnt], isClosed=True, color=col, thickness=thick)
            if card.raw_ocr_text:
                x = int(card.contour[:, 0].min())
                y = int(card.contour[:, 1].min())
                snippet = card.raw_ocr_text[:28] + "?"
                cv2.putText(out, snippet, (x, max(y - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # ── Status panel (bottom-left) ────────────────────────────────────
    matched_n = sum(1 for c in detected if c.matched_name)
    status = [
        (col_l,                          f"Lines : {len(raw_lines)}"),
        (col_q,                          f"Quads : {len(quads)}"),
        (_STYLE[DetectionStep.MATCHED][0], f"Cards : {matched_n}"),
    ]
    for i, (color, text) in enumerate(status):
        yp = fh - 14 - (len(status) - 1 - i) * 20
        cv2.putText(out, text, (10, yp),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # ── Legend (top-right) ────────────────────────────────────────────
    legend = [
        (DetectionStep.LINES,    "Lines detected"),
        (DetectionStep.QUAD,     "Card quad"),
        (DetectionStep.STRIP,    "Name strip (OCR)"),
        (DetectionStep.NO_MATCH, "No match"),
        (DetectionStep.MATCHED,  "Matched"),
    ]
    for i, (step, label) in enumerate(legend):
        col, _ = _STYLE[step]
        yp = 18 + i * 20
        cv2.circle(out, (fw - 140, yp - 4), 5, col, -1)
        cv2.putText(out, label, (fw - 130, yp),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Scanner thread
# ---------------------------------------------------------------------------

class CardScanner:
    """
    Runs video capture and card detection in a background daemon thread.

    frame_queue      — annotated JPEG bytes for the MJPEG stream
    card_event_queue — lists of DetectedCard pushed when new cards are found
    """

    def __init__(self, card_names: list[str]) -> None:
        self.card_names_set: set[str]    = set(card_names)
        self.fuzzy_threshold: int         = FUZZY_MATCH_THRESHOLD

        self.frame_queue:      queue.Queue[bytes]              = queue.Queue(maxsize=4)
        self.card_event_queue: queue.Queue[list[DetectedCard]] = queue.Queue(maxsize=32)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._recent_names: dict[str, float] = {}
        self._recent_ttl = 4.0

        self._video_source: int | str = VIDEO_SOURCE
        self._pending_source: int | str | None = None  # set by switch_source()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="CardScanner"
        )
        self._thread.start()
        log.info("CardScanner started (source=%s)", VIDEO_SOURCE)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        log.info("CardScanner stopped.")

    def update_card_names(self, names: list[str]) -> None:
        self.card_names_set = set(names)

    def switch_source(self, source: int | str) -> None:
        """Request a camera switch. Takes effect at the next frame boundary."""
        self._pending_source = source
        log.info("Camera switch requested → %s", source)

    @property
    def current_source(self) -> int | str:
        return self._video_source

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        def _open(source: int | str) -> cv2.VideoCapture:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap

        cap = _open(self._video_source)
        if not cap.isOpened():
            log.error("Cannot open video source: %s", self._video_source)
            self._running = False
            return

        frame_count    = 0
        last_lines:    list[tuple]         = []
        last_quads:    list[np.ndarray]    = []
        last_detected: list[DetectedCard]  = []

        while self._running:
            # ── Handle pending camera switch ──────────────────────────
            if self._pending_source is not None:
                new_src = self._pending_source
                self._pending_source = None
                new_cap = _open(new_src)
                if new_cap.isOpened():
                    cap.release()
                    cap = new_cap
                    self._video_source = new_src
                    frame_count = 0
                    last_lines = []; last_quads = []; last_detected = []
                    log.info("Camera switched to %s", new_src)
                else:
                    new_cap.release()
                    log.warning("Could not open source %s; keeping %s",
                                new_src, self._video_source)

            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to read frame; retrying…")
                time.sleep(0.1)
                continue

            frame_count += 1

            if frame_count % FRAME_SKIP == 0:
                # Full detection: lines → quads → OCR
                raw_lines, quads = _detect_lines_and_quads(frame)
                last_lines    = raw_lines
                last_quads    = quads

                detected      = self._process_quads(frame, quads)
                last_detected = detected

                new_cards = self._filter_new(detected)
                if new_cards:
                    try:
                        self.card_event_queue.put_nowait(new_cards)
                    except queue.Full:
                        pass

            # Annotate every frame with the latest detection state
            annotated = _annotate(frame, last_lines, last_quads, last_detected)
            jpeg = self._encode_jpeg(annotated)

            try:
                self.frame_queue.put_nowait(jpeg)
            except queue.Full:
                pass

        cap.release()

    # ------------------------------------------------------------------
    # Processing helpers
    # ------------------------------------------------------------------

    def _process_quads(
        self, frame: np.ndarray, quads: list[np.ndarray]
    ) -> list[DetectedCard]:
        detected: list[DetectedCard] = []
        for quad in quads:
            try:
                card_img = _four_point_transform(frame, quad)
            except cv2.error:
                continue

            ch, cw = card_img.shape[:2]
            if cw > ch:                         # ensure portrait orientation
                card_img = cv2.rotate(card_img, cv2.ROTATE_90_CLOCKWISE)

            raw, matched, score, _ = _ocr_card_name(
                card_img, self.card_names_set, self.fuzzy_threshold
            )
            detected.append(DetectedCard(
                raw_ocr_text=raw,
                matched_name=matched,
                confidence=score,
                contour=quad.astype(int),
                card_image=card_img,
            ))
        return detected

    def _filter_new(self, detected: list[DetectedCard]) -> list[DetectedCard]:
        now = time.monotonic()
        new: list[DetectedCard] = []
        for card in detected:
            name = card.matched_name
            if not name:
                continue
            if now - self._recent_names.get(name, 0) > self._recent_ttl:
                self._recent_names[name] = now
                new.append(card)
        self._recent_names = {
            k: v for k, v in self._recent_names.items() if now - v < 60
        }
        return new

    @staticmethod
    def _encode_jpeg(frame: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        return bytes(buf) if ok else b""

    # ------------------------------------------------------------------
    # MJPEG stream
    # ------------------------------------------------------------------

    def latest_jpeg(self) -> Generator[bytes, None, None]:
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            try:
                jpeg = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            yield boundary + jpeg + b"\r\n"
