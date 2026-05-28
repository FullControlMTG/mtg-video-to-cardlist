from __future__ import annotations

import logging
import math
import platform
import queue
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generator, Optional

import cv2
import numpy as np

from config import (
    CAMERA_PROBE_MAX,
    CAMERA_READ_FAIL_LIMIT,
    CAMERA_RECONNECT_DELAY,
    CAMERA_WARMUP_TIMEOUT,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    CARD_ASPECT_MAX,
    CARD_ASPECT_MIN,
    CARD_BORDER_MARGIN,
    CARD_TOP_EXTRA,
    CONFIRM_MIN_MATCH,
    CONFIRM_WINDOW_SIZE,
    JPEG_QUALITY,
    MAX_CARD_AREA_FRACTION,
    MIN_CARD_AREA,
    NAME_COL_FRACTION,
    NAME_ROW_FRACTION,
    OCR_ENGINE,
    OCR_MIN_CONFIDENCE,
    VIDEO_SOURCE,
)

# The per-read matcher injected by the app: raw OCR text -> (card name | None, score).
Matcher = Callable[[str], "tuple[Optional[str], float]"]

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
    t = threading.Thread(target=_get_ocr, daemon=True, name="OCR-prewarm")
    t.start()


class DetectionStep(Enum):
    LINES    = "lines"
    QUAD     = "quad"
    STRIP    = "strip"
    NO_MATCH = "no_match"
    MATCHED  = "matched"


_STYLE: dict[DetectionStep, tuple[tuple[int, int, int], int]] = {
    DetectionStep.LINES:    ((90,  90,  90),  1),
    DetectionStep.QUAD:     ((0,  165, 255),  2),
    DetectionStep.STRIP:    ((0,  255, 255),  2),
    DetectionStep.NO_MATCH: ((0,   60, 220),  2),
    DetectionStep.MATCHED:  ((0,  210,   0),  3),
}


@dataclass
class DetectedCard:
    raw_ocr_text: str
    matched_name: Optional[str]
    confidence: float
    contour: np.ndarray
    card_image: np.ndarray


@dataclass
class ScanFrame:
    jpeg_bytes: bytes
    detected: list[DetectedCard] = field(default_factory=list)
    timestamp: float = field(default_factory=time.monotonic)


def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
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


def _expand_quad(quad: np.ndarray, margin: float, top_extra: float = 0.0) -> np.ndarray:
    """Grow a quad before warping so the OCR crop reliably contains the title.

    `margin` scales the quad outward from its centre on all sides (e.g. 0.08),
    recovering the black border that edge detection tends to crop inside of.

    `top_extra` then pushes the TOP edge further up, by that fraction of the
    card's height. Detection (especially on a stack of bordered cards) often
    lands at or below the title, so without this the band clips the title top;
    extending upward makes the band reach above where detection landed. Corners
    that fall outside the frame warp to black, which is harmless.
    """
    q = _order_points(quad)  # tl, tr, br, bl
    centre = q.mean(axis=0)
    q = centre + (q - centre) * (1.0 + margin)
    if top_extra:
        tl, tr, br, bl = q
        q = np.array([
            tl + (tl - bl) * top_extra,   # push top-left up along the left edge
            tr + (tr - br) * top_extra,   # push top-right up along the right edge
            br,
            bl,
        ], dtype="float32")
    return q


def _name_strip_frame_quad(
    quad: np.ndarray, card_h: int, card_w: int
) -> np.ndarray:
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


def _segment_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180


def _cluster_by_angle(
    segments: list[tuple],
    tolerance: float = 15.0,
) -> list[list[tuple]]:
    # Running-average centroid so later segments are compared against the true
    # cluster mean, not just its first member.
    clusters: list[list[tuple]] = []
    centroids: list[float] = []

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
            n = len(clusters[best_idx])
            centroids[best_idx] = (centroids[best_idx] * (n - 1) + angle) / n
        else:
            clusters.append([seg])
            centroids.append(angle)

    return clusters


def _split_into_line_groups(
    cluster: list[tuple], frame_size: float
) -> list[list[tuple]]:
    """
    Split an angle cluster into spatially coherent sub-groups.

    Segments at the same angle but on opposite sides of the frame (e.g. a
    shadow and a window ledge) belong to different physical lines.  This
    function sub-clusters by perpendicular offset so each group contains only
    segments that lie on the same line.  Groups with fewer than 2 segments are
    discarded as single-segment noise.
    """
    if not cluster:
        return []
    avg_angle = sum(s[4] for s in cluster) / len(cluster)
    perp = math.radians(avg_angle + 90.0)
    px, py = math.cos(perp), math.sin(perp)
    tol = frame_size * 0.05     # 5 % of frame — tight enough to separate lines

    groups:    list[list[tuple]] = []
    centroids: list[float]      = []

    for seg in cluster:
        proj = ((seg[0] + seg[2]) / 2) * px + ((seg[1] + seg[3]) / 2) * py
        best_idx, best_diff = -1, tol
        for k, c in enumerate(centroids):
            d = abs(proj - c)
            if d < best_diff:
                best_diff, best_idx = d, k
        if best_idx >= 0:
            groups[best_idx].append(seg)
            n = len(groups[best_idx])
            centroids[best_idx] = (centroids[best_idx] * (n - 1) + proj) / n
        else:
            groups.append([seg])
            centroids.append(proj)

    return [g for g in groups if len(g) >= 2]


def _group_to_seg(group: list[tuple]) -> tuple:
    """
    Synthesise a representative segment for a line group.

    Uses the average midpoint and average angle of all segments, then extends
    ±200 px in the line direction.  This gives a cleaner line equation for
    intersection than picking any single (potentially noisy) segment.
    """
    avg_angle = sum(s[4] for s in group) / len(group)
    mid_x = sum((s[0] + s[2]) / 2 for s in group) / len(group)
    mid_y = sum((s[1] + s[3]) / 2 for s in group) / len(group)
    dx = math.cos(math.radians(avg_angle)) * 200
    dy = math.sin(math.radians(avg_angle)) * 200
    return (mid_x - dx, mid_y - dy, mid_x + dx, mid_y + dy, avg_angle)


def _to_line_eq(seg: tuple) -> tuple[float, float, float]:
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


def _adaptive_canny(gray: np.ndarray) -> np.ndarray:
    # Thresholds derived from the image's own median so they self-calibrate to
    # the scene's dynamic range — handles low-contrast card borders (card on a
    # similar-coloured surface) without blowing out bright scenes.
    median = float(np.median(gray))
    sigma  = 0.33
    lo = max(10,      int((1.0 - sigma) * median))
    hi = max(lo * 2,  int((1.0 + sigma) * median))
    return cv2.Canny(gray, lo, hi)


def _detect_lines_and_quads(
    frame: np.ndarray,
) -> tuple[list[tuple], list[np.ndarray]]:
    h, w = frame.shape[:2]

    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Unsharp mask: boosts soft edges that appear when the camera hasn't yet
    # refocused on the card, without amplifying broad background gradients.
    _blur  = cv2.GaussianBlur(gray, (0, 0), 3)
    gray   = cv2.addWeighted(gray, 1.5, _blur, -0.5, 0)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray   = clahe.apply(gray)
    smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    edges  = _adaptive_canny(smooth)

    min_len = max(60, int(min(h, w) * 0.08))

    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=65,
        minLineLength=min_len,
        maxLineGap=40,
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

    quads:   list[np.ndarray]  = []
    checked: set[frozenset]    = set()
    min_sep = max(80.0, math.sqrt(MIN_CARD_AREA) * 0.5)
    margin  = min(w, h) * 0.12   # corners must be near the frame

    for i, ci in enumerate(clusters):
        for j, cj in enumerate(clusters):
            if j <= i:
                continue
            key = frozenset([i, j])
            if key in checked:
                continue
            checked.add(key)

            ai = _cluster_angle(ci)
            aj = _cluster_angle(cj)
            diff = min(abs(ai - aj) % 180, 180.0 - abs(ai - aj) % 180)
            if not (65.0 <= diff <= 115.0):
                continue

            # Sub-cluster each direction by spatial position.  Each resulting
            # group represents segments that lie on the same physical line.
            # A valid card edge needs ≥2 segments — single-segment "lines" are
            # almost always background noise.
            gi = _split_into_line_groups(ci, max(w, h))
            gj = _split_into_line_groups(cj, max(w, h))
            if len(gi) < 2 or len(gj) < 2:
                continue

            # Pick the two most-separated line groups in each direction and
            # build representative segments from their averaged geometry.
            # Using group averages rather than single extreme segments prevents
            # one background line from hijacking a card-edge cluster.
            def _boundary_groups(groups: list[list[tuple]], cluster_segs: list[tuple]):
                avg_a = _cluster_angle(cluster_segs)
                pr = math.radians(avg_a + 90.0)
                ppx, ppy = math.cos(pr), math.sin(pr)
                def gproj(g):
                    return sum(((s[0]+s[2])/2)*ppx + ((s[1]+s[3])/2)*ppy
                               for s in g) / len(g)
                ordered = sorted(groups, key=gproj)
                return _group_to_seg(ordered[0]), _group_to_seg(ordered[-1]), \
                       abs(gproj(ordered[-1]) - gproj(ordered[0]))

            li_a, li_b, sep_i = _boundary_groups(gi, ci)
            lj_a, lj_b, sep_j = _boundary_groups(gj, cj)

            if sep_i < min_sep or sep_j < min_sep:
                continue

            # Pre-check aspect ratio from the line separations before the
            # more expensive corner-computation step.
            if sep_i > 0 and sep_j > 0:
                ar_pre = min(sep_i, sep_j) / max(sep_i, sep_j)
                if not (CARD_ASPECT_MIN <= ar_pre <= CARD_ASPECT_MAX):
                    continue

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

            if any(
                cx < -margin or cx > w + margin or cy < -margin or cy > h + margin
                for cx, cy in corners
            ):
                continue

            pts     = np.array(corners, dtype="float32")
            ordered = _order_points(pts)

            if cv2.contourArea(ordered) < MIN_CARD_AREA:
                continue

            if not cv2.isContourConvex(ordered.reshape(-1, 1, 2).astype(int)):
                continue

            tl, tr, br, bl = ordered
            top_w  = np.linalg.norm(tr - tl)
            bot_w  = np.linalg.norm(br - bl)
            left_h = np.linalg.norm(bl - tl)
            rgt_h  = np.linalg.norm(br - tr)

            # Opposite edges of a real card are equal length; large asymmetry
            # means we intersected lines from two different objects.
            if min(top_w, bot_w) > 0 and max(top_w, bot_w) / min(top_w, bot_w) > 1.4:
                continue
            if min(left_h, rgt_h) > 0 and max(left_h, rgt_h) / min(left_h, rgt_h) > 1.4:
                continue

            width_px  = max(top_w, bot_w)
            height_px = max(left_h, rgt_h)
            if height_px < 1:
                continue
            if not (CARD_ASPECT_MIN <= width_px / height_px <= CARD_ASPECT_MAX):
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


def _detect_quads_contour(frame: np.ndarray) -> list[np.ndarray]:
    # Runs on both the normal image and its photometric inverse to catch
    # white-border cards on dark backgrounds and dark-border cards on light ones.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Unsharp mask: same rationale as in _detect_lines_and_quads — compensates
    # for soft-focus images so that card borders survive edge detection.
    _blur = cv2.GaussianBlur(gray, (0, 0), 3)
    gray  = cv2.addWeighted(gray, 1.5, _blur, -0.5, 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    quads: list[np.ndarray] = []

    for enhanced in (gray, cv2.bitwise_not(gray)):
        # Adaptive Canny for normal contrast; fixed low thresholds as a second
        # pass to catch soft borders that the median-derived thresholds miss.
        edges = cv2.bitwise_or(
            _adaptive_canny(enhanced),
            cv2.Canny(enhanced, 15, 40),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
            area = cv2.contourArea(cnt)
            if area < MIN_CARD_AREA:
                break

            peri   = cv2.arcLength(cnt, True)
            # 0.04 (up from 0.03) tolerates slightly imperfect contours from
            # soft-focus frames that approximate to more than 4 vertices.
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) != 4:
                continue

            quad = _order_points(approx.reshape(4, 2).astype("float32"))

            if not cv2.isContourConvex(quad.reshape(-1, 1, 2).astype(int)):
                continue

            tl, tr, br, bl = quad
            cw = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
            ch = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
            if ch < 1:
                continue
            ar = cw / ch
            if not (CARD_ASPECT_MIN <= ar <= CARD_ASPECT_MAX):
                continue

            quads.append(quad)

    return _deduplicate_quads(quads)


def _detect_card_candidates(
    frame: np.ndarray,
) -> tuple[list[tuple], list[np.ndarray]]:
    raw_lines, hough_quads = _detect_lines_and_quads(frame)
    contour_quads          = _detect_quads_contour(frame)
    all_quads = _deduplicate_quads(hough_quads + contour_quads)

    # Drop quads that span essentially the whole frame — the frame border /
    # background being read as one big rectangle. A real card leaves a margin.
    h, w = frame.shape[:2]
    max_area = MAX_CARD_AREA_FRACTION * w * h
    all_quads = [
        q for q in all_quads
        if cv2.contourArea(q.astype(np.float32)) <= max_area
    ]
    return raw_lines, all_quads


def _ocr_name_strip(card_img: np.ndarray) -> str:
    """OCR the card's name strip and return the raw text.

    The raw text is then fuzzy-matched to a card name (the injected matcher), and
    the Confirmer votes on the resolved names — not on this noisy raw string.
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


def _annotate(
    frame: np.ndarray,
    raw_lines: list[tuple],
    quads: list[np.ndarray],
    detected: list[DetectedCard],
) -> np.ndarray:
    out = frame.copy()
    fh, fw = out.shape[:2]

    col_l, th_l = _STYLE[DetectionStep.LINES]
    for seg in raw_lines:
        cv2.line(out, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])),
                 col_l, th_l)

    col_q, th_q = _STYLE[DetectionStep.QUAD]
    for quad in quads:
        pts = quad.reshape(-1, 1, 2).astype(int)
        cv2.polylines(out, [pts], isClosed=True, color=col_q, thickness=th_q)
        for pt in quad.astype(int):
            cv2.circle(out, tuple(pt), 5, col_q, -1)

    for card in detected:
        cnt    = card.contour.reshape(-1, 1, 2).astype(int)
        quad_f = card.contour.astype("float32")
        # The name-strip region is a fixed fraction of the quad, so use the
        # quad's own bounding-box dimensions (the strip math is scale-invariant;
        # it just needs non-zero dims — the old code read these off the OCR crop).
        cimg_w = max(2, int(card.contour[:, 0].max() - card.contour[:, 0].min()))
        cimg_h = max(2, int(card.contour[:, 1].max() - card.contour[:, 1].min()))

        col_s, _ = _STYLE[DetectionStep.STRIP]
        strip_poly = _name_strip_frame_quad(quad_f, cimg_h, cimg_w)
        sp = strip_poly.reshape(-1, 1, 2).astype(int)
        cv2.polylines(out, [sp], isClosed=True, color=col_s, thickness=2)

        if card.matched_name:
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
            col, thick = _STYLE[DetectionStep.NO_MATCH]
            cv2.polylines(out, [cnt], isClosed=True, color=col, thickness=thick)
            if card.raw_ocr_text:
                x = int(card.contour[:, 0].min())
                y = int(card.contour[:, 1].min())
                snippet = card.raw_ocr_text[:28] + "?"
                cv2.putText(out, snippet, (x, max(y - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    matched_n = sum(1 for c in detected if c.matched_name)
    status = [
        (col_l,                            f"Lines : {len(raw_lines)}"),
        (col_q,                            f"Quads : {len(quads)}"),
        (_STYLE[DetectionStep.MATCHED][0], f"Cards : {matched_n}"),
    ]
    for i, (color, text) in enumerate(status):
        yp = fh - 14 - (len(status) - 1 - i) * 20
        cv2.putText(out, text, (10, yp),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if len(quads) == 0:
        if len(raw_lines) == 0:
            hint = "No edges found — try plain background / adjust lighting"
            hint_col = (100, 100, 100)
        else:
            hint = "Edges found — try adjusting card angle or distance"
            hint_col = (0, 165, 255)
        (tw, th2), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        hx = (fw - tw) // 2
        hy = fh - 10
        cv2.rectangle(out, (hx - 6, hy - th2 - 4), (hx + tw + 6, hy + 4),
                      (0, 0, 0), -1)
        cv2.putText(out, hint, (hx, hy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, hint_col, 1, cv2.LINE_AA)
    elif matched_n == 0:
        hint = "Shape found — reading card name…"
        (tw, th2), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        hx = (fw - tw) // 2
        hy = fh - 10
        cv2.rectangle(out, (hx - 6, hy - th2 - 4), (hx + tw + 6, hy + 4),
                      (0, 0, 0), -1)
        cv2.putText(out, hint, (hx, hy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col_q, 1, cv2.LINE_AA)

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


_EMPTY_IMG = np.zeros((1, 1, 3), dtype=np.uint8)


@dataclass
class Detection:
    """A confirmed card — the resolved Scryfall name and its vote confidence."""
    name: str
    confidence: float


# ── Detection confirmation: ONE global name vote (position-independent) ──────
class Confirmer:
    """Confirms the card by voting on the FUZZY-MATCHED card NAME over recent
    frames — globally, not per detected box.

    Why global: a hand-held stack jitters, so keying votes to a box position
    fragments them across many short-lived tracks and nothing ever reaches the
    threshold. Since the user presents one card at a time, a single rolling vote
    over the last CONFIRM_WINDOW_SIZE (N) frames is robust to that jitter: each
    frame contributes the best matched name (or None), and a name is confirmed
    once it wins CONFIRM_MIN_MATCH (M) of those frames. Voting on the matched
    name (not raw OCR) means noisy reads ("Lightning Bo1t") still count toward
    the right card. A different card taking over re-confirms; an empty window
    (card removed) clears the state so it can re-confirm later.
    """

    def __init__(self) -> None:
        self.recent: deque[Optional[str]] = deque(maxlen=CONFIRM_WINDOW_SIZE)
        self.candidate: Optional[str] = None   # current best guess (for overlay)
        self.confirmed: Optional[str] = None    # currently confirmed name (green)
        self.confidence = 0.0
        self._emitted: Optional[str] = None

    def add(self, name: Optional[str]) -> Optional[str]:
        """Record this frame's best matched name; return a name iff NEWLY confirmed."""
        self.recent.append(name)
        votes = Counter(n for n in self.recent if n)
        if not votes:
            self.candidate = self.confirmed = self._emitted = None
            self.confidence = 0.0
            return None

        top, count = votes.most_common(1)[0]
        self.candidate = top
        self.confidence = count / len(self.recent)

        # Forget a prior confirmation once that card stops appearing.
        if self._emitted and self._emitted not in votes:
            self._emitted = None
            self.confirmed = None

        if count >= CONFIRM_MIN_MATCH:
            self.confirmed = top
            if top != self._emitted:
                self._emitted = top
                return top
        return None


# ── Camera: threaded latest-frame grabber + lifecycle state machine ──────────
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


class CardScanner:
    """Orchestrates the camera grabber, the async detection/confirmation loop,
    and the annotated MJPEG stream.

    `matcher` (injected) resolves each OCR read to a card name; the detection
    thread votes on those names and emits the confirmed card on
    ``detection_queue``. Injecting it keeps the scanner decoupled from Scryfall
    while still correcting per-read OCR noise before the vote.
    """

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
            self._publish(self._encode_jpeg(_annotate(frame, lines, quads, detected)))

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

            raw_lines, quads = _detect_card_candidates(frame)
            overlay: list[DetectedCard] = []
            confirmed_name: Optional[str] = None

            if quads:
                # The presented (top) card is the largest valid quad.
                largest = max(quads, key=lambda q: cv2.contourArea(q.astype(np.float32)))
                ocr_quad = _expand_quad(largest, CARD_BORDER_MARGIN, CARD_TOP_EXTRA)

                raw = ""
                try:
                    raw = _ocr_name_strip(_four_point_transform(frame, ocr_quad))
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
