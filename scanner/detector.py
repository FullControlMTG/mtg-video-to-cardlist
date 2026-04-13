from __future__ import annotations

import logging
import math
import platform
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


# Keyed by a coarse centroid+size fingerprint; re-uses the last OCR result while
# a card stays in roughly the same position to avoid redundant inference.
_quad_cache: dict[str, tuple[str, Optional[str], float, float]] = {}
_QUAD_CACHE_TTL    = 3.0   # seconds
_QUAD_CACHE_GRID   = 25    # px — centroid snap
_QUAD_CACHE_WGRID  = 40    # px — size snap


def _quad_key(quad: np.ndarray) -> str:
    cx = round(float(quad[:, 0].mean()) / _QUAD_CACHE_GRID) * _QUAD_CACHE_GRID
    cy = round(float(quad[:, 1].mean()) / _QUAD_CACHE_GRID) * _QUAD_CACHE_GRID
    w  = round(float(quad[:, 0].max() - quad[:, 0].min()) / _QUAD_CACHE_WGRID) * _QUAD_CACHE_WGRID
    h  = round(float(quad[:, 1].max() - quad[:, 1].min()) / _QUAD_CACHE_WGRID) * _QUAD_CACHE_WGRID
    return f"{cx},{cy},{w},{h}"


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
    return raw_lines, all_quads


def _ocr_card_name(
    card_img: np.ndarray,
    card_names_set: set[str],
    fuzzy_threshold: int,
) -> tuple[str, Optional[str], float, np.ndarray]:
    from rapidfuzz import fuzz, process  # noqa: PLC0415

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
        cimg_h, cimg_w = card.card_image.shape[:2]

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


class CardScanner:
    """Background thread for video capture and card detection."""

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
        self._pending_source: int | str | None = None
        self._rotation: int = 0  # degrees: 0, 90, 180, or 270

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="CardScanner"
        )
        self._thread.start()
        log.info("CardScanner started (source=%s)", self._video_source)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        log.info("CardScanner stopped.")

    def update_card_names(self, names: list[str]) -> None:
        self.card_names_set = set(names)

    def switch_source(self, source: int | str) -> None:
        self._pending_source = source
        log.info("Camera switch requested → %s", source)

    def set_rotation(self, degrees: int) -> None:
        """Set feed rotation. degrees must be 0, 90, 180, or 270."""
        self._rotation = degrees % 360
        log.info("Camera rotation set to %d°", self._rotation)

    @property
    def current_source(self) -> int | str:
        return self._video_source

    @property
    def rotation(self) -> int:
        return self._rotation

    def _run(self) -> None:
        def _open(source: int | str) -> cv2.VideoCapture:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ret, _ = cap.read()
                if not ret:
                    cap.release()
                    return cv2.VideoCapture()  # return closed cap
            return cap

        def _open_any() -> tuple[cv2.VideoCapture, int | str]:
            """Try the configured source first, then probe 0–7 for any working camera."""
            cap = _open(self._video_source)
            if cap.isOpened():
                return cap, self._video_source
            log.warning("Source %s unavailable; probing for any camera…", self._video_source)
            for idx in range(8):
                if idx == self._video_source:
                    continue
                cap = _open(idx)
                if cap.isOpened():
                    log.info("Falling back to camera %d", idx)
                    return cap, idx
            return cv2.VideoCapture(), self._video_source

        cap, active_source = _open_any()
        if not cap.isOpened():
            log.error("No working camera found.")
            self._running = False
            return
        self._video_source = active_source

        frame_count    = 0
        last_lines:    list[tuple]        = []
        last_quads:    list[np.ndarray]   = []
        last_detected: list[DetectedCard] = []

        while self._running:
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

            frame = self._rotate_frame(frame)
            frame_count += 1

            if frame_count % FRAME_SKIP == 0:
                raw_lines, quads = _detect_card_candidates(frame)
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

            annotated = _annotate(frame, last_lines, last_quads, last_detected)
            jpeg = self._encode_jpeg(annotated)

            try:
                self.frame_queue.put_nowait(jpeg)
            except queue.Full:
                pass

        cap.release()

    def _rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        if self._rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self._rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self._rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _process_quads(
        self, frame: np.ndarray, quads: list[np.ndarray]
    ) -> list[DetectedCard]:
        now = time.monotonic()
        detected: list[DetectedCard] = []

        for quad in quads:
            key = _quad_key(quad)
            cached = _quad_cache.get(key)
            if cached and cached[3] > now:
                raw, matched, score, _ = cached
                detected.append(DetectedCard(
                    raw_ocr_text=raw,
                    matched_name=matched,
                    confidence=score,
                    contour=quad.astype(int),
                    card_image=np.zeros((1, 1, 3), dtype=np.uint8),
                ))
                continue

            try:
                card_img = _four_point_transform(frame, quad)
            except cv2.error:
                continue

            raw, matched, score, _ = _ocr_card_name(
                card_img, self.card_names_set, self.fuzzy_threshold
            )
            _quad_cache[key] = (raw, matched, score, now + _QUAD_CACHE_TTL)

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

    def latest_jpeg(self) -> Generator[bytes, None, None]:
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            try:
                jpeg = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            yield boundary + jpeg + b"\r\n"
