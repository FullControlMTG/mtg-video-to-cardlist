"""Pure 2-D geometry helpers used by the card detector.

No OpenCV-side-effects, no I/O, no threading — just functions over numpy
arrays and tuples. Kept separate so the detection pipeline modules can
import these without dragging in OCR or camera dependencies.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from config import NAME_COL_FRACTION, NAME_ROW_FRACTION


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return the 4 input points sorted as [top-left, top-right, bottom-right, bottom-left]."""
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Perspective-unwarp `image` so the quad `pts` becomes a flat rectangle."""
    rect = order_points(pts)
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


def expand_quad(quad: np.ndarray, margin: float, top_extra: float = 0.0) -> np.ndarray:
    """Grow a quad before warping so the OCR crop reliably contains the title.

    `margin` scales the quad outward from its centre on all sides (e.g. 0.08),
    recovering the black border that edge detection tends to crop inside of.

    `top_extra` then pushes the TOP edge further up, by that fraction of the
    card's height. Detection (especially on a stack of bordered cards) often
    lands at or below the title, so without this the band clips the title top;
    extending upward makes the band reach above where detection landed. Corners
    that fall outside the frame warp to black, which is harmless.
    """
    q = order_points(quad)  # tl, tr, br, bl
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


def name_strip_frame_quad(
    quad: np.ndarray, card_h: int, card_w: int,
) -> np.ndarray:
    """Project the card's name-strip rectangle back into the original frame's
    coordinate space so it can be drawn as an overlay polygon."""
    rect = order_points(quad.astype("float32"))
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


# ── Hough-segment helpers (used by the line-based quad finder) ───────────────

def segment_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180


def cluster_by_angle(
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


def split_into_line_groups(
    cluster: list[tuple], frame_size: float,
) -> list[list[tuple]]:
    """Split an angle cluster into spatially coherent sub-groups.

    Segments at the same angle but on opposite sides of the frame (e.g. a
    shadow and a window ledge) belong to different physical lines. This
    function sub-clusters by perpendicular offset so each group contains only
    segments that lie on the same line. Groups with fewer than 2 segments are
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


def group_to_seg(group: list[tuple]) -> tuple:
    """Synthesise a representative segment for a line group.

    Uses the average midpoint and average angle of all segments, then extends
    ±200 px in the line direction. This gives a cleaner line equation for
    intersection than picking any single (potentially noisy) segment.
    """
    avg_angle = sum(s[4] for s in group) / len(group)
    mid_x = sum((s[0] + s[2]) / 2 for s in group) / len(group)
    mid_y = sum((s[1] + s[3]) / 2 for s in group) / len(group)
    dx = math.cos(math.radians(avg_angle)) * 200
    dy = math.sin(math.radians(avg_angle)) * 200
    return (mid_x - dx, mid_y - dy, mid_x + dx, mid_y + dy, avg_angle)


def to_line_eq(seg: tuple) -> tuple[float, float, float]:
    x1, y1, x2, y2 = seg[0], seg[1], seg[2], seg[3]
    a = float(y2 - y1)
    b = float(x1 - x2)
    c = a * x1 + b * y1
    return a, b, c


def intersect(s1: tuple, s2: tuple) -> Optional[tuple[float, float]]:
    a1, b1, c1 = to_line_eq(s1)
    a2, b2, c2 = to_line_eq(s2)
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    return (c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det


def adaptive_canny(gray: np.ndarray) -> np.ndarray:
    """Canny with thresholds derived from the image's own median so they
    self-calibrate to the scene's dynamic range — handles low-contrast card
    borders without blowing out bright scenes."""
    median = float(np.median(gray))
    sigma  = 0.33
    lo = max(10,      int((1.0 - sigma) * median))
    hi = max(lo * 2,  int((1.0 + sigma) * median))
    return cv2.Canny(gray, lo, hi)


def deduplicate_quads(
    quads: list[np.ndarray], iou_threshold: float = 0.4,
) -> list[np.ndarray]:
    """Greedy IoU-based dedupe of overlapping quads."""
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
