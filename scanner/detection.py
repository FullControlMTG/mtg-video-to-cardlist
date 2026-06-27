"""Card detection: turn a frame into (lines, quads). Two complementary
detectors — Hough-line intersection and contour approximation — run and
their results are deduplicated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from config import (
    CARD_ASPECT_MAX,
    CARD_ASPECT_MIN,
    MAX_CARD_AREA_FRACTION,
    MIN_CARD_AREA,
)

from .geometry import (
    adaptive_canny,
    cluster_by_angle,
    deduplicate_quads,
    group_to_seg,
    intersect,
    order_points,
    segment_angle,
    split_into_line_groups,
)


# ── Annotation styling (also used by the annotate module) ────────────────────

class DetectionStep(Enum):
    LINES    = "lines"
    QUAD     = "quad"
    STRIP    = "strip"
    NO_MATCH = "no_match"
    MATCHED  = "matched"


STYLE: dict[DetectionStep, tuple[tuple[int, int, int], int]] = {
    DetectionStep.LINES:    ((90,  90,  90),  1),
    DetectionStep.QUAD:     ((0,  165, 255),  2),
    DetectionStep.STRIP:    ((0,  255, 255),  2),
    DetectionStep.NO_MATCH: ((0,   60, 220),  2),
    DetectionStep.MATCHED:  ((0,  210,   0),  3),
}


# ── Per-frame detection results passed from detect loop → annotate ───────────

@dataclass
class DetectedCard:
    raw_ocr_text: str
    matched_name: Optional[str]
    confidence: float
    contour: np.ndarray
    card_image: np.ndarray


@dataclass
class ScanFrame:
    """Wire-format for a processed frame. Kept for API compatibility."""
    jpeg_bytes: bytes
    detected: list[DetectedCard] = field(default_factory=list)
    timestamp: float = 0.0


# ── Line/quad detection ──────────────────────────────────────────────────────

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
    edges  = adaptive_canny(smooth)

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
         segment_angle(x1, y1, x2, y2))
        for x1, y1, x2, y2 in raw[:, 0]
    ]

    clusters = [c for c in cluster_by_angle(segments) if len(c) >= 2]

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

            # Sub-cluster each direction by spatial position. Each resulting
            # group represents segments that lie on the same physical line.
            # A valid card edge needs ≥2 segments — single-segment "lines" are
            # almost always background noise.
            gi = split_into_line_groups(ci, max(w, h))
            gj = split_into_line_groups(cj, max(w, h))
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
                return group_to_seg(ordered[0]), group_to_seg(ordered[-1]), \
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
                    pt = intersect(la, lb)
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
            ordered = order_points(pts)

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

    return segments, deduplicate_quads(quads)


def _detect_quads_contour(frame: np.ndarray) -> list[np.ndarray]:
    """Contour-based quad finder; runs on both the normal image and its
    photometric inverse to catch white-border cards on dark backgrounds and
    dark-border cards on light ones."""
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
            adaptive_canny(enhanced),
            cv2.Canny(enhanced, 15, 40),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
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

            quad = order_points(approx.reshape(4, 2).astype("float32"))

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

    return deduplicate_quads(quads)


def detect_card_candidates(
    frame: np.ndarray,
) -> tuple[list[tuple], list[np.ndarray]]:
    """Top-level frame detector. Returns (raw_lines, quads) where quads are
    the deduplicated, full-frame-rejected card candidates."""
    raw_lines, hough_quads = _detect_lines_and_quads(frame)
    contour_quads          = _detect_quads_contour(frame)
    all_quads = deduplicate_quads(hough_quads + contour_quads)

    # Drop quads that span essentially the whole frame — the frame border /
    # background being read as one big rectangle. A real card leaves a margin.
    h, w = frame.shape[:2]
    max_area = MAX_CARD_AREA_FRACTION * w * h
    all_quads = [
        q for q in all_quads
        if cv2.contourArea(q.astype(np.float32)) <= max_area
    ]
    return raw_lines, all_quads
