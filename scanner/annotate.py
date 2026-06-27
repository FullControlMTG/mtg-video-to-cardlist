"""Draw the live overlay (lines, quads, name strip, status text, legend)
onto a captured frame. Pure presentation — no detection logic here.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detection import STYLE, DetectedCard, DetectionStep
from .geometry import name_strip_frame_quad


def annotate(
    frame: np.ndarray,
    raw_lines: list[tuple],
    quads: list[np.ndarray],
    detected: list[DetectedCard],
) -> np.ndarray:
    out = frame.copy()
    fh, fw = out.shape[:2]

    col_l, th_l = STYLE[DetectionStep.LINES]
    for seg in raw_lines:
        cv2.line(out, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])),
                 col_l, th_l)

    col_q, th_q = STYLE[DetectionStep.QUAD]
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

        col_s, _ = STYLE[DetectionStep.STRIP]
        strip_poly = name_strip_frame_quad(quad_f, cimg_h, cimg_w)
        sp = strip_poly.reshape(-1, 1, 2).astype(int)
        cv2.polylines(out, [sp], isClosed=True, color=col_s, thickness=2)

        if card.matched_name:
            col, thick = STYLE[DetectionStep.MATCHED]
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
            col, thick = STYLE[DetectionStep.NO_MATCH]
            cv2.polylines(out, [cnt], isClosed=True, color=col, thickness=thick)
            if card.raw_ocr_text:
                x = int(card.contour[:, 0].min())
                y = int(card.contour[:, 1].min())
                snippet = card.raw_ocr_text[:28] + "?"
                cv2.putText(out, snippet, (x, max(y - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    matched_n = sum(1 for c in detected if c.matched_name)
    status = [
        (col_l,                           f"Lines : {len(raw_lines)}"),
        (col_q,                           f"Quads : {len(quads)}"),
        (STYLE[DetectionStep.MATCHED][0], f"Cards : {matched_n}"),
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
    # Wider, larger legend so it's actually readable on the live feed.
    lf_scale = 0.6
    lf_thick = 1
    row_h    = 26
    pad      = 8
    label_w  = max(
        cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, lf_scale, lf_thick)[0][0]
        for _, lbl in legend
    )
    panel_w  = label_w + 36
    panel_h  = row_h * len(legend) + pad
    px0      = fw - panel_w - 10
    py0      = 10
    overlay  = out.copy()
    cv2.rectangle(overlay, (px0, py0), (px0 + panel_w, py0 + panel_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    for i, (step, label) in enumerate(legend):
        col, _ = STYLE[step]
        yp = py0 + pad + i * row_h + 6
        cv2.circle(out, (px0 + 14, yp + 3), 6, col, -1)
        cv2.putText(out, label, (px0 + 28, yp + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, lf_scale, (230, 230, 230),
                    lf_thick, cv2.LINE_AA)

    return out
