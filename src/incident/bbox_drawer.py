"""src/incident/bbox_drawer.py — Standalone bbox drawing helpers."""

import cv2
import numpy as np
from typing import List, Dict, Tuple

PALETTE: Dict[str, Tuple[int, int, int]] = {
    "person":  (0,  60, 232),
    "knife":   (0, 128, 255),
    "gun":     (0,   0, 180),
    "car":     (255, 128, 0),
    "truck":   (255, 165, 0),
    "default": (0,  200,  50),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_boxes(
    frame:      np.ndarray,
    detections: List[Dict],
    show_conf:  bool = True,
    show_occ:   bool = True,
    line_width: int  = 2,
) -> np.ndarray:
    """Draw all detections onto *frame* (in-place copy returned)."""
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        label    = d.get("label", "obj")
        conf     = d.get("confidence", 0.0)
        occluded = d.get("occluded", False)
        color    = PALETTE.get(label, PALETTE["default"])

        if occluded and show_occ:
            _dashed_rect(out, x1, y1, x2, y2, color, 1)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, line_width)

        if show_conf:
            text = f"{label} {conf:.0%}" + (" [OCC]" if occluded else "")
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, text, (x1 + 2, y1 - 4),
                        FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_tracks(frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
    """Draw track IDs and trajectory tails."""
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
        tid   = t.get("id", 0)
        label = t.get("label", "obj")
        color = PALETTE.get(label, PALETTE["default"])

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"ID {tid}", (x1, y1 - 6),
                    FONT, 0.45, color, 1, cv2.LINE_AA)

        # Trajectory tail
        traj = t.get("trajectory", [])
        for i in range(1, len(traj)):
            p1 = (int(traj[i-1][0]), int(traj[i-1][1]))
            p2 = (int(traj[i][0]),   int(traj[i][1]))
            alpha = i / len(traj)
            c = tuple(int(v * alpha) for v in color)
            cv2.line(out, p1, p2, c, 1)
    return out


def _dashed_rect(frame, x1, y1, x2, y2, color, thickness, gap=8):
    for i in range(x1, x2, gap):
        cv2.line(frame, (i, y1), (min(i + gap//2, x2), y1), color, thickness)
        cv2.line(frame, (i, y2), (min(i + gap//2, x2), y2), color, thickness)
    for i in range(y1, y2, gap):
        cv2.line(frame, (x1, i), (x1, min(i + gap//2, y2)), color, thickness)
        cv2.line(frame, (x2, i), (x2, min(i + gap//2, y2)), color, thickness)
