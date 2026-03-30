"""
src/utils/visualization.py
Drawing helpers: anomaly heatmap, timeline image, confidence chart.
All functions return numpy BGR images or write to disk.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


# ── colour constants (BGR) ────────────────────────────────────────────────────
C_RED    = (0,   60, 232)
C_AMBER  = (0,  165, 245)
C_TEAL   = (177, 201,   0)
C_BLUE   = (232, 130,  59)
C_WHITE  = (240, 240, 240)
C_DARK   = ( 18,  24,  32)
C_GRAY   = (100, 130, 150)


def draw_anomaly_heatmap(
    scores: List[float],
    width: int = 800,
    height: int = 120,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return a BGR image showing the anomaly score over time."""
    img = np.full((height, width, 3), C_DARK, dtype=np.uint8)
    n   = len(scores)
    if n == 0:
        return img

    # Grid lines
    for pct in [0.25, 0.50, 0.75, 1.0]:
        y = int(height - pct * (height - 20))
        cv2.line(img, (0, y), (width, y), (40, 50, 60), 1)

    # Threshold line
    thresh_y = int(height - threshold * (height - 20))
    cv2.line(img, (0, thresh_y), (width, thresh_y), C_AMBER, 1)
    cv2.putText(img, f"thr={threshold:.2f}", (width - 90, thresh_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_AMBER, 1)

    # Score polygon fill
    pts = [(0, height)]
    for i, s in enumerate(scores):
        x = int(i / max(n - 1, 1) * (width - 1))
        y = int(height - s * (height - 20))
        pts.append((x, y))
    pts.append((width - 1, height))

    poly     = np.array(pts, dtype=np.int32)
    fill_img = img.copy()
    cv2.fillPoly(fill_img, [poly], (20, 40, 80))
    cv2.addWeighted(fill_img, 0.55, img, 0.45, 0, img)

    # Score line
    for i in range(1, n):
        x1 = int((i - 1) / max(n - 1, 1) * (width - 1))
        x2 = int(i       / max(n - 1, 1) * (width - 1))
        y1 = int(height - scores[i - 1] * (height - 20))
        y2 = int(height - scores[i]     * (height - 20))
        color = C_RED if scores[i] > threshold else C_TEAL
        cv2.line(img, (x1, y1), (x2, y2), color, 2)

    # Labels
    cv2.putText(img, "Anomaly Score Timeline", (8, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1)
    return img


def draw_incident_timeline(
    incidents: List[Dict],
    duration: float,
    width: int = 800,
    height: int = 80,
) -> np.ndarray:
    """Return a BGR image showing coloured incident spans on a timeline bar."""
    img = np.full((height, width, 3), C_DARK, dtype=np.uint8)
    if duration <= 0:
        return img

    bar_y1, bar_y2 = 30, 55

    # Background bar
    cv2.rectangle(img, (0, bar_y1), (width, bar_y2), (40, 55, 70), -1)

    # Incident spans
    for inc in incidents:
        x1 = int(inc["start_time"] / duration * width)
        x2 = int(inc["end_time"]   / duration * width)
        x2 = max(x2, x1 + 3)
        cv2.rectangle(img, (x1, bar_y1), (x2, bar_y2), C_RED, -1)

        # Tick labels
        m, s = divmod(int(inc["start_time"]), 60)
        cv2.putText(img, f"{m:02d}:{s:02d}", (max(x1 - 4, 0), bar_y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_RED, 1)

    # Duration markers
    for pct in [0, 0.25, 0.5, 0.75, 1.0]:
        x  = int(pct * (width - 1))
        ts = pct * duration
        m, s = divmod(int(ts), 60)
        cv2.line(img, (x, bar_y2), (x, bar_y2 + 6), C_GRAY, 1)
        cv2.putText(img, f"{m:02d}:{s:02d}", (max(x - 14, 0), height - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_GRAY, 1)

    cv2.putText(img, "Incident Timeline", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_WHITE, 1)
    return img


def draw_confidence_bars(
    incidents: List[Dict],
    width: int = 400,
    row_height: int = 36,
) -> np.ndarray:
    """Return a BGR image with horizontal confidence bars for each incident."""
    n      = max(len(incidents), 1)
    height = n * row_height + 20
    img    = np.full((height, width, 3), C_DARK, dtype=np.uint8)

    for i, inc in enumerate(incidents):
        y   = i * row_height + 10
        bar_w = int(inc["confidence"] * (width - 120))
        # Bar background
        cv2.rectangle(img, (100, y + 6), (width - 10, y + row_height - 8), (40, 55, 70), -1)
        # Bar fill
        color = C_RED if inc["confidence"] > 0.75 else (C_AMBER if inc["confidence"] > 0.5 else C_TEAL)
        cv2.rectangle(img, (100, y + 6), (100 + bar_w, y + row_height - 8), color, -1)
        # Label
        label = f"#{inc['id']} {inc['type'][:8]}"
        cv2.putText(img, label, (4, y + row_height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_WHITE, 1)
        # Pct
        cv2.putText(img, f"{inc['confidence']:.0%}",
                    (100 + bar_w + 4, y + row_height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
    return img


def make_summary_image(
    results: Dict,
    output_path: Optional[str] = None,
    width: int = 820,
) -> np.ndarray:
    """
    Composite a summary image: heatmap + timeline + confidence bars.
    Optionally writes to *output_path* as JPEG.
    """
    heatmap   = draw_anomaly_heatmap(
        results.get("anomaly_scores", []),
        width=width, height=120,
        threshold=results.get("anomaly_threshold", 0.5),
    )
    timeline  = draw_incident_timeline(
        results.get("incidents", []),
        results.get("duration", 1),
        width=width, height=80,
    )
    conf_bars = draw_confidence_bars(
        results.get("incidents", []),
        width=width // 2,
    )
    conf_bars = cv2.resize(conf_bars, (width, conf_bars.shape[0]))

    composite = np.vstack([heatmap, timeline, conf_bars])

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, composite, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return composite


def save_frame_mosaic(
    frames: List[np.ndarray],
    output_path: str,
    cols: int = 4,
    thumb_w: int = 320,
    thumb_h: int = 180,
) -> str:
    """Save a grid mosaic of frames as a JPEG image."""
    import math
    rows = math.ceil(len(frames) / cols)
    mosaic = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        r, c = divmod(idx, cols)
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        thumb = cv2.resize(frame, (thumb_w, thumb_h))
        mosaic[r*thumb_h:(r+1)*thumb_h, c*thumb_w:(c+1)*thumb_w] = thumb

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, mosaic, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return output_path
