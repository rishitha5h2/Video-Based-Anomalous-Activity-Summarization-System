"""
src/annotation/video_annotator.py
Writes an annotated copy of a video with bounding boxes, incident overlays,
and a live anomaly-score bar burned into each frame.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# ── colour palette (BGR) ────────────────────────────────────────────────────
PALETTE = {
    "person":   (0,   60, 232),   # red
    "knife":    (0,  128, 255),   # orange
    "gun":      (0,   0,  200),   # dark red
    "car":      (255, 128,  0),   # blue-ish
    "truck":    (255, 165,  0),
    "default":  (0,  200,  50),   # green
}
INCIDENT_COLOR = (0, 0, 220)      # solid red border during incident


class VideoAnnotator:
    """
    Reads a source video frame-by-frame, draws:
      • Per-frame bounding boxes (solid for confirmed, dashed for occluded)
      • Incident banner at top during active incident windows
      • Anomaly-score progress bar at bottom
      • Frame timestamp (top-left)
    and writes the result to *output_path*.
    """

    def __init__(self, font_scale: float = 0.55, thickness: int = 2):
        self.font       = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness  = thickness

    # ── public API ────────────────────────────────────────────────────────────

    def annotate(
        self,
        video_path: str,
        output_path: str,
        frame_detections: List[List[Dict]],   # one list per extracted frame
        anomaly_scores:   List[float],
        incidents:        List[Dict],
        source_fps:       float = 25.0,
        sample_fps:       float = 5.0,
    ) -> str:
        """
        Parameters
        ----------
        frame_detections : detections[i] corresponds to the i-th *sampled* frame
        anomaly_scores   : one score per sampled frame  (same length)
        incidents        : list of incident dicts with start_time / end_time
        source_fps       : native FPS of the source video
        sample_fps       : FPS at which frames were sampled for detection
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open: {video_path}")
            return ""

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or source_fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_interval = max(1, int(fps / sample_fps))   # e.g. 25/5 = 5
        native_idx     = 0
        sample_idx     = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Which sampled-frame data corresponds to this native frame?
            sample_idx = min(native_idx // frame_interval, len(frame_detections) - 1)
            timestamp  = native_idx / fps

            dets  = frame_detections[sample_idx] if frame_detections else []
            score = anomaly_scores[sample_idx]   if anomaly_scores   else 0.0

            # Find active incident (if any)
            active = next(
                (inc for inc in incidents
                 if inc["start_time"] <= timestamp <= inc["end_time"]),
                None,
            )

            frame = self._draw_detections(frame, dets)
            frame = self._draw_score_bar(frame, score, w, h)
            frame = self._draw_timestamp(frame, timestamp)
            if active:
                frame = self._draw_incident_banner(frame, active, w)

            out.write(frame)
            native_idx += 1

        cap.release()
        out.release()
        logger.info(f"Annotated video saved → {output_path}")
        return output_path

    # ── drawing helpers ───────────────────────────────────────────────────────

    def _draw_detections(self, frame: np.ndarray, dets: List[Dict]) -> np.ndarray:
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            label     = d.get("label", "obj")
            conf      = d.get("confidence", 0.0)
            occluded  = d.get("occluded", False)
            color     = PALETTE.get(label, PALETTE["default"])

            if occluded:
                self._draw_dashed_rect(frame, x1, y1, x2, y2, color, 1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

            text  = f"{label} {conf:.0%}"
            tw, th = cv2.getTextSize(text, self.font, self.font_scale, 1)[0]
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, text,
                        (x1 + 2, y1 - 3),
                        self.font, self.font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def _draw_dashed_rect(self, frame, x1, y1, x2, y2, color, thickness, gap=8):
        for pts in [
            [(x1 + i, y1, min(x1+i+gap//2, x2), y1) for i in range(0, x2-x1, gap)],
            [(x1 + i, y2, min(x1+i+gap//2, x2), y2) for i in range(0, x2-x1, gap)],
            [(x1, y1 + i, x1, min(y1+i+gap//2, y2)) for i in range(0, y2-y1, gap)],
            [(x2, y1 + i, x2, min(y1+i+gap//2, y2)) for i in range(0, y2-y1, gap)],
        ]:
            for seg in pts:
                cv2.line(frame, (seg[0], seg[1]), (seg[2], seg[3]), color, thickness)

    def _draw_score_bar(self, frame, score, w, h):
        bar_h  = 6
        filled = int(w * min(score, 1.0))
        # Background
        cv2.rectangle(frame, (0, h - bar_h), (w, h), (30, 30, 30), -1)
        # Filled portion — green → amber → red
        if score < 0.5:
            color = (0, 200, 0)
        elif score < 0.75:
            color = (0, 165, 255)
        else:
            color = (0, 0, 220)
        cv2.rectangle(frame, (0, h - bar_h), (filled, h), color, -1)
        return frame

    def _draw_timestamp(self, frame, ts: float):
        m, s = divmod(int(ts), 60)
        text = f"{m:02d}:{s:02d}"
        # semi-transparent backing
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (90, 28), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, text, (6, 20),
                    self.font, 0.65, (0, 220, 255), 1, cv2.LINE_AA)
        return frame

    def _draw_incident_banner(self, frame, incident: Dict, w: int):
        label = incident["type"].upper()
        conf  = incident["confidence"]
        text  = f"  INCIDENT: {label}  |  CONF {conf:.0%}  "

        # Red top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, text, (8, 25),
                    self.font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Red border
        cv2.rectangle(frame, (0, 0), (w - 1, frame.shape[0] - 1),
                      INCIDENT_COLOR, 3)
        return frame
