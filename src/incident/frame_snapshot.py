"""src/incident/frame_snapshot.py — Save annotated keyframes for each incident."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FrameSnapshotExtractor:
    """
    Given a source video and a list of incident dicts, extracts *n* keyframes
    per incident, draws bounding boxes + timestamp overlays, and saves them
    as JPEG files.
    """

    def __init__(self, output_dir: str = "data/outputs/incident_frames", quality: int = 92):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quality    = quality

    def extract(
        self,
        video_path: str,
        incidents:  List[Dict],
        detector,
        video_name: str = "video",
        snapshots_per_incident: int = 5,
    ) -> List[Dict]:
        """
        For every incident, seek to *snapshots_per_incident* evenly-spaced
        timestamps, detect objects, draw bboxes + overlay, and save.

        Returns the same *incidents* list with 'frame_paths' populated.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return incidents

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        for incident in incidents:
            start = incident["start_time"]
            end   = incident["end_time"]
            times = np.linspace(start, end, snapshots_per_incident)
            paths = []

            for ts in times:
                frame_idx = int(ts * native_fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Detect and annotate
                dets      = detector.detect(frame)
                annotated = detector.draw_detections(frame.copy(), dets)
                annotated = self._draw_overlay(annotated, incident, ts)

                fname = (
                    f"{video_name}_inc{incident['id']}"
                    f"_t{int(ts):04d}s.jpg"
                )
                fpath = str(self.output_dir / fname)
                cv2.imwrite(fpath, annotated,
                            [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                paths.append(fpath)

            incident["frame_paths"]     = paths
            incident["snapshot_times"]  = [round(float(t), 2) for t in times]

        cap.release()
        return incidents

    # ── private ──────────────────────────────────────────────────────────────

    def _draw_overlay(self, frame: np.ndarray, incident: Dict, ts: float) -> np.ndarray:
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Dark top strip
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 44), (8, 10, 18), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # Timestamp
        m, s = divmod(int(ts), 60)
        cv2.putText(frame, f"{m:02d}:{s:02d}",
                    (10, 30), font, 0.85, (0, 220, 255), 2, cv2.LINE_AA)

        # Type + confidence
        label = f"[{incident['type'].upper()}]  {incident['confidence']:.0%}"
        cv2.putText(frame, label,
                    (110, 30), font, 0.75, (80, 80, 255), 2, cv2.LINE_AA)

        # Red border for high-confidence incidents
        if incident["confidence"] >= 0.70:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 210), 4)

        # Bottom anomaly bar
        score = incident.get("peak_anomaly_score", incident["confidence"])
        bar_w = int(score * w)
        cv2.rectangle(frame, (0, h - 5), (w, h), (30, 30, 30), -1)
        color = (0, 0, 210) if score > 0.75 else (0, 165, 245)
        cv2.rectangle(frame, (0, h - 5), (bar_w, h), color, -1)

        return frame

    def save_mosaic(self, frame_paths: List[str], output_path: str,
                    cols: int = 3) -> Optional[str]:
        """Save a grid mosaic of saved snapshots."""
        frames = []
        for p in frame_paths:
            img = cv2.imread(p)
            if img is not None:
                frames.append(cv2.resize(img, (320, 180)))

        if not frames:
            return None

        import math
        rows     = math.ceil(len(frames) / cols)
        mosaic   = np.zeros((rows * 180, cols * 320, 3), dtype=np.uint8)
        for idx, f in enumerate(frames):
            r, c = divmod(idx, cols)
            mosaic[r*180:(r+1)*180, c*320:(c+1)*320] = f

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, mosaic, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return output_path
