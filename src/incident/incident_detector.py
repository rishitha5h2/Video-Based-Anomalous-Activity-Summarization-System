import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)


class IncidentDetector:
    """Fuse object detections, action labels, and anomaly scores into incidents."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        min_duration: float = 2.0,
        merge_gap: float = 3.0,
        snapshot_count: int = 5,
        output_dir: str = "data/outputs",
    ):
        self.confidence_threshold = confidence_threshold
        self.min_duration = min_duration
        self.merge_gap = merge_gap
        self.snapshot_count = snapshot_count
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "incident_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def detect_incidents(
        self,
        video_path: str,
        frame_detections: List[List[Dict]],
        action_results: List[Dict],
        anomaly_scores: List[float],
        anomaly_threshold: float,
        fps: float = 5.0,
    ) -> List[Dict]:
        """Combine all signals into a list of incident events."""
        n_frames = len(frame_detections)
        incident_flags = np.zeros(n_frames, dtype=bool)

        # Flag frames with anomalous action
        for ar in action_results:
            if ar.get("is_anomalous"):
                s, e = ar.get("frame_start", 0), ar.get("frame_end", 0)
                incident_flags[s:e] = True

        # Flag frames with high anomaly score
        for i, score in enumerate(anomaly_scores):
            if score > anomaly_threshold:
                incident_flags[i] = True

        # Flag frames with high-confidence detections of multiple persons
        for i, dets in enumerate(frame_detections):
            persons = [d for d in dets if d["label"] == "person"]
            if len(persons) >= 2:
                incident_flags[i] = True

        # Group consecutive flagged frames into segments
        segments = self._group_segments(incident_flags, fps)

        # Merge close segments
        segments = self._merge_segments(segments, self.merge_gap)

        # Filter by minimum duration
        segments = [s for s in segments if (s[1] - s[0]) >= self.min_duration]

        # Build incident objects
        incidents = []
        for idx, (start_t, end_t) in enumerate(segments):
            start_frame = int(start_t * fps)
            end_frame = int(end_t * fps)

            # Gather action predictions in window
            actions_in_window = [
                ar for ar in action_results
                if ar.get("frame_start", 0) >= start_frame and ar.get("frame_end", 0) <= end_frame
            ]
            action_label = self._majority_action(actions_in_window)
            confidence = self._compute_confidence(
                anomaly_scores[start_frame:end_frame], actions_in_window
            )

            if confidence < self.confidence_threshold:
                continue

            # Select snapshot frame indices
            snap_times = np.linspace(start_t, end_t, self.snapshot_count)

            incident = {
                "id": idx + 1,
                "start_time": round(start_t, 2),
                "end_time": round(end_t, 2),
                "duration": round(end_t - start_t, 2),
                "type": action_label,
                "confidence": round(confidence, 4),
                "frame_start": start_frame,
                "frame_end": end_frame,
                "snapshot_times": [round(t, 2) for t in snap_times],
                "frame_paths": [],
                "peak_anomaly_score": round(float(max(anomaly_scores[start_frame:end_frame] or [0])), 4),
                "num_persons": self._count_persons(frame_detections[start_frame:end_frame]),
            }
            incidents.append(incident)

        logger.info(f"Detected {len(incidents)} incidents")
        return incidents

    def extract_snapshots(
        self,
        video_path: str,
        incidents: List[Dict],
        detector,
        video_name: str = "video",
    ) -> List[Dict]:
        """Extract and annotate keyframe snapshots for each incident."""
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        for incident in incidents:
            frame_paths = []
            for snap_t in incident["snapshot_times"]:
                frame_idx = int(snap_t * video_fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                # Detect and draw bboxes
                detections = detector.detect(frame)
                annotated = detector.draw_detections(frame.copy(), detections)
                # Overlay timestamp and incident info
                annotated = self._draw_overlay(annotated, incident, snap_t)
                # Save
                fname = f"{video_name}_incident{incident['id']}_t{int(snap_t)}s.jpg"
                fpath = str(self.frames_dir / fname)
                cv2.imwrite(fpath, annotated)
                frame_paths.append(fpath)

            incident["frame_paths"] = frame_paths

        cap.release()
        return incidents

    def _draw_overlay(self, frame: np.ndarray, incident: Dict, timestamp: float) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 48), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        mins, secs = divmod(int(timestamp), 60)
        ts_text = f"T: {mins:02d}:{secs:02d}"
        type_text = f"[{incident['type'].upper()}]  conf: {incident['confidence']:.0%}"
        cv2.putText(frame, ts_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.putText(frame, type_text, (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 60, 255), 2)
        # Red border for high-confidence incidents
        if incident["confidence"] > 0.7:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), 4)
        return frame

    def _group_segments(self, flags: np.ndarray, fps: float) -> List[Tuple[float, float]]:
        segments = []
        in_seg = False
        start_i = 0
        for i, f in enumerate(flags):
            if f and not in_seg:
                in_seg = True
                start_i = i
            elif not f and in_seg:
                in_seg = False
                segments.append((start_i / fps, i / fps))
        if in_seg:
            segments.append((start_i / fps, len(flags) / fps))
        return segments

    def _merge_segments(self, segments: List[Tuple], gap: float) -> List[Tuple]:
        if not segments:
            return []
        merged = [segments[0]]
        for start, end in segments[1:]:
            if start - merged[-1][1] <= gap:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def _majority_action(self, action_results: List[Dict]) -> str:
        if not action_results:
            return "suspicious"
        from collections import Counter

        NON_NORMAL = {"fighting", "assault", "running", "theft", "vandalism",
                      "loitering", "suspicious", "anomaly", "explosion", "robbery"}

        # First: actions flagged as anomalous
        anomalous = [ar.get("action", "") for ar in action_results if ar.get("is_anomalous")]
        if anomalous:
            return Counter(anomalous).most_common(1)[0][0]

        # Second: any non-normal action by highest confidence
        non_normal = [(ar.get("action",""), ar.get("confidence", 0))
                      for ar in action_results if ar.get("action","") in NON_NORMAL]
        if non_normal:
            return sorted(non_normal, key=lambda x: x[1], reverse=True)[0][0]

        # Third: highest confidence action — if still normal, call it suspicious
        best = sorted(action_results, key=lambda x: x.get("confidence", 0), reverse=True)
        top_action = best[0].get("action", "suspicious")
        return "suspicious" if top_action == "normal" else top_action

    def _compute_confidence(self, scores: List[float], action_results: List[Dict]) -> float:
        score_conf = float(np.mean(scores)) if len(scores) > 0 else 0.5
        action_conf = float(np.mean([ar.get("confidence", 0) for ar in action_results])) if action_results else 0.5
        return min(1.0, (score_conf + action_conf) / 2 + 0.2)

    def _count_persons(self, frame_dets: List[List[Dict]]) -> int:
        if not frame_dets:
            return 0
        counts = [len([d for d in fd if d["label"] == "person"]) for fd in frame_dets]
        return int(np.max(counts)) if counts else 0