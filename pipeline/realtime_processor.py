"""
Real-time RTSP/webcam stream processor for VIGIL.

Usage:
    from pipeline.realtime_processor import RealtimeProcessor
    proc = RealtimeProcessor(pipeline, rtsp_url='rtsp://...')
    proc.start()
"""

import cv2
import threading
import time
import logging
from typing import Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class RealtimeProcessor:
    """Process live video streams frame-by-frame with windowed incident detection."""

    def __init__(
        self,
        pipeline,
        source: str = "0",
        window_seconds: int = 30,
        callback: Optional[Callable] = None,
    ):
        self.pipeline = pipeline
        self.source = source
        self.window_seconds = window_seconds
        self.callback = callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_buffer = []
        self._incident_log = []

    def start(self):
        """Start stream processing in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info(f"Realtime processor started: {self.source}")

    def stop(self):
        """Stop the processing loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Realtime processor stopped")

    def _process_loop(self):
        src = int(self.source) if self.source.isdigit() else self.source
        cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            logger.error(f"Cannot open source: {self.source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        window_frames = int(fps * self.window_seconds)
        frame_count = 0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Stream ended or frame read failed")
                time.sleep(0.1)
                continue

            self._frame_buffer.append(frame)
            frame_count += 1

            # Process window when full
            if len(self._frame_buffer) >= window_frames:
                self._process_window(fps)
                # Keep last 10% of frames for overlap
                overlap = max(1, window_frames // 10)
                self._frame_buffer = self._frame_buffer[-overlap:]

        cap.release()

    def _process_window(self, fps: float):
        """Run detection on the current frame buffer window."""
        try:
            from src.detection.object_detector import ObjectDetector
            from src.detection.anomaly_scorer import AnomalyScorer
            from src.detection.action_recognizer import ActionRecognizer
            from src.incident.incident_detector import IncidentDetector

            detector = self.pipeline.detector
            scorer = self.pipeline.anomaly_scorer
            action_rec = self.pipeline.action_recognizer
            incident_det = self.pipeline.incident_detector

            frame_dets = [detector.detect(f) for f in self._frame_buffer[::5]]
            scores = scorer.score_sequence(self._frame_buffer[::5])
            actions = action_rec.recognize_sequence(self._frame_buffer, step=8)

            incidents = incident_det.detect_incidents(
                video_path="",
                frame_detections=frame_dets,
                action_results=actions,
                anomaly_scores=scores,
                anomaly_threshold=self.pipeline.anomaly_threshold,
                fps=fps / 5,
            )

            if incidents:
                logger.warning(f"LIVE ALERT: {len(incidents)} incident(s) detected!")
                self._incident_log.extend(incidents)
                if self.callback:
                    self.callback(incidents)

        except Exception as e:
            logger.error(f"Window processing error: {e}")

    @property
    def incident_log(self):
        return self._incident_log.copy()
