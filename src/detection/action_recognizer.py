import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

ACTION_CLASSES = {
    "fighting": ["punch", "kick", "hit", "fight", "brawl", "assault"],
    "running": ["run", "sprint", "chase", "flee"],
    "falling": ["fall", "collapse", "trip", "stumble"],
    "loitering": ["stand", "linger", "wait", "loiter"],
    "theft": ["grab", "steal", "snatch", "take"],
    "vandalism": ["break", "smash", "spray", "damage"],
    "normal": ["walk", "sit", "talk", "work"],
}


class ActionRecognizer:
    """Clip-based action recognition using motion features + heuristics."""

    def __init__(self, clip_length: int = 16, confidence_threshold: float = 0.5):
        self.clip_length = clip_length
        self.confidence_threshold = confidence_threshold
        self._load_model()

    def _load_model(self):
        """Try to load a pretrained model; fall back to heuristic."""
        try:
            import torch
            self.model = None  # Replace with actual SlowFast/I3D if available
            logger.info("Action recognizer initialized (heuristic mode)")
        except Exception as e:
            logger.warning(f"Could not load action model: {e}")
            self.model = None

    def recognize(self, clip: List[np.ndarray]) -> Dict:
        """Recognize action in a video clip."""
        if len(clip) < 2:
            return {"action": "unknown", "confidence": 0.0, "is_anomalous": False}

        motion_score = self._compute_motion_score(clip)
        density_score = self._compute_person_density(clip)
        action, confidence = self._classify_by_motion(motion_score, density_score)

        return {
            "action": action,
            "confidence": confidence,
            "motion_score": motion_score,
            "density_score": density_score,
            "is_anomalous": action not in ["normal", "unknown"] and confidence > self.confidence_threshold,
        }

    def _compute_motion_score(self, clip: List[np.ndarray]) -> float:
        scores = []
        for i in range(1, len(clip)):
            f1 = clip[i - 1]
            f2 = clip[i]
            if f1.dtype != np.uint8:
                f1 = (f1 * 255).astype(np.uint8)
                f2 = (f2 * 255).astype(np.uint8)
            g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY) if len(f1.shape) == 3 else f1
            g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY) if len(f2.shape) == 3 else f2
            diff = cv2.absdiff(g1, g2)
            scores.append(float(np.mean(diff)) / 255.0)
        return float(np.mean(scores)) if scores else 0.0

    def _compute_person_density(self, clip: List[np.ndarray]) -> float:
        """Approximate person density via edge density."""
        if not clip:
            return 0.0
        frame = clip[len(clip) // 2]
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        edges = cv2.Canny(gray, 100, 200)
        return float(np.sum(edges > 0)) / (frame.shape[0] * frame.shape[1])

    def _classify_by_motion(self, motion: float, density: float) -> Tuple[str, float]:
        # Thresholds tuned for low-res CCTV footage (320x240, 5fps)
        # Motion scores tend to be much lower than HD footage
        if motion > 0.06 and density > 0.08:
            return "fighting", min(0.95, 0.55 + motion * 3 + density * 0.5)
        elif motion > 0.05 and density > 0.12:
            return "assault", min(0.90, 0.50 + motion * 2 + density)
        elif motion > 0.04:
            return "running", min(0.88, 0.45 + motion * 3)
        elif motion > 0.03 and density > 0.10:
            return "theft", min(0.82, 0.42 + motion * 2 + density)
        elif motion > 0.02 and density > 0.08:
            return "suspicious", min(0.75, 0.38 + motion * 2 + density)
        elif motion < 0.005 and density > 0.05:
            return "loitering", min(0.72, 0.32 + density * 2)
        elif motion < 0.008:
            return "normal", 0.85
        else:
            return "normal", 0.55

    def recognize_sequence(self, frames: List[np.ndarray], step: int = 8) -> List[Dict]:
        results = []
        for i in range(0, len(frames) - self.clip_length, step):
            clip = frames[i:i + self.clip_length]
            result = self.recognize(clip)
            result["frame_start"] = i
            result["frame_end"] = i + self.clip_length
            results.append(result)
        return results