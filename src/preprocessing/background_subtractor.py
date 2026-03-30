import cv2
import numpy as np
from typing import List


class BackgroundSubtractor:
    """MOG2-based background subtraction for motion detection."""

    def __init__(self, method: str = "MOG2", history: int = 200, threshold: float = 16.0):
        self.method = method
        if method == "MOG2":
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=threshold, detectShadows=True
            )
        else:
            self.subtractor = cv2.createBackgroundSubtractorKNN(history=history)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        mask = self.subtractor.apply(frame)
        # Remove shadows (gray pixels with value 127)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def get_motion_score(self, frame: np.ndarray) -> float:
        mask = self.apply(frame)
        return float(np.sum(mask > 0)) / (mask.shape[0] * mask.shape[1])

    def process_sequence(self, frames: List[np.ndarray]) -> List[float]:
        return [self.get_motion_score(f) for f in frames]
