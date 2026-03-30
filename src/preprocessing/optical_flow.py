import cv2
import numpy as np
from typing import List, Tuple, Optional


class OpticalFlowExtractor:
    """Compute dense optical flow between consecutive frames."""

    def __init__(self, method: str = "farneback"):
        self.method = method

    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        if frame1.dtype != np.uint8:
            f1 = (frame1 * 255).astype(np.uint8)
            f2 = (frame2 * 255).astype(np.uint8)
        else:
            f1, f2 = frame1, frame2

        gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY) if len(f1.shape) == 3 else f1
        gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY) if len(f2.shape) == 3 else f2

        if self.method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        else:
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow

    def flow_to_magnitude(self, flow: np.ndarray) -> np.ndarray:
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return mag

    def compute_flow_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        flows = []
        for i in range(1, len(frames)):
            flow = self.compute_flow(frames[i - 1], frames[i])
            flows.append(flow)
        return flows

    def visualize_flow(self, flow: np.ndarray) -> np.ndarray:
        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
