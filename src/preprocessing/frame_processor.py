import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FrameProcessor:
    """Extract, resize and normalize frames from video files."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 360),
        fps: int = 5,
        normalize: bool = True,
    ):
        self.target_size = target_size
        self.fps = fps
        self.normalize = normalize

    def extract_frames(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> List[np.ndarray]:
        """Extract frames from a video file at the configured FPS."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0

        if end_time is None:
            end_time = duration

        frame_interval = max(1, int(video_fps / self.fps))
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        frame_idx = start_frame

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % frame_interval == 0:
                frame = self._process_frame(frame)
                frames.append(frame)

                if max_frames and len(frames) >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {Path(video_path).name}")
        return frames

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, self.target_size)
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        return frame

    def get_video_info(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {}
        info = {
            "path": str(video_path),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(cap.get(cv2.CAP_PROP_FPS), 1),
        }
        cap.release()
        return info

    def extract_frame_at(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.resize(frame, self.target_size)
        return None
