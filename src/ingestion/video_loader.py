import cv2
import os
from pathlib import Path
from typing import Optional, Generator
import logging

logger = logging.getLogger(__name__)


class VideoLoader:
    """Load video from file, RTSP stream, or local path."""

    SUPPORTED_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.m4v', '.wmv'}

    def __init__(self, max_size_mb: int = 500):
        self.max_size_mb = max_size_mb

    def load(self, source: str) -> Optional[cv2.VideoCapture]:
        """Open a video source and return a VideoCapture object."""
        if source.startswith('rtsp://') or source.startswith('http'):
            return self._load_stream(source)
        return self._load_file(source)

    def _load_file(self, path: str) -> Optional[cv2.VideoCapture]:
        p = Path(path)
        if not p.exists():
            logger.error(f"File not found: {path}")
            return None
        if p.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            logger.error(f"Unsupported format: {p.suffix}")
            return None
        size_mb = p.stat().st_size / 1024 / 1024
        if size_mb > self.max_size_mb:
            logger.error(f"File too large: {size_mb:.1f} MB > {self.max_size_mb} MB limit")
            return None
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {path}")
            return None
        logger.info(f"Loaded video: {p.name} ({size_mb:.1f} MB)")
        return cap

    def _load_stream(self, url: str) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            logger.error(f"Cannot open stream: {url}")
            return None
        logger.info(f"Connected to stream: {url}")
        return cap

    def iter_frames(self, cap: cv2.VideoCapture) -> Generator:
        """Yield frames one by one from a VideoCapture."""
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()

    def validate_video(self, path: str) -> dict:
        """Return metadata dict or empty dict if invalid."""
        cap = self.load(path)
        if cap is None:
            return {}
        info = {
            'path': path,
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'valid': True,
        }
        info['duration'] = info['frame_count'] / max(info['fps'], 1)
        cap.release()
        return info

    def find_videos(self, directory: str) -> list:
        """Recursively find all video files in a directory."""
        videos = []
        for root, _, files in os.walk(directory):
            for f in files:
                if Path(f).suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    videos.append(os.path.join(root, f))
        return sorted(videos)
