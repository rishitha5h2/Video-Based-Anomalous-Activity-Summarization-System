"""src/ingestion/stream_manager.py — RTSP/webcam stream buffering."""

import cv2
import threading
import queue
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class StreamManager:
    """
    Thread-safe frame buffer for live video streams (RTSP, webcam, HTTP).
    Decodes frames in a background thread and exposes them via get_frame().
    """

    def __init__(self, source: str, buffer_size: int = 30, reconnect: bool = True):
        self.source      = source
        self.buffer_size = buffer_size
        self.reconnect   = reconnect

        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._cap:   Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock    = threading.Lock()
        self.connected = False

    # ── public ────────────────────────────────────────────────────────────────

    def start(self):
        """Open the stream and begin buffering frames."""
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"StreamManager started: {self.source}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
        logger.info("StreamManager stopped")

    def get_frame(self, timeout: float = 1.0):
        """
        Return the latest frame or None on timeout.
        Non-blocking consumer — always returns the newest available frame.
        """
        try:
            # Drain the queue to get the freshest frame
            frame = None
            while True:
                frame = self._queue.get_nowait()
        except queue.Empty:
            pass
        return frame

    @property
    def fps(self) -> float:
        if self._cap and self._cap.isOpened():
            return self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        return 25.0

    @property
    def resolution(self):
        if self._cap and self._cap.isOpened():
            return (
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        return (0, 0)

    # ── private ───────────────────────────────────────────────────────────────

    def _capture_loop(self):
        while self._running:
            self._cap = self._open_source()
            if self._cap is None:
                if self.reconnect:
                    logger.warning(f"Reconnecting in 3s…")
                    time.sleep(3)
                    continue
                break

            self.connected = True
            while self._running:
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning("Stream read failed — reconnecting…")
                    self.connected = False
                    break

                try:
                    if self._queue.full():
                        self._queue.get_nowait()   # drop oldest
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass

            self._cap.release()

    def _open_source(self) -> Optional[cv2.VideoCapture]:
        src = int(self.source) if self.source.isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            logger.error(f"Cannot open source: {self.source}")
            return None
        # Buffer only 1 frame in the internal capture buffer for low latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
