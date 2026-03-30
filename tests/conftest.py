"""tests/conftest.py — shared fixtures."""
import pytest
import numpy as np


@pytest.fixture
def sample_frame():
    """640×360 uint8 BGR frame."""
    return np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frames(sample_frame):
    """List of 20 frames."""
    return [np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(20)]


@pytest.fixture
def sample_incident():
    return {
        "id": 1, "start_time": 10.0, "end_time": 30.0,
        "duration": 20.0, "type": "fighting", "confidence": 0.88,
        "num_persons": 2, "peak_anomaly_score": 0.92,
        "frame_paths": [], "narrative": "Test narrative.",
        "snapshot_times": [10.0, 15.0, 20.0, 25.0, 30.0],
    }


@pytest.fixture
def sample_results(sample_incident):
    return {
        "video_name": "test_video.mp4",
        "video_path": "/tmp/test_video.mp4",
        "duration": 120.0, "resolution": "1280x720",
        "fps": 25, "frame_count": 600,
        "incidents": [sample_incident],
        "incident_count": 1, "is_anomalous": True,
        "max_confidence": 0.88,
        "summary": "Test summary.",
        "processing_time": 12.5,
    }
