"""tests/test_incident.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.incident.incident_detector   import IncidentDetector
from src.incident.timeframe_extractor import extract_timeframes, flags_from_scores, format_timeframe
from src.incident.bbox_drawer         import draw_boxes, draw_tracks


class TestTimeframeExtractor:
    def test_empty_input(self):
        result = extract_timeframes([], fps=5.0)
        assert result == []

    def test_single_long_anomaly(self):
        flags  = [False]*5 + [True]*20 + [False]*5
        result = extract_timeframes(flags, fps=5.0, min_duration=2.0, merge_gap=1.0)
        assert len(result) == 1
        start, end = result[0]
        assert start == 1.0          # frame 5 / fps 5
        assert end   == 5.0          # frame 25 / fps 5

    def test_short_segment_filtered(self):
        flags  = [False]*10 + [True]*2 + [False]*10
        result = extract_timeframes(flags, fps=5.0, min_duration=5.0)
        assert result == []

    def test_merge_gap(self):
        # Two segments 1s apart — should merge with merge_gap=2
        flags  = [True]*10 + [False]*5 + [True]*10
        result = extract_timeframes(flags, fps=5.0, min_duration=1.0, merge_gap=2.0)
        assert len(result) == 1

    def test_flags_from_scores_threshold(self):
        scores = [0.1, 0.2, 0.8, 0.9, 0.85, 0.1]
        flags  = flags_from_scores(scores, threshold=0.5, window=1)
        assert flags[2] is True
        assert flags[0] is False

    def test_format_timeframe(self):
        s = format_timeframe(65.0, 130.0)
        assert "01:05" in s
        assert "02:10" in s


class TestIncidentDetector:
    def setup_method(self):
        self.det = IncidentDetector(
            confidence_threshold=0.3,
            min_duration=1.0,
            merge_gap=2.0,
            output_dir="/tmp/vigil_test",
        )

    def _make_person_det(self):
        return [
            {"label": "person", "confidence": 0.9, "bbox": [10, 10, 80, 200], "occluded": False},
            {"label": "person", "confidence": 0.85,"bbox": [90, 15, 160, 190],"occluded": False},
        ]

    def test_detect_incidents_returns_list(self):
        n = 50
        fd = [self._make_person_det() for _ in range(n)]
        ar = [{"is_anomalous": True,  "action": "fighting",
               "confidence": 0.88, "frame_start": 0, "frame_end": n}]
        sc = [0.75] * n

        incidents = self.det.detect_incidents(
            video_path="dummy.mp4",
            frame_detections=fd,
            action_results=ar,
            anomaly_scores=sc,
            anomaly_threshold=0.5,
            fps=5.0,
        )
        assert isinstance(incidents, list)

    def test_no_incidents_for_normal(self):
        n  = 30
        fd = [[{"label": "person", "confidence": 0.7, "bbox": [10,10,60,150], "occluded": False}]] * n
        ar = [{"is_anomalous": False, "action": "normal", "confidence": 0.9,
               "frame_start": 0, "frame_end": n}]
        sc = [0.1] * n

        incidents = self.det.detect_incidents(
            video_path="dummy.mp4",
            frame_detections=fd,
            action_results=ar,
            anomaly_scores=sc,
            anomaly_threshold=0.5,
            fps=5.0,
        )
        assert incidents == []

    def test_incident_has_required_keys(self):
        n  = 60
        fd = [self._make_person_det()] * n
        ar = [{"is_anomalous": True, "action": "fighting",
               "confidence": 0.9, "frame_start": 0, "frame_end": n}]
        sc = [0.8] * n

        incidents = self.det.detect_incidents(
            video_path="dummy.mp4",
            frame_detections=fd,
            action_results=ar,
            anomaly_scores=sc,
            anomaly_threshold=0.5,
            fps=5.0,
        )
        if incidents:
            inc = incidents[0]
            for key in ["id","start_time","end_time","duration","type","confidence"]:
                assert key in inc, f"Missing key: {key}"


class TestBboxDrawer:
    def test_draw_boxes_returns_image(self):
        import numpy as np
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        dets  = [{"label": "person", "confidence": 0.9,
                  "bbox": [50, 50, 200, 300], "occluded": False}]
        out   = draw_boxes(frame, dets)
        assert out.shape == frame.shape

    def test_draw_boxes_occluded(self):
        import numpy as np
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        dets  = [{"label": "person", "confidence": 0.6,
                  "bbox": [50, 50, 200, 300], "occluded": True}]
        out   = draw_boxes(frame, dets)
        assert out.shape == frame.shape
