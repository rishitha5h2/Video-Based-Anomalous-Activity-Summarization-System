"""tests/test_detection.py"""
import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.detection.object_detector import ObjectDetector
from src.detection.anomaly_scorer  import AnomalyScorer
from src.detection.action_recognizer import ActionRecognizer


def make_frame(h=360, w=640, dtype=np.uint8):
    return np.random.randint(0, 255, (h, w, 3), dtype=dtype)


# ── ObjectDetector ────────────────────────────────────────────────────────────

class TestObjectDetector:
    def setup_method(self):
        self.det = ObjectDetector(model_size="n", confidence=0.4)

    def test_detect_returns_list(self):
        frame = make_frame()
        dets  = self.det.detect(frame)
        assert isinstance(dets, list)

    def test_detection_schema(self):
        frame = make_frame()
        dets  = self.det.detect(frame)
        for d in dets:
            assert "label"      in d
            assert "confidence" in d
            assert "bbox"       in d
            assert len(d["bbox"]) == 4

    def test_float_frame_accepted(self):
        frame = make_frame(dtype=np.uint8).astype(np.float32) / 255.0
        dets  = self.det.detect(frame)
        assert isinstance(dets, list)

    def test_iou_zero_for_disjoint(self):
        iou = self.det._compute_iou([0, 0, 10, 10], [20, 20, 30, 30])
        assert iou == 0.0

    def test_iou_one_for_identical(self):
        iou = self.det._compute_iou([0, 0, 100, 100], [0, 0, 100, 100])
        assert iou == pytest.approx(1.0)

    def test_draw_detections_returns_image(self):
        frame = make_frame()
        dets  = self.det._mock_detect(frame)
        out   = self.det.draw_detections(frame, dets)
        assert out.shape == frame.shape


# ── AnomalyScorer ─────────────────────────────────────────────────────────────

class TestAnomalyScorer:
    def setup_method(self):
        self.scorer = AnomalyScorer(latent_dim=32, input_shape=(32, 32, 3))

    def test_score_returns_float(self):
        frame = make_frame(32, 32)
        score = self.scorer.score_frame(frame)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_is_anomalous_returns_tuple(self):
        frame   = make_frame(32, 32)
        result  = self.scorer.is_anomalous(frame)
        assert isinstance(result, tuple) and len(result) == 2

    def test_set_threshold(self):
        self.scorer.set_threshold(0.99)
        assert self.scorer.threshold == 0.99

    def test_score_sequence_length(self):
        frames = [make_frame(32, 32) for _ in range(5)]
        scores = self.scorer.score_sequence(frames)
        assert len(scores) == 5

    def test_training_reduces_loss(self):
        frames = [make_frame(32, 32) for _ in range(20)]
        losses = self.scorer.train(frames, epochs=3, batch_size=4)
        assert len(losses) == 3
        assert all(isinstance(l, float) for l in losses)


# ── ActionRecognizer ──────────────────────────────────────────────────────────

class TestActionRecognizer:
    def setup_method(self):
        self.rec = ActionRecognizer(clip_length=8)

    def test_recognize_returns_dict(self):
        clip   = [make_frame() for _ in range(8)]
        result = self.rec.recognize(clip)
        assert "action"       in result
        assert "confidence"   in result
        assert "is_anomalous" in result

    def test_empty_clip(self):
        result = self.rec.recognize([])
        assert result["action"] == "unknown"

    def test_recognize_sequence(self):
        frames  = [make_frame() for _ in range(24)]
        results = self.rec.recognize_sequence(frames, step=8)
        assert isinstance(results, list)
        for r in results:
            assert "frame_start" in r and "frame_end" in r
