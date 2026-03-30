"""tests/test_preprocessing.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.preprocessing.frame_processor      import FrameProcessor
from src.preprocessing.optical_flow         import OpticalFlowExtractor
from src.preprocessing.background_subtractor import BackgroundSubtractor


def make_frame(h=360, w=640, val=128, dtype=np.uint8):
    return np.full((h, w, 3), val, dtype=dtype)


class TestFrameProcessor:
    def setup_method(self):
        self.proc = FrameProcessor(target_size=(160, 90), fps=5, normalize=True)

    def test_process_frame_shape(self):
        frame = make_frame()
        out   = self.proc._process_frame(frame)
        assert out.shape[:2] == (90, 160)

    def test_process_frame_normalized(self):
        frame = make_frame(val=255)
        out   = self.proc._process_frame(frame)
        assert out.max() <= 1.0

    def test_no_normalization(self):
        proc  = FrameProcessor(target_size=(160, 90), fps=5, normalize=False)
        frame = make_frame(val=200)
        out   = proc._process_frame(frame)
        assert out.dtype == np.uint8


class TestOpticalFlow:
    def setup_method(self):
        self.extractor = OpticalFlowExtractor(method="farneback")

    def test_flow_shape(self):
        f1   = make_frame(h=90, w=160, val=50)
        f2   = make_frame(h=90, w=160, val=80)
        flow = self.extractor.compute_flow(f1, f2)
        assert flow.shape == (90, 160, 2)

    def test_zero_flow_for_identical(self):
        f    = make_frame(h=90, w=160, val=100)
        flow = self.extractor.compute_flow(f, f)
        mag  = self.extractor.flow_to_magnitude(flow)
        assert mag.mean() < 1e-3

    def test_sequence_length(self):
        frames = [make_frame(h=90, w=160) for _ in range(5)]
        flows  = self.extractor.compute_flow_sequence(frames)
        assert len(flows) == 4


class TestBackgroundSubtractor:
    def setup_method(self):
        self.sub = BackgroundSubtractor(method="MOG2")

    def test_apply_returns_mask(self):
        frame = make_frame()
        mask  = self.sub.apply(frame)
        assert mask.ndim == 2
        assert mask.dtype == np.uint8

    def test_motion_score_is_float(self):
        frame = make_frame()
        score = self.sub.get_motion_score(frame)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_sequence_returns_list(self):
        frames = [make_frame() for _ in range(4)]
        scores = self.sub.process_sequence(frames)
        assert len(scores) == 4
