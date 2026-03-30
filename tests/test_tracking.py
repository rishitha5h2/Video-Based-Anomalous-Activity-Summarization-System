"""tests/test_tracking.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tracking.tracker           import MultiObjectTracker, Track
from src.tracking.trajectory_analyzer import TrajectoryAnalyzer
from src.tracking.id_manager        import IDManager


class TestMultiObjectTracker:
    def setup_method(self):
        self.tracker = MultiObjectTracker(max_age=5, min_hits=1, iou_threshold=0.3)

    def test_empty_update(self):
        result = self.tracker.update([])
        assert isinstance(result, list)

    def test_single_detection_creates_track(self):
        dets = [{"label": "person", "confidence": 0.9, "bbox": [10, 10, 80, 200]}]
        self.tracker.update(dets)
        result = self.tracker.update(dets)   # second update — hits >= min_hits
        assert len(result) >= 1

    def test_ids_are_stable(self):
        dets = [{"label": "person", "confidence": 0.9, "bbox": [10, 10, 80, 200]}]
        r1 = self.tracker.update(dets)
        r2 = self.tracker.update(dets)
        if r1 and r2:
            assert r1[0]["id"] == r2[0]["id"]

    def test_iou_disjoint(self):
        t = MultiObjectTracker()
        assert t._iou([0,0,10,10], [50,50,60,60]) == 0.0

    def test_iou_identical(self):
        import pytest
        t = MultiObjectTracker()
        assert t._iou([0,0,100,100], [0,0,100,100]) == pytest.approx(1.0)

    def test_reset_clears_tracks(self):
        dets = [{"label": "person", "confidence": 0.9, "bbox": [10, 10, 80, 200]}]
        self.tracker.update(dets)
        self.tracker.reset()
        assert self.tracker.tracks == []


class TestTrajectoryAnalyzer:
    def setup_method(self):
        self.analyzer = TrajectoryAnalyzer(
            loiter_distance_threshold=30.0, loiter_min_frames=5, crowd_threshold=3
        )

    def test_analyze_empty(self):
        result = self.analyzer.analyze([])
        assert result["crowd_detected"] is False
        assert result["anomaly_score"] == 0.0

    def test_crowd_detected(self):
        tracks = [
            {"id": i, "label": "person", "bbox": [i*10,0,i*10+8,50],
             "confidence": 0.9, "trajectory": [(i*10+4, 25)], "occluded": False}
            for i in range(4)
        ]
        result = self.analyzer.analyze(tracks)
        assert result["crowd_detected"] is True

    def test_loitering_detected(self):
        # Trajectory that stays within a 10px circle
        traj = [(100 + i % 3, 100 + i % 3) for i in range(10)]
        assert self.analyzer._is_loitering(traj) is True

    def test_not_loitering_when_moving(self):
        traj = [(i * 20, i * 20) for i in range(10)]
        assert self.analyzer._is_loitering(traj) is False


class TestIDManager:
    def setup_method(self):
        self.mgr = IDManager(grace_frames=5, max_distance_px=60)

    def test_first_assignment_gets_id_1(self):
        ids = self.mgr.assign([(100.0, 100.0)])
        assert ids[0] == 1

    def test_stable_id_across_frames(self):
        ids1 = self.mgr.assign([(100.0, 100.0)])
        ids2 = self.mgr.assign([(102.0, 101.0)])   # small movement
        assert ids1[0] == ids2[0]

    def test_new_id_for_distant_centre(self):
        ids1 = self.mgr.assign([(10.0, 10.0)])
        ids2 = self.mgr.assign([(500.0, 500.0)])   # far away
        assert ids1[0] != ids2[0]

    def test_reset(self):
        self.mgr.assign([(1.0, 1.0)])
        self.mgr.reset()
        ids = self.mgr.assign([(1.0, 1.0)])
        assert ids[0] == 1   # back to ID 1
