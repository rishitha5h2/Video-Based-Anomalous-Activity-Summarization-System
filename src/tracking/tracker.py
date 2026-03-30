import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class Track:
    """Single object track with persistent ID across frames."""

    _next_id = 1

    def __init__(self, bbox: List[int], label: str, confidence: float):
        self.id = Track._next_id
        Track._next_id += 1
        self.label = label
        self.confidence = confidence
        self.bbox = bbox
        self.history: List[List[int]] = [bbox]
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.occluded = False

    def update(self, bbox: List[int], confidence: float):
        self.bbox = bbox
        self.confidence = confidence
        self.history.append(bbox)
        self.hits += 1
        self.time_since_update = 0

    def predict(self):
        """Simple constant velocity prediction."""
        if len(self.history) >= 2:
            dx = self.history[-1][0] - self.history[-2][0]
            dy = self.history[-1][1] - self.history[-2][1]
            predicted = [
                self.bbox[0] + dx, self.bbox[1] + dy,
                self.bbox[2] + dx, self.bbox[3] + dy,
            ]
            return predicted
        return self.bbox

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    @property
    def trajectory(self) -> List[Tuple[float, float]]:
        return [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in self.history]


class MultiObjectTracker:
    """IoU-based multi-object tracker with occlusion awareness."""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []

    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracks with new detections. Returns confirmed track states."""
        # Predict positions for existing tracks
        for t in self.tracks:
            t.age += 1
            t.time_since_update += 1

        if not detections:
            self._cleanup()
            return self._get_confirmed()

        det_bboxes = [d['bbox'] for d in detections]

        # Match detections to tracks
        if self.tracks:
            matched, unmatched_dets, unmatched_trks = self._match(det_bboxes)

            for trk_idx, det_idx in matched:
                self.tracks[trk_idx].update(
                    detections[det_idx]['bbox'],
                    detections[det_idx]['confidence']
                )

            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                d = detections[det_idx]
                self.tracks.append(Track(d['bbox'], d['label'], d['confidence']))

            # Mark occluded tracks
            for trk_idx in unmatched_trks:
                self.tracks[trk_idx].occluded = True
        else:
            for d in detections:
                self.tracks.append(Track(d['bbox'], d['label'], d['confidence']))

        self._cleanup()
        return self._get_confirmed()

    def _match(self, det_bboxes: List[List[int]]) -> Tuple[List, List, List]:
        iou_matrix = np.zeros((len(self.tracks), len(det_bboxes)))
        for t_i, track in enumerate(self.tracks):
            for d_i, det_bbox in enumerate(det_bboxes):
                iou_matrix[t_i, d_i] = self._iou(track.bbox, det_bbox)

        matched_indices, unmatched_dets, unmatched_trks = [], [], []
        if iou_matrix.size > 0:
            # Greedy matching
            assigned_tracks = set()
            assigned_dets = set()
            while True:
                if iou_matrix.max() < self.iou_threshold:
                    break
                t_i, d_i = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                matched_indices.append((t_i, d_i))
                assigned_tracks.add(t_i)
                assigned_dets.add(d_i)
                iou_matrix[t_i, :] = -1
                iou_matrix[:, d_i] = -1
            unmatched_dets = [i for i in range(len(det_bboxes)) if i not in assigned_dets]
            unmatched_trks = [i for i in range(len(self.tracks)) if i not in assigned_tracks]
        else:
            unmatched_dets = list(range(len(det_bboxes)))
            unmatched_trks = list(range(len(self.tracks)))

        return matched_indices, unmatched_dets, unmatched_trks

    def _cleanup(self):
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

    def _get_confirmed(self) -> List[Dict]:
        return [
            {
                'id': t.id,
                'label': t.label,
                'bbox': t.bbox,
                'confidence': t.confidence,
                'hits': t.hits,
                'occluded': t.occluded,
                'trajectory': t.trajectory[-20:],
                'center': t.center,
            }
            for t in self.tracks
            if t.hits >= self.min_hits or t.time_since_update == 0
        ]

    def _iou(self, box_a: List[int], box_b: List[int]) -> float:
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
        area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def reset(self):
        self.tracks = []
        Track._next_id = 1
