"""
src/detection/occlusion_handler.py

Handles detection of objects that are partially or fully behind other objects.
Uses depth-ordering heuristics, trajectory extrapolation, and IoU overlap
analysis to infer occluded object positions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class OcclusionHandler:
    """
    Augments raw detections with occlusion-awareness:
      1. Marks partially-occluded detections (IoU overlap with larger objects)
      2. Extrapolates fully-occluded objects from trajectory history
      3. Estimates depth order (objects lower in frame = closer = in front)
    """

    def __init__(
        self,
        iou_occlude_min: float = 0.10,   # min overlap to mark as occluded
        iou_occlude_max: float = 0.60,   # above this → assume same object
        depth_alpha:     float = 0.7,    # weight for y-position in depth estimate
    ):
        self.iou_occlude_min = iou_occlude_min
        self.iou_occlude_max = iou_occlude_max
        self.depth_alpha     = depth_alpha

        # Track history for extrapolation: id → list of bboxes
        self._history: Dict[int, List[List[int]]] = {}

    def process(
        self,
        detections:  List[Dict],
        frame_shape: Tuple[int, int],
        tracks:      Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Main entry point.

        Parameters
        ----------
        detections  : raw YOLOv8 detections for this frame
        frame_shape : (height, width)
        tracks      : optional tracker output (for extrapolation)

        Returns
        -------
        Augmented detections with 'occluded', 'depth_rank', and optionally
        'extrapolated' keys added.
        """
        detections = self._assign_depth(detections, frame_shape)
        detections = self._mark_occluded(detections)

        if tracks:
            extrapolated = self._extrapolate_occluded(tracks)
            detections.extend(extrapolated)

        return detections

    # ── depth assignment ──────────────────────────────────────────────────────

    def _assign_depth(self, detections: List[Dict], frame_shape: Tuple) -> List[Dict]:
        """
        Assign a depth rank (0 = closest/foreground, N = furthest).
        Heuristic: lower foot position (y2) → closer to camera.
        """
        h = frame_shape[0]
        for d in detections:
            foot_y  = d["bbox"][3]                    # bottom of bbox
            area    = self._area(d["bbox"])
            # normalised foot position + area factor
            d["depth_score"] = (foot_y / h) * self.depth_alpha + \
                                (area / (h * frame_shape[1])) * (1 - self.depth_alpha)

        # Rank: highest depth_score = closest (rank 0)
        ranked = sorted(detections, key=lambda d: d["depth_score"], reverse=True)
        for rank, d in enumerate(ranked):
            d["depth_rank"] = rank
        return detections

    # ── occlusion marking ─────────────────────────────────────────────────────

    def _mark_occluded(self, detections: List[Dict]) -> List[Dict]:
        """
        For each pair (A, B):
          if A is smaller than B AND they overlap significantly → A is occluded by B.
        """
        for i, det_a in enumerate(detections):
            for j, det_b in enumerate(detections):
                if i == j:
                    continue
                iou    = self._iou(det_a["bbox"], det_b["bbox"])
                area_a = self._area(det_a["bbox"])
                area_b = self._area(det_b["bbox"])

                if self.iou_occlude_min < iou < self.iou_occlude_max:
                    if area_a < area_b:
                        detections[i]["occluded"]   = True
                        detections[i]["occluded_by"] = det_b.get("label", "object")
        return detections

    # ── extrapolation for fully-occluded objects ──────────────────────────────

    def _extrapolate_occluded(self, tracks: List[Dict]) -> List[Dict]:
        """
        For tracks that disappeared (time_since_update > 0),
        extrapolate their position using constant velocity.
        """
        extrapolated = []
        for t in tracks:
            if t.get("time_since_update", 0) == 0:
                continue              # visible this frame — skip
            if t.get("time_since_update", 0) > 5:
                continue              # too stale — ignore

            traj = t.get("trajectory", [])
            if len(traj) < 2:
                continue

            # Constant-velocity extrapolation
            dx   = traj[-1][0] - traj[-2][0]
            dy   = traj[-1][1] - traj[-2][1]
            last = t["bbox"]
            predicted_bbox = [
                int(last[0] + dx), int(last[1] + dy),
                int(last[2] + dx), int(last[3] + dy),
            ]

            extrapolated.append({
                "label":        t.get("label", "person"),
                "confidence":   t.get("confidence", 0.5) * 0.8,  # reduced conf
                "bbox":         predicted_bbox,
                "class_id":     0,
                "occluded":     True,
                "extrapolated": True,
                "track_id":     t.get("id"),
                "depth_rank":   999,
                "depth_score":  0.0,
            })

        return extrapolated

    # ── geometry helpers ──────────────────────────────────────────────────────

    def _iou(self, a: List[int], b: List[int]) -> float:
        xa = max(a[0], b[0]); ya = max(a[1], b[1])
        xb = min(a[2], b[2]); yb = min(a[3], b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        union = self._area(a) + self._area(b) - inter
        return inter / union if union > 0 else 0.0

    def _area(self, bbox: List[int]) -> float:
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
