"""src/tracking/id_manager.py — persistent ID management across occlusion gaps."""

from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class IDManager:
    """
    Assigns and maintains stable numeric IDs for tracked objects.
    When an object disappears (occlusion) and re-appears within *grace_frames*,
    it is reassigned its original ID rather than getting a new one.
    """

    def __init__(self, grace_frames: int = 15, max_distance_px: float = 80.0):
        self.grace_frames    = grace_frames
        self.max_distance_px = max_distance_px

        self._next_id: int          = 1
        # id → last known centre (cx, cy)
        self._active:  Dict[int, Tuple[float, float]] = {}
        # id → frames since last seen
        self._missing: Dict[int, int]                  = {}

    # ── public API ─────────────────────────────────────────────────────────────

    def assign(self, centres: List[Tuple[float, float]]) -> List[int]:
        """
        Given a list of detected object centres for the current frame,
        return a list of stable IDs (same order as *centres*).
        """
        assigned_ids: List[Optional[int]] = [None] * len(centres)
        used_active_ids: set = set()

        # Try to match each centre to an existing active or recently-missing ID
        for det_i, ctr in enumerate(centres):
            best_id, best_dist = None, float("inf")

            for existing_id, last_ctr in {**self._active, **{k: self._active.get(k, ctr)
                                            for k in self._missing}}.items():
                if existing_id in used_active_ids:
                    continue
                dist = _euclidean(ctr, last_ctr)
                if dist < best_dist and dist < self.max_distance_px:
                    best_dist = dist
                    best_id   = existing_id

            if best_id is not None:
                assigned_ids[det_i] = best_id
                used_active_ids.add(best_id)
                self._active[best_id] = ctr
                self._missing.pop(best_id, None)

        # Assign new IDs to unmatched detections
        for det_i, ctr in enumerate(centres):
            if assigned_ids[det_i] is None:
                new_id = self._next_id
                self._next_id += 1
                assigned_ids[det_i] = new_id
                self._active[new_id] = ctr

        # Age missing IDs; evict stale ones
        for tid in list(self._active):
            if tid not in used_active_ids:
                self._missing[tid] = self._missing.get(tid, 0) + 1
                if self._missing[tid] > self.grace_frames:
                    del self._active[tid]
                    del self._missing[tid]

        return [i for i in assigned_ids]   # type: ignore[return-value]

    def reset(self):
        self._next_id = 1
        self._active.clear()
        self._missing.clear()

    @property
    def active_count(self) -> int:
        return len(self._active)


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
