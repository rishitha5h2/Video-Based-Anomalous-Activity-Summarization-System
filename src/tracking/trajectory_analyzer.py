import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TrajectoryAnalyzer:
    """Analyse object trajectories for loitering, crowd, and erratic movement."""

    def __init__(
        self,
        loiter_distance_threshold: float = 50.0,
        loiter_min_frames: int = 30,
        crowd_threshold: int = 4,
    ):
        self.loiter_distance_threshold = loiter_distance_threshold
        self.loiter_min_frames = loiter_min_frames
        self.crowd_threshold = crowd_threshold

    def analyze(self, tracks: List[Dict]) -> Dict:
        """Return a summary of trajectory-based anomalies for this frame."""
        results = {
            'loitering_ids': [],
            'crowd_detected': False,
            'crowd_size': 0,
            'erratic_ids': [],
            'anomaly_score': 0.0,
        }

        person_tracks = [t for t in tracks if t['label'] == 'person']
        results['crowd_size'] = len(person_tracks)
        results['crowd_detected'] = len(person_tracks) >= self.crowd_threshold

        for track in person_tracks:
            traj = track.get('trajectory', [])
            if len(traj) >= self.loiter_min_frames:
                if self._is_loitering(traj):
                    results['loitering_ids'].append(track['id'])
                if self._is_erratic(traj):
                    results['erratic_ids'].append(track['id'])

        # Compute combined anomaly score
        score = 0.0
        if results['crowd_detected']:
            score += min(0.4, (results['crowd_size'] - self.crowd_threshold) * 0.1)
        score += len(results['loitering_ids']) * 0.2
        score += len(results['erratic_ids']) * 0.15
        results['anomaly_score'] = min(1.0, score)

        return results

    def _is_loitering(self, trajectory: List[Tuple[float, float]]) -> bool:
        """Check if the object has stayed within a small area."""
        if len(trajectory) < self.loiter_min_frames:
            return False
        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]
        span = ((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2) ** 0.5
        return span < self.loiter_distance_threshold

    def _is_erratic(self, trajectory: List[Tuple[float, float]]) -> bool:
        """Check if movement direction changes sharply and frequently."""
        if len(trajectory) < 6:
            return False
        angles = []
        for i in range(2, len(trajectory)):
            dx1 = trajectory[i-1][0] - trajectory[i-2][0]
            dy1 = trajectory[i-1][1] - trajectory[i-2][1]
            dx2 = trajectory[i][0] - trajectory[i-1][0]
            dy2 = trajectory[i][1] - trajectory[i-1][1]
            if (dx1**2+dy1**2)**0.5 < 2 or (dx2**2+dy2**2)**0.5 < 2:
                continue
            dot = dx1*dx2 + dy1*dy2
            mag = ((dx1**2+dy1**2)**0.5) * ((dx2**2+dy2**2)**0.5)
            if mag > 0:
                cos_a = max(-1, min(1, dot/mag))
                angles.append(np.degrees(np.arccos(cos_a)))
        if not angles:
            return False
        sharp_turns = sum(1 for a in angles if a > 90)
        return sharp_turns / len(angles) > 0.4

    def compute_flow_direction(self, trajectories: List[List[Tuple]]) -> Tuple[float, float]:
        """Compute the dominant flow direction across all trajectories."""
        vx, vy = [], []
        for traj in trajectories:
            if len(traj) >= 2:
                vx.append(traj[-1][0] - traj[0][0])
                vy.append(traj[-1][1] - traj[0][1])
        if not vx:
            return (0.0, 0.0)
        return (float(np.mean(vx)), float(np.mean(vy)))
