"""src/incident/timeframe_extractor.py — Extract precise start/end timestamps."""

from typing import List, Tuple
import numpy as np


def extract_timeframes(
    anomaly_flags: List[bool],
    fps: float,
    min_duration: float = 2.0,
    merge_gap: float = 3.0,
) -> List[Tuple[float, float]]:
    """
    Convert a boolean flag array (one per sampled frame) into a list of
    (start_seconds, end_seconds) tuples representing anomalous segments.

    Parameters
    ----------
    anomaly_flags  : True where anomaly detected
    fps            : sampling rate (frames per second)
    min_duration   : minimum segment length to keep (seconds)
    merge_gap      : merge segments closer than this (seconds)
    """
    if not anomaly_flags:
        return []

    # 1. Find raw segments
    segments: List[Tuple[float, float]] = []
    in_seg, start_i = False, 0

    for i, flag in enumerate(anomaly_flags):
        if flag and not in_seg:
            in_seg  = True
            start_i = i
        elif not flag and in_seg:
            in_seg = False
            segments.append((start_i / fps, i / fps))

    if in_seg:
        segments.append((start_i / fps, len(anomaly_flags) / fps))

    # 2. Merge close segments
    merged: List[Tuple[float, float]] = []
    for seg in segments:
        if merged and (seg[0] - merged[-1][1]) <= merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(seg)

    # 3. Filter short segments
    return [(s, e) for s, e in merged if (e - s) >= min_duration]


def flags_from_scores(
    scores: List[float],
    threshold: float,
    window: int = 3,
) -> List[bool]:
    """
    Smooth scores with a rolling average and threshold.

    Parameters
    ----------
    scores    : per-frame anomaly scores
    threshold : flag frame if smoothed score exceeds this
    window    : rolling average window size
    """
    if not scores:
        return []
    arr      = np.array(scores, dtype=float)
    smoothed = np.convolve(arr, np.ones(window) / window, mode="same")
    return [bool(s > threshold) for s in smoothed]


def format_timeframe(start: float, end: float) -> str:
    """Return human-readable 'MM:SS → MM:SS' string."""
    def fmt(t: float) -> str:
        m, s = divmod(int(t), 60)
        return f"{m:02d}:{s:02d}"
    return f"{fmt(start)} → {fmt(end)}"
