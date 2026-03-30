import subprocess
import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_video_duration(path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"ffprobe failed: {e}")
        return 0.0


def cut_video_segment(input_path: str, output_path: str, start: float, end: float) -> bool:
    """Extract a video segment using ffmpeg."""
    try:
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ss', str(start), '-to', str(end),
            '-c', 'copy', output_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=60, check=True)
        return True
    except Exception as e:
        logger.error(f"ffmpeg cut failed: {e}")
        return False


def add_bbox_overlay(input_path: str, output_path: str, incidents: list) -> bool:
    """Add text-based incident markers to video using ffmpeg drawtext."""
    try:
        filters = []
        for inc in incidents:
            s, e = inc['start_time'], inc['end_time']
            label = inc['type'].upper()
            filters.append(
                f"drawtext=text='{label}':x=10:y=10:fontsize=24:fontcolor=red:enable='between(t,{s},{e})'"
            )
        filter_str = ','.join(filters) if filters else 'null'
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-vf', filter_str,
            '-c:a', 'copy', output_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=300, check=True)
        return True
    except Exception as e:
        logger.error(f"ffmpeg overlay failed: {e}")
        return False


def get_video_thumbnail(video_path: str, timestamp: float, output_path: str) -> bool:
    """Extract a single frame as JPEG thumbnail."""
    try:
        cmd = [
            'ffmpeg', '-y', '-ss', str(timestamp),
            '-i', video_path, '-vframes', '1', output_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=15, check=True)
        return os.path.exists(output_path)
    except Exception as e:
        logger.error(f"Thumbnail extraction failed: {e}")
        return False
