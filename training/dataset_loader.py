"""
UCF-Crime dataset loader using kagglehub.

Usage:
    from training.dataset_loader import UCFCrimeDataset
    dataset = UCFCrimeDataset()
    dataset.download()
    videos = dataset.get_videos(category='Fighting', limit=10)
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict
from collections import defaultdict

logger = logging.getLogger(__name__)

CATEGORIES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
    'Explosion', 'Fighting', 'RoadAccidents', 'Robbery',
    'Shooting', 'Shoplifting', 'Stealing', 'Vandalism', 'Normal_Videos',
]

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov'}


class UCFCrimeDataset:
    """Wrapper for the UCF-Crime dataset, downloaded via kagglehub."""

    KAGGLE_ID = 'odins0n/ucf-crime-dataset'

    def __init__(self, local_path: Optional[str] = None):
        self.dataset_path: Optional[str] = local_path
        self._index: Dict[str, List[str]] = defaultdict(list)
        if local_path and Path(local_path).exists():
            self._build_index()

    def download(self) -> str:
        """Download the dataset using kagglehub and build the index."""
        try:
            import kagglehub
            logger.info(f"Downloading {self.KAGGLE_ID} via kagglehub...")
            path = kagglehub.dataset_download(self.KAGGLE_ID)
            self.dataset_path = path
            self._build_index()
            logger.info(f"Dataset ready at: {path}")
            return path
        except ImportError:
            raise RuntimeError("kagglehub not installed. Run: pip install kagglehub")
        except Exception as e:
            raise RuntimeError(f"Download failed: {e}")

    def _build_index(self):
        """Walk dataset directory and index videos by category."""
        self._index = defaultdict(list)
        for root, _, files in os.walk(self.dataset_path):
            for f in files:
                if Path(f).suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                cat = self._infer_category(root)
                self._index[cat].append(os.path.join(root, f))
        total = sum(len(v) for v in self._index.values())
        logger.info(f"Indexed {total} videos across {len(self._index)} categories")

    def _infer_category(self, path: str) -> str:
        for cat in CATEGORIES:
            if cat in path:
                return cat
        return 'Unknown'

    def get_videos(self, category: Optional[str] = None, limit: Optional[int] = None) -> List[str]:
        """Return video paths for a given category (or all if None)."""
        if not self._index:
            raise RuntimeError("Dataset not loaded. Call download() first.")
        if category:
            videos = self._index.get(category, [])
        else:
            videos = [v for vlist in self._index.values() for v in vlist]
        if limit:
            videos = videos[:limit]
        return videos

    def get_normal_videos(self, limit: Optional[int] = None) -> List[str]:
        return self.get_videos('Normal_Videos', limit)

    def get_anomalous_videos(self, limit: Optional[int] = None) -> List[str]:
        vids = [v for cat, vlist in self._index.items() if cat != 'Normal_Videos' for v in vlist]
        return vids[:limit] if limit else vids

    @property
    def stats(self) -> Dict:
        return {
            'total': sum(len(v) for v in self._index.values()),
            'by_category': {k: len(v) for k, v in self._index.items()},
            'categories': list(self._index.keys()),
            'dataset_path': self.dataset_path,
        }

    @property
    def categories(self) -> List[str]:
        return list(self._index.keys())
