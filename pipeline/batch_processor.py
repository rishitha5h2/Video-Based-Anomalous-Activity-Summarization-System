import os
import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BatchProcessor:
    def __init__(self, pipeline, max_workers: int = 2):
        self.pipeline = pipeline
        self.max_workers = max_workers

    def process_batch(self, video_paths: List[str], generate_reports: bool = True) -> List[Dict]:
        results = []
        with tqdm(total=len(video_paths), desc="Batch Processing") as pbar:
            for vp in video_paths:
                try:
                    r = self.pipeline.process_video(vp, generate_report=generate_reports)
                    results.append(r)
                except Exception as e:
                    logger.error(f"Failed {vp}: {e}")
                    results.append({"video_path": vp, "error": str(e), "incident_count": 0})
                pbar.update(1)
        return results
