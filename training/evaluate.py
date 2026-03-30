"""
Evaluate VIGIL pipeline on UCF-Crime dataset.

Usage:
    python training/evaluate.py \
        --dataset_path /path/to/ucf-crime \
        --max_videos 20
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

ANOMALOUS_CATEGORIES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
    'Explosion', 'Fighting', 'RoadAccidents', 'Robbery',
    'Shooting', 'Shoplifting', 'Stealing', 'Vandalism',
]


def get_ground_truth(video_path: str) -> int:
    """Return 1 if anomalous, 0 if normal based on folder name."""
    path_str = str(video_path)
    for cat in ANOMALOUS_CATEGORIES:
        if cat in path_str:
            return 1
    return 0


def evaluate(dataset_path: str, max_videos: int = 20, model_size: str = 'n'):
    from pipeline.pipeline import VIGILPipeline

    pipeline = VIGILPipeline(
        yolo_model_size=model_size,
        confidence_threshold=0.4,
        output_dir='data/outputs/eval',
    )

    y_true, y_pred, y_scores = [], [], []
    video_results = []

    # Collect sample videos
    sample_videos = []
    cat_counts = defaultdict(int)
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if not f.lower().endswith(('.mp4', '.avi')):
                continue
            cat = next((c for c in ANOMALOUS_CATEGORIES + ['Normal_Videos'] if c in root), 'Unknown')
            if cat_counts[cat] >= 2:
                continue
            sample_videos.append(os.path.join(root, f))
            cat_counts[cat] += 1
        if len(sample_videos) >= max_videos:
            break

    logger.info(f"Evaluating on {len(sample_videos)} videos...")

    for i, vp in enumerate(sample_videos):
        logger.info(f"[{i+1}/{len(sample_videos)}] {Path(vp).name}")
        gt = get_ground_truth(vp)
        try:
            result = pipeline.process_video(vp, generate_report=False)
            pred = 1 if result.get('incident_count', 0) > 0 else 0
            score = result.get('max_confidence', 0.0)
        except Exception as e:
            logger.error(f"Failed: {e}")
            pred, score = 0, 0.0

        y_true.append(gt)
        y_pred.append(pred)
        y_scores.append(score)
        video_results.append({'path': vp, 'gt': gt, 'pred': pred, 'score': score})

    # Metrics
    try:
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            confusion_matrix, classification_report
        )
        import numpy as np

        logger.info("\n" + "="*50)
        logger.info("EVALUATION RESULTS")
        logger.info("="*50)
        logger.info(classification_report(y_true, y_pred, target_names=['Normal', 'Anomalous']))

        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            logger.info(f"AUC-ROC:           {auc:.4f}")
            logger.info(f"Average Precision: {ap:.4f}")
        else:
            logger.warning("Only one class in ground truth — skipping AUC metrics")

        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")

    except ImportError:
        logger.warning("scikit-learn not installed — skipping metrics")

    # Save results
    out_path = 'data/outputs/eval/evaluation_results.json'
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'video_results': video_results, 'y_true': y_true, 'y_pred': y_pred}, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--max_videos', type=int, default=20)
    parser.add_argument('--model_size', type=str, default='n')
    args = parser.parse_args()
    evaluate(args.dataset_path, args.max_videos, args.model_size)


if __name__ == '__main__':
    main()
