"""
Train the VIGIL anomaly autoencoder on UCF-Crime normal videos.

Usage:
    python training/train_anomaly_model.py \
        --dataset_path /path/to/ucf-crime \
        --epochs 50 \
        --output models/weights/anomaly_autoencoder.pth
"""

import argparse
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.frame_processor import FrameProcessor
from src.detection.anomaly_scorer import AnomalyScorer

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def collect_normal_frames(dataset_path: str, max_videos: int = 50, frames_per_video: int = 30):
    """Collect frames from Normal_Videos category."""
    processor = FrameProcessor(target_size=(64, 64), fps=2)
    normal_frames = []

    normal_dirs = ['Normal_Videos', 'Normal', 'normal']
    video_count = 0

    for root, dirs, files in os.walk(dataset_path):
        if not any(nd in root for nd in normal_dirs):
            continue
        for f in files:
            if not f.lower().endswith(('.mp4', '.avi', '.mkv')):
                continue
            video_path = os.path.join(root, f)
            frames = processor.extract_frames(video_path, max_frames=frames_per_video)
            normal_frames.extend(frames)
            video_count += 1
            logger.info(f"[{video_count}/{max_videos}] Loaded {len(frames)} frames from {f}")
            if video_count >= max_videos:
                break
        if video_count >= max_videos:
            break

    logger.info(f"Total normal frames collected: {len(normal_frames)}")
    return normal_frames


def main():
    parser = argparse.ArgumentParser(description="Train VIGIL anomaly autoencoder")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to UCF-Crime dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--max_videos', type=int, default=50)
    parser.add_argument('--frames_per_video', type=int, default=30)
    parser.add_argument('--output', type=str, default='models/weights/anomaly_autoencoder.pth')
    args = parser.parse_args()

    # Collect frames
    logger.info("Collecting normal frames for training...")
    frames = collect_normal_frames(args.dataset_path, args.max_videos, args.frames_per_video)

    if len(frames) < 32:
        logger.error(f"Too few frames ({len(frames)}). Need at least 32. Check dataset path.")
        sys.exit(1)

    # Train
    scorer = AnomalyScorer(latent_dim=args.latent_dim, input_shape=(64, 64, 3))
    logger.info(f"Training on {len(frames)} frames for {args.epochs} epochs...")
    losses = scorer.train(frames, epochs=args.epochs, batch_size=args.batch_size)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    scorer.save(args.output)
    logger.info(f"Model saved to {args.output}")
    logger.info(f"Final loss: {losses[-1]:.6f} | Threshold: {scorer.threshold:.6f}")

    # Plot loss curve
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(losses, linewidth=2, color='#e8453c')
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.title('Anomaly Autoencoder Training Loss')
        plt.grid(True, alpha=0.3)
        loss_plot = args.output.replace('.pth', '_loss.png')
        plt.savefig(loss_plot, dpi=150, bbox_inches='tight')
        logger.info(f"Loss plot saved: {loss_plot}")
    except ImportError:
        pass


if __name__ == '__main__':
    main()
