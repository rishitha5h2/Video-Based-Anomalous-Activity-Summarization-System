"""
training/finetune_yolo.py — Fine-tune YOLOv8 on UCF-Crime frames.

This script:
  1. Extracts frames from UCF-Crime videos
  2. Creates a minimal YOLO dataset structure (images + labels)
  3. Fine-tunes YOLOv8 for person/activity detection

Usage:
  python training/finetune_yolo.py \
      --dataset_path /path/to/ucf-crime \
      --output_dir   models/finetune_yolo \
      --epochs 20
"""

import argparse
import os
import sys
import shutil
import random
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CATEGORIES = [
    "Abuse","Arrest","Arson","Assault","Burglary",
    "Explosion","Fighting","RoadAccidents","Robbery",
    "Shooting","Shoplifting","Stealing","Vandalism","Normal_Videos",
]


def extract_frames_for_yolo(
    dataset_path: str,
    output_dir: str,
    frames_per_video: int = 10,
    max_videos_per_cat: int = 5,
) -> dict:
    """Extract frames and create YOLO directory structure."""
    import cv2

    images_dir = Path(output_dir) / "images"
    labels_dir = Path(output_dir) / "labels"
    for split in ("train", "val"):
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    frame_count = {"train": 0, "val": 0}

    for cat in CATEGORIES:
        videos = []
        for root, _, files in os.walk(dataset_path):
            if cat not in root:
                continue
            for f in files:
                if f.lower().endswith((".mp4", ".avi")):
                    videos.append(os.path.join(root, f))
            if len(videos) >= max_videos_per_cat:
                break
        videos = videos[:max_videos_per_cat]

        for vid_path in videos:
            cap   = cv2.VideoCapture(vid_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step  = max(1, total // frames_per_video)

            for i in range(frames_per_video):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
                ret, frame = cap.read()
                if not ret:
                    break

                split   = "val" if random.random() < 0.2 else "train"
                stem    = f"{cat}_{Path(vid_path).stem}_f{i}"
                img_path = images_dir / split / f"{stem}.jpg"
                lbl_path = labels_dir / split / f"{stem}.txt"

                cv2.imwrite(str(img_path), frame)

                # Auto-label: person class (0) full frame — placeholder
                # In a real pipeline, use a pre-trained detector to generate labels
                h, w = frame.shape[:2]
                with open(lbl_path, "w") as lf:
                    lf.write("0 0.5 0.5 0.9 0.9\n")  # cx cy w h normalised

                frame_count[split] += 1
            cap.release()

        logger.info(f"  {cat}: processed {len(videos)} videos")

    return frame_count


def create_yaml(output_dir: str) -> str:
    yaml_path = os.path.join(output_dir, "vigil.yaml")
    content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val:   images/val
nc: 1
names:
  0: person
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on UCF-Crime")
    parser.add_argument("--dataset_path",        required=True)
    parser.add_argument("--output_dir",          default="models/finetune_yolo")
    parser.add_argument("--epochs",      type=int, default=20)
    parser.add_argument("--batch",       type=int, default=16)
    parser.add_argument("--imgsz",       type=int, default=640)
    parser.add_argument("--model",                default="yolov8n.pt")
    parser.add_argument("--frames_per_video", type=int, default=10)
    parser.add_argument("--max_videos",      type=int, default=5)
    args = parser.parse_args()

    logger.info("Step 1/3 — Extracting frames for YOLO dataset...")
    counts = extract_frames_for_yolo(
        args.dataset_path, args.output_dir,
        args.frames_per_video, args.max_videos,
    )
    logger.info(f"  Train frames: {counts['train']}  Val frames: {counts['val']}")

    logger.info("Step 2/3 — Creating dataset YAML...")
    yaml_path = create_yaml(args.output_dir)

    logger.info("Step 3/3 — Fine-tuning YOLOv8...")
    try:
        from ultralytics import YOLO
        model  = YOLO(args.model)
        results = model.train(
            data=yaml_path,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=args.output_dir,
            name="vigil_yolo",
            exist_ok=True,
            verbose=True,
        )
        save_path = os.path.join(args.output_dir, "vigil_yolo", "weights", "best.pt")
        logger.info(f"Fine-tuned weights saved: {save_path}")
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
