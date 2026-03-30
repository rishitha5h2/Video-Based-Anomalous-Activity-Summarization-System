#!/usr/bin/env python3
"""
run.py — VIGIL command-line entry point.

Usage examples:

  # Analyse a single video
  python run.py analyse --video path/to/video.mp4

  # Batch analyse a directory
  python run.py batch --dir data/raw/Fighting/ --limit 5

  # Download UCF-Crime dataset
  python run.py download

  # Start the FastAPI server
  python run.py serve

  # Train the anomaly model
  python run.py train --dataset /path/to/ucf-crime
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vigil")


# ── SUB-COMMANDS ──────────────────────────────────────────────────────────────

def cmd_analyse(args):
    from pipeline.pipeline import VIGILPipeline

    if not Path(args.video).exists():
        logger.error(f"File not found: {args.video}")
        sys.exit(1)

    pipeline = VIGILPipeline(
        yolo_model_size=args.model,
        confidence_threshold=args.confidence,
        anomaly_threshold=args.anomaly_threshold,
        output_dir=args.output_dir,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    logger.info(f"Analysing: {args.video}")
    results = pipeline.process_video(args.video)

    print(f"\n{'='*55}")
    print(f"  VIGIL Analysis — {results['video_name']}")
    print(f"{'='*55}")
    print(f"  Duration   : {results['duration']:.1f}s")
    print(f"  Incidents  : {results['incident_count']}")
    print(f"  Status     : {'⚠  ANOMALOUS' if results['is_anomalous'] else '✓  NORMAL'}")

    for inc in results["incidents"]:
        ms, ss = divmod(int(inc["start_time"]), 60)
        me, se = divmod(int(inc["end_time"]),   60)
        print(f"\n  Incident #{inc['id']}  [{inc['type'].upper()}]")
        print(f"    Time       : {ms:02d}:{ss:02d} → {me:02d}:{se:02d}")
        print(f"    Confidence : {inc['confidence']:.0%}")
        print(f"    Persons    : {inc['num_persons']}")

    if results.get("pdf_report"):
        print(f"\n  PDF Report : {results['pdf_report']}")
    if results.get("json_report"):
        print(f"  JSON Report: {results['json_report']}")
    print()


def cmd_batch(args):
    from pipeline.pipeline       import VIGILPipeline
    from pipeline.batch_processor import BatchProcessor
    from src.ingestion.video_loader import VideoLoader

    loader  = VideoLoader()
    videos  = loader.find_videos(args.dir)
    if args.limit:
        videos = videos[:args.limit]

    if not videos:
        logger.error(f"No video files found in: {args.dir}")
        sys.exit(1)

    logger.info(f"Found {len(videos)} videos in {args.dir}")

    pipeline = VIGILPipeline(
        yolo_model_size=args.model,
        output_dir=args.output_dir,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    batch   = BatchProcessor(pipeline)
    results = batch.process_batch(videos)

    print(f"\n{'='*65}")
    print(f"  VIGIL Batch Results  ({len(results)} videos)")
    print(f"{'='*65}")
    anom = sum(1 for r in results if r.get("is_anomalous"))
    print(f"  Anomalous : {anom} / {len(results)}")
    print(f"  Total incidents : {sum(r.get('incident_count',0) for r in results)}")
    print()


def cmd_download(args):
    from training.dataset_loader import UCFCrimeDataset
    ds = UCFCrimeDataset()
    path = ds.download()
    print(f"\n  Dataset ready at: {path}")
    print(f"  Stats: {ds.stats}\n")


def cmd_serve(args):
    import uvicorn
    logger.info(f"Starting VIGIL API on http://{args.host}:{args.port}")
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload)


def cmd_train(args):
    from training.train_anomaly_model import main as train_main

    sys.argv = [
        "train",
        "--dataset_path", args.dataset,
        "--epochs",       str(args.epochs),
        "--output",       args.output,
    ]
    train_main()


# ── ARGUMENT PARSER ───────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="vigil",
        description="VIGIL — Video Anomalous Activity Detection System",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # analyse
    sp = sub.add_parser("analyse", help="Analyse a single video file")
    sp.add_argument("--video",             required=True,  help="Path to video file")
    sp.add_argument("--model",             default="n",    help="YOLOv8 model size (n/s/m/l/x)")
    sp.add_argument("--confidence",        type=float, default=0.4)
    sp.add_argument("--anomaly-threshold", type=float, default=0.6, dest="anomaly_threshold")
    sp.add_argument("--output-dir",        default="data/outputs",  dest="output_dir")
    sp.set_defaults(func=cmd_analyse)

    # batch
    sp = sub.add_parser("batch", help="Batch analyse a directory of videos")
    sp.add_argument("--dir",        required=True)
    sp.add_argument("--limit",      type=int, default=None)
    sp.add_argument("--model",      default="n")
    sp.add_argument("--output-dir", default="data/outputs", dest="output_dir")
    sp.set_defaults(func=cmd_batch)

    # download
    sp = sub.add_parser("download", help="Download UCF-Crime dataset via kagglehub")
    sp.set_defaults(func=cmd_download)

    # serve
    sp = sub.add_parser("serve", help="Start the FastAPI server")
    sp.add_argument("--host",   default="0.0.0.0")
    sp.add_argument("--port",   type=int, default=8000)
    sp.add_argument("--reload", action="store_true")
    sp.set_defaults(func=cmd_serve)

    # train
    sp = sub.add_parser("train", help="Train the anomaly autoencoder")
    sp.add_argument("--dataset", required=True, help="Path to UCF-Crime dataset")
    sp.add_argument("--epochs",  type=int, default=50)
    sp.add_argument("--output",  default="models/weights/anomaly_autoencoder.pth")
    sp.set_defaults(func=cmd_train)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)
