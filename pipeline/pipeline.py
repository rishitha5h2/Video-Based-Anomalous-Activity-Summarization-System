import os
import cv2
import time
import logging
from pathlib import Path
from typing import Dict, Optional, List

from src.preprocessing.frame_processor import FrameProcessor
from src.preprocessing.background_subtractor import BackgroundSubtractor
from src.detection.object_detector import ObjectDetector
from src.detection.anomaly_scorer import AnomalyScorer
from src.detection.action_recognizer import ActionRecognizer
from src.incident.incident_detector import IncidentDetector
from src.summarization.llm_summarizer import LLMSummarizer
from src.reporting.pdf_generator import PDFReportGenerator
from src.reporting.json_exporter import JSONExporter

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
class VIGILPipeline:
    """End-to-end video anomaly detection and reporting pipeline."""

    def __init__(
        self,
        yolo_model_size: str = "x",
        confidence_threshold: float = 0.4,
        anomaly_threshold: float = 0.6,
        output_dir: str = "data/outputs",
        api_key: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.anomaly_threshold = anomaly_threshold

        logger.info("Initializing VIGIL pipeline components...")
        self.frame_processor = FrameProcessor(target_size=(640, 360), fps=5)
        self.bg_subtractor = BackgroundSubtractor()
        self.detector = ObjectDetector(model_size=yolo_model_size, confidence=confidence_threshold)
        self.anomaly_scorer = AnomalyScorer(latent_dim=128, input_shape=(64, 64, 3))
        self.action_recognizer = ActionRecognizer(clip_length=16)
        self.incident_detector = IncidentDetector(
            confidence_threshold=confidence_threshold,
            output_dir=str(output_dir),
        )
        self.summarizer = LLMSummarizer(api_key=api_key)
        self.pdf_gen = PDFReportGenerator()
        self.json_exp = JSONExporter()
        logger.info("Pipeline ready.")

    def process_video(self, video_path: str, original_filename: str = None, generate_report: bool = True) -> Dict:
        t0 = time.time()
        video_path = str(video_path)
        # Use original filename if provided, strip extension; else use stem (which may be UUID)
        if original_filename:
            video_name = Path(original_filename).stem
        else:
            video_name = Path(video_path).stem
        logger.info(f"Processing: {video_name}")

        # ── 1. VIDEO INFO ──
        info = self.frame_processor.get_video_info(video_path)
        duration = info.get("duration", 0)
        fps_native = info.get("fps", 25)
        resolution = f"{info.get('width', 0)}x{info.get('height', 0)}"

        # ── 2. FRAME EXTRACTION ──
        logger.info("Extracting frames...")
        frames = self.frame_processor.extract_frames(video_path, max_frames=300)
        if not frames:
            logger.error("No frames extracted.")
            return {"error": "No frames extracted", "video_name": video_name}

        # ── 3. OBJECT DETECTION ──
        logger.info(f"Running object detection on {len(frames)} frames...")
        frame_detections = [self.detector.detect(f) for f in frames]

        # ── 4. ANOMALY SCORING ──
        logger.info("Computing anomaly scores...")
        # Use background subtraction motion as proxy if autoencoder not trained
        motion_scores = self.bg_subtractor.process_sequence(frames)
        anomaly_scores = [min(1.0, s * 5.0) for s in motion_scores]

        # ── 5. ACTION RECOGNITION ──
        logger.info("Running action recognition...")
        action_results = self.action_recognizer.recognize_sequence(frames, step=8)

        # ── 6. INCIDENT DETECTION ──
        logger.info("Detecting incidents...")
        incidents = self.incident_detector.detect_incidents(
            video_path=video_path,
            frame_detections=frame_detections,
            action_results=action_results,
            anomaly_scores=anomaly_scores,
            anomaly_threshold=self.anomaly_threshold,
            fps=self.frame_processor.fps,
        )

        # ── 7. SNAPSHOT EXTRACTION ──
        if incidents:
            logger.info("Extracting incident snapshots...")
            incidents = self.incident_detector.extract_snapshots(
                video_path=video_path,
                incidents=incidents,
                detector=self.detector,
                video_name=video_name,
            )

        # ── 8. SUMMARIZATION ──
        logger.info("Generating summaries...")
        results = {
            "video_name": video_name,
            "video_path": video_path,
            "duration": duration,
            "resolution": resolution,
            "fps": fps_native,
            "frame_count": len(frames),
            "incidents": incidents,
            "incident_count": len(incidents),
            "is_anomalous": len(incidents) > 0,
            "processing_time": 0,
            "summary": "",
            "max_confidence": max((i["confidence"] for i in incidents), default=0),
        }

        for incident in results["incidents"]:
            incident["narrative"] = self.summarizer.summarize_incident(incident, video_name)

        results["summary"] = self.summarizer.summarize_video(results)

        # ── 9. REPORT GENERATION ──
        if generate_report:
            reports_dir = self.output_dir / "reports"
            reports_dir.mkdir(exist_ok=True)
            pdf_path = self.pdf_gen.generate(results, str(reports_dir / f"{video_name}_report.pdf"))
            json_path = self.json_exp.export(results, str(reports_dir / f"{video_name}_report.json"))
            results["pdf_report"] = pdf_path
            results["json_report"] = json_path

        results["processing_time"] = round(time.time() - t0, 2)
        logger.info(f"Done in {results['processing_time']}s — {len(incidents)} incidents found.")
        return results

    def train_anomaly_model(self, normal_video_paths: List[str], epochs: int = 30):
        """Train the autoencoder on normal videos."""
        normal_frames = []
        for vp in normal_video_paths:
            fp = FrameProcessor(target_size=(64, 64), fps=2)
            frames = fp.extract_frames(vp, max_frames=50)
            normal_frames.extend(frames)
        logger.info(f"Training anomaly model on {len(normal_frames)} normal frames...")
        losses = self.anomaly_scorer.train(normal_frames, epochs=epochs)
        save_path = "models/weights/anomaly_autoencoder.pth"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.anomaly_scorer.save(save_path)
        return losses