import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

CRIME_RELEVANT_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon",
    56: "chair", 57: "couch", 60: "dining table", 63: "laptop",
    64: "mouse", 67: "cell phone", 73: "book", 76: "scissors",
}

BBOX_COLORS = {
    "person": (0, 0, 255),
    "knife": (0, 128, 255),
    "gun": (0, 0, 200),
    "car": (255, 128, 0),
    "truck": (255, 165, 0),
    "default": (0, 200, 0),
}


class ObjectDetector:
    """YOLOv8-based object detector with occlusion-aware detection."""

    def __init__(self, model_size: str = "x", confidence: float = 0.4, iou: float = 0.5):
        self.confidence = confidence
        self.iou = iou
        self.model = None
        self._load_model(model_size)

    def _load_model(self, model_size: str):
        try:
            from ultralytics import YOLO
            model_name = f"yolov8{model_size}.pt"
            self.model = YOLO(model_name)
            logger.info(f"Loaded YOLOv8{model_size}")
        except Exception as e:
            logger.warning(f"Could not load YOLO: {e}. Using mock detector.")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Dict]:
        if frame.dtype != np.uint8:
            frame_uint8 = (frame * 255).astype(np.uint8)
        else:
            frame_uint8 = frame

        if self.model is None:
            return self._mock_detect(frame_uint8)

        try:
            results = self.model(frame_uint8, conf=self.confidence, iou=self.iou, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = r.names.get(cls_id, "unknown")
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "label": label,
                        "confidence": float(box.conf[0]),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "class_id": cls_id,
                        "occluded": False,
                    })
            detections = self._handle_occlusions(detections, frame_uint8.shape[:2])
            return detections
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def _handle_occlusions(self, detections: List[Dict], frame_shape: Tuple) -> List[Dict]:
        """Mark detections that are partially occluded by other objects."""
        for i, det_a in enumerate(detections):
            for j, det_b in enumerate(detections):
                if i == j:
                    continue
                iou = self._compute_iou(det_a["bbox"], det_b["bbox"])
                if 0.1 < iou < 0.5:
                    area_a = (det_a["bbox"][2] - det_a["bbox"][0]) * (det_a["bbox"][3] - det_a["bbox"][1])
                    area_b = (det_b["bbox"][2] - det_b["bbox"][0]) * (det_b["bbox"][3] - det_b["bbox"][1])
                    if area_a < area_b:
                        detections[i]["occluded"] = True
        return detections

    def _compute_iou(self, box_a: List[int], box_b: List[int]) -> float:
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _mock_detect(self, frame: np.ndarray) -> List[Dict]:
        h, w = frame.shape[:2]
        return [
            {"label": "person", "confidence": 0.87, "bbox": [int(w*0.2), int(h*0.1), int(w*0.4), int(h*0.9)], "class_id": 0, "occluded": False},
            {"label": "person", "confidence": 0.79, "bbox": [int(w*0.5), int(h*0.15), int(w*0.75), int(h*0.85)], "class_id": 0, "occluded": False},
        ]

    def draw_detections(self, frame: np.ndarray, detections: List[Dict], show_occluded: bool = True) -> np.ndarray:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            occluded = det.get("occluded", False)
            color = BBOX_COLORS.get(label, BBOX_COLORS["default"])
            thickness = 1 if occluded else 2
            style = "--" if occluded else "-"
            if occluded:
                for i in range(0, (x2 - x1), 10):
                    cv2.line(out, (x1 + i, y1), (min(x1 + i + 5, x2), y1), color, thickness)
                    cv2.line(out, (x1 + i, y2), (min(x1 + i + 5, x2), y2), color, thickness)
                for i in range(0, (y2 - y1), 10):
                    cv2.line(out, (x1, y1 + i), (x1, min(y1 + i + 5, y2)), color, thickness)
                    cv2.line(out, (x2, y1 + i), (x2, min(y1 + i + 5, y2)), color, thickness)
            else:
                cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            text = f"{label} {conf:.0%}" + (" [OCC]" if occluded else "")
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out
