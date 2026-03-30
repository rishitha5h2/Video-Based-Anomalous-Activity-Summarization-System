from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from datetime import datetime


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float
    occluded: bool = False


class IncidentSchema(BaseModel):
    id: int
    start_time: float
    end_time: float
    duration: float
    type: str
    confidence: float
    num_persons: int = 0
    peak_anomaly_score: float = 0.0
    snapshot_times: List[float] = []
    frame_paths: List[str] = []
    narrative: str = ""


class VideoResultSchema(BaseModel):
    video_name: str
    video_path: str
    duration: float
    resolution: str
    fps: float
    frame_count: int
    is_anomalous: bool
    incident_count: int
    max_confidence: float
    incidents: List[IncidentSchema]
    summary: str
    processing_time: float
    pdf_report: Optional[str] = None
    json_report: Optional[str] = None


class JobSchema(BaseModel):
    id: str
    status: str  # uploaded | processing | completed | failed
    filename: str
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    job_id: str
    status: str
    filename: str


class ProcessRequest(BaseModel):
    yolo_model_size: str = "n"
    confidence_threshold: float = 0.4
    anomaly_threshold: float = 0.6
    generate_report: bool = True
