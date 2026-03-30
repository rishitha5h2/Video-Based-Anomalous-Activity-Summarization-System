"""api/routers/process.py — Processing trigger and status endpoints."""

import asyncio
import logging
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, HTTPException, BackgroundTasks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/process", tags=["process"])

# Shared job store (imported by main.py too)
jobs: Dict[str, Dict] = {}


def register_job(job_id: str, filename: str, path: str):
    jobs[job_id] = {
        "id":           job_id,
        "status":       "uploaded",
        "filename":     filename,
        "path":         path,
        "created_at":   datetime.now().isoformat(),
        "completed_at": None,
        "results":      None,
        "error":        None,
    }


@router.post("/{job_id}")
async def trigger_processing(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found — upload first")
    if jobs[job_id]["status"] == "processing":
        raise HTTPException(400, "Already processing")
    jobs[job_id]["status"] = "processing"
    background_tasks.add_task(_run, job_id, jobs[job_id]["path"])
    return {"job_id": job_id, "status": "processing"}


async def _run(job_id: str, video_path: str):
    try:
        from pipeline.pipeline import VIGILPipeline
        import os
        pipeline = VIGILPipeline(
            yolo_model_size="n",
            confidence_threshold=0.4,
            output_dir="data/outputs",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        loop    = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, pipeline.process_video, video_path)
        jobs[job_id]["status"]       = "completed"
        jobs[job_id]["results"]      = results
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
    except Exception as e:
        logger.error(f"Pipeline error [{job_id}]: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"]  = str(e)


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id].copy()
    j.pop("results", None)
    return j


@router.get("/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j["status"] != "completed":
        raise HTTPException(400, f"Job not completed — status: {j['status']}")
    # Trim heavy arrays before returning
    r = dict(j["results"])
    for inc in r.get("incidents", []):
        inc.pop("frame_start", None)
        inc.pop("frame_end",   None)
    return r
