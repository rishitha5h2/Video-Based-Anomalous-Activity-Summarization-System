"""api/routers/reports.py — PDF and JSON report download endpoints."""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/report", tags=["reports"])


def _get_jobs():
    from api.routers.process import jobs
    return jobs


@router.get("/{job_id}/pdf")
async def download_pdf(job_id: str):
    jobs = _get_jobs()
    _check_completed(jobs, job_id)
    pdf_path = jobs[job_id]["results"].get("pdf_report")
    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(404, "PDF report not found — ensure pipeline ran with generate_report=True")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=Path(pdf_path).name,
    )


@router.get("/{job_id}/json")
async def download_json(job_id: str):
    jobs = _get_jobs()
    _check_completed(jobs, job_id)
    json_path = jobs[job_id]["results"].get("json_report")
    if not json_path or not Path(json_path).exists():
        raise HTTPException(404, "JSON report not found")
    return FileResponse(
        json_path,
        media_type="application/json",
        filename=Path(json_path).name,
    )


@router.get("/{job_id}/frame/{incident_id}/{frame_idx}")
async def get_annotated_frame(job_id: str, incident_id: int, frame_idx: int):
    jobs = _get_jobs()
    _check_completed(jobs, job_id)
    incidents = jobs[job_id]["results"].get("incidents", [])
    inc = next((i for i in incidents if i["id"] == incident_id), None)
    if not inc:
        raise HTTPException(404, f"Incident #{incident_id} not found")
    paths = inc.get("frame_paths", [])
    if frame_idx >= len(paths):
        raise HTTPException(404, f"Frame index {frame_idx} out of range (0–{len(paths)-1})")
    fpath = paths[frame_idx]
    if not Path(fpath).exists():
        raise HTTPException(404, "Frame file not found on disk")
    return FileResponse(fpath, media_type="image/jpeg")


def _check_completed(jobs: dict, job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    if jobs[job_id]["status"] != "completed":
        raise HTTPException(400, f"Job not completed — status: {jobs[job_id]['status']}")
