"""api/routers/incidents.py — Incident query endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List

router = APIRouter(prefix="/incidents", tags=["incidents"])


def _get_jobs():
    """Import job store lazily to avoid circular imports."""
    from api.routers.process import jobs
    return jobs


@router.get("/")
async def list_all_incidents(
    job_id:    Optional[str]   = Query(None, description="Filter by job"),
    type_filter: Optional[str] = Query(None, description="Filter by incident type"),
    min_conf:  float           = Query(0.0,  description="Minimum confidence"),
):
    """Return a flat list of all incidents across all completed jobs."""
    jobs = _get_jobs()
    results = []

    target_jobs = (
        {job_id: jobs[job_id]} if job_id and job_id in jobs else jobs
    )

    for jid, job in target_jobs.items():
        if job["status"] != "completed" or not job.get("results"):
            continue
        for inc in job["results"].get("incidents", []):
            if inc["confidence"] < min_conf:
                continue
            if type_filter and inc["type"].lower() != type_filter.lower():
                continue
            results.append({
                "job_id":     jid,
                "video_name": job["results"]["video_name"],
                **inc,
            })

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return {"total": len(results), "incidents": results}


@router.get("/{job_id}")
async def get_job_incidents(job_id: str):
    jobs = _get_jobs()
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j["status"] != "completed":
        raise HTTPException(400, f"Job not completed — status: {j['status']}")
    return {
        "job_id":    job_id,
        "video":     j["results"]["video_name"],
        "incidents": j["results"]["incidents"],
    }


@router.get("/{job_id}/{incident_id}")
async def get_incident_detail(job_id: str, incident_id: int):
    jobs = _get_jobs()
    if job_id not in jobs or jobs[job_id]["status"] != "completed":
        raise HTTPException(404, "Job not ready")
    incidents = jobs[job_id]["results"].get("incidents", [])
    inc = next((i for i in incidents if i["id"] == incident_id), None)
    if not inc:
        raise HTTPException(404, f"Incident #{incident_id} not found")
    return inc
