"""api/routers/stream.py — WebSocket real-time status updates."""

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["stream"])


def _get_jobs():
    from api.routers.process import jobs
    return jobs


@router.websocket("/ws/{job_id}")
async def websocket_status(websocket: WebSocket, job_id: str):
    """
    Streams job status updates every second until completion or disconnection.

    Message schema:
        { "job_id": str, "status": str, "incident_count": int,
          "progress_pct": float, "message": str }
    """
    await websocket.accept()
    jobs = _get_jobs()

    step_messages = [
        "Extracting frames…",
        "Running object detection…",
        "Computing anomaly scores…",
        "Recognising actions…",
        "Fusing signals into incidents…",
        "Extracting annotated snapshots…",
        "Generating LLM summaries…",
        "Building report…",
    ]
    tick = 0

    try:
        while True:
            job = jobs.get(job_id)
            if job is None:
                await websocket.send_json({"error": "Job not found"})
                break

            status         = job["status"]
            incident_count = 0
            if status == "completed" and job.get("results"):
                incident_count = job["results"].get("incident_count", 0)

            progress = min(95, (tick * 12)) if status == "processing" else \
                       100 if status == "completed" else 0

            msg_idx = min(tick, len(step_messages) - 1)
            message = step_messages[msg_idx] if status == "processing" else status

            await websocket.send_json({
                "job_id":         job_id,
                "status":         status,
                "incident_count": incident_count,
                "progress_pct":   progress,
                "message":        message,
            })

            if status in ("completed", "failed"):
                break

            tick += 1
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
