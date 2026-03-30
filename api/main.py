"""
api/main.py — VIGIL FastAPI — single shared job store, no routing conflicts.
"""
import uuid
import logging
import asyncio
import aiofiles
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="VIGIL API", version="1.0.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ONE shared job store (no more split between routers) ──────────────────────
JOBS: dict = {}

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_MIME = {
    "video/mp4", "video/avi", "video/x-msvideo",
    "video/quicktime", "video/x-matroska", "video/webm",
    "application/octet-stream",   # some browsers send this for .mp4
}

# ── Frontend static files ─────────────────────────────────────────────────────
FRONTEND_DIR = Path("frontend")
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Serve the frontend dashboard."""
    index = Path("frontend/index.html")
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return {"service": "VIGIL API", "version": "1.0.0", "status": "running", "docs": "/docs"}

@app.get("/app")
async def frontend():
    """Alias for frontend."""
    return await root()

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# ── Upload ────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    # Accept any video/* or octet-stream (browser inconsistency)
    ct = file.content_type or ""
    if not (ct.startswith("video/") or ct == "application/octet-stream"):
        # Try accepting anyway — just log a warning
        logger.warning(f"Unexpected content-type: {ct} — accepting anyway")

    job_id    = str(uuid.uuid4())
    ext       = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    save_path = UPLOAD_DIR / f"{job_id}{ext}"

    size = 0
    async with aiofiles.open(save_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            await f.write(chunk)

    JOBS[job_id] = {
        "id":           job_id,
        "status":       "uploaded",
        "filename":     file.filename,
        "path":         str(save_path),
        "size_mb":      round(size / 1024 / 1024, 2),
        "created_at":   datetime.now().isoformat(),
        "completed_at": None,
        "results":      None,
        "error":        None,
    }
    logger.info(f"Uploaded {file.filename} → job {job_id} ({JOBS[job_id]['size_mb']} MB)")
    return {"job_id": job_id, "status": "uploaded",
            "filename": file.filename, "size_mb": JOBS[job_id]["size_mb"]}

# ── Process ───────────────────────────────────────────────────────────────────
@app.post("/process/{job_id}")
async def process_video(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found — upload first")
    job = JOBS[job_id]
    if job["status"] == "processing":
        raise HTTPException(400, "Already processing")
    if job["status"] == "completed":
        return {"job_id": job_id, "status": "completed"}

    JOBS[job_id]["status"] = "processing"
    background_tasks.add_task(_run_pipeline, job_id, job["path"], job.get("filename"))
    return {"job_id": job_id, "status": "processing"}

async def _run_pipeline(job_id: str, video_path: str, original_filename: str = None):
    try:
        import os
        from pipeline.pipeline import VIGILPipeline
        from functools import partial

        pipeline = VIGILPipeline(
            yolo_model_size="n",
            confidence_threshold=0.4,
            output_dir="data/outputs",
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        # Wrap call to inject original_filename into video results after processing
        loop    = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, pipeline.process_video, video_path
        )
        # Patch video_name with original filename if available
        if original_filename and results:
            from pathlib import Path as _P
            results["video_name"] = _P(original_filename).stem

        JOBS[job_id]["status"]       = "completed"
        JOBS[job_id]["results"]      = results
        JOBS[job_id]["completed_at"] = datetime.now().isoformat()
        logger.info(f"Job {job_id} completed — {results.get('incident_count',0)} incidents")

    except Exception as e:
        logger.error(f"Pipeline error [{job_id}]: {e}", exc_info=True)
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"]  = str(e)

# ── Status ────────────────────────────────────────────────────────────────────
@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    j = dict(JOBS[job_id])
    j.pop("results", None)   # don't send heavy results in status
    return j

# ── Results ───────────────────────────────────────────────────────────────────
@app.get("/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    job = JOBS[job_id]
    if job["status"] != "completed":
        raise HTTPException(400, f"Job status: {job['status']}")

    r = dict(job["results"])
    # Remove heavy/non-serialisable fields
    for inc in r.get("incidents", []):
        inc.pop("frame_start", None)
        inc.pop("frame_end",   None)
    return r

# ── Jobs list ─────────────────────────────────────────────────────────────────
@app.get("/jobs")
async def list_jobs():
    return [
        {"id": j["id"], "status": j["status"],
         "filename": j["filename"], "created_at": j["created_at"]}
        for j in JOBS.values()
    ]

# ── Reports ───────────────────────────────────────────────────────────────────
@app.get("/report/{job_id}/pdf")
async def download_pdf(job_id: str):
    job = _get_completed(job_id)
    pdf = job["results"].get("pdf_report", "")
    if not pdf or not Path(pdf).exists():
        raise HTTPException(404, "PDF not found")
    return FileResponse(pdf, media_type="application/pdf", filename=Path(pdf).name)

@app.get("/report/{job_id}/json")
async def download_json_report(job_id: str):
    job = _get_completed(job_id)
    jp  = job["results"].get("json_report", "")
    if not jp or not Path(jp).exists():
        raise HTTPException(404, "JSON report not found")
    return FileResponse(jp, media_type="application/json", filename=Path(jp).name)

@app.get("/report/{job_id}/frame/{incident_id}/{frame_idx}")
async def get_frame(job_id: str, incident_id: int, frame_idx: int):
    job = _get_completed(job_id)
    incs = job["results"].get("incidents", [])
    inc  = next((i for i in incs if i["id"] == incident_id), None)
    if not inc:
        raise HTTPException(404, f"Incident {incident_id} not found")
    paths = inc.get("frame_paths", [])
    if frame_idx >= len(paths):
        raise HTTPException(404, "Frame index out of range")
    fp = paths[frame_idx]
    if not Path(fp).exists():
        raise HTTPException(404, "Frame file not found")
    return FileResponse(fp, media_type="image/jpeg")

def _get_completed(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    if JOBS[job_id]["status"] != "completed":
        raise HTTPException(400, f"Job not completed — status: {JOBS[job_id]['status']}")
    return JOBS[job_id]

# ── WebSocket ─────────────────────────────────────────────────────────────────
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/{job_id}")
async def ws_status(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            job = JOBS.get(job_id)
            if not job:
                await websocket.send_json({"error": "Job not found"})
                break
            payload = {
                "job_id":  job_id,
                "status":  job["status"],
                "incident_count": job["results"].get("incident_count", 0)
                    if job["results"] else 0,
            }
            await websocket.send_json(payload)
            if job["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)