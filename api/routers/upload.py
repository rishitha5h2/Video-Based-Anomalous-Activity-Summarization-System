"""api/routers/upload.py — Video upload endpoint."""

import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException
import aiofiles

router = APIRouter(prefix="/upload", tags=["upload"])

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_TYPES  = {"video/mp4", "video/avi", "video/quicktime",
                  "video/x-msvideo", "video/x-matroska", "video/webm"}
MAX_SIZE_BYTES = 500 * 1024 * 1024   # 500 MB


@router.post("/")
async def upload_video(file: UploadFile = File(...)):
    """Accept a video file and return a job_id for later processing."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported media type: {file.content_type}")

    job_id  = str(uuid.uuid4())
    ext     = Path(file.filename).suffix.lower() or ".mp4"
    dest    = UPLOAD_DIR / f"{job_id}{ext}"

    size = 0
    async with aiofiles.open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):   # 1 MB chunks
            size += len(chunk)
            if size > MAX_SIZE_BYTES:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "File exceeds 500 MB limit")
            await out.write(chunk)

    return {
        "job_id":   job_id,
        "status":   "uploaded",
        "filename": file.filename,
        "size_mb":  round(size / 1024 / 1024, 2),
        "path":     str(dest),
    }
