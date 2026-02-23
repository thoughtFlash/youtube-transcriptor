import os
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.transcriptor import download_audio, transcribe_audio

app = FastAPI(
    title="YouTube Transcriptor",
    description="Transcribes YouTube videos using yt-dlp and faster-whisper.",
    version="2.0.0",
)

DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "medium")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))

AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
jobs: dict[str, dict] = {}


class TranscribeRequest(BaseModel):
    url: str
    model: Optional[str] = None
    language: Optional[str] = None  # ISO 639-1, e.g. "de", "en" — None = auto-detect
    format: str = "json"  # "json" or "txt"


def _run_job(job_id: str, req: TranscribeRequest) -> None:
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()

    tmpdir = tempfile.mkdtemp()
    try:
        audio_path, video_title, duration = download_audio(
            req.url, os.path.join(tmpdir, "audio")
        )

        model_size = req.model or DEFAULT_MODEL
        segments, info = transcribe_audio(audio_path, model_size, req.language)

        full_text = " ".join(seg.text.strip() for seg in segments)

        jobs[job_id].update(
            {
                "status": "done",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "result": full_text
                if req.format == "txt"
                else {
                    "title": video_title,
                    "duration_seconds": duration,
                    "language": info.language,
                    "language_probability": round(info.language_probability, 4),
                    "model": model_size,
                    "text": full_text,
                    "segments": [
                        {
                            "start": round(seg.start, 2),
                            "end": round(seg.end, 2),
                            "text": seg.text.strip(),
                        }
                        for seg in segments
                    ],
                },
            }
        )
    except Exception as exc:
        jobs[job_id].update(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/health")
def health():
    return {"status": "ok", "active_jobs": sum(1 for j in jobs.values() if j["status"] == "processing")}


@app.get("/models")
def list_models():
    return {
        "available": AVAILABLE_MODELS,
        "default": DEFAULT_MODEL,
    }


@app.post("/transcribe", status_code=202)
def transcribe(req: TranscribeRequest):
    model_size = req.model or DEFAULT_MODEL

    if model_size not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_size}'. Available: {AVAILABLE_MODELS}",
        )

    if req.format not in ("json", "txt"):
        raise HTTPException(status_code=400, detail="format must be 'json' or 'txt'")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "url": req.url,
        "model": model_size,
        "format": req.format,
    }

    executor.submit(_run_job, job_id, req)

    return {"job_id": job_id, "status": "pending"}


@app.get("/jobs")
def list_jobs():
    return list(jobs.values())


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == "done" and job.get("format") == "txt":
        return PlainTextResponse(job["result"])

    return job


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "processing":
        raise HTTPException(status_code=409, detail="Cannot delete a running job")
    del jobs[job_id]
