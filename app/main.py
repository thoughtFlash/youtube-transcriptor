import os
import shutil
import tempfile
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, HttpUrl

from app.transcriptor import download_audio, transcribe_audio

app = FastAPI(
    title="YouTube Transcriptor",
    description="Transcribes YouTube videos using yt-dlp and faster-whisper.",
    version="1.0.0",
)

DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "medium")

AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]


class TranscribeRequest(BaseModel):
    url: str
    model: Optional[str] = None
    language: Optional[str] = None  # ISO 639-1, e.g. "de", "en" — None = auto-detect
    format: str = "json"  # "json" or "txt"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    return {
        "available": AVAILABLE_MODELS,
        "default": DEFAULT_MODEL,
    }


@app.post("/transcribe")
def transcribe(req: TranscribeRequest, background_tasks: BackgroundTasks):
    model_size = req.model or DEFAULT_MODEL

    if model_size not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_size}'. Available: {AVAILABLE_MODELS}",
        )

    if req.format not in ("json", "txt"):
        raise HTTPException(
            status_code=400,
            detail="format must be 'json' or 'txt'",
        )

    tmpdir = tempfile.mkdtemp()
    background_tasks.add_task(shutil.rmtree, tmpdir, True)

    try:
        audio_path, video_title, duration = download_audio(
            req.url, os.path.join(tmpdir, "audio")
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Download failed: {exc}")

    try:
        segments, info = transcribe_audio(audio_path, model_size, req.language)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")

    full_text = " ".join(seg.text.strip() for seg in segments)

    if req.format == "txt":
        return PlainTextResponse(full_text)

    return {
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
    }
