import os
import tempfile
import shutil
from typing import Optional

import yt_dlp
from faster_whisper import WhisperModel

MODELS_DIR = os.getenv("WHISPER_MODELS_DIR", "/models")

_model_cache: dict[str, WhisperModel] = {}


def get_model(model_size: str) -> WhisperModel:
    if model_size not in _model_cache:
        _model_cache[model_size] = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root=MODELS_DIR,
        )
    return _model_cache[model_size]


def download_audio(url: str, output_path: str) -> str:
    """Download audio from a YouTube URL and return the path to the audio file."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "unknown")
        duration = info.get("duration", 0)

    audio_file = output_path + ".mp3"
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio download failed, expected file: {audio_file}")

    return audio_file, title, duration


def transcribe_audio(
    audio_path: str,
    model_size: str,
    language: Optional[str],
) -> tuple:
    """Transcribe audio file using faster-whisper and return segments and metadata."""
    model = get_model(model_size)
    segments, info = model.transcribe(
        audio_path,
        language=language if language else None,
        beam_size=5,
        vad_filter=True,
    )
    # Consume the generator before the temp file is deleted
    segments_list = list(segments)
    return segments_list, info
