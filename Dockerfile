FROM python:3.11-slim

# ffmpeg is required by yt-dlp for audio extraction
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Directory for cached whisper models (mount as volume to persist between restarts)
RUN mkdir -p /models

ENV WHISPER_MODEL=medium
ENV WHISPER_MODELS_DIR=/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
