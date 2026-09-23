# Pipecat Cloud supplies the process manager and discovers bot.py:bot through
# this base image. Docker Compose overrides the entrypoint to run the local
# Goodbox/Plivo FastAPI server instead.
FROM dailyco/pipecat-base:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VOICEAGENT_RUNTIME_LOG=/app/logs/voiceagent.log

WORKDIR /app

# pyproject.toml is the canonical dependency manifest for this V2 service.
# Copying the complete source is necessary because the voice_agent package is
# installed with the application rather than as an unrelated example project.
COPY Appzo-VoiceAgent-V2/ ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    && mkdir -p /app/logs /app/.runtime-cache

EXPOSE 8000
