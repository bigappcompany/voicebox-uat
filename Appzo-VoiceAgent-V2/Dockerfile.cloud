FROM dailyco/pipecat-base:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VOICEAGENT_RUNTIME_LOG=/app/logs/voiceagent.log

WORKDIR /app

COPY Appzo-VoiceAgent-V2/ ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    && mkdir -p /app/logs /app/.runtime-cache

EXPOSE 8000
