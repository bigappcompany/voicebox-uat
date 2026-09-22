# The agent currently runs on Python 3.13 locally; keep the container runtime
# aligned with that supported environment.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GOODBOX_SERVER_HOST=0.0.0.0 \
    GOODBOX_SERVER_PORT=8000 \
    VOICEAGENT_RUNTIME_LOG=/app/logs/voiceagent.log

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# The service needs writable locations for the persistent greeting cache and
# structured runtime logs, but should not otherwise run as root.
RUN useradd --create-home --uid 10001 voiceagent \
    && mkdir -p /app/logs /app/.runtime-cache \
    && chown -R voiceagent:voiceagent /app

USER voiceagent

EXPOSE 8000

CMD ["python", "goodbox_server.py"]
