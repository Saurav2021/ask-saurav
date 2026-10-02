# Runs on Render's free web service (512 MB RAM) or any Docker host.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000
RUN useradd -m -u 1000 app
WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app . .
USER app

# Download the ONNX embedding model and build the Chroma index at build time,
# so a cold start only has to load files from disk.
RUN python -m app.ingest

EXPOSE 8000
# Render injects $PORT; a single worker keeps memory well under 512 MB.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips '*'"]
