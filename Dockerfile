# Hugging Face Spaces (Docker SDK) image. Free CPU tier: 2 vCPU, 16 GB RAM.
FROM python:3.11-slim

# Spaces run containers as uid 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
RUN mkdir -p /home/user/app && chown user:user /home/user/app
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY --chown=user . .
USER user

# Download the embedding model and build the Chroma index at build time,
# so a cold start only has to load files from disk.
RUN python -m app.ingest

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
