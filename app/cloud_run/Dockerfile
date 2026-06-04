# ── Build ─────────────────────────────────────────────────────────────────────
# pytorch/pytorch ships CUDA 12.4 + cuDNN 9 + torch 2.4 — matches Cloud Run L4.
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# ── Runtime ───────────────────────────────────────────────────────────────────
# Cloud Run injects PORT; default 8080.
ENV PORT=8080

# HuggingFace cache lives on the container's writable layer.
# For faster cold starts, mount a pre-warmed GCS FUSE bucket here:
#   --update-env-vars HF_HOME=/mnt/hf_cache
ENV HF_HOME=/tmp/hf_cache

# FLUX.1-dev is a gated model — inject your token via Cloud Run secret manager:
#   --update-secrets HUGGING_FACE_HUB_TOKEN=hf-token:latest
ENV HUGGING_FACE_HUB_TOKEN=""

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1"]
