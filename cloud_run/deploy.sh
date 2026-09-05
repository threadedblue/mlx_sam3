#!/usr/bin/env bash
# Deploy the FLUX inference service to Cloud Run with an NVIDIA L4 GPU.
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
#   gcloud services enable run.googleapis.com artifactregistry.googleapis.com
#
# Usage:
#   chmod +x deploy.sh
#   HF_TOKEN=hf_xxx ./deploy.sh

set -euo pipefail

PROJECT=$(gcloud config get-value project)
REGION=${REGION:-us-central1}
SERVICE=flux-infer
REPO=gcr.io/${PROJECT}/${SERVICE}
HF_TOKEN=${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-""}}

echo "==> Building image …"
docker build -t "${REPO}:latest" .

echo "==> Pushing to Container Registry …"
docker push "${REPO}:latest"

echo "==> Deploying to Cloud Run (L4 GPU) …"
gcloud run deploy "${SERVICE}" \
  --image "${REPO}:latest" \
  --region "${REGION}" \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --memory 32Gi \
  --cpu 8 \
  --timeout 3600 \
  --concurrency 1 \
  --min-instances 0 \
  --max-instances 1 \
  --no-allow-unauthenticated \
  --set-env-vars "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}"

echo ""
echo "==> Service URL:"
gcloud run services describe "${SERVICE}" --region "${REGION}" \
  --format "value(status.url)"
echo ""
echo "Paste the URL above into the Flutter app's Cloud Run field."
