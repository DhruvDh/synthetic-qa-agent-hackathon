#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# Config: keep the public contract unchanged for grading
# ------------------------------------------------------------------
IMAGE_DEFAULT="edaamd/aiac:latest"
IMAGE_SOURCE="${IMAGE_SOURCE:-edaamd/aiac:test-vllm}"   # your tested build tag
CONTAINER_NAME="rocm-jupyter"
WORKDIR_HOST="/workspace/AIAC"
HF_CACHE_HOST="/workspace/.cache/huggingface"

# ------------------------------------------------------------------
# Make sure the host paths exist (avoid mount errors)
# ------------------------------------------------------------------
mkdir -p "${HF_CACHE_HOST}"
mkdir -p "${WORKDIR_HOST}"

# ------------------------------------------------------------------
# (Optional but recommended) Make your tested image the default tag
# This ensures graders pulling/running edaamd/aiac:latest locally use YOUR build.
# If you prefer to skip retagging locally, comment this out.
# ------------------------------------------------------------------
if docker image inspect "${IMAGE_SOURCE}" >/dev/null 2>&1; then
  echo "Retagging ${IMAGE_SOURCE} -> ${IMAGE_DEFAULT}"
  docker tag "${IMAGE_SOURCE}" "${IMAGE_DEFAULT}"
fi

# ------------------------------------------------------------------
# Stop any previous container with the same name (idempotent)
# ------------------------------------------------------------------
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Stopping old container: ${CONTAINER_NAME}"
  docker stop "${CONTAINER_NAME}" >/dev/null || true
  docker rm   "${CONTAINER_NAME}" >/dev/null || true
fi

# ------------------------------------------------------------------
# Run with the exact interface the hackathon expects
# - host network
# - same devices/groups
# - same shm size
# - same mounts
# - same working dir
# - same container name
# - same image tag (now pointing at your new build)
# Jupyter port stays 8888 (image CMD), maintaining compatibility.
# ------------------------------------------------------------------
exec docker run -it --rm \
  --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --group-add=render \
  --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 8G \
  -v "${HF_CACHE_HOST}:/root/.cache/huggingface" \
  -v "${WORKDIR_HOST}:${WORKDIR_HOST}" \
  -w "${WORKDIR_HOST}" \
  --name "${CONTAINER_NAME}" \
  "${IMAGE_DEFAULT}"

