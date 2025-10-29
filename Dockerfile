# syntax=docker/dockerfile:1.6
ARG BASE=rocm/vllm-dev:nightly

FROM ${BASE} AS base_clean
RUN set -eux; \
  for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do \
    [ -f "$f" ] || continue; \
    sed -i '/repo\.radeon\.com\/amdgpu\/7\.0\/ubuntu/d' "$f"; \
  done

# ---------- Stage 0: build vLLM wheel from the commit you want ----------
FROM base_clean AS vllm_builder
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=main           # <-- set to a tag or commit SHA you need
ARG PYTORCH_ROCM_ARCH=gfx942
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}

RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

RUN git clone ${VLLM_REPO} /src/vllm && cd /src/vllm && \
    git fetch -v --tags && git checkout ${VLLM_REF} && \
    python3 -m pip install -U pip && \
    python3 -m pip install -r requirements/rocm.txt && \
    python3 setup.py clean --all && \
    python3 setup.py bdist_wheel --dist-dir=/out

# ---------- Stage 1: your AIAC runtime with the replaced vLLM ----------
FROM base_clean AS aiac
ARG PYTORCH_ROCM_ARCH=gfx942
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}

# Keep your existing base packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git tmux wget && \
    rm -rf /var/lib/apt/lists/*

# Swap in the freshly built vLLM wheel (no deps to preserve base ABI)
COPY --from=vllm_builder /out/*.whl /tmp/vllm/
RUN python3 -m pip uninstall -y vllm || true && \
    python3 -m pip install --no-deps /tmp/vllm/*.whl

# Unsloth (as you have it)
RUN pip uninstall unsloth_zoo unsloth -y || true
RUN pip install --upgrade --no-cache-dir --no-deps \
      git+https://github.com/unslothai/unsloth_zoo && \
    pip install --upgrade --no-cache-dir --no-deps \
      git+https://github.com/unslothai/unsloth

# Synthetic-Data-Kit
RUN pip install synthetic-data-kit --ignore-installed blinker

# Your pinned userland
RUN pip install "wheel>=0.42.0" packaging torchvision numpy tqdm psutil jupyter tyro protobuf \
    "sentencepiece>=0.2.0" \
    "datasets>=3.4.1,!=4.0.*,!=4.1.0" \
    "accelerate>=0.34.1" \
    "peft>=0.7.1,!=0.11.0" \
    "huggingface_hub>=0.34.0" \
    hf_transfer diffusers \
    "transformers>=4.51.3,!=4.52.0,!=4.52.1,!=4.52.2,!=4.52.3,!=4.53.0,!=4.54.0,!=4.55.0,!=4.55.1,<=4.56.2" \
    "trl>=0.7.9,!=0.9.0,!=0.9.1,!=0.9.2,!=0.9.3,!=0.15.0,!=0.19.0,<=0.23.0"

# Build bitsandbytes for gfx942 only
RUN git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git /tmp/bitsandbytes && \
    cd /tmp/bitsandbytes && \
    cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx942" -S . && \
    make && \
    pip install . && \
    cd / && rm -rf /tmp/bitsandbytes

# vLLM / NCCL env as you had
ENV GLOO_SOCKET_IFNAME=lo
ENV NCCL_SOCKET_IFNAME=lo

# Your workspace
WORKDIR /workspace
COPY . /workspace/AIAC/

# Default command (unchanged)
CMD ["jupyter-lab","--ip=0.0.0.0","--port=8888","--no-browser","--allow-root"]

