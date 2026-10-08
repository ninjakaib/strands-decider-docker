# syntax=docker/dockerfile:1.7
#
# Strands Decider served over HTTP, with every weight baked into the image so the container
# never contacts Hugging Face. Build this somewhere that can reach huggingface.co (a GitHub
# Actions runner), push it to a registry, then pull and run it locally.
#
#   docker run --rm --gpus all -p 127.0.0.1:8000:8000 <image>
#   curl -s localhost:8000/health
#   curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
#     "state": "Help! My payouts have been failing for 3 days.",
#     "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}
#   }'
#
# The server has no authentication, so publish the port on 127.0.0.1 only.
# Versions follow the upstream reference environment (strands-decider training/README.md).

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

# --- base: Python plus the C compiler Triton needs at runtime to build its CUDA launcher ---
FROM ${PYTHON_IMAGE} AS base
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# --- deps: the virtualenv with torch, the Gated DeltaNet kernels and strands-decider ---
FROM base AS deps
ARG TORCH_VERSION=2.7.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG TRANSFORMERS_VERSION=5.17.0
ARG PEFT_VERSION=0.21.0
ARG FLA_VERSION=0.5.2
# 0.1.0 on PyPI predates the device extras, so install a pinned commit of the repo.
ARG DECIDER_GIT=https://github.com/strands-labs/strands-decider
ARG DECIDER_COMMIT=8550e887254485a8db4a4321b11442e351d96055
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv "$VIRTUAL_ENV"
RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
# Repeating the torch pin keeps the second resolve from swapping in a different torch build.
# causal-conv1d is left out, as in the upstream reference environment: it needs nvcc to build,
# and transformers falls back to its PyTorch conv without it.
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_STRANDS_DECIDER=0.1.0+g${DECIDER_COMMIT} \
    pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" \
        "transformers==${TRANSFORMERS_VERSION}" \
        "peft==${PEFT_VERSION}" \
        "flash-linear-attention==${FLA_VERSION}" \
        "strands-decider[cuda] @ git+${DECIDER_GIT}@${DECIDER_COMMIT}"

# --- weights: the decider checkpoint plus the base model revision it was trained on ---
FROM deps AS weights
ARG DECIDER_REPO=StrandsAgents/strands-decider-2B-hobson-v21
ARG DECIDER_REVISION=main
ENV HF_HOME=/models/hf \
    HF_HUB_DISABLE_TELEMETRY=1
# The checkpoint holds only the LoRA adapter and the pointer head. StrandsDeciderModel.load
# then fetches the base torso (Qwen/Qwen3.5-2B-Base, about 4.5 GB) at the revision the
# checkpoint names. The strands_decider helpers resolve that name and revision here, the same
# way load() does at runtime, so the cache holds exactly what the server will ask for.
RUN <<'EOF'
python - <<'PY'
import os

from huggingface_hub import snapshot_download
from strands_decider.modeling import StrandsDeciderConfig, base_revision, config_path

ckpt = "/models/strands-decider"
snapshot_download(
    os.environ["DECIDER_REPO"], revision=os.environ["DECIDER_REVISION"], local_dir=ckpt
)
config = StrandsDeciderConfig.from_json(config_path(ckpt))
revision = base_revision(ckpt, config)
print(f"base model {config.base_model} at revision {revision or 'main'}")
snapshot_download(
    config.base_model,
    revision=revision,
    ignore_patterns=["*.md", "*.bin", "*.pth", "*.gguf", "original/*"],
)
PY
rm -rf /models/strands-decider/.cache
EOF
# Fail the build if anything the loader reads is not cached. This resolves the files with
# downloads disabled; it skips the 2B forward pass because the build runner has no GPU.
RUN <<'EOF'
HF_HUB_OFFLINE=1 python - <<'PY'
from huggingface_hub import snapshot_download
from strands_decider.modeling import (
    StrandsDeciderConfig, base_revision, checkpoint_dir, config_path, load_head_state,
)
from transformers import AutoConfig, AutoTokenizer

ckpt = checkpoint_dir("/models/strands-decider")
config = StrandsDeciderConfig.from_json(config_path(ckpt))
revision = base_revision(ckpt, config)
load_head_state(ckpt)
AutoTokenizer.from_pretrained(ckpt)
AutoConfig.from_pretrained(config.base_model, revision=revision)
snapshot_download(config.base_model, revision=revision, local_files_only=True)
print("offline resolution ok")
PY
EOF

# --- runtime ---
FROM base AS runtime
ARG DECIDER_REPO=StrandsAgents/strands-decider-2B-hobson-v21
ARG DECIDER_REVISION=main
ARG DECIDER_COMMIT=8550e887254485a8db4a4321b11442e351d96055
LABEL org.opencontainers.image.title="strands-decider" \
      org.opencontainers.image.source="https://github.com/strands-labs/strands-decider" \
      decider.model="${DECIDER_REPO}" \
      decider.model-revision="${DECIDER_REVISION}" \
      decider.code-commit="${DECIDER_COMMIT}"
RUN useradd --create-home --uid 1000 decider
COPY --from=deps /opt/venv /opt/venv
COPY --from=weights --chown=decider:decider /models /models
# HF_HUB_OFFLINE stops any Hub call at runtime. /usr/lib/wsl/lib is where the NVIDIA
# Container Toolkit mounts libcuda under WSL2; Triton looks for libcuda on LD_LIBRARY_PATH,
# and the directory is simply absent on other hosts.
ENV HF_HOME=/models/hf \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRITON_CACHE_DIR=/home/decider/.triton \
    LD_LIBRARY_PATH=/usr/lib/wsl/lib
USER decider
WORKDIR /home/decider
EXPOSE 8000
# Loading 2B weights takes a while, so give startup room before health checks count.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
# serve runs one uvicorn worker. Upstream has not verified concurrent requests, so keep it
# at one replica per GPU. Arguments after the image name replace CMD, e.g. --max-batch 8.
# This image needs a GPU: flash-linear-attention raises on import when Triton finds no CUDA
# driver, so --device cpu fails here.
ENTRYPOINT ["strands-decider", "serve", "/models/strands-decider", "--host", "0.0.0.0", "--port", "8000"]
CMD ["--device", "cuda"]
