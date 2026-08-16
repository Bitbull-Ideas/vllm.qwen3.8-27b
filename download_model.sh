#!/usr/bin/env bash
# Download the exact tested NVFP4 model revision from Hugging Face.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
MODEL_REPO="${MODEL_REPO:-Inferact/Qwen3.8-27B-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-6128240ebaf4eaa7bad2b3d1c72c37d677c5f462}"
MODEL_NAME="${MODEL_REPO##*/}"
MODEL_PATH="${SCRIPT_DIR}/models/${MODEL_NAME}"

[[ -x "${PYTHON}" ]] || { echo "Run setup_vllm.sh first" >&2; exit 1; }
mkdir -p "${SCRIPT_DIR}/models"

MODEL_REPO="${MODEL_REPO}" MODEL_REVISION="${MODEL_REVISION}" MODEL_PATH="${MODEL_PATH}" \
"${PYTHON}" - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    revision=os.environ["MODEL_REVISION"],
    local_dir=os.environ["MODEL_PATH"],
    ignore_patterns=["*.md", "*.h5", "*.ot", "*.msgpack"],
)
print(path)
PY

test -s "${MODEL_PATH}/config.json"
test -s "${MODEL_PATH}/model.safetensors.index.json"
printf 'Downloaded %s at revision %s\n' "${MODEL_REPO}" "${MODEL_REVISION}"
du -sh "${MODEL_PATH}"
