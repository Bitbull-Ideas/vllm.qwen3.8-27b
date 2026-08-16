#!/usr/bin/env bash
# Install a pinned vLLM environment for DGX Spark (GB10, ARM64).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
VLLM_VERSION="${VLLM_VERSION:-0.26.0}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv for ${USER}..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi

command -v uv >/dev/null 2>&1 || {
  echo "uv was installed but is not on PATH" >&2
  exit 1
}

uv venv "${VENV_DIR}" --python 3.12
# vLLM 0.26 supports Qwen3.8; transformers 5.8 matches its processor config.
uv pip install --python "${VENV_DIR}/bin/python" \
  "vllm==${VLLM_VERSION}" \
  "transformers>=5.8.0" \
  "huggingface-hub>=0.34" \
  --torch-backend=auto

mkdir -p "${SCRIPT_DIR}/models"
"${VENV_DIR}/bin/vllm" --version
"${VENV_DIR}/bin/python" - <<'PY'
import transformers
print(f"transformers {transformers.__version__}")
PY
