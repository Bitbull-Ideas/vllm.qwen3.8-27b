#!/usr/bin/env bash
# Serve Qwen3.8-27B NVFP4 on NVIDIA GB10 with a conservative memory budget.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
VLLM="${VENV_DIR}/bin/vllm"
# FlashInfer and torch.compile invoke build tools such as ninja by name.
export PATH="${VENV_DIR}/bin:${PATH}"
MODEL_REPO="${MODEL_REPO:-RadixArk/Qwen3.8-27B-NVFP4}"
MODEL_NAME="${MODEL_REPO##*/}"
# MODEL_PATH can be overridden explicitly to avoid collisions when different
# repos share the same trailing path segment (e.g. .../Qwen3.8-27B-NVFP4).
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/models/${MODEL_NAME}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.8-27B}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.50}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
LANGUAGE_ONLY="${LANGUAGE_ONLY:-true}"

[[ -x "${VLLM}" ]] || { echo "Run setup_vllm.sh first" >&2; exit 1; }
[[ -s "${MODEL_PATH}/config.json" ]] || { echo "Run download_model.sh first" >&2; exit 1; }

case "${GPU_MEM_UTIL}" in
  0.[0-5]|0.[0-5][0-9]|.5|.50) ;;
  *) echo "GPU_MEM_UTIL must be <= 0.59; systemd enforces the hard 60% cap" >&2; exit 1 ;;
esac

ARGS=(
  "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --trust-remote-code
)

if [[ "${LANGUAGE_ONLY}" == "true" ]]; then
  ARGS+=(--language-model-only)
else
  ARGS+=(--mm-encoder-tp-mode data)
fi

if (( NUM_SPEC_TOKENS > 0 )); then
  ARGS+=(--speculative-config '{"method":"mtp","num_speculative_tokens":'"${NUM_SPEC_TOKENS}"'}')
fi

exec "${VLLM}" serve "${ARGS[@]}"
