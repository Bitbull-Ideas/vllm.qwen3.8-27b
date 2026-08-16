#!/usr/bin/env bash
# Functional OpenAI-compatible API test.
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
BASE_URL="http://${HOST}:${PORT}"

models="$(curl --fail --silent --show-error --max-time 30 "${BASE_URL}/v1/models")"
jq -e --arg model "${MODEL}" '.data[] | select(.id == $model)' <<<"${models}" >/dev/null

response="$(curl --fail --silent --show-error --max-time 300 \
  "${BASE_URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"${MODEL}"'",
    "messages": [{"role":"user","content":"Reply with exactly: SPARK_OK"}],
    "temperature": 0,
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false}
  }')"

jq -e '.choices[0].message.content | strings | length > 0' <<<"${response}" >/dev/null
jq '{model, content: .choices[0].message.content, usage}' <<<"${response}"
