#!/usr/bin/env bash
# Install and enable the boot-persistent systemd user service.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR="${HOME}/.config/systemd/user"
mkdir -p "${UNIT_DIR}"
install -m 0644 "${SCRIPT_DIR}/vllm.service" "${UNIT_DIR}/vllm.service"
systemctl --user daemon-reload
systemctl --user enable --now vllm.service
