# vLLM on DGX Spark — Qwen3.8-27B NVFP4

A reproducible vLLM deployment for **Qwen3.8-27B** on NVIDIA DGX Spark (GB10, ARM64, 128 GB unified memory).

## Model choice

This setup uses [`Inferact/Qwen3.8-27B-NVFP4`](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4), pinned to revision `6128240ebaf4eaa7bad2b3d1c72c37d677c5f462`.

Why this checkpoint:

- NVFP4 is optimized for NVIDIA Blackwell and is the vLLM recipe's low-latency variant.
- It is smaller than the official FP8 checkpoint, leaving more unified memory for KV cache and other workloads.
- Qwen3.8 includes an MTP draft head; this setup enables three speculative tokens.
- The upstream model is Apache-2.0 licensed.

References:

- [vLLM Qwen3.8-27B recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Inferact/Qwen3.8-27B-NVFP4](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4)

## Defaults

| Setting | Value |
|---|---:|
| API port | `8000` |
| Served model name | `Qwen/Qwen3.8-27B` |
| vLLM | `0.26.0` |
| Native context | `262144` tokens |
| vLLM memory utilization | `0.50` |
| systemd soft memory limit | `55%` |
| systemd hard memory limit | `60%` |
| KV cache | FP8 |
| MTP speculative tokens | `3` |
| Vision encoder | disabled by default |

The systemd `MemoryMax=60%` limit is the hard protection for total service memory on DGX Spark's unified-memory architecture. `GPU_MEM_UTIL=0.50` keeps vLLM's own allocation target below that ceiling.

## Install

Run as a dedicated unprivileged user whose `/srv/vllm3` directory already exists and is writable:

```bash
cd /srv/vllm3
git clone https://github.com/Bitbull-Ideas/vllm.qwen3.8-27b.git .
./setup_vllm.sh
./download_model.sh
./install_service.sh
```

For user services to start without an interactive login, an administrator must enable lingering once:

```bash
sudo loginctl enable-linger vllm3
```

## Verify

```bash
systemctl --user status vllm.service --no-pager
journalctl --user -u vllm.service -n 100 --no-pager
./test_vllm.sh

systemctl --user show vllm.service   -p ActiveState -p SubState -p UnitFileState   -p MemoryCurrent -p MemoryHigh -p MemoryMax
```

A real host reboot is the strongest boot test. Without rebooting, verify `Linger=yes`, `UnitFileState=enabled`, and that the service survives a manual restart.

## API example

```bash
curl http://127.0.0.1:8000/v1/chat/completions   -H 'Content-Type: application/json'   -d '{
    "model": "Qwen/Qwen3.8-27B",
    "messages": [{"role":"user","content":"Hello"}],
    "max_tokens": 128
  }'
```

Thinking is enabled by default. Disable it per request with:

```json
{"chat_template_kwargs":{"enable_thinking":false}}
```

## Tuning

All launch settings are environment variables. Use a systemd user override rather than editing the unit:

```bash
systemctl --user edit vllm.service
```

Example:

```ini
[Service]
Environment=MAX_MODEL_LEN=131072
Environment=NUM_SPEC_TOKENS=0
Environment=LANGUAGE_ONLY=false
```

Then reload and restart:

```bash
systemctl --user daemon-reload
systemctl --user restart vllm.service
```

Do not increase `GPU_MEM_UTIL` beyond `0.59`; the wrapper rejects it and systemd independently enforces the 60% hard cap.

## Rollback

```bash
systemctl --user disable --now vllm.service
rm -f ~/.config/systemd/user/vllm.service
systemctl --user daemon-reload
```

Restore `/srv/vllm3` from the target's approved backup or remove it only after preserving required evidence and model data.
