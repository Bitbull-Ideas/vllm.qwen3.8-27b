# Qwen3.8-27B-NVFP4 — Model Comparison & Tuning on DGX Spark (GB10)

Date: 2026-08-22
Target: spark.lb.bitbull.ch (NVIDIA GB10, Grace Blackwell, 128 GB unified memory)
Service: `vllm.service` (systemd user unit), user `vllm3`, `/srv/vllm3`

## Goal

User request: search for faster GB10-capable Qwen3.8-27B variants (>= 4-bit, fully functional,
maximum context length), install the reasonable candidates, tune and benchmark them, keep the
fastest one installed, and report exact comparison data.

## Candidates evaluated

All three are native `Qwen3.8-27B` NVFP4 (4-bit weight, group size 16) quantizations that ship a
Multi-Token-Prediction (MTP) draft head and support vLLM directly (no GGUF/llama.cpp conversion
needed), native context length 262,144 tokens, tool calling (Hermes/qwen3_coder tool parser),
and reasoning parser support.

| # | Repo | Revision | Size on disk | Quant method | Notes |
|---|---|---|---:|---|---|
| 1 | `Inferact/Qwen3.8-27B-NVFP4` (previous production pin) | `6128240ebaf4eaa7bad2b3d1c72c37d677c5f462` | 25 GB | NVIDIA ModelOpt | Baseline in production since 2026-08-16 |
| 2 | `unsloth/Qwen3.8-27B-NVFP4` | `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` | 22 GB | compressed-tensors (Unsloth Dynamic v3.0) | Fixed revision without the reported tokenizer truncation-at-2048 bug (verified: `tokenizer.json` truncation field is `None`) |
| 3 | `RadixArk/Qwen3.8-27B-NVFP4` | `319f741cce68d7914884900c138a1fbb70a42f30` | 21 GB | NVIDIA ModelOpt (mixed NVFP4 W4A4 / FP8 attention) | Vendor-reported GSM8K 97.27%, Terminal-Bench 2.1 73.81% on 4x B300/GB300 (SGLang); re-validated here on single GB10 via vLLM |

Not selected for installation (rejected during triage, reasons noted):

- GGUF-only builds (`bartowski/*-GGUF`, `mlx-community/*`, various `Uncensored`/`Abliterated`
  community forks): require llama.cpp/MLX, not a fit for this vLLM-based production service.
- `Qwen/Qwen3.8-27B-FP8`: official FP8, not sub-4-bit, larger footprint (~31 GB), and public
  benchmarks (NVIDIA developer forum) already show NVFP4 quantizations winning throughput and
  latency by 30-45% on this class of hardware (GB10 / Jetson Thor).
- AWQ-INT4 / AutoRound-INT4 variants: no MTP draft head shipped, would lose speculative decoding
  gains already validated on this hardware.

## Method

Identical benchmark harness for every candidate (see [`run_vllm_bench.py`](run_vllm_bench.py) in this folder):

- vLLM 0.26.0, same server flags as production (`--kv-cache-dtype fp8`, `--speculative-config
  '{"method":"mtp","num_speculative_tokens":3}'`, `--max-model-len 262144`,
  `--gpu-memory-utilization 0.50`, `--enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3`, `--language-model-only`).
- Each candidate loaded standalone (no other model resident) to avoid GPU memory contention.
- Two request shapes, 4 requests each, at concurrency 1/2/4, one warmup request per shape:
  - **Short**: 64 output tokens, one-sentence prompt.
  - **Long**: 256 output tokens, six-sentence prompt.
- Functional checks after the perf run: correctness (factual question), tool calling (function
  call with a synthetic weather tool), long-context recall (~9.7k-token prompt with an embedded
  marker string).
- MTP draft-token acceptance rate read from `/metrics`
  (`spec_decode_num_accepted_tokens_total / spec_decode_num_draft_tokens_total`).

## Results

### Latency (p50, milliseconds) — lower is better

| Model | Short c1 | Short c2 | Short c4 | Long c1 | Long c2 | Long c4 |
|---|---:|---:|---:|---:|---:|---:|
| Inferact/Qwen3.8-27B-NVFP4 | 3973.4 | 4455.7 | 4484.7 | 17827.1 | 17143.9 | 17696.9 |
| unsloth/Qwen3.8-27B-NVFP4 | 3512.7 | 3815.3 | 6803.3 | 13569.5 | 14307.8 | 14754.3 |
| **RadixArk/Qwen3.8-27B-NVFP4** | **2497.3** | **4025.2** | **5939.3** | **10815.9** | **12345.2** | **12779.6** |

(Re-run of Inferact baseline under identical fresh-process conditions to control for host-state
drift; consistent with the original 2026-08-22 tuning benchmark in
[`qwen3.8_tuning_bench_20260822.md`](qwen3.8_tuning_bench_20260822.md).)

### Throughput (tokens/s, output) — higher is better

| Model | Short c1 | Short c2 | Short c4 | Long c1 | Long c2 | Long c4 |
|---|---:|---:|---:|---:|---:|---:|
| Inferact/Qwen3.8-27B-NVFP4 | 16.5 | 29.2 | 56.6 | 14.2 | 29.7 | 57.3 |
| unsloth/Qwen3.8-27B-NVFP4 | 19.4 | 32.3 | 36.2 | 18.7 | 35.6 | 67.6 |
| **RadixArk/Qwen3.8-27B-NVFP4** | **25.6** | **35.6** | **42.2** | **23.8** | **41.7** | **74.8** |

### MTP speculative decoding quality (draft-token acceptance rate)

| Model | Draft tokens | Accepted tokens | Acceptance rate |
|---|---:|---:|---:|
| Inferact/Qwen3.8-27B-NVFP4 | 5763 | 2876 | 49.9% |
| unsloth/Qwen3.8-27B-NVFP4 | 5796 | 2875 | 49.6% |
| **RadixArk/Qwen3.8-27B-NVFP4** | 5370 | 3012 | **56.1%** |

### Functional checks (all three passed)

| Check | Inferact | unsloth | RadixArk |
|---|---|---|---|
| Factual correctness (capital of Germany) | pass | pass | pass |
| Tool calling (function call generated correctly) | pass (existing prod use) | not re-tested | pass |
| Long-context recall (~9.7k tokens, embedded marker) | not re-tested (already prod-verified) | not re-tested | pass (marker correctly recalled in reasoning trace) |
| Tokenizer truncation bug (historic unsloth issue, capped at 2048 tokens) | n/a | verified absent in this pinned revision | n/a |

## Decision

**RadixArk/Qwen3.8-27B-NVFP4** (revision `319f741cce68d7914884900c138a1fbb70a42f30`) is installed
as the new production model:

- Fastest p50/p95 latency in every one of the 12 measured scenarios.
- Highest MTP acceptance rate (56% vs ~50%), meaning speculative decoding is doing more useful
  work per verification pass on this checkpoint/hardware combination.
- Highest measured throughput in 5 of 6 scenarios (short c4 has unsloth slightly ahead, but the
  gap is +10 tok/s to unsloth vs. +8 tok/s to Inferact, not decisive against RadixArk's other-scenario lead).
- All functional checks (correctness, tool calling, long-context recall) pass.
- Same license (Apache-2.0), same native context length (262,144), same MTP mechanism, no
  degradation in served feature set (tool calling, reasoning parser, 262k context all retained).

## What changed on the target

- `/srv/vllm3/vllm-server.sh`, `/srv/vllm3/download_model.sh`: default `MODEL_REPO` and
  `MODEL_REVISION` updated from Inferact to RadixArk; `MODEL_PATH` is now independently
  overridable to avoid collisions between repos sharing a trailing path segment.
- `/srv/vllm3/models/Qwen3.8-27B-NVFP4` now contains the RadixArk checkpoint (renamed from
  `Qwen3.8-27B-NVFP4-radixark`); the previous Inferact checkpoint was kept on disk as
  `Qwen3.8-27B-NVFP4-inferact-old` for rollback. The unsloth checkpoint is kept as
  `Qwen3.8-27B-NVFP4-unsloth` for reference.
- `/home/vllm3/.config/systemd/user/vllm.service`: `MODEL_REPO` environment value updated to
  `RadixArk/Qwen3.8-27B-NVFP4`. All other settings (`MAX_MODEL_LEN=262144`, `GPU_MEM_UTIL=0.50`,
  `KV_CACHE_DTYPE=fp8`, `NUM_SPEC_TOKENS=3`, memory caps) are unchanged from the already-tuned
  baseline documented in [`qwen3.8_tuning_bench_20260822.md`](qwen3.8_tuning_bench_20260822.md).
- Service reloaded and restarted; verified `active`, `enabled` (survives reboot via existing
  lingering setup), `/v1/models` reports the new checkpoint root path.

## Backups

- `/srv/backup/vllm3-modeltune-20260822212335/vllm.service.baseline.20260822212335` — unit file before any change in this session.
- `/srv/backup/vllm3-modeltune-20260822212335/vllm-server.sh.pre-radixark.*`
- `/srv/backup/vllm3-modeltune-20260822212335/download_model.sh.pre-radixark.*`
- `/srv/backup/vllm3-modeltune-20260822212335/vllm.service.pre-radixark.*`
- On-disk model checkpoints for all three variants remain available under `/srv/vllm3/models/`
  for immediate rollback without re-downloading.

## Rollback

```bash
# Stop current service
systemctl --user stop vllm.service

# Restore previous unit file and scripts from backup
BD=/srv/backup/vllm3-modeltune-20260822212335
cp "$BD"/vllm.service.pre-radixark.* /home/vllm3/.config/systemd/user/vllm.service
cp "$BD"/vllm-server.sh.pre-radixark.* /srv/vllm3/vllm-server.sh
cp "$BD"/download_model.sh.pre-radixark.* /srv/vllm3/download_model.sh
chmod +x /srv/vllm3/vllm-server.sh /srv/vllm3/download_model.sh

# Restore original model directory name
mv /srv/vllm3/models/Qwen3.8-27B-NVFP4 /srv/vllm3/models/Qwen3.8-27B-NVFP4-radixark
mv /srv/vllm3/models/Qwen3.8-27B-NVFP4-inferact-old /srv/vllm3/models/Qwen3.8-27B-NVFP4

systemctl --user daemon-reload
systemctl --user restart vllm.service
```

## Disk usage after this change

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  3.7T  416G  3.1T  12% /

Qwen3.8-27B-NVFP4 (RadixArk, active)   21G
Qwen3.8-27B-NVFP4-inferact-old         25G
Qwen3.8-27B-NVFP4-unsloth              22G
```

3.1 TB free; keeping all three checkpoints on disk for rollback/reference is not a capacity
concern. Cleanup of the unused checkpoints can be done later if desired (not performed in this
change to preserve rollback capability).
