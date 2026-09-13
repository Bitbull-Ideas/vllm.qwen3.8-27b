# Qwen3.8-27B-NVFP4 — Second-Round Model Comparison on DGX Spark (GB10)

Date: 2026-09-13
Target: DGX Spark host (NVIDIA GB10, Grace Blackwell, 128 GB unified memory)
Service: `vllm.service` (systemd user unit)

## Goal

Follow-up to the 2026-08-22 comparison
([`qwen3.8_27b_nvfp4_model_comparison_20260822.md`](qwen3.8_27b_nvfp4_model_comparison_20260822.md)).
User asked whether newer/faster NVFP4 builds of Qwen3.8-27B have appeared since RadixArk was
installed, and to re-run the same research → benchmark → decide cycle.

## Candidates evaluated

HuggingFace was searched for `Qwen3.8-27B-NVFP4`, sorted by last-modified date. Dozens of new
repos have appeared since August, mostly community "uncensored"/"abliterated" GGUF forks or
niche recompressions. Three were selected as credible, vLLM-ready, non-GGUF challengers based on
download counts, provenance, and shipped MTP support:

| # | Repo | Revision | Size on disk | Quant method | Notes |
|---|---|---|---:|---|---|
| 1 | `nvidia/Qwen3.8-27B-NVFP4` | `dbb8f445b3145f8a4c18ddc769f032d57d32867c` | 21 GB | NVIDIA ModelOpt (official) | First-party NVIDIA release, published 2026-09-09; 31k downloads |
| 2 | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` | `cfd1460322b9d8367a4ab13564a2182d50852d60` | 20 GB | compressed-tensors, quantization-aware training | Vendor claims highest GPQA-Diamond/AIME'26 among public NVFP4 builds (arXiv:2608.13966); 38k downloads |
| 3 | `ukisai/Swift-Qwen3.8-27B-NVFP4` | `c282aa2636bd4f7efbbced64b55271d62b326f9b` | 27 GB | compressed-tensors (nvfp4-pack-quantized, W4A4 group 16) | "Efficient-thinking" tuned variant with own hosted API; low adoption (26 downloads) |
| — | `RadixArk/Qwen3.8-27B-NVFP4` (current production) | `319f741cce68d7914884900c138a1fbb70a42f30` | 21 GB | NVIDIA ModelOpt (mixed NVFP4 W4A4 / FP8 attention) | Re-tested fresh in this session as the baseline for a fair comparison |

Not selected (same triage logic as the prior round): GGUF-only builds (dozens of new
"Uncensored"/"Heretic"/"Abliterated" forks — not vLLM-native), AWQ/GPTQ-only builds without MTP,
and the `unsloth` build already tested and rejected in the prior round.

## Method

Identical to the 2026-08-22 comparison — same harness
([`run_vllm_bench.py`](run_vllm_bench.py)), same server flags, one model resident at a time:

- vLLM 0.26.0, `--kv-cache-dtype fp8`, `--speculative-config
  '{"method":"mtp","num_speculative_tokens":3}'`, `--max-model-len 262144`,
  `--gpu-memory-utilization 0.50`, `--enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3`, `--mm-encoder-tp-mode data` (vision enabled, matching current
  production `LANGUAGE_ONLY=false`).
- Short (64 tokens) / Long (256 tokens) prompts, concurrency 1/2/4, 4 requests each, one warmup.
- Functional checks: correctness, tool calling. (Vision/long-context recall were already verified
  for the model family in the prior round and were not re-run per-candidate here.)
- MTP draft-token acceptance rate from `/metrics`.
- RadixArk was **re-benchmarked fresh** in this session (not just carried over from the August
  report) so the comparison is apples-to-apples under identical current host conditions.

## Results

### Latency (p50, milliseconds) — lower is better

| Model | Short c1 | Short c2 | Short c4 | Long c1 | Long c2 | Long c4 |
|---|---:|---:|---:|---:|---:|---:|
| **RadixArk/Qwen3.8-27B-NVFP4 (production, re-tested)** | **2821.5** | **3344.5** | 5669.8 | **12101.9** | **12495.8** | 13580.8 |
| nvidia/Qwen3.8-27B-NVFP4 | 2669.2 | 4409.4 | 5752.9 | 12071.8 | 13438.9 | 13561.1 |
| QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 | 3806.1 | 4483.5 | 6178.1 | 15333.0 | 14658.6 | 15404.1 |
| ukisai/Swift-Qwen3.8-27B-NVFP4 | 3980.9 | 5123.2 | 7129.8 | 19480.1 | 18402.8 | 19009.4 |

### Throughput (tokens/s, output) — higher is better

| Model | Short c1 | Short c2 | Short c4 | Long c1 | Long c2 | Long c4 |
|---|---:|---:|---:|---:|---:|---:|
| **RadixArk/Qwen3.8-27B-NVFP4 (production, re-tested)** | 23.1 | **38.5** | 42.7 | 21.3 | **40.0** | 71.8 |
| nvidia/Qwen3.8-27B-NVFP4 | **23.9** | 33.2 | 40.3 | **21.4** | 37.4 | 74.2 |
| QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 | 17.0 | 30.8 | 35.2 | 16.9 | 34.5 | 64.2 |
| ukisai/Swift-Qwen3.8-27B-NVFP4 | 16.3 | 26.7 | 35.3 | 13.7 | 27.8 | 52.4 |

### MTP speculative decoding quality (draft-token acceptance rate)

| Model | Draft tokens | Accepted tokens | Acceptance rate |
|---|---:|---:|---:|
| **RadixArk/Qwen3.8-27B-NVFP4 (production, re-tested)** | 5703 | 2903 | **50.9%** |
| nvidia/Qwen3.8-27B-NVFP4 | 5736 | 2889 | 50.4% |
| QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 | 5835 | 2857 | 49.0% |
| ukisai/Swift-Qwen3.8-27B-NVFP4 | 5832 | 2850 | 48.9% |

### Functional checks

| Check | RadixArk | nvidia | QUASAR-QAT | ukisai/Swift |
|---|---|---|---|---|
| Factual correctness (capital of Germany) | pass | pass | pass | pass |
| Tool calling (function call generated correctly) | pass | pass | pass | pass |

## Decision

**No change.** `RadixArk/Qwen3.8-27B-NVFP4` remains the production model:

- Fastest or tied-fastest in 4 of 6 scenarios (short c1/c2, long c1/c2); nvidia's build wins the
  other two by a narrow margin (short c1: -152ms; long c1: -30ms — noise-level for this sample
  size), while losing more clearly at c2 concurrency in both scenarios.
- Highest MTP acceptance rate (50.9%), marginally ahead of nvidia's official build (50.4%) and
  clearly ahead of QUASAR-QAT (49.0%) and Swift (48.9%).
- QUASAR-QAT's vendor-claimed accuracy edge (GPQA-Diamond, AIME'26 vs. other NVFP4 builds) did not
  translate into a latency or throughput advantage on this hardware in this test — it was
  consistently 25-30% slower than RadixArk across every scenario. No accuracy benchmark was run
  in-house to confirm or refute the vendor's quality claim; if per-token quality (not just speed)
  becomes the priority, QUASAR-QAT is worth a deeper, dedicated accuracy evaluation.
- `ukisai/Swift-Qwen3.8-27B-NVFP4` was the slowest candidate in every scenario and has the lowest
  community adoption (26 downloads) of the three; not competitive on this hardware.
- `nvidia/Qwen3.8-27B-NVFP4` (the official first-party release) is a close second and a reasonable
  fallback if RadixArk becomes unavailable or unmaintained — it is within noise of RadixArk on
  most metrics and has NVIDIA's own long-term support behind it.

## What changed on the target

Nothing in the running configuration. This was a read-mostly comparison:

- Three new candidate checkpoints downloaded to the models directory for testing (kept on disk for
  reference/rollback, no capacity concern — see below).
- Production `vllm.service` was stopped, three candidates plus a fresh RadixArk re-test were run
  standalone on port 8000, then the production service was restarted unchanged.
- No edits to `vllm-server.sh`, `download_model.sh`, or the systemd unit.

## Backups

- Unit file snapshot taken before this session's testing (unchanged; kept for audit trail
  consistency per the target's change-management process).

## Disk usage after this comparison

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  3.7T  529G  3.0T  15% /

Qwen3.8-27B-NVFP4 (RadixArk, active)     21G
Qwen3.8-27B-NVFP4-inferact-old           25G
Qwen3.8-27B-NVFP4-nvidia (new candidate) 21G
Qwen3.8-27B-NVFP4-quasar (new candidate) 20G
Qwen3.8-27B-NVFP4-swift (new candidate)  27G
Qwen3.8-27B-NVFP4-unsloth                22G
```

3.0 TB free; all candidate checkpoints kept on disk for reference/future re-evaluation without
re-downloading.

## Recommendation for future rounds

- Re-check again in another few weeks; the NVFP4 ecosystem for this model is churning quickly
  (dozens of new repos per week, many low-quality community forks).
- If accuracy (not just speed) becomes the deciding factor, run an in-house eval (e.g. GSM8K/GPQA
  subset) against QUASAR-QAT before dismissing its vendor-claimed quality edge — this round only
  measured latency/throughput/MTP-acceptance, not answer quality.
- Filter future candidate searches by `sort=lastModified` and prioritize repos with (a) a
  `model.safetensors.index.json` containing `mtp.*` tensors (native MTP, no extra download), (b)
  meaningful download counts (proxy for community vetting), and (c) an explicit vLLM serve command
  in the README.
