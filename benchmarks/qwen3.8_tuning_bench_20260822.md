# Qwen3.8-27B Tuning & Benchmark Report

Date: 2026-08-22
Target: spark.lb.bitbull.ch
Scope: vLLM serving on /srv/vllm3 (service `vllm.service`, user vllm3)

## Configs tested

Baseline/Production baseline used in final state:
- `MAX_MODEL_LEN=262144`
- `GPU_MEM_UTIL=0.50`
- `KV_CACHE_DTYPE=fp8`
- `NUM_SPEC_TOKENS=3`
- `--language-model-only=true`
- `--enable-auto-tool-choice` and `--tool-call-parser qwen3_coder` **kept enabled**
- `--reasoning-parser qwen3` in `/srv/vllm3/vllm-server.sh`

## Bench harness

For each config, 6 requests per scenario at concurrencies 1/2/4, with one warmup request per scenario:
- Short: 64 output tokens, single-sentence prompt
- Long: 256 output tokens, 6-sentence prompt
- End-to-end POST to `/v1/chat/completions`

## Results (ms)

Columns: `scenario`, `conc`, `wall_s`, `p50_ms`, `p95_ms`, `avg_ms`, `throughput_rps`, `throughput_tokens_s`

### A) `NUM_SPEC_TOKENS=2` (while `MAX_MODEL_LEN=262144`)

| scenario | conc | wall_s | p50_ms | p95_ms | avg_ms | throughput_rps | throughput_tokens_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 1 | 24.559 | 4073.7 | 4288.2 | 4092.7 | 0.24 | 15.6 |
| short | 2 | 13.257 | 4363.3 | 4757.4 | 4393.7 | 0.45 | 29.0 |
| short | 4 | 8.366 | 4239.4 | 4450.2 | 4243.2 | 0.72 | 45.9 |
| long  | 1 | 105.768 | 18210.9 | 18250.9 | 17627.7 | 0.06 | 14.5 |
| long  | 2 | 50.147 | 16464.6 | 16590.1 | 16448.0 | 0.12 | 30.6 |
| long  | 4 | 35.072 | 17618.3 | 17620.6 | 17294.9 | 0.17 | 43.8 |

### B) `NUM_SPEC_TOKENS=3` (`MAX_MODEL_LEN=262144`) — main baseline/validation

| scenario | conc | wall_s | p50_ms | p95_ms | avg_ms | throughput_rps | throughput_tokens_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 1 | 23.287 | 3924.5 | 3934.1 | 3880.8 | 0.26 | 16.5 |
| short | 2 | 12.574 | 4083.3 | 4700.4 | 4136.4 | 0.48 | 30.5 |
| short | 4 | 7.940 | 3504.2 | 3757.6 | 3700.4 | 0.76 | 48.4 |
| long  | 1 | 106.156 | 17635.2 | 17842.8 | 17692.4 | 0.06 | 14.5 |
| long  | 2 | 54.746 | 18249.9 | 18449.4 | 17651.5 | 0.11 | 28.1 |
| long  | 4 | 33.784 | 17083.3 | 17223.2 | 16865.3 | 0.18 | 45.5 |

### C) `NUM_SPEC_TOKENS=4` (`MAX_MODEL_LEN=262144`)

| scenario | conc | wall_s | p50_ms | p95_ms | avg_ms | throughput_rps | throughput_tokens_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 1 | 25.119 | 4506.1 | 4608.6 | 4186.1 | 0.24 | 15.3 |
| short | 2 | 16.443 | 4052.5 | 7498.1 | 5262.1 | 0.36 | 23.4 |
| short | 4 | 8.228 | 4399.8 | 4402.0 | 4057.6 | 0.73 | 46.7 |
| long  | 1 | 107.418 | 17637.4 | 18134.3 | 17902.8 | 0.06 | 14.3 |
| long  | 2 | 51.084 | 17052.0 | 17116.0 | 16900.5 | 0.12 | 30.1 |
| long  | 4 | 36.100 | 18895.6 | 19823.8 | 18663.4 | 0.17 | 42.5 |

### D) `MAX_MODEL_LEN=131072` (temporary, `NUM_SPEC_TOKENS=3`)

| scenario | conc | wall_s | p50_ms | p95_ms | avg_ms | throughput_rps | throughput_tokens_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 1 | 25.969 | 4240.3 | 4458.2 | 4327.7 | 0.23 | 14.8 |
| short | 2 | 14.759 | 4931.2 | 5062.3 | 4749.7 | 0.41 | 26.0 |
| short | 4 | 11.242 | 6843.4 | 6844.3 | 6037.5 | 0.53 | 34.2 |
| long  | 1 | 104.852 | 17745.1 | 17784.4 | 17474.9 | 0.06 | 14.6 |
| long  | 2 | 51.126 | 17114.9 | 17353.1 | 16928.1 | 0.12 | 30.0 |
| long  | 4 | 35.219 | 17768.0 | 18764.7 | 17854.3 | 0.17 | 43.6 |

### Final fresh run on production config (`spec=3`, `256k`, after all adjustments)

| scenario | conc | wall_s | p50_ms | p95_ms | avg_ms | throughput_rps | throughput_tokens_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 1 | 15.386 | 3933.3 | 3933.3 | 3846.1 | 0.26 | 16.6 |
| short | 2 | 8.604 | 4390.1 | 4390.1 | 4295.0 | 0.46 | 29.8 |
| short | 4 | 4.456 | 4420.3 | 4420.3 | 4156.1 | 0.90 | 57.5 |
| long  | 1 | 69.727 | 17581.5 | 17581.5 | 17431.5 | 0.06 | 14.7 |
| long  | 2 | 34.108 | 16766.0 | 16766.0 | 16677.2 | 0.12 | 30.0 |
| long  | 4 | 18.066 | 17583.0 | 17583.0 | 17506.0 | 0.22 | 56.7 |

## Quality/safety proxy from metrics endpoint (live after final run)
- `spec_decode_num_drafts_total=1919`
- `spec_decode_num_draft_tokens_total=5757`
- `spec_decode_num_accepted_tokens_total=2880`
- Draft acceptance ratio ~50.0%

## Recommendation
- Keep production at `NUM_SPEC_TOKENS=3`, `MAX_MODEL_LEN=262144`.
- `NUM_SPEC_TOKENS=4` and reduced context length did not show stable improvement.
- Short-latency with high concurrency is best at spec=3; 256k context performs better for throughput consistency in this workload.

## Note on tooling capability
- Tooling parameters were not removed in any benchmark-only run; current command-line still contains `--enable-auto-tool-choice` and tool parsers.
- MTP remains active and available for model-internal speculative decoding in this configuration.
