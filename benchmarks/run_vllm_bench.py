#!/usr/bin/env python3
"""Benchmark harness used for vLLM tuning and model-checkpoint comparisons on DGX Spark.

Runs two request shapes (short/long output) at several concurrency levels against a running
vLLM OpenAI-compatible server and reports p50/p95 latency and throughput. Used to produce the
data in ../qwen3.8_tuning_bench_20260822.md and ../qwen3.8_27b_nvfp4_model_comparison_20260822.md.

Usage:
    python3 run_vllm_bench.py --url http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B
"""

from concurrent.futures import ThreadPoolExecutor
from statistics import mean
import argparse
import requests
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8000', help='vLLM endpoint')
    p.add_argument('--model', default='Qwen/Qwen3.8-27B', help='served model name')
    p.add_argument('--short-tokens', type=int, default=64)
    p.add_argument('--long-tokens', type=int, default=256)
    p.add_argument('--repeat', type=int, default=6)
    p.add_argument('--conc', nargs='+', type=int, default=[1, 2, 4])
    return p.parse_args()


def wait_ready(base: str, timeout_s: int = 420):
    for _ in range(timeout_s):
        try:
            r = requests.get(f"{base}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def call_once(url, payload):
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    latency = time.perf_counter() - t0
    usage = r.json().get('usage', {})
    return latency, usage.get('completion_tokens', 0)


def bench_case(url, payload, conc, repeat):
    # warmup
    requests.post(url, json=payload, timeout=180)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        rows = [f.result() for f in [ex.submit(call_once, url, payload) for _ in range(repeat)]]
    wall = time.perf_counter() - start
    times = [x[0] for x in rows]
    toks = [x[1] for x in rows]

    ps = sorted(times)
    p50 = ps[len(ps) // 2]
    p95 = ps[max(0, int(len(ps) * 0.95) - 1)]
    return {
        'requests': repeat,
        'wall_s': round(wall, 3),
        'p50_ms': round(p50 * 1000, 1),
        'p95_ms': round(p95 * 1000, 1),
        'avg_ms': round(mean(times) * 1000, 1),
        'avg_tokens': round(mean(toks), 1),
        'throughput_rps': round(repeat / wall, 2),
        'throughput_tokens_s': round(sum(toks) / wall, 1),
    }


def main():
    a = parse_args()
    if not wait_ready(a.url):
        raise SystemExit('service-not-ready')

    url = f"{a.url}/v1/chat/completions"
    cases = [
        ('short', {
            'model': a.model,
            'messages': [{'role': 'user', 'content': 'Erkläre in einem Satz den Unterschied zwischen CPU und GPU.'}],
            'temperature': 0.2,
            'max_tokens': a.short_tokens,
            'top_p': 0.95,
        }),
        ('long', {
            'model': a.model,
            'messages': [{'role': 'user', 'content': 'Erkläre präzise in 6 Sätzen, warum Speicherhierarchie für LLM-Inferenz wichtig ist.'}],
            'temperature': 0.2,
            'max_tokens': a.long_tokens,
            'top_p': 0.95,
        }),
    ]

    print('BENCH_START')
    for name, payload in cases:
        for c in a.conc:
            r = bench_case(url, payload, c, a.repeat)
            print(name, r)
    print('BENCH_END')


if __name__ == '__main__':
    main()
