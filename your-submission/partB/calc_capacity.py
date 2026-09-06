#!/usr/bin/env python3
"""
calc_capacity.py - Reproducible Part B arithmetic from bench/model_spec.md
and bench/bench_log.csv.

Run from starter_kit/:
    python your-submission/partB/calc_capacity.py
"""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH_LOG = ROOT / "bench" / "bench_log.csv"

# Constants copied from bench/model_spec.md.
LAYERS = 28
KV_HEADS = 8
HEAD_DIM = 128
BYTES_PER_FP16 = 2
GPU_GIB = 24
GPU_MEMORY_UTILIZATION = 0.92
PARAMETERS = 4.2e9
RUNTIME_OVERHEAD_GB = 1.6
MAX_MODEL_LEN = 4096

BYTES_PER_GIB = 1024 ** 3
BYTES_PER_MIB = 1024 ** 2


def load_rows():
    with BENCH_LOG.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_int(row, key):
    return int(row[key])


def as_float(row, key):
    return float(row[key])


def main():
    rows = load_rows()

    kv_bytes_per_token = LAYERS * 2 * KV_HEADS * HEAD_DIM * BYTES_PER_FP16
    kv_mib_per_sequence = (MAX_MODEL_LEN * kv_bytes_per_token) / BYTES_PER_MIB

    usable_gib = GPU_GIB * GPU_MEMORY_UTILIZATION
    weights_gib = (PARAMETERS * BYTES_PER_FP16) / BYTES_PER_GIB
    runtime_overhead_gib = (RUNTIME_OVERHEAD_GB * 1e9) / BYTES_PER_GIB
    kv_pool_gib = usable_gib - weights_gib - runtime_overhead_gib
    ideal_sequences = (kv_pool_gib * 1024) / kv_mib_per_sequence

    print("=== B1 KV Cache Arithmetic ===")
    print(f"KV bytes/token = {LAYERS} * 2 * {KV_HEADS} * {HEAD_DIM} * {BYTES_PER_FP16} = {kv_bytes_per_token:,}")
    print(f"KV per 4096-token sequence = {kv_mib_per_sequence:.1f} MiB")
    print(f"Usable memory = {GPU_GIB} GiB * {GPU_MEMORY_UTILIZATION} = {usable_gib:.2f} GiB")
    print(f"Weights = 4.2B * 2 bytes = {weights_gib:.2f} GiB")
    print(f"Runtime overhead = 1.6 GB = {runtime_overhead_gib:.2f} GiB")
    print(f"KV pool = {kv_pool_gib:.2f} GiB")
    print(f"Ideal max 4096-token sequences = {ideal_sequences:.2f}")

    print("\n=== Long-Context Sweep ===")
    print("batch,prompt,gen,wall_s,reported_tok_s,output_tok_s,reported_output_adjusted_tok_s,preempted,kv_util")
    for row in rows:
        if as_int(row, "prompt_len") != 3584:
            continue
        batch = as_int(row, "batch_size")
        prompt_len = as_int(row, "prompt_len")
        gen_len = as_int(row, "gen_len")
        wall_clock_s = as_float(row, "wall_clock_s")
        reported_tok_s = as_float(row, "reported_tok_s")
        total_len = prompt_len + gen_len
        output_tok_s = batch * gen_len / wall_clock_s
        adjusted_tok_s = reported_tok_s * gen_len / total_len
        print(
            f"{batch},{prompt_len},{gen_len},{wall_clock_s:.2f},{reported_tok_s:.1f},"
            f"{output_tok_s:.2f},{adjusted_tok_s:.2f},{as_int(row, 'preempted_seqs')},"
            f"{as_float(row, 'kv_cache_util'):.2f}"
        )

    batch24 = next(
        row for row in rows
        if as_int(row, "prompt_len") == 3584 and as_int(row, "batch_size") == 24
    )
    goodput_direct = as_int(batch24, "batch_size") * as_int(batch24, "gen_len") / as_float(batch24, "wall_clock_s")
    goodput_adjusted = as_float(batch24, "reported_tok_s") * as_int(batch24, "gen_len") / (
        as_int(batch24, "prompt_len") + as_int(batch24, "gen_len")
    )

    print("\n=== B3 Batch-24 Goodput ===")
    print(f"Direct output goodput = 24 * 512 / 61.16 = {goodput_direct:.2f} output tok/s")
    print(f"Reported-throughput adjusted = 1607.4 * 512 / 4096 = {goodput_adjusted:.2f} output tok/s")


if __name__ == "__main__":
    main()
