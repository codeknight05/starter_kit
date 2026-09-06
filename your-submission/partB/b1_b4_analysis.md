# Part B: Capacity Reconciliation

Source files: `bench/model_spec.md`, `bench/bench_log.csv`. The same calculations are reproducible with `python your-submission/partB/calc_capacity.py`; captured output is in `partB/calc_capacity_output.txt`.

## B1. KV Cache Math

For one token in one layer:

`K + V = 2 * kv_heads * head_dim = 2 * 8 * 128 = 2048 fp16 elements`

`2048 elements * 2 bytes = 4096 bytes per token per layer`

Across 28 layers:

`28 * 4096 = 114,688 bytes/token = 112 KiB/token`

For a full 4096-token sequence:

`4096 * 114,688 = 469,762,048 bytes = 448 MiB`

Approximate ideal concurrent 4096-token sequences from the spec:

`24 GiB * 0.92 = 22.08 GiB usable`

Weights: `4.2B params * 2 bytes = 8.4 GB = 7.82 GiB`  
Runtime overhead: `1.6 GB = 1.49 GiB`

KV pool:

`22.08 - 7.82 - 1.49 = 12.77 GiB`

Ideal sequence capacity:

`12.77 GiB / 448 MiB = 29.2 sequences`

This is **theoretical spec-only capacity**, not an operational promise. The log shows an observed operational limit around 24-26 long-context sequences: batch 24 reaches `kv_cache_util=0.93` with zero preemptions, while batch 32 reaches `0.97` and preempts 7 sequences. The gap is consistent with block allocation granularity, scheduler reservations/watermarks, fragmentation, and runtime overhead not itemized in the simplified arithmetic.

## B2. Long-Context Throughput Anomaly

In the long-context sweep, `reported_tok_s` rises through batch 24, then falls:

| batch | reported_tok_s | wall_clock_s | preempted_seqs | kv_cache_util |
|---:|---:|---:|---:|---:|
| 16 | 1311.4 | 49.97 | 0 | 0.62 |
| 24 | 1607.4 | 61.16 | 0 | 0.93 |
| 32 | 1384.0 | 94.71 | 7 | 0.97 |
| 48 | 1298.5 | 151.41 | 23 | 0.97 |

The anomaly is the drop after batch 24. The mechanism is KV-cache saturation: once the long prompts no longer fit, the scheduler preempts sequences. Rescheduled sequences must re-prefill the 3584-token prompt, spending GPU time on repeated prompt work rather than new output tokens.

Recommended change: cap active long-context concurrency at `max_num_seqs=24` per L4 replica and add another replica before admitting a 48-request long-context burst to one GPU. First-order projection, not a measured two-replica benchmark: `2 * 200.92 = 401.84 output tok/s`, assuming each replica sustains batch-24 goodput. `max_num_seqs=24` is a deployment recommendation from the observed zero-preemption point, not a universal vLLM optimum.

## B3. Goodput vs Reported Throughput

`REPORT_v0.md` misread `reported_tok_s` as generated-token throughput. The harness throughput includes prompt tokens:

`reported_tok_s = num_requests * (prompt_len + gen_len) / wall_clock_s`

For the batch-24 long-prompt row, 3584 of 4096 tokens are prompt tokens, so 87.5% of the reported throughput is prefill accounting, not delivered answer tokens.

Honest output-token goodput, two equivalent derivations:

1. Direct from generated tokens:

`24 requests * 512 gen tokens / 61.16 s = 200.92 output tok/s`

2. From the harness counter after removing prompt-token accounting:

`1607.4 reported tok/s * (512 / 4096) = 200.93 output tok/s`

The report should have said: long prompts inflate raw token throughput because prompt prefill is counted. On L4, long-context generated-token goodput peaks around **201 output tok/s** at batch 24, and larger active batches trigger KV preemption and worse latency.

## B4. Counter To Confirm Mechanism

Pull vLLM's preemption counter, `vllm:num_preemptions_total`, alongside GPU cache usage (`vllm:gpu_cache_usage_perc`). I expect 0 preemptions through batch 24, then increments matching the log: 7 at batch 32 and 23 at batch 48, with cache usage pinned near 0.97.
