# Production Validation Plan

The checked-in measurements are an audit baseline. They do not replace validation on production traffic or a real serving deployment.

## 1. Production language/payload sample

Sample a privacy-reviewed, stratified set of real user turns by detected language and route. Preserve the original turn distribution, including short turns, code-switching, transliteration, slang, spelling variation, and multi-turn context. Report the sample counts and detection confidence before comparing routes.

Primary tokenizer metric:

`input_plus_output_tokens_per_turn` by language and route.

Use the same request sample for every candidate tokenizer/model route. Compare ratios to English and report uncertainty or bootstrap intervals; do not assume one translated sentence represents every production turn.

## 2. End-to-end serving benchmark

For each candidate route, hold model weights, hardware, request mix, prompt template, maximum context, and generation policy constant. Measure separately:

- prefill throughput and time to first token;
- decode throughput and inter-token latency;
- output-token goodput;
- p50/p95/p99 end-to-end latency;
- GPU memory and KV-cache utilization;
- scheduler preemptions and request failures.

Tokenizer counts can motivate this test, but cannot substitute for it.

## 3. Two-replica validation

Run the same long-context workload used in `bench/bench_log.csv` on two independent replicas. Compare:

- one-replica batch-48 behavior;
- two replicas receiving 24 requests each;
- aggregate generated-token goodput;
- per-request p95 latency;
- preemptions and cache utilization on each replica.

Only replace the projection `2 * 200.92 = 401.84 output tok/s` with a measured result after this benchmark has completed. Until then, it remains a capacity projection.

## 4. Offline reproducibility

After tokenizer files are cached, rerun without network access:

```bash
python your-submission/partA/corrected_analysis.py --offline
```

If a requested Hugging Face tokenizer is not cached, the command should fail rather than silently substituting a different tokenizer.
