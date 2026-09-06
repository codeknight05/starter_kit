# Live Defense Prep

## Commands To Rerun

From `starter_kit/`:

```bash
python your-submission/partA/build_corpus.py
python your-submission/partA/audit_fertility.py
python your-submission/partA/corrected_analysis.py
python fertility.py --corpus eng=corpus_sample/eng_sample.txt --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2
python your-submission/partB/calc_capacity.py
```

The corrected analysis downloads XLM-R and Qwen tokenizer files on first use. Token counts are not latency benchmarks; the run requires network access or a local Hugging Face cache.

## Tokenizer Numbers To Remember

The original v0 smoke test reproduces:

- English: 1.27 tok/word, 0.226 tok/char
- Hindi: 7.45 tok/word, 1.579 tok/char
- Ratio: 5.89x Hindi/English fertility

Corrected FLORES-200 sentence-cost ratios:

| Tokenizer | Hindi | Kannada | Tamil |
|---|---:|---:|---:|
| GPT-2 | 7.36x | 13.57x | 15.50x |
| XLM-RoBERTa-base | 1.26x | 1.37x | 1.37x |
| Qwen2.5-7B | 4.32x | 6.85x | 6.04x |

Main defense sentence: capacity planning should use tokens per comparable semantic request payload, not tokens per whitespace word.

Do not say that XLM-R token reductions prove serving is equally faster. They demonstrate tokenizer/model-vocabulary dependence and motivate an end-to-end serving benchmark.

## KV Cache Derivation

Bytes per token:

`2 * layers * kv_heads * head_dim * bytes_per_element`

`2 * 28 * 8 * 128 * 2 = 114,688 bytes/token = 112 KiB/token`

Memory per 4096-token sequence:

`4096 * 114,688 = 469,762,048 bytes = 448 MiB`

Ideal spec-only capacity:

`24 GiB * 0.92 = 22.08 GiB usable`

`4.2B * 2 bytes = 8.4 GB = 7.82 GiB weights`

`1.6 GB = 1.49 GiB runtime overhead`

`(22.08 - 7.82 - 1.49) GiB / 448 MiB = 29.2 ideal sequences`

Log check: theoretical spec-only capacity is 29.2 sequences; observed operational behavior is about 24-26 long-context sequences. Batch 24 has no preemptions at 0.93 cache utilization; batch 32 preempts 7; batch 48 preempts 23.

## Goodput Derivation

For batch 24, prompt 3584, generation 512:

Direct output-token goodput:

`24 * 512 / 61.16 = 200.92 output tok/s`

Equivalent derivation from harness throughput:

`1607.4 reported tok/s * 512 / (3584 + 512) = 200.93 output tok/s`

Do not use `batch / itl` as the final goodput answer here; it estimates median steady-state decode cadence and does not include the full wall-clock effects that the assignment asks to derive from the log.

## Counterfactuals

If KV cache uses FP8 instead of FP16, KV bytes/token halves to 57,344 bytes and ideal capacity doubles from about 29 to about 58 full 4096-token sequences. In practice, expected capacity would also be below ideal because of allocation and scheduler overhead, but batch 48 should stop preempting if model quality tolerates FP8 KV.

If the model used full MHA with 24 KV heads instead of GQA with 8 KV heads, KV bytes/token triples to 344,064 bytes/token and ideal full-context capacity drops to about 9-10 sequences.

If product insists on SFT for Part C, scope it only to Hindi/Kannada first, because those are the only languages with native reviewer coverage in the stated constraints.

## Defense Questions

- **Why not tok/word?** Word segmentation and morphology differ, so words do not hold semantic payload constant. Parallel sentences are a better first-order aligned unit.
- **Why not tok/byte or tok/grapheme?** They describe text representation, not comparable user-turn payload; use them diagnostically, not as the primary routing denominator.
- **Why does longer prompt appear faster?** `reported_tok_s` includes prefill tokens. It is not generated-answer goodput.
- **Why not batch 48?** KV saturation causes preemptions and prompt recomputation. The observed batch-48 output goodput is only 162.31 tok/s, not 3200 tok/s.
- **What is the two-replica number?** `2 * 200.92 = 401.84 output tok/s` is a projection assuming two replicas each sustain batch-24 goodput, not a measured benchmark.
- **Why prompt engineering?** It is reversible and fits the reviewer budget; SFT and a rewriter add validation or serving cost that the constraints do not support yet.
