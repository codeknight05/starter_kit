# NOTEBOOK: Chronological Audit Log

## 1. Initial Inspection

**Hypothesis:** `REPORT_v0.md` is making broad routing claims from very small evidence.

**Experiment:** Read `REPORT_v0.md`, `fertility.py`, `bench/model_spec.md`, and `bench/bench_log.csv`.

**Command:** `python fertility.py --corpus eng=corpus_sample/eng_sample.txt --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2`

**Result:** The report uses ~10 English/Hindi sample sentences, GPT-2 only, and treats `tokens/word` and harness `reported_tok_s` as direct cost signals.

**Revision:** Split the audit into tokenizer evidence (Part A), capacity reconciliation (Part B), and product decision memo (Part C).

## 2. Corpus Construction Attempt

**Hypothesis:** A real cross-language comparison needs a shared multilingual eval corpus, not the toy sample.

**Experiment:** Tried official FLORES through `datasets.load_dataset("facebook/flores", ...)`.

**Command:** `python -c "from datasets import load_dataset; load_dataset('facebook/flores', split='devtest')"`

**Result:** The official Hub dataset was gated/unauthenticated in this environment.

**Revision:** First built a runnable fallback from `Helsinki-NLP/opus-100` English-target pairs. This produced 250 lines/language, but it was only pairwise parallel, not a shared multi-way corpus, so it was weaker than the assignment ideally wants.

## 3. Corpus Upgrade To FLORES-200

**Hypothesis:** A public FLORES-200 mirror can provide true multi-way rows for English, Hindi, Kannada, and Tamil.

**Experiment:** Probed `yash9439/flores200` and then updated `partA/build_corpus.py` to use `eng_Latn`, `hin_Deva`, `kan_Knda`, and `tam_Taml` from the `devtest` split, with OPUS-100 retained as a fallback.

**Result:** `python your-submission/partA/build_corpus.py` generated 250 multi-way parallel FLORES sentences per language.

**Revision:** Reran all Part A analysis on the regenerated FLORES corpus and replaced stale OPUS-based memo numbers. Corpus details are documented in `partA/CORPUS.md`.

## 4. Fertility Script Audit

**Hypothesis:** `fertility.py` has both code-level and metric-level problems.

**Experiment:** Ran `python your-submission/partA/audit_fertility.py`.

**Result:**

- `line.split(" ")` creates empty word entries on repeated spaces. On sample English line 7, fertility moves from `10/8 = 1.2500` to `10/7 = 1.4286`, so the naive split deflates that line by 12.5%.
- Macro-averaging per-line fertility differs from corpus-level micro-averaging. On the FLORES eval set, macro vs micro deltas are +1.22% English, +0.50% Hindi, +0.91% Kannada, and +0.34% Tamil.
- `tokens/word` is not the right cross-language routing denominator. On FLORES with GPT-2, Kannada is 17.5x English by `tok/word` but 13.6x by `tok/sentence`; the word denominator overstates the sentence-payload ratio by 29.1%.

**Revision:** Treat repeated-space splitting and macro averaging as measurable code/aggregation flaws, and treat `tokens per comparable semantic payload` as the routing denominator.

## 5. Harmless/Suspicious Feature Check

**Hypothesis:** `line.lower()` might corrupt Indic text or meaningfully alter token counts.

**Experiment:** Compared original vs `lower()` tokenization across 250 Hindi, Kannada, and Tamil FLORES sentences.

**Result:** Lowercasing touched only a small number of lines with caseful embedded Latin text. Net GPT-2 token deltas were +6 Hindi (+0.012%), +11 Kannada (+0.012%), and +14 Tamil (+0.013%).

**Revision:** Do not claim `lower()` is a Unicode-corruption bug. It is slightly noisy but far too small to explain `REPORT_v0`'s conclusions.

## 6. Corrected Tokenizer Analysis

**Hypothesis:** GPT-2's Indic penalty is mostly vocabulary mismatch.

**Experiment:** Ran `python your-submission/partA/corrected_analysis.py` across GPT-2, XLM-RoBERTa-base, and Qwen2.5-7B, using `tok/sentence`, `tok/word`, `tok/grapheme`, and `tok/byte`.

**Result:** On FLORES, GPT-2 sentence ratios were Hindi 7.36x, Kannada 13.57x, and Tamil 15.50x vs English. XLM-RoBERTa sentence ratios were Hindi 1.26x, Kannada 1.37x, and Tamil 1.37x. GPT-2 to XLM-R token reductions by sentence were 5.17x Hindi, 8.73x Kannada, and 9.97x Tamil.

**Revision:** The routing memo now recommends against the 6x Hindi cost assumption and recommends a multilingual/Indic-aware tokenizer/model route validated on production traffic.

**Evidence files:** `partA/results/v0_fertility_gpt2.txt`, `partA/results/audit_fertility_output.txt`, `partA/results/corrected_analysis_output.txt`, `partA/corrected_benchmark.csv`, and `partA/relative_ratios.csv`.

**Command:** `python your-submission/partA/corrected_analysis.py`

## 7. Capacity Reconciliation

**Hypothesis:** The serving anomaly is memory/scheduler related, not "longer prompts improve throughput."

**Experiment:** Derived KV cache size from model spec and compared with long-context rows in `bench_log.csv`.

**Result:** Exact KV cache is `28 * 2 * 8 * 128 * 2 = 114,688 bytes/token`. One 4096-token sequence needs 448 MiB. Ideal spec-only capacity is about 29 sequences, while the log shows effective capacity around 25-26 long sequences: batch 24 has no preemptions at 0.93 cache use, batch 32 preempts 7, and batch 48 preempts 23.

**Revision:** Explain the gap as allocation/watermark overhead not included in the simple spec. Recommend capping active long-context concurrency at 24 per L4 replica.

**Evidence file:** `partB/calc_capacity_output.txt`, generated by `partB/calc_capacity.py`.

**Command:** `python your-submission/partB/calc_capacity.py`

## 8. Goodput Correction

**Hypothesis:** `REPORT_v0` misread `reported_tok_s`.

**Experiment:** Recomputed generated-token goodput for the batch-24, prompt-3584 row.

**Result:** Direct generated-token goodput is `24 * 512 / 61.16 = 200.92 output tok/s`. The same number comes from stripping prompt accounting out of the harness metric: `1607.4 * 512 / 4096 = 200.93 output tok/s`.

**Revision:** The Part B writeup now says long prompts inflate reported throughput because prompt prefill is counted; they do not imply batch 48 will deliver 3200 generated tok/s.

**Command:** `python your-submission/partB/calc_capacity.py`

## 9. Part C Decision Memo

**Hypothesis:** With reviewer coverage for only Hindi and Kannada, SFT or a rewriter is higher risk than prompt-only launch tuning.

**Experiment:** Estimated reviewer throughput and generation cost for prompt sweeps.

**Result:** Under the planning assumption of about 4 minutes per paired judgment, 20 reviewer hours at approximately 15 paired judgments/hour gives about 300 reviewed comparisons, enough to choose prompt variants but not enough to validate six-language SFT. Prompt sweeps are cheap on one A100; a rewriter adds a second decode pass to every request.

**Revision:** Recommend prompt engineering for launch review, with a Day 7 kill criterion and a narrow Hindi/Kannada LoRA/SFT fallback only if prompt-only fails.
