# AI Team Intern Assignment — The Audit

A reproducible audit of multilingual tokenization, L4 inference capacity, long-context throughput, and a practical launch decision for a multilingual rewriter.

The repository contains the original starter kit plus the completed submission. **All completed work is under [`your-submission/`](your-submission/).**

## Executive summary

This submission audits three connected questions:

1. **Part A — Multilingual tokenization:** Does a simple fertility-style metric accurately describe tokenization cost across English, Hindi, Kannada, and Tamil? The audit compares multiple tokenizers and multiple denominators on aligned multilingual text.
2. **Part B — Inference capacity:** How should an L4 serving report be interpreted when prompt-heavy workloads, KV-cache pressure, and preemption interact? The analysis derives theoretical KV capacity, reconciles reported throughput with generated-token goodput, and identifies a practical batch limit.
3. **Part C — Product decision:** Under one A100-80GB, one Hindi+Kannada reviewer at 10 h/week, a 3-week launch window, and no external API, should the team use prompting, a small rewriter model, or SFT? The decision memo recommends a prompt-first launch experiment with explicit success and kill criteria.

The central principle throughout the submission is **evidence before conclusion**: measured results, calculations, assumptions, projections, and production recommendations are kept separate so that a reviewer can reproduce the claims rather than relying on unsupported narrative.

## Repository structure

```text
starter_kit/
├── fertility.py                         # Original starter-kit baseline
├── corpus_sample/                       # Original small sample used by the baseline
├── your-submission/
│   ├── NOTEBOOK.md                      # Submission roadmap and experiment record
│   ├── AI_USAGE.md                      # AI-assisted workflow and verification log
│   ├── DEFENSE_PREP.md                  # Live-defense questions, calculations, and counterfactuals
│   ├── PRODUCTION_VALIDATION.md         # What the benchmark does and does not prove operationally
│   ├── CORPUS.md                        # Corpus provenance and construction details
│   ├── requirements.txt                 # Python dependencies
│   ├── partA/
│   │   ├── build_corpus.py              # Builds the multilingual audit corpus
│   │   ├── audit_fertility.py           # Audits the original fertility implementation/metric
│   │   ├── corrected_analysis.py        # Multi-tokenizer, multi-denominator comparison
│   │   ├── memo.md                      # Part A conclusions and recommendation
│   │   └── results/                     # Captured Part A evidence/results
│   ├── partB/
│   │   ├── calc_capacity.py             # KV-cache capacity calculations
│   │   ├── b1_b4_analysis.md            # Part B analysis and evidence
│   │   └── calc_capacity_output.txt     # Captured calculation output
│   └── partC/
│       └── memo.md                      # Part C launch decision memo
└── README.md
```

## Submission map

| Requirement | Where to look |
|---|---|
| A1 — real multilingual corpus | [`your-submission/CORPUS.md`](your-submission/CORPUS.md), [`your-submission/partA/build_corpus.py`](your-submission/partA/build_corpus.py) |
| A2 — audit of original fertility code/metric | [`your-submission/partA/audit_fertility.py`](your-submission/partA/audit_fertility.py), [`your-submission/partA/memo.md`](your-submission/partA/memo.md) |
| A3 — corrected multilingual comparison | [`your-submission/partA/corrected_analysis.py`](your-submission/partA/corrected_analysis.py), [`your-submission/partA/results/`](your-submission/partA/results/) |
| A4 — short memo | [`your-submission/partA/memo.md`](your-submission/partA/memo.md) |
| B1–B4 — serving analysis | [`your-submission/partB/b1_b4_analysis.md`](your-submission/partB/b1_b4_analysis.md), [`your-submission/partB/calc_capacity.py`](your-submission/partB/calc_capacity.py) |
| C — launch recommendation | [`your-submission/partC/memo.md`](your-submission/partC/memo.md) |
| Reproducibility | [`your-submission/NOTEBOOK.md`](your-submission/NOTEBOOK.md), commands below |
| AI disclosure | [`your-submission/AI_USAGE.md`](your-submission/AI_USAGE.md) |
| Production caveats | [`your-submission/PRODUCTION_VALIDATION.md`](your-submission/PRODUCTION_VALIDATION.md) |
| Defense preparation | [`your-submission/DEFENSE_PREP.md`](your-submission/DEFENSE_PREP.md) |

## Quick start

From the repository root:

```bash
python -m pip install -r your-submission/requirements.txt
```

Python 3.10+ is recommended.

The corrected Part A analysis uses Hugging Face tokenizer implementations for XLM-RoBERTa and Qwen2.5-7B. On first use, those tokenizer files may need to be downloaded or otherwise be present in the local Hugging Face cache. After the cache is populated, the analysis can be rerun without network access using `--offline`.

```bash
python your-submission/partA/corrected_analysis.py --offline
```

If the cache has not been populated, run once without `--offline` while network access is available:

```bash
python your-submission/partA/corrected_analysis.py
```

`--offline` means **cache-only**, not that the repository contains the model/tokenizer files themselves.

## Reproduce Part A

Build the checked-in multilingual corpus:

```bash
python your-submission/partA/build_corpus.py
```

Run the audit of the original fertility implementation:

```bash
python your-submission/partA/audit_fertility.py
```

Run the corrected comparison across the selected tokenizers and denominators:

```bash
python your-submission/partA/corrected_analysis.py
```

For a no-network rerun after the Hugging Face files have been cached:

```bash
python your-submission/partA/corrected_analysis.py --offline
```

The original starter-kit baseline is also preserved for direct comparison:

```bash
python fertility.py \
  --corpus eng=corpus_sample/eng_sample.txt \
  --corpus hin=corpus_sample/hin_sample.txt \
  --tokenizer gpt2
```

### Part A methodology

The corrected analysis evaluates English, Hindi, Kannada, and Tamil using aligned multilingual material rather than the original tiny English/Hindi toy sample. The benchmark reports token counts under several denominators, including whitespace words, graphemes, UTF-8 bytes, and aligned parallel sentences.

The recommended routing/cost denominator is **tokens per comparable semantic request payload**, operationalized here with tokens per aligned parallel sentence. The report explicitly treats that as a proxy: aligned sentences approximately hold message meaning constant, while whitespace-word counts, grapheme counts, and byte counts are measures of different physical properties.

The analysis includes three tokenizer families:

- GPT-2 tokenizer
- XLM-RoBERTa-base tokenizer
- Qwen2.5-7B tokenizer

The Part A memo also documents the distinction between **tokenization evidence** and **end-to-end serving behavior**. A lower token count does not by itself establish a proportional reduction in latency, throughput, or production cost because those outcomes also depend on the model, prefill/decode balance, batching, KV-cache behavior, serving stack, and workload distribution.

## Reproduce Part B

Run the capacity calculation:

```bash
python your-submission/partB/calc_capacity.py
```

The captured output is stored in:

[`your-submission/partB/calc_capacity_output.txt`](your-submission/partB/calc_capacity_output.txt)

The analysis covers:

- per-token KV-cache memory from the model configuration;
- theoretical maximum 4096-token sequence concurrency under the stated memory assumptions;
- observed operational behavior around batch 24, 32, and 48;
- the long-context throughput anomaly;
- generated-token goodput versus the report's prompt+generation token throughput;
- a production metric/counter to monitor for preemption and cache pressure.

Important terminology used in the report:

**Theoretical capacity** is a specification-based calculation. It is not an operational concurrency guarantee.

**Measured benchmark values** are rows actually observed in the supplied serving report/benchmark.

**Projected values** are derived from measured values under stated assumptions. For example, the two-replica 401.84 output-token/s figure is explicitly a first-order projection rather than a measured two-GPU benchmark.

## Reproduce Part C

The Part C memo is a one-page-style decision document:

[`your-submission/partC/memo.md`](your-submission/partC/memo.md)

Recommendation: **prompt engineering first**, rather than immediately training a rewriter model or performing SFT.

The memo makes its operating assumptions explicit, including:

- one A100-80GB;
- three-week launch horizon;
- no external API;
- one reviewer covering Hindi and Kannada for 10 h/week;
- an explicit reviewer throughput assumption derived from 4 minutes per paired judgment;
- an explicit generation-throughput assumption used only for first-order capacity arithmetic;
- a human-evaluation success threshold and kill criterion;
- a Day-1 experiment designed to reduce the search space quickly.

The memo treats these thresholds as **preselected decision criteria**, not as experimentally established truths.

## Key results at a glance

The main numbers worth understanding before reviewing the repository are:

| Finding | Result | Interpretation |
|---|---:|---|
| GPT-2 Kannada ratio, tokens/word | **17.5×** | High apparent cost under a whitespace-word denominator |
| GPT-2 Kannada ratio, tokens/parallel sentence | **13.6×** | Lower ratio when the denominator better approximates comparable payload |
| Denominator effect on Kannada ratio | **29.1%** | Shows that the denominator materially changes the apparent cross-language cost |
| XLM-R Hindi token reduction vs GPT-2 | **5.17× fewer tokens** | Tokenizer result only; not proof of equal serving-speed reduction |
| KV-cache footprint | **114,688 bytes/token = 112 KiB/token** | Derived from model/config assumptions |
| 4096-token KV footprint | **448 MiB/sequence** | Theoretical per-sequence KV allocation |
| Theoretical 4096-token concurrency | **29.2 sequences** | Memory-model estimate, not an operational guarantee |
| Batch-24 generated goodput | **200.92 output tok/s** | Derived directly from 24 × 512 / 61.16 s |
| Batch-24 adjusted goodput | **200.93 output tok/s** | Cross-check using reported token throughput |

These values are reproduced and explained in the checked-in analysis files rather than treated as standalone claims.

## Evidence and interpretation policy

This repository intentionally separates four categories of statements:

1. **Measured:** values directly observed from an experiment or supplied serving report.
2. **Calculated:** values obtained by deterministic arithmetic from stated inputs.
3. **Assumed:** values introduced for planning, especially in Part C.
4. **Projected/recommended:** forward-looking values or deployment guidance derived from measured/calculated inputs and explicit assumptions.

This distinction matters. For example, the recommendation to use `max_num_seqs=24` on an L4 is a deployment recommendation for the tested workload, not a universal statement about the optimal vLLM configuration. Likewise, a projected two-replica throughput figure is not described as a benchmark result.

## Production-validation boundary

The repository includes [`your-submission/PRODUCTION_VALIDATION.md`](your-submission/PRODUCTION_VALIDATION.md) to make the benchmark boundary explicit.

The checked-in evidence does **not** claim to prove:

- production latency for a live multilingual application;
- end-to-end cost savings from tokenizer choice alone;
- capacity on a different GPU, model, serving stack, or workload distribution;
- that a formal FLORES benchmark is identical to real production traffic;
- that a projected two-replica result is equivalent to a real multi-GPU benchmark.

The intended conclusion is narrower: the experiments identify what the current benchmark supports, expose where the original analysis overreaches, and specify what production validation should measure next.

## Desmos companion

Interactive visualization:

**https://www.desmos.com/calculator/vykqbj6s4o**

The Desmos graph visualizes the measured **English-normalized sentence ratios** for GPT-2, XLM-RoBERTa-base, and Qwen2.5-7B across English, Hindi, Kannada, and Tamil.

It is a companion visualization of the checked-in benchmark evidence, **not an additional serving benchmark**.

## Reproducibility notes

The repository is designed so that a reviewer can start at the root and follow the commands above into the submission directory. Generated/captured evidence is checked in where practical so that the written conclusions can be inspected without having to infer missing intermediate results.

For fully clean reruns, note the following:

- Hugging Face tokenizer downloads may require network access on first use.
- `--offline` is cache-only and therefore assumes those files already exist locally.
- The formal multilingual corpus used for the primary benchmark is documented in `CORPUS.md`.
- The production-serving conclusions are tied to the supplied benchmark configuration and should not be generalized beyond it without validation.

## Reading order for a reviewer

For a fast technical review, read in this order:

1. [`your-submission/NOTEBOOK.md`](your-submission/NOTEBOOK.md) — what was changed and where the evidence lives.
2. [`your-submission/partA/memo.md`](your-submission/partA/memo.md) — main multilingual audit conclusion.
3. [`your-submission/partB/b1_b4_analysis.md`](your-submission/partB/b1_b4_analysis.md) — capacity, throughput, and production metric analysis.
4. [`your-submission/partC/memo.md`](your-submission/partC/memo.md) — launch recommendation.
5. [`your-submission/DEFENSE_PREP.md`](your-submission/DEFENSE_PREP.md) — calculations and likely follow-up questions.
6. [`your-submission/PRODUCTION_VALIDATION.md`](your-submission/PRODUCTION_VALIDATION.md) — explicit limits of what the experiments establish.

For implementation details, inspect the Python scripts under `your-submission/partA/` and `your-submission/partB/` and compare their outputs with the checked-in evidence files.

## Submission

For the assignment submission form, use this repository as the primary deliverable:

**GitHub:** https://github.com/codeknight05/starter_kit

The completed audit is contained in [`your-submission/`](your-submission/).
