# Final Submission Audit

This audit records the evidence currently present in the repository. `PASS` means the requirement is addressed by a checked-in script, memo, or captured output; it does not mean the result is a production guarantee.

| Rubric | Requirement | Evidence file/script | Status |
|---|---|---|---|
| A1 | Four-language aligned corpus including English, Hindi, and two Dravidian languages | `partA/CORPUS.md`; `partA/eval_corpus/`; `partA/build_corpus.py` | PASS |
| A1 | Source, split, language codes, size, preprocessing, domain, and limitations documented | `partA/CORPUS.md` | PASS |
| A2 | Code bug: repeated-space splitting measured before/after | `partA/audit_fertility.py`; `partA/results/audit_fertility_output.txt` | PASS |
| A2 | Macro versus micro aggregation measured on the evaluation corpus | `partA/audit_fertility.py`; captured audit output | PASS |
| A2 | Conceptual denominator problem measured on aligned data | `partA/audit_fertility.py`; `partA/memo.md` | PASS |
| A2 | Suspicious `lower()` behavior tested and shown immaterial | `partA/audit_fertility.py`; captured audit output | PASS |
| A2 | Tokenizer/model-vocabulary effect measured without claiming equivalent serving speed | `partA/audit_fertility.py`; `partA/memo.md` | PASS |
| A3 | At least two tokenizers and multilingual tokenizer included | `partA/corrected_analysis.py`; `corrected_benchmark.csv` | PASS |
| A3 | Sentence, word, grapheme, and byte denominators reported | `partA/corrected_analysis.py`; both CSV outputs | PASS |
| A3 | Special tokens excluded consistently for Hugging Face tokenizers | `partA/corrected_analysis.py` uses `add_special_tokens=False` | PASS |
| A3 | Tokenizer counts distinguished from latency/cost | `partA/corrected_analysis.py`; `partA/memo.md` | PASS |
| A4 | One-page routing memo with recommendation, caveat, and production monitor | `partA/memo.md` | PASS |
| B1 | Exact KV-cache arithmetic and theoretical sequence capacity | `partB/b1_b4_analysis.md`; `partB/calc_capacity.py` | PASS |
| B1 | Theoretical capacity distinguished from observed operational capacity | `partB/b1_b4_analysis.md` | PASS |
| B2 | Long-context anomaly, mechanism, rows, and deployment recommendation | `partB/b1_b4_analysis.md`; `bench/bench_log.csv` | PASS |
| B2 | Multi-replica result labeled as projection rather than measurement | `partB/b1_b4_analysis.md` | PASS |
| B3 | Two independent batch-24 goodput derivations | `partB/b1_b4_analysis.md`; `partB/calc_capacity.py` | PASS |
| B4 | Preemption and cache metrics proposed as mechanism checks | `partB/b1_b4_analysis.md` | PASS |
| C | Prompt-only recommendation addresses constraints and alternatives | `partC/memo.md` | PASS |
| C | Assumptions, data volume, reviewer throughput, serving estimate, and arithmetic documented | `partC/memo.md` | PASS |
| C | Human metrics separated from automated guardrails | `partC/memo.md` | PASS |
| C | Numeric success metric, Day 7 kill criterion, and Day 1 experiment | `partC/memo.md` | PASS |
| Reproducibility | Pinned dependencies and root-relative commands | `your-submission/requirements.txt`; `README.md` | PASS |
| Defense | Numbers, why-answers, counterfactuals, and live commands | `your-submission/DEFENSE_PREP.md` | PASS |
| Transparency | Chronological work log and AI disclosure preserved | `your-submission/NOTEBOOK.md`; `your-submission/AI_USAGE.md` | PASS |

## Remaining limitations

- The checked-in corpus is formal FLORES text, not sampled production traffic. The required production sampling and acceptance procedure is documented in `PRODUCTION_VALIDATION.md`; no production data was available in this environment.
- XLM-R and Qwen tokenizer counts do not establish end-to-end generation latency or serving cost. The validation document separates tokenizer-count, prefill, decode, and end-to-end measurements.
- The two-replica `401.84 output tok/s` figure is a first-order projection; no two-replica serving benchmark is included because the available environment has no two-GPU serving setup. The document specifies the benchmark needed to replace the projection.
- The OPUS-100 fallback is pairwise rather than shared multi-way data and is not equivalent to the checked-in FLORES evidence.
- The Part C success thresholds are preselected decision criteria, not measured outcomes.
- Hugging Face tokenizer downloads require network access or a local cache on first run; `corrected_analysis.py --offline` now supports cache-only reruns.
