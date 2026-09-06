# AI Usage Summary

AI assistance was used to inspect the starter kit, draft Python benchmark scripts, improve memo structure, and catch inconsistencies between measured outputs and written claims.

## Where AI Helped

- Built the repeatable corpus pipeline in `partA/build_corpus.py`, including NFC normalization, whitespace cleanup, FLORES-200 column selection, and an OPUS-100 fallback.
- Wrote the tokenizer benchmark harness in `partA/corrected_analysis.py` for GPT-2, XLM-RoBERTa-base, and Qwen2.5-7B across sentence, word, grapheme, and byte denominators.
- Structured the Part B KV-cache arithmetic and goodput derivations so every capacity claim has a reproducible calculation.
- Found stale memo numbers after rerunning the scripts and updated the written deliverables to match the generated CSVs.
- Added CLI controls for sample size, languages, and tokenizer selection; verified the reduced GPT-2 path and restored the canonical 250-row artifacts afterward.

## Where AI Misled Or Needed Correction

- It initially treated the OPUS-100 fallback as if it were a fully shared multi-way parallel corpus. That was too strong. The corpus builder was revised to use a public FLORES-200 mirror for true multi-way rows, with OPUS-100 only as a fallback.
- It initially framed `line.lower()` as an exact no-op for Indic text. The rerun showed small nonzero token deltas from embedded Latin text: +0.012% Hindi, +0.012% Kannada, and +0.013% Tamil on the FLORES sample. The conclusion was corrected to "tiny effect, not a driver," not "exact zero."
- It initially overclaimed that the spec-only KV capacity estimate "perfectly" matched the log. The corrected Part B writeup now separates ideal capacity (~29 sequences) from effective observed capacity (~25-26 sequences), and explains the likely scheduler/block-allocation gap.
- It initially described the two-replica `401.84 output tok/s` figure as a predicted effect. That number is now explicitly labeled a first-order projection, not a measured two-replica benchmark.

## Self-Verification

Commands rerun after corrections:

```bash
python your-submission/partA/build_corpus.py
python your-submission/partA/audit_fertility.py
python your-submission/partA/corrected_analysis.py
python your-submission/partB/calc_capacity.py
```

The generated files `partA/corrected_benchmark.csv` and `partA/relative_ratios.csv` are the source of the final Part A memo numbers.

I independently checked the corpus row counts, alignment, NFC normalization, empty-line absence, and duplicate counts with a Python validation command. Model tokenizer runs also require network access or a populated Hugging Face cache on first use.
