# AI Team Intern Assignment: The Audit

This repository contains the starter kit and completed submission for the audit assignment. Completed work is under `your-submission/`.

## Setup

From the repository root:

```bash
python -m pip install -r your-submission/requirements.txt
```

The corrected tokenizer analysis needs network access or a populated Hugging Face cache on first use for XLM-R and Qwen tokenizer files. Once cached, use `--offline` for deterministic no-network reruns:

```bash
python your-submission/partA/corrected_analysis.py --offline
```

## Reproduce

Part A corpus and audit:

```bash
python your-submission/partA/build_corpus.py
python your-submission/partA/audit_fertility.py
python your-submission/partA/corrected_analysis.py
```

Original baseline:

```bash
python fertility.py --corpus eng=corpus_sample/eng_sample.txt --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2
```

Part B:

```bash
python your-submission/partB/calc_capacity.py
```

The Part A memos are in `your-submission/partA/memo.md`, the Part B analysis is in `your-submission/partB/b1_b4_analysis.md`, and the Part C decision memo is in `your-submission/partC/memo.md`. Captured evidence outputs are in `your-submission/partA/results/` and `your-submission/partB/calc_capacity_output.txt`.

The boundary between tokenizer evidence and production serving validation is documented in [PRODUCTION_VALIDATION.md](your-submission/PRODUCTION_VALIDATION.md).
