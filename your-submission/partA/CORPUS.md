# Part A Corpus Metadata

## Source

Primary source: public Hugging Face mirror `yash9439/flores200`, split `devtest`.

The official `facebook/flores` dataset was not usable without authentication in this environment, so the builder uses the public consolidated mirror. `build_corpus.py` retains an OPUS-100 fallback for reproducibility if the mirror is unavailable, but the checked-in corpus was regenerated from FLORES-200.

## Languages

| Output file | FLORES column | Language |
|---|---|---|
| `eval_corpus/eng_eval.txt` | `eng_Latn` | English |
| `eval_corpus/hin_eval.txt` | `hin_Deva` | Hindi |
| `eval_corpus/kan_eval.txt` | `kan_Knda` | Kannada |
| `eval_corpus/tam_eval.txt` | `tam_Taml` | Tamil |

The builder also supports optional Telugu (`tel_Telu`) via `--langs eng,hin,tel`;
the checked-in evidence corpus remains the original four-language evaluation.

## Size And Domain

The corpus contains 250 multi-way parallel sentences per language from the FLORES-200 devtest split. FLORES is a professionally translated multilingual benchmark with formal, general-domain sentences drawn from sources such as news and Wikipedia-like text.

## Preprocessing

`build_corpus.py` applies:

- Unicode NFC normalization.
- Leading/trailing whitespace stripping.
- Internal whitespace collapse via `" ".join(text.split())`.
- A minimum cleaned length filter of 10 characters per language row.

No lowercasing, punctuation stripping, transliteration, language filtering, or deduplication is applied after source selection.

## Caveats

This corpus is strong for comparing tokenizer cost on aligned semantic content, but it is not a production chat eval. It under-represents code-mixing, slang, spelling variation, transliterated Indic text, short UI commands, and conversational assistant turns. Treat the results as a clean audit baseline, then validate the chosen tokenizer route on sampled production prompts before final capacity commitments.

## Rebuild Command

From `starter_kit/`:

```bash
python your-submission/partA/build_corpus.py
```
