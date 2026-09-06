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

250 rows were selected because they are large enough to expose stable tokenizer and denominator differences for this 10-hour audit while keeping reruns and live-defense experiments practical. This is sufficient for a reproducible audit baseline, not for production capacity commitments or broad language-quality claims.

## Preprocessing

`build_corpus.py` applies:

- Unicode NFC normalization.
- Leading/trailing whitespace stripping.
- Internal whitespace collapse via `" ".join(text.split())`.
- A minimum cleaned length filter of 10 characters per language row.

No lowercasing, punctuation stripping, transliteration, language filtering, or deduplication is applied after source selection.

## Caveats

FLORES is formal translated text from a limited general-domain sample. It does not reliably represent conversational language, code-switching, transliteration-heavy inputs, slang, spelling noise, or the production prompt distribution. It is strong for comparing tokenizer counts on aligned semantic content, but it is not a production chat eval. Validate any tokenizer route and capacity commitment on sampled production traffic first.

## Rebuild Command

From `starter_kit/`:

```bash
python your-submission/partA/build_corpus.py
```

The checked-in corpus was produced with the default command above. If the FLORES mirror cannot be fetched, the script reports the failure and uses OPUS-100 pairwise English-target data as a fallback; that fallback is not a shared multi-way corpus and should not be described as equivalent evidence.
