# Part A4: Tokenizer Routing Memo

**Date:** September 6, 2026  
**Source:** 250 multi-way parallel FLORES-200 devtest sentences from public mirror `yash9439/flores200`; NFC normalization and whitespace cleanup only. See `CORPUS.md` and `results/` for corpus metadata and captured command output.

`REPORT_v0.md` is not safe for routing or capacity planning. It used a ~10-line toy sample, GPT-2 only, and `tokens / whitespace word` as the main cross-language denominator. On the corrected corpus, GPT-2 still makes Indic text look expensive, but that is mostly tokenizer vocabulary mismatch, not an inherent property of Hindi or Dravidian scripts.

| Language | GPT-2 tok/sent | Qwen2.5-7B tok/sent | XLM-R tok/sent | XLM-R cost vs English |
|---|---:|---:|---:|---:|
| English | 27.0 | 27.9 | 30.5 | 1.00x |
| Hindi | 198.5 | 120.5 | 38.4 | 1.26x |
| Kannada | 366.1 | 191.1 | 41.9 | 1.37x |
| Tamil | 418.2 | 168.6 | 41.9 | 1.37x |

The single number I would use for routing/cost decisions is **tokens per comparable semantic request payload**, operationalized here as tokens per parallel sentence. Parallel sentences approximately hold meaning constant; whitespace words, graphemes, and UTF-8 bytes do not. On the same FLORES rows, GPT-2 makes Kannada look **17.5x** worse by `tok/word`, but **13.6x** worse by `tok/sentence`; the word denominator produces a 29.1% higher apparent ratio. In production, use tokens per actual user-turn distribution rather than treating every sentence as equivalent.

**Recommendation:** do not budget Hindi at 6x English serving cost from `REPORT_v0`. For Indic traffic, evaluate a multilingual or Indic-aware model family before making routing decisions. XLM-R produces 5.17x fewer Hindi tokens per aligned sentence than GPT-2 here and keeps Hindi/Kannada/Tamil within 1.26x-1.37x of English; this is tokenizer evidence, not proof of equivalent end-to-end serving speed.

**Biggest caveat:** FLORES is formal translated text, not production chat. It will under-sample code-mixing, slang, spelling variation, transliteration, and messy user turns. I would treat these numbers as a clean lower-variance audit, then validate on sampled production prompts before committing capacity.

**Production monitor:** `avg_input_plus_output_tokens_per_turn` by detected language and route, compared to English. The **1.75x** threshold is an operational alert chosen for launch monitoring, not a universal scientific boundary; alert if it is exceeded for a full day after moving to the multilingual route.
