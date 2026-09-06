# Part C: Indic Casual Tone Decision Memo

**Recommendation:** choose **(c) prompt engineering only** for the 3-week launch review. Do not run SFT yet, and do not add a rewriter model yet.

## Assumptions

We have one A100-80GB for 14 days, one native-speaker reviewer for Hindi and Kannada only, 10 h/week for 2 weeks before launch review, and no external API budget. Target languages are Hindi, Kannada, Tamil, Telugu, Bengali, and Marathi. The launch bar is improved casualness without meaning drift or grammar regressions.

## Back-Of-Envelope Arithmetic

Reviewer budget: `10 h/week * 2 weeks = 20 h`. At about 15 paired judgments/hour, the reviewer can score `20 * 15 = 300` comparisons total, roughly 150 Hindi and 150 Kannada. That is enough to choose among prompt variants, but not enough to validate a synthetic SFT set across six languages.

Prompt sweep: `6 languages * 50 prompts * 6 prompt variants = 1,800 generations`. Even at a conservative 100 output tok/s and 200 output tokens each, this is about `360,000 / 100 = 3,600 s`, or 1 GPU-hour plus overhead. Serving cost for the chosen prompt is only the added prefill text, roughly 30-80 tokens/request. A rewriter adds a second decode pass to every request, and SFT creates a larger validation burden than our reviewer budget can support.

## Success Metric

Blind pairwise human evaluation on Hindi and Kannada: prompt variant beats baseline formal style in at least **70%** of reviewed pairs, with fluency at least **4.5/5** and semantic-error rate under **2%**. For Tamil, Telugu, Bengali, and Marathi, use automated filters only as a guardrail: response length change under **25%** and multilingual embedding similarity at least **0.85** versus the baseline answer.

## Kill Criterion

Kill prompt-only by **end of Day 7** if Hindi/Kannada win rate is below **65%**, if semantic-error rate is above **2%**, or if average response length grows by more than **30%**. The fallback is a narrow Hindi/Kannada LoRA/SFT experiment only, because those are the languages we can actually review.

## Day 1 Experiment

Create 50 realistic user prompts covering support, chit-chat, factual Q&A, refusal, and code-mixed requests. Generate baseline answers plus six system-prompt variants for all six languages. Send the Hindi/Kannada pairs to the reviewer immediately, and use the first 50 judgments to prune to the best two variants before expanding the review set.
