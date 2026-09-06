#!/usr/bin/env python3
"""
audit_fertility.py — Empirical audit of fertility.py (A2).
Applies the Evidence Rule:
For every claimed flaw, isolates it, measures before/after numbers,
and states the exact direction and magnitude of distortion.
"""

import os
import sys
import unicodedata
import argparse
import tiktoken
import grapheme
from transformers import AutoTokenizer

def load_corpora(langs, sample_size):
    sample_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "corpus_sample"))
    eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "eval_corpus"))
    
    samples = {}
    for lang in [lang for lang in langs if lang in ["eng", "hin"]]:
        p = os.path.join(sample_dir, f"{lang}_sample.txt")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                samples[lang] = [unicodedata.normalize("NFC", line.strip()) for line in f if line.strip()][:sample_size]

    evals = {}
    for lang in langs:
        p = os.path.join(eval_dir, f"{lang}_eval.txt")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                evals[lang] = [unicodedata.normalize("NFC", line.strip()) for line in f if line.strip()][:sample_size]

    return samples, evals

def exp1_whitespace_split(samples, enc):
    """Flaw 1 (Code Bug): line.split(' ') vs line.split() on multiple spaces."""
    print("=== EXPERIMENT 1: split(' ') vs split() ===")
    print("Hypothesis: repeated spaces create empty word entries and deflate fertility.")
    print("Command: python your-submission/partA/audit_fertility.py")
    print("Input: corpus_sample/eng_sample.txt, line 7 and the full 10-line sample.")
    lines = samples["eng"]
    
    # Isolate sentence with double space (Line 7)
    line7 = lines[6]
    toks = len(enc.encode(line7.lower()))
    words_naive = len(line7.lower().split(" "))
    words_clean = len(line7.lower().split())
    
    fert_naive = toks / words_naive
    fert_clean = toks / words_clean
    
    print(f"Target Line: '{line7}'")
    print(f"  Tokens: {toks}")
    print(f"  Naive line.split(' '): {words_naive} words -> Fertility = {fert_naive:.4f}")
    print(f"  Clean line.split():    {words_clean} words -> Fertility = {fert_clean:.4f}")
    print(f"  Absolute delta: {fert_clean - fert_naive:+.4f}")
    print(f"  Relative distortion: {(1 - fert_naive/fert_clean)*100:.2f}% deflation")

    # Overall sample corpus impact
    tot_toks = sum(len(enc.encode(l.lower())) for l in lines)
    tot_words_naive = sum(len(l.lower().split(" ")) for l in lines)
    tot_words_clean = sum(len(l.lower().split()) for l in lines)
    print(f"English Sample Corpus Overall:")
    print(f"  Naive sum(split(' ')): {tot_words_naive} words -> Fertility = {tot_toks/tot_words_naive:.4f}")
    print(f"  Clean sum(split()):    {tot_words_clean} words -> Fertility = {tot_toks/tot_words_clean:.4f}\n")
    print(f"  Corpus-level relative distortion: {(1 - (tot_toks/tot_words_naive)/(tot_toks/tot_words_clean))*100:.2f}% deflation")
    print("Interpretation: the implementation bug is real, but its measured corpus-level effect is small on this sample.\n")

def exp2_macro_vs_micro(samples, evals, enc):
    """Flaw 2 (Code Bug): Macro-average sum(r_i)/N vs Micro-average sum(toks)/sum(words)."""
    print("=== EXPERIMENT 2: Macro-average vs micro-average ===")
    print("Hypothesis: the two aggregation rules produce different corpus statistics when sentences have different word counts.")
    print("Command: python your-submission/partA/audit_fertility.py")
    print("Input: corpus_sample/* and partA/eval_corpus/*; GPT-2; clean split().")
    
    eval_size = len(next(iter(evals.values()))) if evals else 0
    for name, corp in [("Sample Corpus", samples), (f"Eval Corpus ({eval_size} lines)", evals)]:
        print(f"--- {name} ---")
        for lang, lines in corp.items():
            per_line_ratios = []
            tot_toks = 0
            tot_words = 0
            for l in lines:
                toks = len(enc.encode(l.lower()))
                words = len(l.lower().split())
                per_line_ratios.append(toks / words if words > 0 else 0)
                tot_toks += toks
                tot_words += words
            
            macro_avg = sum(per_line_ratios) / len(per_line_ratios)
            micro_avg = tot_toks / tot_words
            delta = macro_avg - micro_avg
            pct_diff = (delta / micro_avg) * 100
            
            print(f"  [{lang}] Macro-Avg: {macro_avg:.4f} | Micro-Avg: {micro_avg:.4f} | Delta: {delta:+.4f} ({pct_diff:+.2f}%)")
        print()
    print("Interpretation: macro gives every sentence equal weight; micro gives aggregate tokens/words equal weight. Micro is the relevant choice for corpus-level token cost, but macro is not universally invalid.\n")

def exp3_conceptual_word_unit(evals, enc):
    """Flaw 3 (Conceptual Flaw): Word as cross-lingual unit (Agglutination & Morphology)."""
    print("=== EXPERIMENT 3: Whitespace words as a cross-language denominator ===")
    print("Hypothesis: whitespace-word counts do not hold comparable semantic payload constant across aligned languages.")
    print("Command: python your-submission/partA/audit_fertility.py")
    print("Input: 250 aligned FLORES rows; GPT-2; Kannada compared with English.")
    lines_eng = evals["eng"]
    lines_kan = evals["kan"]
    
    tot_toks_eng = sum(len(enc.encode(l)) for l in lines_eng)
    tot_words_eng = sum(len(l.split()) for l in lines_eng)
    
    tot_toks_kan = sum(len(enc.encode(l)) for l in lines_kan)
    tot_words_kan = sum(len(l.split()) for l in lines_kan)
    
    fert_eng = tot_toks_eng / tot_words_eng
    fert_kan = tot_toks_kan / tot_words_kan
    
    sent_ratio = tot_toks_kan / tot_toks_eng
    word_ratio = fert_kan / fert_eng
    words_per_sent_kan = tot_words_kan / len(lines_kan)
    words_per_sent_eng = tot_words_eng / len(lines_eng)
    
    print(f"English Eval Set:   {tot_words_eng} total words ({words_per_sent_eng:.1f} words/sent) | Tok/Word: {fert_eng:.2f}")
    print(f"Kannada Eval Set:   {tot_words_kan} total words ({words_per_sent_kan:.1f} words/sent) | Tok/Word: {fert_kan:.2f}")
    print(f"Tok/Word Ratio (Kan/Eng): {word_ratio:.2f}x worse")
    print(f"Tok/Sentence Ratio (Kan/Eng): {sent_ratio:.2f}x worse")
    distortion = (word_ratio / sent_ratio - 1) * 100
    print(f"Absolute ratio difference: {word_ratio - sent_ratio:+.2f}x")
    print(f"Relative distortion: {distortion:.1f}% higher by tok/word")
    print("Interpretation: parallel sentences approximately hold meaning constant, while whitespace words, graphemes, and bytes do not. Sentence tokens are a useful first-order aligned comparison, not a universal production denominator.\n")

def exp4_script_vs_tokenizer(evals):
    """Flaw 4 (Conceptual Flaw): Character count vs Graphemes & Script vs Tokenizer Vocab."""
    print("=== EXPERIMENT 4: Tokenizer vocabulary effect ===")
    print("Hypothesis: token count inflation depends strongly on tokenizer/model vocabulary, not only on script.")
    print("Command: python your-submission/partA/audit_fertility.py")
    print("Input: 250 aligned FLORES rows; GPT-2 versus XLM-R; grapheme denominator.")
    gpt2_enc = tiktoken.get_encoding("gpt2")
    xlmr_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    langs = ["eng", "hin", "kan", "tam"]
    print(f"{'lang':<6}{'GPT2 Tok/Graph':>16}{'XLM-R Tok/Graph':>18}{'XLM-R/GPT2':>14}{'Reduction':>13}")
    print("-" * 77)
    
    for lang in langs:
        lines = evals[lang]
        tot_graphs = sum(len(list(grapheme.graphemes(l))) for l in lines)
        
        gpt2_toks = sum(len(gpt2_enc.encode(l)) for l in lines)
        xlmr_toks = sum(len(xlmr_tok.encode(l, add_special_tokens=False)) for l in lines)
        
        gpt2_tpg = gpt2_toks / tot_graphs
        xlmr_tpg = xlmr_toks / tot_graphs
        ratio = gpt2_tpg / xlmr_tpg
        reduction = (1 - xlmr_tpg / gpt2_tpg) * 100
        
        print(f"{lang:<6}{gpt2_tpg:>16.3f}{xlmr_tpg:>18.3f}{xlmr_tpg/gpt2_tpg:>14.2f}x{reduction:>12.1f}%")
    print("\nInterpretation: XLM-R produces fewer tokens per aligned sentence on this corpus, demonstrating tokenizer/model-vocabulary dependence. This does not by itself prove an equal end-to-end serving-cost or latency improvement.\n")

def exp5_harmless_features(samples, evals, enc):
    """Audit Harmless Features: line.lower() on Indic text & NFC normalization."""
    print("=== EXPERIMENT 5: Suspicious feature - lower() ===")
    print("Hypothesis: lower() might materially alter Indic text or token counts.")
    print("Command: python your-submission/partA/audit_fertility.py")
    print("Input: 250 Hindi, Kannada, and Tamil FLORES rows; GPT-2 before/after lower().")
    
    for lang in ["hin", "kan", "tam"]:
        lines = evals[lang]
        diff_count = 0
        tok_diff = 0
        token_delta = 0
        original_tokens = 0
        for l in lines:
            if l.lower() != l:
                diff_count += 1
            t1 = len(enc.encode(l))
            t2 = len(enc.encode(l.lower()))
            original_tokens += t1
            token_delta += (t2 - t1)
            if t1 != t2:
                tok_diff += 1
                
        pct_delta = (token_delta / original_tokens) * 100 if original_tokens else 0
        print(
            f"[{lang}] Total Sentences: {len(lines)} | String lower() diffs: {diff_count} | "
            f"Original tokens: {original_tokens} | Lowered tokens: {original_tokens + token_delta} | "
            f"Token count diffs: {tok_diff} | Net token delta: {token_delta:+d} ({pct_delta:+.3f}%)"
        )
    print(
        "Result: lower() is not an Indic Unicode-corruption bug. Its measured effect is tiny "
        "and comes from caseful embedded Latin text, so it is not a driver of the v0 cross-language claim.\n"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--langs", default="eng,hin,kan,tam")
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    required = {"eng", "hin", "kan", "tam"}
    if set(langs) != required:
        parser.error("--langs must include exactly eng,hin,kan,tam for the full audit")
    samples, evals = load_corpora(langs, args.sample_size)
    gpt2_enc = tiktoken.get_encoding("gpt2")
    
    exp1_whitespace_split(samples, gpt2_enc)
    exp2_macro_vs_micro(samples, evals, gpt2_enc)
    exp3_conceptual_word_unit(evals, gpt2_enc)
    exp4_script_vs_tokenizer(evals)
    exp5_harmless_features(samples, evals, gpt2_enc)

if __name__ == "__main__":
    main()
