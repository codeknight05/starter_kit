#!/usr/bin/env python3
"""
corrected_analysis.py — Corrected Cross-Language Tokenizer Analysis (A3).
Evaluates 3 tokenizers (GPT-2, XLM-RoBERTa-base, Qwen2.5-7B) across selected
languages using 4 denominators (whitespace word, grapheme cluster,
UTF-8 byte, and parallel sentence).
"""

import os
import sys
import unicodedata
import argparse
import tiktoken
import grapheme
import pandas as pd
from transformers import AutoTokenizer

EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "eval_corpus"))

def load_eval_corpus(langs):
    corpus = {}
    for lang in langs:
        p = os.path.join(EVAL_DIR, f"{lang}_eval.txt")
        with open(p, "r", encoding="utf-8") as f:
            lines = [unicodedata.normalize("NFC", line.strip()) for line in f if line.strip()]
            corpus[lang] = lines
    return corpus

def run_corrected_benchmark(sample_size=250, langs=None, tokenizer_names=None):
    langs = langs or ["eng", "hin", "kan", "tam"]
    tokenizer_names = tokenizer_names or ["gpt2", "xlmr", "qwen"]
    corpus = load_eval_corpus(langs)
    min_sents = min(sample_size, min(len(corpus[l]) for l in corpus))
    print(f"Loaded eval corpus: {min_sents} parallel sentences per language.\n")

    tokenizer_map = {
        "gpt2": "GPT-2 (tiktoken)",
        "xlmr": "XLM-RoBERTa-base",
        "qwen": "Qwen2.5-7B",
    }
    unknown = sorted(set(tokenizer_names) - set(tokenizer_map))
    if unknown:
        raise ValueError(f"Unsupported tokenizers: {', '.join(unknown)}")

    print("Loading tokenizers...")
    tokenizers = {}
    if "gpt2" in tokenizer_names:
        tokenizers["GPT-2 (tiktoken)"] = tiktoken.get_encoding("gpt2").encode
    if "xlmr" in tokenizer_names:
        xlmr_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
        tokenizers["XLM-RoBERTa-base"] = lambda s: xlmr_tok.encode(s, add_special_tokens=False)
    if "qwen" in tokenizer_names:
        qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        tokenizers["Qwen2.5-7B"] = lambda s: qwen_tok.encode(s, add_special_tokens=False)

    results = []

    for tok_name, encode in tokenizers.items():
        print(f"Benchmarking {tok_name}...")
        for lang in langs:
            lines = corpus[lang][:min_sents]
            
            tot_tokens = 0
            tot_words = 0
            tot_graphemes = 0
            tot_bytes = 0
            
            for line in lines:
                toks = len(encode(line))
                words = len(line.split())
                graphs = len(list(grapheme.graphemes(line)))
                utf8_bytes = len(line.encode("utf-8"))
                
                tot_tokens += toks
                tot_words += words
                tot_graphemes += graphs
                tot_bytes += utf8_bytes

            tok_per_sent = tot_tokens / min_sents
            tok_per_word = tot_tokens / tot_words
            tok_per_graph = tot_tokens / tot_graphemes
            tok_per_byte = tot_tokens / tot_bytes

            results.append({
                "Tokenizer": tok_name,
                "Language": lang,
                "Total Tokens": tot_tokens,
                "Tok / Sentence": tok_per_sent,
                "Tok / Word": tok_per_word,
                "Tok / Grapheme": tok_per_graph,
                "Tok / Byte": tok_per_byte,
            })

    df = pd.DataFrame(results)
    print("\n================================ FULL BENCHMARK MATRIX ================================")
    print(df.to_string(index=False))

    # Compute Relative Ratio Tables against English
    print("\n================ RELATIVE COST RATIOS (NORMALIZED TO ENGLISH = 1.00) ================")
    ratio_rows = []
    for tok_name in tokenizers:
        eng_row = df[(df["Tokenizer"] == tok_name) & (df["Language"] == "eng")].iloc[0]
        for lang in langs:
            row = df[(df["Tokenizer"] == tok_name) & (df["Language"] == lang)].iloc[0]
            ratio_rows.append({
                "Tokenizer": tok_name,
                "Language": lang,
                "Sent Ratio (Cost)": row["Tok / Sentence"] / eng_row["Tok / Sentence"],
                "Word Ratio": row["Tok / Word"] / eng_row["Tok / Word"],
                "Grapheme Ratio": row["Tok / Grapheme"] / eng_row["Tok / Grapheme"],
                "Byte Ratio": row["Tok / Byte"] / eng_row["Tok / Byte"],
            })
    df_ratios = pd.DataFrame(ratio_rows)
    print(df_ratios.to_string(index=False))

    # Save outputs to CSV for reporting
    df.to_csv(os.path.join(os.path.dirname(__file__), "corrected_benchmark.csv"), index=False)
    df_ratios.to_csv(os.path.join(os.path.dirname(__file__), "relative_ratios.csv"), index=False)
    print(f"\nSaved results to corrected_benchmark.csv and relative_ratios.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--langs", default="eng,hin,kan,tam")
    parser.add_argument("--tokenizers", default="gpt2,xlmr,qwen")
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    tokenizer_names = [name.strip() for name in args.tokenizers.split(",") if name.strip()]
    valid_tokenizers = {"gpt2", "xlmr", "qwen"}
    unknown = sorted(set(tokenizer_names) - valid_tokenizers)
    if unknown:
        parser.error(f"unsupported tokenizers: {', '.join(unknown)}")
    if "eng" not in langs:
        parser.error("--langs must include eng for relative ratios")
    run_corrected_benchmark(args.sample_size, langs, tokenizer_names)
