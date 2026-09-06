#!/usr/bin/env python3
"""
build_corpus.py - Assembles multilingual eval set (A1).

Primary source: public consolidated FLORES-200 mirror `yash9439/flores200`,
devtest split. FLORES rows are multi-way parallel across languages.

Fallback source: Helsinki-NLP/opus-100 English-target test pairs. This keeps the
script runnable if the public FLORES mirror is unavailable, but the fallback is
pairwise parallel rather than a shared multi-way corpus.
"""

import argparse
import os
import unicodedata

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eval_corpus")

LANGS = {
    "eng": "eng_Latn",
    "hin": "hin_Deva",
    "kan": "kan_Knda",
    "tam": "tam_Taml",
    "tel": "tel_Telu",
}
DEFAULT_LANGS = ["eng", "hin", "kan", "tam"]


def clean_text(text):
    if not text:
        return ""
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text


def write_corpus(corpus, langs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    min_len = min(len(corpus[l]) for l in langs)
    print(f"\nSuccessfully assembled eval corpus with {min_len} lines per language.")

    for lang_code in langs:
        file_path = os.path.join(OUTPUT_DIR, f"{lang_code}_eval.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            for line in corpus[lang_code][:min_len]:
                f.write(line + "\n")
        print(f"Saved {min_len} lines to {file_path}")


def build_from_flores(sample_size, langs):
    print("Fetching multi-way parallel FLORES-200 devtest rows...")
    ds = load_dataset("yash9439/flores200", split="devtest")

    corpus = {lang: [] for lang in langs}
    for row in ds:
        cleaned = {lang: clean_text(row[LANGS[lang]]) for lang in langs}
        if all(len(text) > 10 for text in cleaned.values()):
            for lang, text in cleaned.items():
                corpus[lang].append(text)
            if len(corpus[langs[0]]) >= sample_size:
                break

    if min(len(lines) for lines in corpus.values()) < sample_size:
        raise RuntimeError("FLORES source did not yield enough complete rows.")

    return corpus


def build_from_opus100(sample_size, langs):
    print("Falling back to Helsinki-NLP/opus-100 pairwise English-target test sets...")
    configs = [
        ("en-hi", "hi", "hin"),
        ("en-kn", "kn", "kan"),
        ("en-ta", "ta", "tam"),
        ("en-te", "te", "tel"),
    ]

    corpus = {lang: [] for lang in langs}
    eng_sets = {}

    selected_configs = [config for config in configs if config[2] in langs]
    if not selected_configs:
        raise RuntimeError("OPUS-100 fallback requires at least one supported target language.")

    for pair, target_key, lang_code in selected_configs:
        print(f"Downloading {pair} test set from Helsinki-NLP/opus-100...")
        path = hf_hub_download(
            repo_id="Helsinki-NLP/opus-100",
            filename=f"{pair}/test-00000-of-00001.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(path)

        target_lines = []
        eng_lines = []
        for row in df["translation"]:
            en_txt = clean_text(row.get("en", ""))
            tgt_txt = clean_text(row.get(target_key, ""))
            if len(en_txt) > 10 and len(tgt_txt) > 10:
                eng_lines.append(en_txt)
                target_lines.append(tgt_txt)
                if len(target_lines) >= sample_size:
                    break
        corpus[lang_code] = target_lines
        eng_sets[lang_code] = eng_lines

    corpus["eng"] = eng_sets[selected_configs[0][2]][:sample_size]
    return corpus


def build_eval_corpus(sample_size=250, langs=None):
    langs = langs or list(LANGS)
    unknown = sorted(set(langs) - set(LANGS))
    if unknown:
        raise ValueError(f"Unsupported languages: {', '.join(unknown)}")
    if len(langs) < 1:
        raise ValueError("At least one language is required.")
    try:
        corpus = build_from_flores(sample_size, langs)
    except Exception as exc:
        print(f"FLORES fetch failed: {exc}")
        corpus = build_from_opus100(sample_size, langs)

    write_corpus(corpus, langs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--langs", default=",".join(DEFAULT_LANGS))
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    build_eval_corpus(
        sample_size=args.sample_size,
        langs=[lang.strip() for lang in args.langs.split(",") if lang.strip()],
    )
