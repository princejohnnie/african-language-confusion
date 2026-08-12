"""
Computes Line Pass Rate (LPR) and Word Pass Rate (WPR) for the African Language
Confusion sweep outputs.
The core algorithm (`compute_metrics`/`compute_all_metrics`) is unchanged; what's
different is the language-ID model.

Language-ID model choice -- GlotLID

Two replacements were evaluated:
  - GlotLID (cis-lmu/glotlid, fastText architecture, 2102 labels): tested against
    one real native-language prompt per language from this project's own
    prompts/monolingual/*/*.csv -- 18/18 correct, all >= 0.95 confidence.
  - AfroLID (UBC-NLP/afrolid_1.5, transformer/SERENGETI, 517 African languages):
    per its paper (Adebara et al. 2022), purpose-built for African languages and
    reports higher African-specific accuracy (F1 95.89 on its own test set,
    beating general LID tools on most of 16 shared African languages tested).
    GlotLID's own paper (Kargaran et al. 2023) explicitly excludes AfroLID from
    its baselines "despite its excellent coverage of African languages," citing
    efficiency (transformer vs fastText), not accuracy -- so AfroLID likely has
    a genuine accuracy edge here. It's not used because its HF checkpoint's
    tokenizer fails to load under the transformers/tokenizers versions installed
    for this project (`TypeError: argument 'vocab': 'dict' object cannot be
    converted to 'Sequence'`, reproduced with both fast and slow tokenizer
    paths). Revisit if that gets fixed upstream.
"""

import collections
import csv
import itertools
import os
import string
from typing import Iterable

import numpy as np
import requests

EN_WORDS_URL = "https://gist.githubusercontent.com/wchargin/8927565/raw/d9783627c731268fb2935a731a618aa8e95cf465/words"
EN_WORDS_PATH = "words"

# key -> GlotLID label (ISO 639-3 + script). Verified against the actual model's
# get_labels() output -- see module docstring.
LANGUAGE_TO_GLOTLID = {
    "acholi": "ach_Latn",
    "amharic": "amh_Ethi",
    "chichewa": "nya_Latn",
    "fongbe": "fon_Latn",
    "ga": "gaa_Latn",
    "hausa": "hau_Latn",
    "igbo": "ibo_Latn",
    "kinyarwanda": "kin_Latn",
    "lingala": "lin_Latn",
    "malagasy": "plt_Latn",
    "ndebele": "nbl_Latn",
    "sepedi": "nso_Latn",
    "shona": "sna_Latn",
    "swahili": "swh_Latn",
    "tigrinya": "tir_Ethi",
    "twi": "twi_Latn",
    "wolof": "wol_Latn",
    "yoruba": "yor_Latn",
}

_lid_model = None
_en_words = None


def _script(lang: str) -> str:
    return LANGUAGE_TO_GLOTLID[lang].split("_", 1)[1]


def _patch_numpy_for_fasttext() -> None:
    """
    fasttext-wheel's .predict() calls np.array(probs, copy=False), which numpy>=2.0
    raises on when a copy is actually needed (its copy=False semantics changed from
    numpy 1.x's "avoid copying if possible" to "error if a copy would be required").
    Patch np.array to fall back to a copying asarray in that one case, rather than
    pinning numpy down -- pinning would conflict with pandas 3.0.3's requirement.
    """
    if getattr(np, "_fasttext_copy_patch", False):
        return
    _orig_array = np.array

    def _patched_array(*args, **kwargs):
        if kwargs.get("copy") is False:
            kwargs.pop("copy")
            return np.asarray(*args, **kwargs)
        return _orig_array(*args, **kwargs)

    np.array = _patched_array
    np._fasttext_copy_patch = True


def _get_lid_model():
    global _lid_model
    if _lid_model is None:
        _patch_numpy_for_fasttext()
        
        import fasttext
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download("cis-lmu/glotlid", "model.bin")
        _lid_model = fasttext.load_model(model_path)
    return _lid_model


def _get_en_words() -> set:
    global _en_words
    if _en_words is None:
        if not os.path.exists(EN_WORDS_PATH):
            r = requests.get(EN_WORDS_URL, timeout=60)
            r.raise_for_status()
            with open(EN_WORDS_PATH, "w", encoding="utf-8") as f:
                f.write(r.text)
        with open(EN_WORDS_PATH, encoding="utf-8") as f:
            words = [line.strip() for line in f]
        _en_words = {w for w in words if w.islower() and len(w) > 3}
    return _en_words


def normalize(text: str) -> str:
    text = text.split("\nQ:")[0].strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.replace("—", " ")
    text = text.replace("،", "")
    return text


def tokenize(line: str) -> list[str]:
    return line.split()


def langid(line: str) -> str:
    model = _get_lid_model()
    labels, scores = model.predict(line)
    label = labels[0].removeprefix("__label__")
    score = float(scores[0])
    return label if score > 0.3 else "unknown"


def compute_metrics(completions: Iterable[str], lang: str) -> dict[str, float]:
    """
    Compute Line Pass Rate (LPR) and Word Pass Rate (WPR) over the given completions,
    whose expected language is given (one of our internal language keys, e.g. "igbo").
    WPR is only computed for non-Latin-script languages (here: amharic, tigrinya) --
    same rationale as the reference script's Latin-script restriction: an English word
    list lookup is unreliable for Latin-script African languages due to loanwords/
    cognates/short function-word overlap with English.
    """
    if lang not in LANGUAGE_TO_GLOTLID:
        raise ValueError(f"No GlotLID mapping for language {lang!r}; add it to LANGUAGE_TO_GLOTLID.")
    target_label = LANGUAGE_TO_GLOTLID[lang]
    wpr_eligible = _script(lang) != "Latn"
    en_words = _get_en_words()

    with_word_errors = 0
    with_line_errors = 0
    non_skipped = 0
    line_acc = []

    for completion in completions:
        print("Completion: ", completion)
        completion = normalize(completion)
        print("Normalized Completion: ", completion)
        lines = completion.split("\n")
        print("Lines: ", lines)
        line_tokens = [tokenize(line) for line in lines]
        print("line_tokens: ", line_tokens)
        # remove lines that are too short
        indices = [i for i, tokens in enumerate(line_tokens) if len(tokens) >= 5]
        lines = [lines[i] for i in indices]
        # print("Lines -> ", lines)
        line_tokens = [line_tokens[i] for i in indices]
        if lines:
            for a_line in lines:
                print("\nA Line ", a_line)
                print("Lang ID per line ->", langid(a_line))
            non_skipped += 1
            line_errors = sum(langid(line) != target_label for line in lines)
            if line_errors > 0:
                with_line_errors += 1
            elif wpr_eligible and any(token.strip() in en_words for tokens in line_tokens for token in tokens):
                with_word_errors += 1
            line_acc.append(1 - line_errors / len(lines))

    metrics = {}
    metrics["acc"] = sum(line_acc) / len(line_acc) if line_acc else 1.0
    metrics["lpr"] = 1 - with_line_errors / max(1, non_skipped)
    if wpr_eligible:
        metrics["wpr"] = 1 - with_word_errors / max(1, non_skipped - with_line_errors)
    return metrics


def compute_all_metrics(outputs: list[dict]) -> dict[tuple, dict[str, float]]:
    """
    Takes the crosslingual or monolingual outputs from a model and returns all the WPR
    and LPR metrics (WPR and LPR per dataset and averages per language and per source).
    The provided outputs should be dictionaries with 'source', 'language' and
    'completion' fields, for instance:

    ```
    outputs = [
        {'source': 'okapi', 'language': 'igbo', 'completion': 'Amaghị m'},
        {'source': 'okapi', 'language': 'igbo', 'completion': 'I do not know'},
    ]

    compute_all_metrics(outputs)
    {
        ('okapi', 'igbo'): {'lpr': 0.5},               # scores for Igbo Okapi
        ('okapi', 'all'): {'lpr': 0.5},                # averages over the Okapi source
        ('all', 'igbo'): {'lpr': 0.5},                 # averages over the Igbo language
        ('all', 'all'): {'lpr': 0.5},                  # overall average
    }
    ```
    """
    all_metrics = {}
    metrics_per_lang = collections.defaultdict(list)
    metrics_per_source = collections.defaultdict(list)

    group_key = lambda output: (output["source"], output["language"])
    outputs = sorted(outputs, key=group_key)

    for (source, lang), grouped in itertools.groupby(outputs, key=group_key):
        completions = [output["completion"] for output in grouped]
        metrics = compute_metrics(completions, lang)
        all_metrics[(source, lang)] = metrics
        metrics_per_lang[lang].append(metrics)
        metrics_per_source[source].append(metrics)

    def average_metrics(metrics_list: list[dict]) -> dict:
        averages = {}
        for key in ("acc", "lpr", "wpr"):
            values = [
                m[key] for m in metrics_list
                if m.get(key) is not None  # WPR can be missing for some languages
            ]
            if values:
                averages[key] = sum(values) / len(values)
        return averages

    average_per_source = {
        (source, "all"): average_metrics(metrics)
        for source, metrics in metrics_per_source.items()
    }
    all_metrics.update(average_per_source)

    average_per_lang = {
        ("all", lang): average_metrics(metrics)
        for lang, metrics in metrics_per_lang.items()
    }
    all_metrics.update(average_per_lang)

    average = average_metrics(list(average_per_lang.values()))
    all_metrics[("all", "all")] = average

    return all_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute WPR and LPR over the model completions in given CSV file. "
                                     "Example: `python compute_metrics.py outputs/qwen.csv`")
    parser.add_argument("csv_file", help="CSV file with the same schema as the sweep outputs "
                        "(with 'task', 'model', 'source', 'language' and 'completion' fields)")
    args = parser.parse_args()

    with open(args.csv_file, encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        outputs = list(reader)

    print("task", "model", "source", "language", "lpr", "wpr", sep="\t")

    group_key = lambda output: (output["task"], output["model"])
    outputs = sorted(outputs, key=group_key)
    for (task, model), outputs_ in itertools.groupby(outputs, key=group_key):
        all_metrics = compute_all_metrics(outputs_)

        for (source, lang), metrics in all_metrics.items():
            lpr = f"{metrics['lpr']:.2%}"
            wpr = f"{metrics['wpr']:.2%}" if "wpr" in metrics else "N/A"
            print(task, model, source, lang, lpr, wpr, sep="\t")
