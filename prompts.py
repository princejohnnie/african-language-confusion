"""
Builds the African Language Confusion prompt set from the sources:
 Aya, Dolly, PolyWrite and AfriQA (monolingual -- a native
speaker prompts the model in their own language), plus Okapi and ShareGPT
(crosslingual -- an English instruction asks the model to answer in a
target African language).

Output schema:
    id, prompt, source, task, language

`task` is either "monolingual" or "crosslingual".
"""
import io
import json
import random
import zipfile

import pandas as pd
import requests

import model_clients as mc
from languages import LANGUAGES, TARGET_COUNTS, RANDOM_SEED

LCB_TEST_SETS_URL = "https://github.com/Cohere-Labs-Community/language-confusion/raw/main/test_sets.zip"


def _sample(df: pd.DataFrame, n: int, lang_key: str, source: str) -> pd.DataFrame:
    """Sample up to n rows, warning (not silently truncating) when short."""
    if len(df) < n:
        print(f"  [WARN] {source}/{lang_key}: requested {n}, only {len(df)} available -- using all of them")
        return df
    return df.sample(n=n, random_state=RANDOM_SEED)


# --------------------------------------------------------------------------
# Monolingual sources
# --------------------------------------------------------------------------

def load_aya() -> pd.DataFrame:
    from datasets import load_dataset

    aya = load_dataset("CohereForAI/aya_dataset", token=mc.get_hf_token())
    df = pd.concat([aya[s].to_pandas().assign(split=s) for s in aya], ignore_index=True)

    n = TARGET_COUNTS["aya"]
    frames = []
    for key, info in LANGUAGES.items():
        code = info.get("aya")
        if code is None:
            continue
        subset = df[df["language"] == code]
        if subset.empty:
            print(f"  [WARN] aya/{key}: no rows found for '{code}'")
            continue
        subset = _sample(subset, n, key, "aya")
        frames.append(pd.DataFrame({
            "prompt": subset["inputs"].values,
            "source": "aya",
            "task": "monolingual",
            "language": key,
        }))
    return pd.concat(frames, ignore_index=True)


def load_dolly() -> pd.DataFrame:
    from datasets import load_dataset

    df = load_dataset(
        "CohereForAI/aya_evaluation_suite",
        "dolly_machine_translated",
        split="test",
        token=mc.get_hf_token(),
    ).to_pandas()

    n = TARGET_COUNTS["dolly"]
    frames = []
    for key, info in LANGUAGES.items():
        code = info.get("dolly")
        if code is None:
            continue
        subset = df[df["language"] == code]
        if subset.empty:
            print(f"  [WARN] dolly/{key}: no rows found for '{code}'")
            continue
        subset = _sample(subset, n, key, "dolly")
        frames.append(pd.DataFrame({
            "prompt": subset["inputs"].values,
            "source": "dolly",
            "task": "monolingual",
            "language": key,
        }))
    return pd.concat(frames, ignore_index=True)


def load_polywrite() -> pd.DataFrame:
    from datasets import load_dataset

    df = load_dataset("MaLA-LM/PolyWrite", split="train", token=mc.get_hf_token()).to_pandas()

    n = TARGET_COUNTS["polywrite"]
    frames = []
    for key, info in LANGUAGES.items():
        code = info.get("polywrite")
        if code is None:
            continue
        subset = df[df["lang_script"] == code]
        if subset.empty:
            print(f"  [WARN] polywrite/{key}: no rows found for '{code}'")
            continue
        subset = _sample(subset, n, key, "polywrite")
        frames.append(pd.DataFrame({
            "prompt": subset["prompt_translated"].values,
            "source": "polywrite",
            "task": "monolingual",
            "language": key,
        }))
    return pd.concat(frames, ignore_index=True)


def load_afriqa() -> pd.DataFrame:
    base_url = "https://raw.githubusercontent.com/masakhane-io/afriqa/main/data/queries"
    # Fon and Wolof pivot through French in the source repo, the rest through English.
    pivot = {"fon": "fr", "wol": "fr"}

    n = TARGET_COUNTS["afriqa"]
    frames = []
    for key, info in LANGUAGES.items():
        code = info.get("afriqa")
        if code is None:
            continue
        pv = pivot.get(code, "en")
        parts = []
        for split in ["train", "dev", "test"]:
            url = f"{base_url}/{code}/queries.afriqa.{code}.{pv}.{split}.json"
            try:
                parts.append(pd.read_json(url, lines=True))
            except Exception as e:
                print(f"  [WARN] afriqa/{key}: failed to fetch {split} split ({e})")
        if not parts:
            continue
        subset = pd.concat(parts, ignore_index=True)
        subset = _sample(subset, n, key, "afriqa")
        frames.append(pd.DataFrame({
            "prompt": subset["question"].values,
            "source": "afriqa",
            "task": "monolingual",
            "language": key,
        }))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Crosslingual sources (Okapi, ShareGPT)
#
# Each source contributes a pool of clean English base prompts (no
# instruction phrase attached), which we render once per African target
# language by inserting an instruction phrase like "Reply in {language}."
# at the Start or End of the prompt -- see _assign_phrasing/_render_crosslingual.
#
# Okapi: the actual upstream host (nlp.uoregon.edu, per Okapi's own
# scripts/download.sh) is unreachable -- confirmed via curl and Python's
# urllib, both timing out on connect (not just a slow response), with DNS
# resolving fine and other hosts (GitHub, HF) reachable, so it's the
# specific server that's down, not a local network issue. We fall back to
# the language-confusion benchmark's own bundled monolingual English Okapi
# test set, which is the same unaltered upstream Okapi text, just reliably
# hosted on GitHub.
#
# ShareGPT: LCB doesn't bundle a clean English ShareGPT set the way it does
# for Okapi, so we pull the first turn (the user's message) of each
# conversation directly from ShareGPT_Vicuna_unfiltered. We tried the raw
# RyokoAI/ShareGPT52K scrape first (no cleaning/splitting pipeline in
# between), but its conversations turned out to be noisier in practice --
# reverted to the cleaned/deduped version instead, accepting the tradeoff
# that its split_long_conversation.py step means turns[0] isn't guaranteed to
# be a conversation's true opening message.
# --------------------------------------------------------------------------

CROSSLINGUAL_PHRASINGS = [
    "reply in {language_requested}",
    "write in {language_requested}",
    "respond in {language_requested}",
    "answer in {language_requested}",
]


def _download_lcb_test_sets() -> None:
    import os
    if os.path.isdir("lcb_test_sets/test_sets"):
        return
    r = requests.get(LCB_TEST_SETS_URL, timeout=60)
    r.raise_for_status()
    zipfile.ZipFile(io.BytesIO(r.content)).extractall("lcb_test_sets")


def _load_okapi_crosslingual_base() -> pd.DataFrame:
    """Clean English base prompts for Okapi crosslingual rendering, from LCB's
    own unaltered monolingual Okapi test set (nlp.uoregon.edu, the real Okapi
    data host, is unreachable -- see module note above)."""
    _download_lcb_test_sets()
    df = pd.read_csv("lcb_test_sets/test_sets/monolingual/okapi/en.csv")
    n = TARGET_COUNTS["crosslingual_okapi"]
    sampled = _sample(df, n, "en", "crosslingual_okapi_base")
    return sampled.rename(columns={"prompt": "base_prompt"})[["base_prompt"]]


def _load_sharegpt_crosslingual_base() -> pd.DataFrame:
    """Clean English base prompts for ShareGPT crosslingual rendering: the
    first turn (the user's message, not ChatGPT's reply) of each conversation
    in ShareGPT_Vicuna_unfiltered, the cleaned/deduped derivative widely used
    for LLM training/eval (rather than the raw RyokoAI/ShareGPT52K scrape,
    which turned out to be dirtier in practice -- see module note above).
    Turns aren't English-only, so we still language-detect each one and keep
    only the English ones -- the crosslingual design requires an *English*
    instruction asking for a reply in the target language. Note this doesn't
    catch every case: a turn that's mostly non-English text wrapped in an
    English instruction template can still get misclassified as English by
    langdetect, since it scores the whole string at once rather than
    per-segment."""

    from datasets import load_dataset
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = RANDOM_SEED  # langdetect is otherwise non-deterministic

    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
        split="train",
        token=mc.get_hf_token(),
    )

    prompts = []
    n_non_english = 0
    for row in ds:
        turns = row["conversations"]
        if not turns:
            continue
        first = turns[0]
        first = json.loads(first) if isinstance(first, str) else first
        value = (first.get("value") or "").strip()
        if first.get("from") != "human" or not value:
            continue
        try:
            if detect(value) != "en":
                n_non_english += 1
                continue
        except LangDetectException:
            continue
        prompts.append(value)

    print(f"  crosslingual_sharegpt_base: {len(prompts)} English turns kept, "
          f"{n_non_english} non-English turns dropped")

    df = pd.DataFrame({"prompt": prompts})
    n = TARGET_COUNTS["crosslingual_sharegpt"]
    sampled = _sample(df, n, "en", "crosslingual_sharegpt_base")
    return sampled.rename(columns={"prompt": "base_prompt"})[["base_prompt"]]


def _assign_phrasing(base_df: pd.DataFrame) -> pd.DataFrame:
    """Reproducibly assign each base prompt a Start/End insertion point and an
    instruction phrasing (CROSSLINGUAL_PHRASINGS), for _render_crosslingual to
    apply per target language."""
    rng = random.Random(RANDOM_SEED)
    locations = [rng.choice(["Start", "End"]) for _ in range(len(base_df))]
    phrasings = [rng.choice(CROSSLINGUAL_PHRASINGS) for _ in range(len(base_df))]
    return base_df.assign(location=locations, phrasing=phrasings)


def _render_crosslingual(base_df: pd.DataFrame, source: str) -> pd.DataFrame:
    n = TARGET_COUNTS[f"crosslingual_{source}"]
    if len(base_df) < n:
        print(f"  [WARN] crosslingual/{source}: requested {n} prompts/language, "
              f"only {len(base_df)} unique base prompts available upstream -- using all of them")
    sampled = base_df if len(base_df) <= n else base_df.sample(n=n, random_state=RANDOM_SEED)

    frames = []
    for key, info in LANGUAGES.items():
        lang_name = info["name"]
        rows = []
        for _, r in sampled.iterrows():
            phrase = r["phrasing"].format(language_requested=lang_name)
            phrase_cap = phrase[0].upper() + phrase[1:]
            if r["location"] == "Start":
                prompt = f"{phrase_cap}. {r['base_prompt']}"
            else:
                prompt = f"{r['base_prompt']} {phrase_cap}."
            rows.append(prompt)
        frames.append(pd.DataFrame({
            "prompt": rows,
            "source": source,
            "task": "crosslingual",
            "language": key,
        }))
    return pd.concat(frames, ignore_index=True)


def load_crosslingual_okapi() -> pd.DataFrame:
    base_df = _assign_phrasing(_load_okapi_crosslingual_base())
    return _render_crosslingual(base_df, "okapi")


def load_crosslingual_sharegpt() -> pd.DataFrame:
    base_df = _assign_phrasing(_load_sharegpt_crosslingual_base())
    return _render_crosslingual(base_df, "sharegpt")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

LOADERS = {
    "aya": load_aya,
    "dolly": load_dolly,
    "polywrite": load_polywrite,
    "afriqa": load_afriqa,
    "crosslingual_okapi": load_crosslingual_okapi,
    "crosslingual_sharegpt": load_crosslingual_sharegpt,
}


def build_all_prompts(sources=None) -> pd.DataFrame:
    """Load every source (or just `sources`), concatenate, and assign a stable id."""
    sources = sources or list(LOADERS.keys())
    frames = []
    for name in sources:
        print(f"Loading {name}...")
        frames.append(LOADERS[name]())
    all_prompts = pd.concat(frames, ignore_index=True)
    all_prompts.insert(0, "id", range(len(all_prompts)))
    return all_prompts


def save_test_sets(df: pd.DataFrame, out_dir: str = "prompts") -> None:
    """Write one CSV per (task, source, language), mirroring language-confusion's test_sets/ layout."""
    import os
    for (task, source, language), group in df.groupby(["task", "source", "language"]):
        path = os.path.join(out_dir, task, source)
        os.makedirs(path, exist_ok=True)
        group[["id", "prompt", "source", "task", "language"]].to_csv(
            os.path.join(path, f"{language}.csv"), index=False
        )
    df.to_csv(os.path.join(out_dir, "all_prompts.csv"), index=False)
    print(f"Saved {len(df)} prompts to {out_dir}/ ({df['task'].nunique()} tasks, "
          f"{df['source'].nunique()} sources, {df['language'].nunique()} languages)")
