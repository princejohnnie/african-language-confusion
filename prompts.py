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

AYA_CANDIDATES_DIR = "candidates/aya_candidates"
AYA_CURATED_DIR = "candidates/aya_curated"


def generate_aya_candidates(out_dir: str = AYA_CANDIDATES_DIR) -> dict:
    """Pull Aya prompts per language, keep only the 5-20 word ones, sample
    TARGET_COUNTS['aya_base'] candidates per language, and save one CSV per
    language under `out_dir` for manual review.

    Review each file, delete the rows you don't want, and save the result
    (down to TARGET_COUNTS['aya']) to the corresponding path under
    AYA_CURATED_DIR before calling load_aya()."""
    import os

    from datasets import load_dataset

    aya = load_dataset("CohereForAI/aya_dataset", token=mc.get_hf_token())
    df = pd.concat([aya[s].to_pandas().assign(split=s) for s in aya], ignore_index=True)

    n = TARGET_COUNTS["aya_base"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(AYA_CURATED_DIR, exist_ok=True)
    frames = {}
    for key, info in LANGUAGES.items():
        code = info.get("aya")
        if code is None:
            continue
        subset = df[df["language"] == code]
        word_counts = subset["inputs"].str.split().str.len()
        subset = subset[word_counts.between(5, 20)]
        if subset.empty:
            print(f"  [WARN] aya_candidates/{key}: no rows found for '{code}'")
            continue
        subset = _sample(subset, n, key, "aya_candidates")
        candidates = pd.DataFrame({"prompt": subset["inputs"].values})
        path = os.path.join(out_dir, f"{key}.csv")
        candidates.to_csv(path, index=False)
        frames[key] = candidates
        print(f"  Saved {len(candidates)} candidates to {path} -- review and prune to "
              f"{TARGET_COUNTS['aya']}, then save as {os.path.join(AYA_CURATED_DIR, f'{key}.csv')}")
    return frames


def _load_aya_curated(curated_dir: str = AYA_CURATED_DIR) -> pd.DataFrame:
    """Load the manually curated Aya candidate pool (see
    generate_aya_candidates), one CSV per language, as monolingual prompts."""
    import os

    n = TARGET_COUNTS["aya"]
    frames = []
    for key, info in LANGUAGES.items():
        if info.get("aya") is None:
            continue
        path = os.path.join(curated_dir, f"{key}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- run generate_aya_candidates() first, review "
                f"the resulting candidates CSV, prune it down to {n} prompts, and "
                f"save it to {path}."
            )
        subset = pd.read_csv(path)
        if len(subset) != n:
            print(f"  [WARN] aya/{key}: curated file has {len(subset)} prompts, expected {n}")
        frames.append(pd.DataFrame({
            "prompt": subset["prompt"].values,
            "source": "aya",
            "task": "monolingual",
            "language": key,
        }))
    return pd.concat(frames, ignore_index=True)


def load_aya() -> pd.DataFrame:
    return _load_aya_curated()


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
        subset = subset[~subset["inputs"].str.contains("Context:", na=False)]
        subset = subset[subset["inputs"].str.split().str.len() >= 5]
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
# conversation directly from the raw RyokoAI/ShareGPT52K scrape. We'd
# previously switched to ShareGPT_Vicuna_unfiltered (the cleaned/deduped
# derivative) because the raw scrape was noisier, but its
# split_long_conversation.py step means turns[0] isn't guaranteed to be a
# conversation's true opening message. Back on the raw scrape now, with
# tighter filtering to compensate: English-only, 5-20 words. Since that
# filtering still lets some low-quality prompts through,
# generate_sharegpt_candidates() over-samples a pool of
# crosslingual_sharegpt_base candidates for manual review -- prune that pool
# down to crosslingual_sharegpt curated prompts by hand before rendering.
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


SHAREGPT_CANDIDATES_PATH = "candidates/sharegpt_candidates.csv"
SHAREGPT_CURATED_PATH = "candidates/sharegpt_curated.csv"


def generate_sharegpt_candidates(path: str = SHAREGPT_CANDIDATES_PATH) -> pd.DataFrame:
    """Pull the first turn (the user's message, not ChatGPT's reply) of each
    conversation in the raw RyokoAI/ShareGPT52K scrape, keep only the
    English, 5-20 word ones, sample TARGET_COUNTS['crosslingual_sharegpt_base']
    candidates, and save them to `path` for manual review.

    Turns aren't English-only, so we still language-detect each one -- the
    crosslingual design requires an *English* instruction asking for a reply
    in the target language. Note this doesn't catch every case: a turn
    that's mostly non-English text wrapped in an English instruction
    template can still get misclassified as English by langdetect, since it
    scores the whole string at once rather than per-segment. That, plus the
    scrape's general noisiness, is why this pool still needs a manual pass:
    open `path`, delete the rows you don't want, and save the rest (down to
    TARGET_COUNTS['crosslingual_sharegpt']) to SHAREGPT_CURATED_PATH before
    calling load_crosslingual_sharegpt()."""
    import os

    from datasets import load_dataset
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = RANDOM_SEED  # langdetect is otherwise non-deterministic

    ds = load_dataset("RyokoAI/ShareGPT52K", split="train", token=mc.get_hf_token())

    prompts = []
    n_wrong_length = 0
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
        if not (5 <= len(value.split()) <= 20):
            n_wrong_length += 1
            continue
        try:
            if detect(value) != "en":
                n_non_english += 1
                continue
        except LangDetectException:
            continue
        prompts.append(value)

    print(f"  crosslingual_sharegpt_candidates: {len(prompts)} kept, "
          f"{n_wrong_length} dropped for length, {n_non_english} dropped as non-English")

    df = pd.DataFrame({"base_prompt": prompts})
    n = TARGET_COUNTS["crosslingual_sharegpt_base"]
    sampled = _sample(df, n, "en", "crosslingual_sharegpt_candidates")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    sampled.to_csv(path, index=False)
    print(f"  Saved {len(sampled)} candidates to {path} -- review and prune to "
          f"{TARGET_COUNTS['crosslingual_sharegpt']}, then save as {SHAREGPT_CURATED_PATH}")
    return sampled


def _load_sharegpt_crosslingual_base(path: str = SHAREGPT_CURATED_PATH) -> pd.DataFrame:
    """Load the manually curated ShareGPT candidate pool (see
    generate_sharegpt_candidates) as clean English base prompts for
    crosslingual rendering."""
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run generate_sharegpt_candidates() first, review "
            f"the resulting candidates CSV, prune it down to "
            f"{TARGET_COUNTS['crosslingual_sharegpt']} prompts, and save it to {path}."
        )
    df = pd.read_csv(path)
    n = TARGET_COUNTS["crosslingual_sharegpt"]
    if len(df) != n:
        print(f"  [WARN] crosslingual_sharegpt: curated file has {len(df)} prompts, expected {n}")
    return df[["base_prompt"]]


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
    # "afriqa": load_afriqa,
    # "crosslingual_okapi": load_crosslingual_okapi,  # no longer used -- see module note above
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
