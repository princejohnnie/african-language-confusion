"""
Shared plumbing for running the African Language Confusion model sweeps,
used by both Open_Source_Models.ipynb (OpenRouter) and
Closed_Source_Models.ipynb (CMU AI Gateway). Each notebook supplies its own
backend-specific `call_fn(prompt: str) -> str` closure; this module owns
the part that's identical either way: iterating prompts, retrying
transient failures, and writing/resuming the output CSV.

Design note -- one independent call per prompt, not a running conversation:
the exploratory Open_Source_Models.ipynb / Closed_Source_Models.ipynb blueprints
accumulated conversation turns across prompts within a language (each new
prompt saw every earlier reply as context). For a benchmark sweep that's a
confound -- it means language confusion in prompt N could be caused by
prompt N-1's reply rather than prompt N itself, and it doesn't match the
`id, model, completion, task, source, language` schema used by the
upstream language-confusion repo's `compute_metrics.py` (one row = one
independently-scored completion). So `run_benchmark` below sends every
prompt as a fresh single-turn message.
"""
import csv
import os
import time

import pandas as pd

OUTPUT_FIELDS = ["id", "model", "completion", "task", "source", "language"]

_env_loaded = False


def get_api_key(name: str) -> str:
    """
    Read an API key from the environment, loading `.env` (via python-dotenv)
    on first use. Copy `.env.example` to `.env` and fill in your keys --
    `.env` is gitignored, so real keys never get committed.
    """
    global _env_loaded
    if not _env_loaded:
        from dotenv import load_dotenv
        load_dotenv()
        _env_loaded = True

    key = os.getenv(name)
    if not key:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env in this folder and fill in {name}."
        )
    return key


def get_hf_token():
    """
    Read HF_TOKEN from the environment, loading `.env` on first use, same as
    get_api_key -- but optional. Hugging Face dataset downloads work fine
    anonymously; a token just raises the rate limit and speeds things up, so
    this returns None instead of raising when it's unset.
    """
    global _env_loaded
    if not _env_loaded:
        from dotenv import load_dotenv
        load_dotenv()
        _env_loaded = True
    return os.getenv("HF_TOKEN") or None


def get_prompts(df: pd.DataFrame, language=None, source=None, task=None, n=None) -> pd.DataFrame:
    """Filter the consolidated prompts table; handy for a quick check on one
    language/source before running the full sweep. `language`/`source` can be
    a single string or a list of strings."""
    out = df
    if language is not None:
        langs = [language] if isinstance(language, str) else language
        out = out[out["language"].isin(langs)]
    if source is not None:
        sources = [source] if isinstance(source, str) else source
        out = out[out["source"].isin(sources)]
    if task is not None:
        out = out[out["task"] == task]
    if n is not None:
        out = out.sample(n=min(n, len(out)), random_state=42)
    return out


def _load_done_ids(output_path: str) -> set:
    if not os.path.exists(output_path):
        return set()
    with open(output_path, newline="", encoding="utf-8") as f:
        return {int(row["id"]) for row in csv.DictReader(f)}


def run_benchmark(
    model_key: str,
    call_fn,
    prompts_df: pd.DataFrame,
    output_dir: str = "outputs",
    delay: float = 0.0,
    max_retries: int = 3,
    retry_backoff: float = 5.0,
) -> str:
    """
    Send every row in `prompts_df` through `call_fn` (a single-argument
    `call_fn(prompt) -> completion_str` closure) and append results to
    `{output_dir}/{model_key}.csv`. Resumable: prompt ids already present in
    that file are skipped, so a sweep interrupted by a rate limit or a
    restarted runtime can just be re-run. Failures that survive all retries
    are logged with an empty completion (and printed) rather than silently
    dropped, so they stay visible in the output file and in metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_key}.csv")

    done_ids = _load_done_ids(output_path)
    todo = prompts_df[~prompts_df["id"].isin(done_ids)]
    print(f"[{model_key}] {len(done_ids)} already done, {len(todo)} remaining "
          f"({len(prompts_df)} total) -> {output_path}")

    file_exists = os.path.exists(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if not file_exists:
            writer.writeheader()

        for i, (_, row) in enumerate(todo.iterrows()):
            completion = None
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    completion = call_fn(row["prompt"])
                    break
                except Exception as e:
                    last_error = e
                    print(f"  [{model_key}] id={row['id']} attempt {attempt}/{max_retries} failed: {e}")
                    if attempt < max_retries:
                        time.sleep(retry_backoff * attempt)

            if completion is None:
                print(f"  [{model_key}] id={row['id']} FAILED after {max_retries} attempts "
                      f"({last_error}) -- logging empty completion")
                completion = ""

            writer.writerow({
                "id": row["id"],
                "model": model_key,
                "completion": completion,
                "task": row["task"],
                "source": row["source"],
                "language": row["language"],
            })
            f.flush()

            if (i + 1) % 25 == 0:
                print(f"  [{model_key}] {i + 1}/{len(todo)}")
            if delay:
                time.sleep(delay)

    print(f"[{model_key}] done -> {output_path}")
    return output_path


def run_sweep(models: dict, call_fn_factory, prompts_df: pd.DataFrame, output_dir: str = "outputs", **kwargs):
    """
    Run `run_benchmark` for every model in `models` (a {model_key: model_id}
    dict). `call_fn_factory(model_id) -> call_fn` builds the backend-specific
    single-prompt call for that model_id -- see the two notebooks for the
    OpenRouter / CMU gateway versions.
    """
    paths = {}
    for model_key, model_id in models.items():
        call_fn = call_fn_factory(model_id)
        paths[model_key] = run_benchmark(model_key, call_fn, prompts_df, output_dir=output_dir, **kwargs)
    return paths
