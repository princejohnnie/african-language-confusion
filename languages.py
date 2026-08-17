"""
Canonical language registry for the African Language Confusion study.

Each entry maps our internal `key` (used everywhere downstream as the
`language` column / output filenames) to:
  - name: English display name, used when we have to *say* the language
    inside an instruction prompt (crosslingual templates).
  - per-source codes: the value that identifies this language inside each
    upstream dataset. A missing key means that source has no data for the
    language
"""

# key -> {name, aya, dolly, polywrite, afriqa}
LANGUAGES = {
    "acholi":       {"name": "Acholi",         "polywrite": "ach_Latn"},
    "amharic":      {"name": "Amharic",        "aya": "Amharic",           "dolly": "amh", "polywrite": "amh_Ethi"},
    "chichewa":     {"name": "Chichewa",       "aya": "Nyanja",                            "polywrite": "nya_Latn"},
    "fongbe":       {"name": "Fongbe",                                                      "polywrite": "fon_Latn", "afriqa": "fon"},
    "ga":           {"name": "Ga",                                                          "polywrite": "gaa_Latn"},
    "hausa":        {"name": "Hausa",          "aya": "Hausa",             "dolly": "hau",  "polywrite": "hau_Latn", "afriqa": "hau"},
    "igbo":         {"name": "Igbo",           "aya": "Igbo",              "dolly": "ibo",  "polywrite": "ibo_Latn", "afriqa": "ibo"},
    "kinyarwanda":  {"name": "Kinyarwanda",                                                 "polywrite": "kin_Latn", "afriqa": "kin"},
    "lingala":      {"name": "Lingala",                                                     "polywrite": "lin_Latn"},
    "malagasy":     {"name": "Malagasy",       "aya": "Plateau Malagasy",  "dolly": "plt",  "polywrite": "mlg_Latn"},
    "ndebele":      {"name": "Ndebele",                                                     "polywrite": "nbl_Latn"},
    "sepedi":       {"name": "Northern Sotho", "aya": "Northern Sotho",    "dolly": "nso",  "polywrite": "nso_Latn"},
    "shona":        {"name": "Shona",          "aya": "Shona",             "dolly": "sna",  "polywrite": "sna_Latn"},
    "swahili":      {"name": "Swahili",        "aya": "Swahili",          "dolly": "swh",   "polywrite": "swa_Latn", "afriqa": "swa"},
    "tigrinya":     {"name": "Tigrinya",                                                    "polywrite": "tir_Ethi"},
    "twi":          {"name": "Twi",                                                         "polywrite": "aka_Latn", "afriqa": "twi"},
    "wolof":        {"name": "Wolof",          "aya": "Wolof",                              "polywrite": "wol_Latn", "afriqa": "wol"},
    "yoruba":       {"name": "Yoruba",         "aya": "Yoruba",            "dolly": "yor",  "polywrite": "yor_Latn", "afriqa": "yor"},
}

# The union of every language covered by at least one monolingual source.
# This is also the target-language list for the crosslingual generation
# task.
CROSSLINGUAL_TARGETS = list(LANGUAGES.keys())

# Per-language, per-source prompt counts to sample.
# Actual availability is often lower for some languages (e.g. Ga in
# PolyWrite, Northern Sotho in Aya) -- the loaders sample min(target,
# available) and report the shortfall rather than silently padding.
TARGET_COUNTS = {
    # Candidate pool pulled per language (see generate_aya_candidates in
    # prompts.py) for manual curation down to aya below.
    "aya_base": 200,
    "aya": 100,
    # Candidate pool pulled per language (see generate_dolly_candidates in
    # prompts.py) for manual curation down to dolly below.
    "dolly_base": 250,
    "dolly": 150,
    # Candidate pool pulled per language (see generate_polywrite_candidates in
    # prompts.py) for manual curation down to polywrite below.
    "polywrite_base": 150,
    "polywrite": 100,
    # "afriqa": 250,
    "crosslingual_okapi": 100,
    # Candidate pool pulled from the raw RyokoAI/ShareGPT52K scrape (see
    # generate_sharegpt_candidates in prompts.py) for manual curation down to
    # crosslingual_sharegpt below.
    "crosslingual_sharegpt_base": 400,
    "crosslingual_sharegpt": 200,
}

RANDOM_SEED = 42
