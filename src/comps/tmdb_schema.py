"""TMDB schema helpers for the comps pipeline.

Copied from notebooks/eda/src/data_processing.py (Sarah's EDA code) so src/comps
runs on its own without importing from the notebooks folder. Keep the two in
sync if the canonical schema changes.
"""

import ast

import pandas as pd

# Canonical schema used throughout the project (matches tmdb_movies_2024_2026.xlsx).
CANONICAL_COLUMNS = [
    "id", "title", "original_title", "release_date", "status", "runtime_minutes",
    "genres", "overview", "tagline", "vote_average", "vote_count", "popularity",
    "budget", "revenue", "original_language", "spoken_languages",
    "production_companies", "production_countries", "director", "writers",
    "producers", "top_10_cast", "full_cast", "keywords", "homepage", "imdb_id",
    "poster_url", "backdrop_url", "adult", "belongs_to_collection",
]

# Maps the big Kaggle dataset's columns onto the canonical schema above.
BIG_DATASET_RENAME = {
    "runtime": "runtime_minutes",
    "genres_list": "genres",
    "Director": "director",
    "Writer": "writers",
    "Producers": "producers",
    "Cast_list": "full_cast",
    "poster_path": "poster_url",
    "backdrop_path": "backdrop_url",
}

def _listlike_to_csv(value):
    """Turn a stringified Python list (e.g. "['Action', 'Drama']") into "Action, Drama".

    The big Kaggle dataset stores genres/keywords this way while the rest of the
    project uses plain comma-separated strings; leaves already-plain strings as-is.
    """
    if not isinstance(value, str) or not value.startswith("["):
        return value
    try:
        items = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    return ", ".join(str(item) for item in items)

def standardize_schema(df, source):
    """Rename/align a raw dataset's columns onto CANONICAL_COLUMNS and tag its origin."""
    df = df.rename(columns=BIG_DATASET_RENAME)
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[CANONICAL_COLUMNS].copy()
    for col in ("genres", "keywords"):
        df[col] = df[col].map(_listlike_to_csv)
    df["data_source"] = source
    return df

# On overlapping ids the big dataset wins: it's the larger, more complete source,
# while the 2024-2026 pull has more missing values. The 2024-2026 rows only fill in
# movies the big dataset doesn't have (e.g. belongs_to_collection is unique to it).
SOURCE_PRIORITY = {"big_dataset": 0, "tmdb_2024_2026": 1}

def combine_datasets(small_df, big_df):
    """Union the 2024-2026 TMDB pull with the big Kaggle dataset on the canonical schema.

    Rows are deduplicated by TMDB `id` using SOURCE_PRIORITY to pick a winner.
    """
    small_std = standardize_schema(small_df, "tmdb_2024_2026")
    big_std = standardize_schema(big_df, "big_dataset")
    combined = pd.concat([small_std, big_std], ignore_index=True)
    combined["_priority"] = combined["data_source"].map(SOURCE_PRIORITY)
    combined = combined.sort_values("_priority").drop_duplicates(subset="id", keep="first")
    combined = combined.drop(columns="_priority")
    return combined.reset_index(drop=True)
