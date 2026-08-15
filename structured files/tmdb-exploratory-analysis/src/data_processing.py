import ast
import pandas as pd
import glob

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

def load_data(file_pattern):
    """Load data from an Excel file matching the given pattern."""
    excel_file = glob.glob(file_pattern)[0]
    print(f"Loading: {excel_file}")
    df = pd.read_excel(excel_file)
    return df

def clean_data(df):
    """Clean the DataFrame by converting budget and revenue to numeric and filtering out invalid rows."""
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    return df

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

MIN_PLAUSIBLE_DOLLARS = 1000  # below this, budget/revenue is placeholder junk (e.g. $1), not real data

def clean_combined_data(df):
    """Drop rows with zeroed-out, placeholder, or missing core fields so only usable movies remain."""
    df = df.copy()
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df["runtime_minutes"] = pd.to_numeric(df["runtime_minutes"], errors="coerce")

    # budget/revenue of a few dollars (sometimes just $1) are scraping placeholders,
    # not real financials -- require a plausible minimum, not just > 0.
    df = df[(df["budget"] >= MIN_PLAUSIBLE_DOLLARS) & (df["revenue"] >= MIN_PLAUSIBLE_DOLLARS)]
    df = df[df["runtime_minutes"] > 0]
    df = df[df["status"] == "Released"]
    df = df[df["adult"] == False]  # noqa: E712 - pandas bool comparison
    df = df[df["genres"].notna() & (df["genres"].astype(str).str.strip() != "")]
    df = df[df["production_countries"].notna() & (df["production_countries"].astype(str).str.strip() != "")]

    return df.reset_index(drop=True)

def build_processed_dataset(small_file, big_file, output_file):
    """Load both raw sources, combine, clean, and save the analysis-ready dataset."""
    small_df = load_data(small_file)
    big_df = load_data(big_file)

    combined = combine_datasets(small_df, big_df)
    cleaned = clean_combined_data(combined)

    save_cleaned_data(cleaned, output_file)
    return cleaned

def save_cleaned_data(df, output_file):
    """Save the cleaned DataFrame to an Excel file."""
    df.to_excel(output_file, index=False)
    print(f"Saved cleaned data to {output_file}")
