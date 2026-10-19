"""P5 Comps, box 1: Clean and label.

Builds the labeled film table the Comps search, Comps expert and Combiner train on.

What it does (per the team plan, "P5 Comps -> Clean and label"):
  1. Loads the big TMDB dataset (and optionally the 2024-2026 pull) and merges them
     with the existing combine_datasets() from data_processing.py.
  2. Cleans rows, logging every dropped row and the reason (shared rule).
  3. Keys everything on imdb_id and adds as_of_date (shared handoff rule).
  4. Inflation-adjusts budget and revenue with US CPI (FRED CPIAUCSL).
  5. Adds labels: roi_multiple, log_roi, breakeven (>= 2.5x), breakout (>= 5x),
     budget_band, audience_score. streaming_success and legs need sources we don't
     have yet (streaming Top 10, Box Office Mojo opening weekends), so they are left
     empty for now and filled by later scripts.
  6. Assigns the year split: train <= 2017, validate 2018-2019, skip 2020-2021,
     test 2022-2025. Anything else is "excluded".
  7. Prints a report of counts against the plan's minimums.

Usage (from the tmdb-exploratory-analysis folder):

    python src/clean_and_label.py \
        --big "data/raw/TMDB_movie_dataset_v11.csv" \
        --small "data/raw/tmdb_movies_2024_2026.xlsx" \
        --cpi "data/raw/CPIAUCSL.csv"

--small and --cpi are optional. Download the CPI file from
https://fred.stlouisfed.org/series/CPIAUCSL ("Download" -> CSV). Without it the
script still runs but leaves the *_adj columns empty.

Outputs (default data/clean/):
    comps_labeled.csv     one row per film, the handoff table
    comps_dropped_rows.csv   every dropped row with its reason
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processing import CANONICAL_COLUMNS, combine_datasets, standardize_schema  # noqa: E402

# ---- Thresholds from the team plan -------------------------------------------
BREAKEVEN_MULTIPLE = 2.5
BREAKOUT_MULTIPLE = 5.0
MIN_PLAUSIBLE_DOLLARS = 1000      # same placeholder cutoff as data_processing.py
SUSPECT_MULTIPLE = 100            # flagged, not dropped: often a units error in TMDB
AUDIENCE_MIN_VOTES = 50           # vote_average is noise below this many votes

# Budget bands (plan, P1 budget estimator), applied to the inflation-adjusted budget.
BUDGET_BANDS = [
    (0, 1e6, "micro"),
    (1e6, 10e6, "low"),
    (10e6, 40e6, "mid"),
    (40e6, 100e6, "high"),
    (100e6, np.inf, "tentpole"),
]

# Year split.
SPLITS = [
    (None, 2017, "train"),
    (2018, 2019, "validate"),
    (2020, 2021, "skip"),
    (2022, 2025, "test"),
]

# Minimums and "good" targets from Step 1 of the plan.
TARGETS = {
    "films with budget and gross": (3000, 8000),
    "breakout films (5x+)": (300, 1000),
    "validation films (2018-2019)": (300, 600),
    "test films (2022-2025)": (400, 800),
    "synopses with outcomes": (5000, 15000),
}

OUTPUT_COLUMNS = [
    "imdb_id", "as_of_date", "tmdb_id", "title", "release_date", "release_year", "split",
    "genres", "original_language", "production_countries", "runtime_minutes",
    "budget", "revenue", "budget_adj", "revenue_adj", "cpi_base_month",
    "budget_band", "roi_multiple", "log_roi", "breakeven", "breakout",
    "audience_score", "vote_count", "streaming_success", "legs",
    "overview", "has_synopsis", "suspect_multiple", "data_source",
]


# ---- Loading ------------------------------------------------------------------
def read_any(path):
    """Read a CSV or Excel file. The big dataset is too large for Excel, so CSV is normal."""
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def load_and_merge(big_path, small_path=None):
    big = read_any(big_path)
    print(f"Loaded big dataset: {len(big):,} rows from {big_path}")
    if small_path:
        small = read_any(small_path)
        print(f"Loaded 2024-2026 pull: {len(small):,} rows from {small_path}")
        return combine_datasets(small, big)
    return standardize_schema(big, "big_dataset")


# ---- Cleaning with a drop log -----------------------------------------------
class DropLog:
    def __init__(self):
        self.parts = []

    def drop(self, df, mask, reason):
        """Remove rows where mask is True, record them, return the kept rows."""
        mask = mask.fillna(True).astype(bool)
        dropped = df[mask]
        if len(dropped):
            self.parts.append(
                dropped[["id", "imdb_id", "title", "release_date", "budget", "revenue"]]
                .rename(columns={"id": "tmdb_id"})
                .assign(reason=reason)
            )
            print(f"  dropped {len(dropped):>9,}  {reason}")
        return df[~mask]

    def frame(self):
        if not self.parts:
            return pd.DataFrame(columns=["tmdb_id", "imdb_id", "title", "release_date",
                                         "budget", "revenue", "reason"])
        return pd.concat(self.parts, ignore_index=True)


def _blank(series):
    return series.isna() | (series.astype(str).str.strip().isin(["", "nan", "None", "[]"]))


def clean(df, log):
    df = df.copy()
    for col in ("budget", "revenue", "runtime_minutes", "vote_average", "vote_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["imdb_id"] = df["imdb_id"].astype("string").str.strip()

    print("Cleaning:")
    df = log.drop(df, df["budget"].isna() | (df["budget"] < MIN_PLAUSIBLE_DOLLARS),
                  f"budget missing or under ${MIN_PLAUSIBLE_DOLLARS:,}")
    df = log.drop(df, df["revenue"].isna() | (df["revenue"] < MIN_PLAUSIBLE_DOLLARS),
                  f"revenue missing or under ${MIN_PLAUSIBLE_DOLLARS:,}")
    df = log.drop(df, df["status"].astype(str) != "Released", "status is not Released")
    df = log.drop(df, df["adult"].astype(str).str.lower().isin(["true", "1"]), "adult title")
    df = log.drop(df, df["release_date"].isna(), "release date missing")
    df = log.drop(df, ~df["imdb_id"].fillna("").str.match(r"^tt\d+$"),
                  "imdb_id missing or malformed (team joins on imdb_id)")
    df = log.drop(df, _blank(df["genres"]), "genres missing (comps search needs genre)")
    df = log.drop(df, ~(df["runtime_minutes"] > 0), "runtime missing or zero")

    # Same film under two TMDB ids: keep the one with more votes.
    df = df.sort_values("vote_count", ascending=False, na_position="last")
    df = log.drop(df, df.duplicated(subset="imdb_id", keep="first"),
                  "duplicate imdb_id (kept the row with most votes)")
    return df


# ---- Inflation adjustment ----------------------------------------------------
def load_cpi(path):
    """Read FRED's CPIAUCSL CSV into a month -> CPI series."""
    cpi = pd.read_csv(path)
    date_col = [c for c in cpi.columns if "date" in c.lower()][0]
    value_col = [c for c in cpi.columns if c != date_col][0]
    cpi[date_col] = pd.to_datetime(cpi[date_col])
    cpi[value_col] = pd.to_numeric(cpi[value_col], errors="coerce")
    series = cpi.set_index(date_col)[value_col].sort_index()
    series.index = series.index.to_period("M")
    # FRED has blank months (e.g. Oct 2025, skipped during the government shutdown).
    # Fill them from the neighbouring months so no film loses its adjustment.
    missing = series[series.isna()].index
    if len(missing):
        print(f"CPI blank for {', '.join(map(str, missing))}: filled by interpolation.")
        series = series.interpolate(limit_area="inside")
    return series.dropna()


def inflation_adjust(df, cpi):
    if cpi is None:
        print("No CPI file given: budget_adj and revenue_adj left empty, "
              "budget bands use nominal budget.")
        df["budget_adj"] = np.nan
        df["revenue_adj"] = np.nan
        df["cpi_base_month"] = pd.NA
        return df

    base_month = cpi.index.max()
    base = cpi.loc[base_month]
    month = df["release_date"].dt.to_period("M")
    # Films before CPI starts (1947) use the first month; films after it ends use the last.
    month = month.clip(lower=cpi.index.min(), upper=base_month)
    factor = base / month.map(cpi)
    df["budget_adj"] = (df["budget"] * factor).round(0)
    df["revenue_adj"] = (df["revenue"] * factor).round(0)
    df["cpi_base_month"] = str(base_month)
    print(f"Inflation-adjusted to {base_month} dollars.")
    return df


# ---- Labels and split --------------------------------------------------------
def budget_band(amount):
    if pd.isna(amount):
        return pd.NA
    for low, high, name in BUDGET_BANDS:
        if low <= amount < high:
            return name
    return pd.NA


def split_for_year(year):
    for low, high, name in SPLITS:
        if (low is None or year >= low) and year <= high:
            return name
    return "excluded"


def add_labels(df):
    df["roi_multiple"] = df["revenue"] / df["budget"]   # same with or without inflation
    df["log_roi"] = np.log(df["roi_multiple"])
    df["breakeven"] = (df["roi_multiple"] >= BREAKEVEN_MULTIPLE).astype(int)
    df["breakout"] = (df["roi_multiple"] >= BREAKOUT_MULTIPLE).astype(int)
    df["suspect_multiple"] = (df["roi_multiple"] > SUSPECT_MULTIPLE).astype(int)

    band_source = df["budget_adj"].where(df["budget_adj"].notna(), df["budget"])
    df["budget_band"] = band_source.map(budget_band)

    # Audience score: TMDB vote average (0-10) only where enough people voted.
    df["audience_score"] = df["vote_average"].where(df["vote_count"] >= AUDIENCE_MIN_VOTES)

    # Need sources we haven't pulled yet (streaming Top 10, Box Office Mojo).
    df["streaming_success"] = pd.NA
    df["legs"] = pd.NA

    df["has_synopsis"] = (~_blank(df["overview"])).astype(int)

    df["release_year"] = df["release_date"].dt.year.astype(int)
    df["split"] = df["release_year"].map(split_for_year)
    # A film's comps must come from before its release, so as_of_date = release date.
    df["as_of_date"] = df["release_date"].dt.date
    df["release_date"] = df["release_date"].dt.date
    return df.rename(columns={"id": "tmdb_id"})


# ---- Report ------------------------------------------------------------------
def report(df):
    usable = df[df["split"] != "excluded"]
    counts = {
        "films with budget and gross": len(df),
        "breakout films (5x+)": int(df["breakout"].sum()),
        "validation films (2018-2019)": int((df["split"] == "validate").sum()),
        "test films (2022-2025)": int((df["split"] == "test").sum()),
        "synopses with outcomes": int(df["has_synopsis"].sum()),
    }
    print("\nAgainst the plan's targets:")
    for name, (minimum, good) in TARGETS.items():
        n = counts[name]
        status = "GOOD" if n >= good else ("ok (min met)" if n >= minimum else "BELOW MINIMUM")
        print(f"  {name:<32} {n:>7,}   min {minimum:,} / good {good:,}   {status}")

    print("\nRows by split:")
    print(df["split"].value_counts().reindex(["train", "validate", "skip", "test", "excluded"],
                                             fill_value=0).to_string())
    print("\nBreakeven rate by split:")
    print(usable.groupby("split")["breakeven"].mean().round(3).to_string())
    print("\nFilms by budget band:")
    print(df["budget_band"].value_counts().to_string())
    n_suspect = int(df["suspect_multiple"].sum())
    if n_suspect:
        print(f"\n{n_suspect:,} films have a multiple over {SUSPECT_MULTIPLE}x "
              "(flagged suspect_multiple=1). Spot-check these against The Numbers.")


# ---- Main --------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--big", required=True, help="big TMDB dataset (CSV or Excel)")
    p.add_argument("--small", help="optional 2024-2026 TMDB pull (CSV or Excel)")
    p.add_argument("--cpi", help="FRED CPIAUCSL CSV for inflation adjustment")
    p.add_argument("--out-dir", default="data/clean", help="where to write outputs")
    args = p.parse_args(argv)

    df = load_and_merge(args.big, args.small)
    print(f"Merged: {len(df):,} rows")

    log = DropLog()
    df = clean(df, log)
    df = inflation_adjust(df, load_cpi(args.cpi) if args.cpi else None)
    df = add_labels(df)
    df = df[OUTPUT_COLUMNS].sort_values(["release_date", "imdb_id"]).reset_index(drop=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "comps_labeled.csv")
    drop_path = os.path.join(args.out_dir, "comps_dropped_rows.csv")
    df.to_csv(out_path, index=False)
    log.frame().to_csv(drop_path, index=False)
    print(f"\nWrote {len(df):,} labeled films to {out_path}")
    print(f"Wrote {len(log.frame()):,} dropped rows to {drop_path}")

    report(df)
    return df


if __name__ == "__main__":
    main()
