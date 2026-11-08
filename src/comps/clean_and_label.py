"""P5 Comps, box 1: Clean and label.

Builds the labeled film table the Comps search, Comps expert and Combiner train on.

Steps (team plan, "P5 Comps -> Clean and label"):
  1. Load the TMDB 1M dataset (optionally merged with the 2024-2026 pull).
  2. Fill missing budget / worldwide gross from Wikidata and Wikipedia, joined on
     imdb_id (fill-only: TMDB figures are never overwritten).
  3. Clean rows, logging every dropped row and the reason.
  4. Key on imdb_id with an as_of_date (= release date, so comps come only from earlier films).
  5. Inflation-adjust with US CPI (FRED CPIAUCSL).
  6. Labels: roi_multiple, log_roi, breakeven (>= 2.5x), breakout (>= 5x),
     budget_band, audience_score, legs (domestic gross / opening weekend, from
     pull_box_office.py) and streaming_success (from pull_streaming.py).
  7. Year split: train <= 2017, validate 2018-2019, skip 2020-2021, test 2022-2025.
  8. Print counts against the plan's minimums.

Run from the repo root (see src/comps/README.md for the downloads):

    python src/comps/clean_and_label.py

Reads (defaults; each can be changed with a flag):
    raw/comps/TMDB_movie_dataset_v11.csv     --big       (required)
    raw/comps/CPIAUCSL.csv                   --cpi       (skipped if missing)
    raw/comps/wikidata_financials.csv        --wikidata  (skipped if missing)
    raw/comps/wikipedia_financials.csv       --wikipedia (skipped if missing)
    raw/comps/wikipedia_box_office.csv       --box-office (legs; skipped if missing)
    raw/comps/streaming_success.csv          --streaming (skipped if missing)

Writes:
    clean/comps/comps_labeled.csv            one row per film, the handoff table
    clean/comps/comps_dropped_rows.csv       every dropped row and why
    clean/comps/<source>_unmatched.csv       Wikidata/Wikipedia rows that matched no TMDB film
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tmdb_schema import combine_datasets, standardize_schema  # noqa: E402

# ---- Thresholds from the team plan -------------------------------------------
BREAKEVEN_MULTIPLE = 2.5
BREAKOUT_MULTIPLE = 5.0
MIN_PLAUSIBLE_DOLLARS = 1000      # under this, budget/gross is placeholder junk (e.g. $1)
SUSPECT_MULTIPLE = 100            # flagged, not dropped: often a units error in TMDB
# Gross under 5% of budget: almost always a streaming release with a token theatrical run
# (The Gray Man, White Noise, Pinocchio), not a real box office result. Real bombs earn more.
LOW_GROSS_MULTIPLE = 0.05
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
    "audience_score", "vote_count", "streaming_success", "legs", "domestic_gross", "opening_weekend",
    "overview", "has_synopsis", "suspect_multiple", "data_source", "financials_source", "budget_source", "revenue_source",
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


def apply_fill(df, path, out_dir, source):
    """Fill missing budget/revenue from Wikidata or Wikipedia, joined on imdb_id.

    Fill-only: a TMDB value that already passes the plausibility cutoff is kept.
    budget_source / revenue_source record where each figure came from.
    """
    wd = pd.read_csv(path, dtype={"imdb_id": "string"})
    wd = wd.drop_duplicates("imdb_id").set_index("imdb_id")
    ids = df["imdb_id"].astype("string").str.strip()
    budget = pd.to_numeric(df["budget"], errors="coerce")
    revenue = pd.to_numeric(df["revenue"], errors="coerce")
    had_both = (budget >= MIN_PLAUSIBLE_DOLLARS) & (revenue >= MIN_PLAUSIBLE_DOLLARS)

    wd_budget = ids.map(wd["budget_usd"])
    wd_gross = ids.map(wd["gross_usd"])
    fill_b = (budget.isna() | (budget < MIN_PLAUSIBLE_DOLLARS)) & (wd_budget >= MIN_PLAUSIBLE_DOLLARS)
    fill_r = (revenue.isna() | (revenue < MIN_PLAUSIBLE_DOLLARS)) & (wd_gross >= MIN_PLAUSIBLE_DOLLARS)
    df["budget"] = budget.where(~fill_b, wd_budget)
    df["revenue"] = revenue.where(~fill_r, wd_gross)

    df.loc[fill_b, "budget_source"] = source
    df.loc[fill_r, "revenue_source"] = source

    now_both = (df["budget"] >= MIN_PLAUSIBLE_DOLLARS) & (df["revenue"] >= MIN_PLAUSIBLE_DOLLARS)
    gained = now_both & ~had_both
    years = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    matched = ids.isin(wd.index)
    unmatched = wd[~wd.index.isin(ids)]
    os.makedirs(out_dir, exist_ok=True)
    unmatched.reset_index().to_csv(os.path.join(out_dir, f"{source}_unmatched.csv"), index=False)
    print(f"{source}: {len(wd):,} films, {int(matched.sum()):,} matched on imdb_id; "
          f"filled budget for {int(fill_b.sum()):,}, gross for {int(fill_r.sum()):,}; "
          f"{int(gained.sum()):,} films now have both "
          f"({int((gained & years.between(2022, 2025)).sum()):,} from 2022-2025).")
    return df


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


def clean(df, log, keep_low_gross=False):
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
    if not keep_low_gross:
        df = log.drop(df, df["revenue"] < LOW_GROSS_MULTIPLE * df["budget"],
                      f"gross under {LOW_GROSS_MULTIPLE:.0%} of budget (likely streaming or token release)")
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

    # Filled later in main() from pull_box_office.py / pull_streaming.py output, if present.
    for col in ("streaming_success", "legs", "domestic_gross", "opening_weekend"):
        if col not in df.columns:
            df[col] = pd.NA

    df["has_synopsis"] = (~_blank(df["overview"])).astype(int)

    df["release_year"] = df["release_date"].dt.year.astype(int)
    df["split"] = df["release_year"].map(split_for_year)
    # A film's comps must come from before its release, so as_of_date = release date.
    df["as_of_date"] = df["release_date"].dt.date
    df["release_date"] = df["release_date"].dt.date
    return df.rename(columns={"id": "tmdb_id"})


def add_side_labels(df, box_office_path, streaming_path):
    """legs / domestic gross / opening weekend and streaming_success, joined on imdb_id."""
    if box_office_path:
        bo = pd.read_csv(box_office_path, dtype={"imdb_id": "string"}).drop_duplicates("imdb_id").set_index("imdb_id")
        ids = df["imdb_id"].astype("string")
        df["legs"] = ids.map(bo["legs"]).round(3)
        df["domestic_gross"] = ids.map(bo["domestic_gross"])
        df["opening_weekend"] = ids.map(bo["opening_weekend"])
        print(f"legs: filled for {int(df['legs'].notna().sum()):,} films (median {df['legs'].median():.2f}).")
    if streaming_path:
        st = pd.read_csv(streaming_path, dtype={"imdb_id": "string"}).drop_duplicates("imdb_id").set_index("imdb_id")
        df["streaming_success"] = df["imdb_id"].astype("string").map(st["streaming_success"])
        known = df["streaming_success"].notna()
        print(f"streaming_success: defined for {int(known.sum()):,} films "
              f"({int((df['streaming_success'] == 1).sum()):,} hit Netflix's global Top 10).")
    return df


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
    print("\nTest films by year and financials source:")
    test = df[df["split"] == "test"]
    print(pd.crosstab(test["release_year"], test["financials_source"]).to_string())
    print("\nFilms by budget band:")
    print(df["budget_band"].value_counts().to_string())
    n_suspect = int(df["suspect_multiple"].sum())
    if n_suspect:
        print(f"\n{n_suspect:,} films have a multiple over {SUSPECT_MULTIPLE}x "
              "(flagged suspect_multiple=1). Often a TMDB budget typo; spot-check before training.")


# ---- Main --------------------------------------------------------------------
def optional(path, flag):
    """Use an optional input only if it exists, and say so either way."""
    if path and os.path.exists(path):
        return path
    if path:
        print(f"({flag}: {path} not found, skipping)")
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--big", default="raw/comps/TMDB_movie_dataset_v11.csv", help="TMDB 1M dataset CSV")
    p.add_argument("--small", help="optional 2024-2026 TMDB pull (CSV or Excel)")
    p.add_argument("--cpi", default="raw/comps/CPIAUCSL.csv", help="FRED CPIAUCSL CSV")
    p.add_argument("--wikidata", default="raw/comps/wikidata_financials.csv", help="from pull_wikidata.py")
    p.add_argument("--wikipedia", default="raw/comps/wikipedia_financials.csv", help="from pull_wikipedia.py")
    p.add_argument("--box-office", default="raw/comps/wikipedia_box_office.csv", help="from pull_box_office.py")
    p.add_argument("--streaming", default="raw/comps/streaming_success.csv", help="from pull_streaming.py")
    p.add_argument("--keep-low-gross", action="store_true",
                   help="keep films whose gross is under 5%% of budget (dropped by default as likely streaming)")
    p.add_argument("--out-dir", default="clean/comps", help="where to write outputs")
    args = p.parse_args(argv)
    if not os.path.exists(args.big):
        sys.exit(f"TMDB file not found: {args.big}. Download it (see src/comps/README.md) or pass --big PATH.")

    df = load_and_merge(args.big, args.small)
    print(f"Merged: {len(df):,} rows")
    df["budget_source"] = "tmdb"
    df["revenue_source"] = "tmdb"
    for source, path in (("wikidata", optional(args.wikidata, "--wikidata")),
                         ("wikipedia", optional(args.wikipedia, "--wikipedia"))):
        if path:
            df = apply_fill(df, path, args.out_dir, source)
    # One label per film: "wikipedia", or "tmdb/wikipedia" when budget and gross came from different places.
    df["financials_source"] = df["budget_source"].where(
        df["budget_source"] == df["revenue_source"], df["budget_source"] + "/" + df["revenue_source"])

    log = DropLog()
    df = clean(df, log, keep_low_gross=args.keep_low_gross)
    cpi_path = optional(args.cpi, "--cpi")
    df = inflation_adjust(df, load_cpi(cpi_path) if cpi_path else None)
    df = add_labels(df)
    df = add_side_labels(df, optional(args.box_office, "--box-office"), optional(args.streaming, "--streaming"))
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
