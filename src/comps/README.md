# P5 Comps

Owner: Wesley. Builds the labeled film table for the comps search, comps expert and combiner.

## 1. Downloads (into `raw/comps/`, which is not committed)

| File | Where |
| --- | --- |
| `TMDB_movie_dataset_v11.csv` | [TMDB 1M movies on Kaggle](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies), or `python -c "import kagglehub; print(kagglehub.dataset_download('asaniczka/tmdb-movies-dataset-2023-930k-movies'))"` and copy the CSV in |
| `CPIAUCSL.csv` | [FRED CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL), Download → CSV |

## 2. Fill budget/gross gaps (optional, a few minutes each)

TMDB is missing a budget or a worldwide gross for most recent films, which leaves the
2022–2025 test split short. These two scripts fill gaps from open sources, joined on `imdb_id`:

```
python src/comps/pull_wikidata.py     # -> raw/comps/wikidata_financials.csv
python src/comps/pull_wikipedia.py    # -> raw/comps/wikipedia_financials.csv
```

Both are fill-only: a TMDB figure is never overwritten. US dollars only.

## 3. Clean and label

```
python src/comps/clean_and_label.py   # -> clean/comps/comps_labeled.csv
```

Uses the Wikidata/Wikipedia files automatically if they exist. Prints counts against the plan's minimums.

## 4. Embed synopses

```
python -m pip install -r src/comps/requirements.txt   # once
python src/comps/embed_synopses.py    # -> features/comps/synopsis_embeddings.npz
```

Uses `sentence-transformers/all-MiniLM-L6-v2` (384 dims). The first run downloads the model (~90 MB).

## 5. Comps search

```
python src/comps/comps_search.py      # -> features/comps/comps_features.csv, comps_list.csv
```

For each film, it keeps the 10 most similar films that meet all of these rules:

- same budget band
- includes the film's **main (first-listed) genre**
- released **at least 180 days before** the film's `as_of_date`, so its final gross was already known (`--min-gap-days` to change)
- never the film itself

`comps_features.csv` has one row per film, keyed on `imdb_id` + `as_of_date`, for the comps expert:

| Column | Meaning |
| --- | --- |
| `n_candidates`, `n_comps` | eligible films, comps kept (≤ 10) |
| `comp_log_roi_mean`, `comp_log_roi_std` | comp ROI average and spread (log scale) |
| `comp_roi_median`, `comp_breakeven_rate`, `comp_breakout_rate` | how the comps did |
| `closest_similarity`, `mean_similarity` | cosine similarity to the best / average comp |
| `n_close_comps` | comps with similarity ≥ 0.5 |
| `comp_recency_years` | median years between the comps' release and the as-of date |

`comps_list.csv` has one row per (film, comp) with rank and similarity. It feeds the investor view's "five closest comps".

## Output: `clean/comps/comps_labeled.csv`

One row per film, keyed on `imdb_id` with `as_of_date` (= release date).

| Column | Meaning |
| --- | --- |
| `split` | train ≤ 2017, validate 2018–19, skip 2020–21, test 2022–25 |
| `budget`, `revenue` | nominal USD; `budget_adj`, `revenue_adj` in latest-CPI dollars |
| `roi_multiple`, `log_roi` | gross ÷ budget |
| `breakeven`, `breakout` | 1 if gross ≥ 2.5× / ≥ 5× budget |
| `budget_band` | micro < $1M, low < $10M, mid < $40M, high < $100M, tentpole (inflation-adjusted) |
| `audience_score` | TMDB vote average, only with ≥ 50 votes |
| `budget_source`, `revenue_source`, `financials_source` | tmdb / wikidata / wikipedia |
| `suspect_multiple` | 1 if gross > 100× budget (often a TMDB typo) |
| `streaming_success`, `legs` | empty until streaming and opening-weekend data exist |

## Cleaning rules

Every dropped row is in `clean/comps/comps_dropped_rows.csv` with its reason:

- budget or gross under $1,000
- not released
- adult title
- no release date
- no valid IMDb ID
- no genre
- no runtime
- duplicate IMDb ID

Films whose gross is **under 5% of budget** are also dropped by default. These are almost always streaming releases with a token theatrical run (The Gray Man, White Noise, Pinocchio), not real box office results. Pass `--keep-low-gross` to keep them.

## Current numbers (Nov 7, 2026)

| | Count | Plan |
| --- | --- | --- |
| Labeled films | 10,248 | 3,000 min / 8,000 good |
| Breakouts | 1,987 | 300 / 1,000 |
| Validation (2018–19) | 599 | 300 / 600 |
| Test (2022–25) | 462 | 400 / 800 |

Note: 444 of the 462 test films are from 2022–23. Public box office data for 2024–25 is still thin.
