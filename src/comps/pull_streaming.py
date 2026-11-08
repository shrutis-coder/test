"""P5 Comps: streaming_success label from Netflix's public global Top 10.

Netflix is the only major service that publishes viewing data: a weekly global
Top 10 for films (English and non-English) since the week of June 28, 2021.

Definition:
    streaming_success = 1 if the film reached Netflix's global weekly Top 10
    (films) within 18 months of its release, else 0.
    It is only defined (0 or 1) for films released after the data starts and at
    least 18 months before the latest week in the file, so every film had the
    same chance. Everything else is left blank, not guessed.

Caveats: Netflix only (a hit on Max, Disney+ or Prime doesn't count), and
pre-2021 films are blank, so this mostly covers the 2022-2025 test era.
P3 (Market) also uses the streaming Top 10; this reads their copy from
raw/market/ if it's there, so we don't keep two versions.

Run from the repo root (after clean_and_label.py has made the labeled table):

    python src/comps/pull_streaming.py

Saves raw/comps/streaming_success.csv. clean_and_label.py fills the column from it.
"""

import argparse
import io
import os
import re
import sys
import time
import unicodedata
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pull_wikipedia import CTX, UA  # noqa: E402

URL = "https://www.netflix.com/tudum/top10/data/all-weeks-global.tsv"
LOCAL_COPIES = ["raw/market/all-weeks-global.tsv", "raw/comps/all-weeks-global.tsv"]
DATA_START = pd.Timestamp("2021-06-28")
WINDOW_DAYS = 548          # 18 months
EARLY_DAYS = 30            # Netflix originals can chart days before a listed release date


def norm_title(title):
    if not isinstance(title, str):
        return ""
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    t = t.lower().replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(r"^the ", "", t)


def load_top10():
    for path in LOCAL_COPIES:
        if os.path.exists(path):
            print(f"Using {path}")
            return pd.read_csv(path, sep="\t")
    print(f"Downloading {URL} ...")
    text, last_error = None, None
    for attempt in range(1, 5):     # the connection sometimes drops partway through
        try:
            req = urllib.request.Request(URL, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120, context=CTX) as resp:
                chunks = []
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    chunks.append(chunk)
            text = b"".join(chunks).decode("utf-8")
            break
        except Exception as e:  # noqa: BLE001  (IncompleteRead, timeouts, resets)
            last_error = e
            print(f"  attempt {attempt} failed ({type(e).__name__}); retrying...")
            time.sleep(5 * attempt)
    if text is None:
        sys.exit(f"Couldn't download the Netflix Top 10 file ({last_error}).\n"
                 "Open https://www.netflix.com/tudum/top10, use the 'Download' link for the global "
                 "all-weeks data (TSV), and save it as raw/comps/all-weeks-global.tsv, then rerun.")
    os.makedirs("raw/comps", exist_ok=True)
    with open(LOCAL_COPIES[1], "w") as fh:
        fh.write(text)
    return pd.read_csv(io.StringIO(text), sep="\t")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labeled", default="clean/comps/comps_labeled.csv")
    p.add_argument("--out", default="raw/comps/streaming_success.csv")
    args = p.parse_args(argv)

    top = load_top10()
    top = top[top["category"].astype(str).str.startswith("Films")].copy()
    top["week"] = pd.to_datetime(top["week"])
    top["key"] = top["show_title"].map(norm_title)
    last_week = top["week"].max()
    hits = top.groupby("key").agg(first_week=("week", "min"), weeks_in_top10=("week", "nunique"),
                                  best_rank=("weekly_rank", "min"), netflix_title=("show_title", "first"))
    print(f"Netflix global Top 10: {len(hits):,} distinct films, weeks {top['week'].min().date()} to {last_week.date()}")

    lab = pd.read_csv(args.labeled, usecols=["imdb_id", "title", "release_date"], low_memory=False)
    lab["release"] = pd.to_datetime(lab["release_date"])
    lab["key"] = lab["title"].map(norm_title)
    eligible = (lab["release"] >= DATA_START - pd.Timedelta(days=EARLY_DAYS)) & \
               (lab["release"] <= last_week - pd.Timedelta(days=WINDOW_DAYS))

    m = lab.merge(hits, left_on="key", right_index=True, how="left")
    days = (m["first_week"] - m["release"]).dt.days
    in_window = days.between(-EARLY_DAYS, WINDOW_DAYS)
    # Same title used by two films: keep the hit only for the one released closest before it charted.
    m["gap"] = days.where(in_window)
    best = m.dropna(subset=["gap"]).sort_values("gap").drop_duplicates("key")["imdb_id"]
    m["hit"] = m["imdb_id"].isin(best)

    out = m.loc[eligible, ["imdb_id", "title", "release_date"]].copy()
    out["streaming_success"] = m.loc[eligible, "hit"].astype(int)
    for col in ("netflix_title", "first_week", "weeks_in_top10", "best_rank"):
        out[col] = m.loc[eligible, col].where(m.loc[eligible, "hit"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved {len(out):,} eligible films (released {(DATA_START - pd.Timedelta(days=EARLY_DAYS)).date()} to "
          f"{(last_week - pd.Timedelta(days=WINDOW_DAYS)).date()}) to {args.out}; "
          f"{int(out['streaming_success'].sum()):,} reached the Top 10 within 18 months.")


if __name__ == "__main__":
    main()
