import csv
import glob
import os
from collections import Counter

csv.field_size_limit(10**9)
here = os.path.dirname(os.path.abspath(__file__))
raw = os.path.join(here, "..", "..", "raw_data")

cands = [p for p in glob.glob(os.path.join(raw, "*.csv"))
         if "tmdb" in os.path.basename(p).lower()]
if not cands:
    print("No TMDB csv found in raw_data. Move the big TMDB .csv file there and run again.")
    raise SystemExit
tmdb_path = max(cands, key=os.path.getsize)
print("Using:", tmdb_path)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def band(b):
    if b < 1e6:
        return "1 micro (<$1M)"
    if b < 10e6:
        return "2 low ($1M-$10M)"
    if b < 40e6:
        return "3 mid ($10M-$40M)"
    if b < 100e6:
        return "4 high ($40M-$100M)"
    return "5 tentpole ($100M+)"


ours = {}
with open(os.path.join(raw, "moviesum_clean.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        ours[r["imdb_id"]] = r

found = {}
with open(tmdb_path, encoding="utf-8", newline="") as f:
    rd = csv.DictReader(f)
    print("TMDB columns:", rd.fieldnames)
    need = ["imdb_id", "budget", "revenue", "release_date"]
    miss = [c for c in need if c not in rd.fieldnames]
    if miss:
        print("This file is missing columns:", miss, "- send me the column list above.")
        raise SystemExit
    for row in rd:
        i = row["imdb_id"]
        if i in ours:
            old = found.get(i)
            if old is None or num(row["revenue"]) > num(old["revenue"]):
                found[i] = row

MIN = 10000
rows = []
n_budget = n_rev = 0
for i, r in ours.items():
    t = found.get(i)
    b = num(t["budget"]) if t else 0.0
    g = num(t["revenue"]) if t else 0.0
    b = b if b >= MIN else 0.0
    g = g if g >= MIN else 0.0
    n_budget += b > 0
    n_rev += g > 0
    rows.append({
        "imdb_id": i,
        "movie_name": r["movie_name"],
        "year": r["year"],
        "year_split": r["year_split"],
        "budget": int(b),
        "revenue": int(g),
        "as_of_date": t["release_date"] if t else "",
    })

both = [r for r in rows if r["budget"] > 0 and r["revenue"] > 0]
print("Scripts total:", len(ours))
print("Matched to TMDB:", len(found))
print("With a usable budget (>= $10,000):", n_budget)
print("With a usable worldwide gross:", n_rev)
print("With BOTH budget and gross:", len(both))
print("  by year split:", dict(Counter(r["year_split"] for r in both)))
print("  by budget band:", dict(sorted(Counter(band(r["budget"]) for r in rows if r["budget"] > 0).items())))
print("  breakeven (gross >= 2.5x budget):", sum(r["revenue"] >= 2.5 * r["budget"] for r in both))
print("  breakout (gross >= 5x budget):", sum(r["revenue"] >= 5 * r["budget"] for r in both))

out = os.path.join(raw, "moviesum_with_outcomes.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("Saved:", out)
