"""Budget table v1 (P1, rules only, no LLM).

Puts every budget (and gross) in today's dollars using US CPI, assigns the budget band
from the adjusted budget, and writes the table the Budget estimator will be trained on.
Needs: moviesum_with_outcomes.csv (from check_outcomes.py) and CPIAUCSL.csv (from FRED).
"""
import csv
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

VERSION = "budget_table_v1"
MIN_USABLE = 10000  # budgets under this are treated as missing (same rule as check_outcomes.py)
BANDS = [
    (1e6, "1 micro (<$1M)"),
    (10e6, "2 low ($1M-$10M)"),
    (40e6, "3 mid ($10M-$40M)"),
    (100e6, "4 high ($40M-$100M)"),
    (float("inf"), "5 tentpole ($100M+)"),
]


def band(x):
    for top, name in BANDS:
        if x < top:
            return name


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def month_key(as_of, year):
    s = (as_of or "").strip()
    if len(s) >= 7 and s[4] == "-" and s[:4].isdigit() and s[5:7].isdigit():
        return s[:7], 0
    return "%s-06" % year, 1  # no release date: assume mid-year


def main():
    sys.stdout.reconfigure(errors="replace")
    here = os.path.dirname(os.path.abspath(__file__))
    raw = os.path.join(here, "..", "..", "raw_data")
    out_dir = os.path.join(here, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    cpi_path = os.path.join(raw, "CPIAUCSL.csv")
    if not os.path.exists(cpi_path):
        print("CPIAUCSL.csv not found in raw_data.")
        return
    cpi = {}
    with open(cpi_path, encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            try:
                cpi[r[0][:7]] = float(r[1])
            except (ValueError, IndexError):
                pass
    months = sorted(cpi)
    first_m, last_m = months[0], months[-1]
    base = cpi[last_m]
    print("CPI months: %s to %s | base month for 'today's dollars': %s" % (first_m, last_m, last_m))

    outc_path = os.path.join(raw, "moviesum_with_outcomes.csv")
    if not os.path.exists(outc_path):
        print("moviesum_with_outcomes.csv not found. Run check_outcomes.py first.")
        return
    rows = []
    with open(outc_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            mk, no_date = month_key(r["as_of_date"], r["year"])
            extrap = 0
            if mk < first_m:
                mk, extrap = first_m, 1
            elif mk > last_m:
                mk = last_m
            factor = base / cpi.get(mk, base)
            b = num(r["budget"])
            g = num(r["revenue"])
            usable = 1 if b >= MIN_USABLE else 0
            badj = b * factor if usable else ""
            row = {
                "imdb_id": r["imdb_id"],
                "as_of_date": r["as_of_date"],
                "movie_name": r["movie_name"],
                "year": r["year"],
                "year_split": r["year_split"],
                "budget_nominal": int(b) if usable else "",
                "gross_nominal": int(g) if g >= MIN_USABLE else "",
                "inflation_factor": round(factor, 4),
                "budget_adj": int(round(badj)) if usable else "",
                "gross_adj": int(round(g * factor)) if g >= MIN_USABLE else "",
                "budget_band_adj": band(badj) if usable else "",
                "budget_band_nominal": band(b) if usable else "",
                "flag_budget_usable": usable,
                "flag_no_release_date": no_date,
                "flag_cpi_extrapolated": extrap,
                "budget_table_version": VERSION,
            }
            rows.append(row)

    out_path = os.path.join(out_dir, "p1_budget_table_v1.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # add the new columns to the feature dictionary (without duplicating names)
    dict_path = os.path.join(out_dir, "feature_dictionary_p1.csv")
    new = [
        ("budget_nominal", "budget", "Production budget in dollars of the release year (TMDB); blank if under $10,000", "TMDB budget", "no (label for the budget estimator)"),
        ("gross_nominal", "budget", "Worldwide gross in release-year dollars (TMDB); blank if under $10,000", "TMDB revenue", "no (outcome)"),
        ("inflation_factor", "budget", "US CPI at the latest month divided by CPI at the release month", "FRED CPIAUCSL", "yes"),
        ("budget_adj", "budget", "Budget in today's dollars = budget_nominal x inflation_factor", "TMDB + FRED CPI", "no (label for the budget estimator)"),
        ("gross_adj", "budget", "Gross in today's dollars", "TMDB + FRED CPI", "no (outcome)"),
        ("budget_band_adj", "budget", "micro <$1M, low $1-10M, mid $10-40M, high $40-100M, tentpole $100M+, on budget_adj", "TMDB + FRED CPI", "no (label for the budget estimator)"),
        ("budget_band_nominal", "budget", "Same bands on the unadjusted budget (for comparison only)", "TMDB", "no"),
        ("flag_budget_usable", "flag", "1 if the budget is at least $10,000", "TMDB", "yes"),
        ("flag_no_release_date", "flag", "1 if the release date was missing and mid-year of the title year was used", "TMDB", "yes"),
        ("flag_cpi_extrapolated", "flag", "1 if the film predates the first CPI month (1947) and the first month was used", "FRED CPIAUCSL", "yes"),
        ("budget_table_version", "key", "Version of the budget table code", "this file", "yes"),
    ]
    existing, have = [], set()
    if os.path.exists(dict_path):
        with open(dict_path, encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
            have = {r["name"] for r in existing}
    fields = ["name", "group", "definition", "source", "pre_release", "owner", "version"]
    for n_, g_, d_, s_, p_ in new:
        if n_ not in have:
            existing.append({"name": n_, "group": g_, "definition": d_, "source": s_,
                             "pre_release": p_, "owner": "P1", "version": VERSION})
    with open(dict_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(existing)

    # report
    use = [r for r in rows if r["flag_budget_usable"]]
    print("Rows:", len(rows), "| with a usable budget:", len(use))
    print("Saved:", out_path)
    print("Updated:", dict_path)
    print("Band counts (adjusted):", dict(sorted(Counter(r["budget_band_adj"] for r in use).items())))
    print("Band counts (unadjusted):", dict(sorted(Counter(r["budget_band_nominal"] for r in use).items())))
    changed = sum(1 for r in use if r["budget_band_adj"] != r["budget_band_nominal"])
    print("Films whose band changes after adjusting: %d (%.0f%%)" % (changed, 100.0 * changed / max(1, len(use))))
    print("Usable budgets by year split:", dict(Counter(r["year_split"] for r in use)))
    print("Films with no release date:", sum(r["flag_no_release_date"] for r in rows),
          "| before first CPI month:", sum(r["flag_cpi_extrapolated"] for r in rows))
    by_dec = defaultdict(list)
    for r in use:
        by_dec[(int(r["year"]) // 10) * 10].append(r["budget_adj"])
    print("Median adjusted budget by decade (films with a budget):")
    for d in sorted(by_dec):
        print("   %ds: %3d films | median $%.1fM" % (d, len(by_dec[d]), statistics.median(by_dec[d]) / 1e6))
    use_sorted = sorted(use, key=lambda r: r["budget_adj"])
    print("15 smallest adjusted budgets (check for typos):")
    for r in use_sorted[:15]:
        print("   $%s adj ($%s nominal) | %s" % (format(r["budget_adj"], ","), format(r["budget_nominal"], ","), r["movie_name"]))
    print("10 largest adjusted budgets:")
    for r in use_sorted[-10:]:
        print("   $%s adj ($%s nominal) | %s" % (format(r["budget_adj"], ","), format(r["budget_nominal"], ","), r["movie_name"]))


if __name__ == "__main__":
    main()
