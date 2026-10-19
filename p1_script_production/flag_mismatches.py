"""Flag films where the TMDB match looks like a different film (usually a remake).

Compares the year in the MovieSum title with the TMDB release year. A gap of 2 or more
years sets flag_year_mismatch = 1. Adds year_gap and flag_year_mismatch to the budget table.
"""
import csv
import os
import sys
from collections import Counter

VERSION = "budget_table_v1"


def main():
    sys.stdout.reconfigure(errors="replace")
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "outputs")
    path = os.path.join(out_dir, "p1_budget_table_v1.csv")
    if not os.path.exists(path):
        print("p1_budget_table_v1.csv not found. Run budget_table.py first.")
        return
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        y1 = r["year"][:4]
        y2 = r["as_of_date"][:4]
        if y1.isdigit() and y2.isdigit():
            gap = abs(int(y1) - int(y2))
            r["year_gap"] = gap
            r["flag_year_mismatch"] = 1 if gap >= 2 else 0
        else:
            r["year_gap"] = ""
            r["flag_year_mismatch"] = 0
    fields = [c for c in rows[0].keys() if c not in ("budget_table_version",)] + ["budget_table_version"]
    fields = [c for c in fields if c not in ("year_gap", "flag_year_mismatch")]
    i = fields.index("flag_cpi_extrapolated") + 1
    fields[i:i] = ["year_gap", "flag_year_mismatch"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    dict_path = os.path.join(out_dir, "feature_dictionary_p1.csv")
    with open(dict_path, encoding="utf-8") as f:
        d = list(csv.DictReader(f))
    have = {r["name"] for r in d}
    for n_, g_, de_, s_, p_ in [
        ("year_gap", "flag", "Years between the MovieSum title year and the TMDB release year", "MovieSum + TMDB", "yes"),
        ("flag_year_mismatch", "flag", "1 if year_gap is 2 or more: the TMDB match is probably a different film (e.g. a remake)", "MovieSum + TMDB", "yes"),
    ]:
        if n_ not in have:
            d.append({"name": n_, "group": g_, "definition": de_, "source": s_,
                      "pre_release": p_, "owner": "P1", "version": VERSION})
    with open(dict_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "group", "definition", "source", "pre_release", "owner", "version"])
        w.writeheader()
        w.writerows(d)

    bad = [r for r in rows if r["flag_year_mismatch"] == 1]
    bad_b = [r for r in bad if r["flag_budget_usable"] == "1"]
    print("Films checked:", len(rows))
    print("Flagged as year mismatch (gap 2+ years):", len(bad), "| of those with a usable budget:", len(bad_b))
    print("Year gaps among usable-budget films:", dict(sorted(Counter(r["year_gap"] for r in rows if r["flag_budget_usable"] == "1" and r["year_gap"] != "").items())))
    ok = [r for r in rows if r["flag_budget_usable"] == "1" and r["flag_year_mismatch"] == 0]
    print("Usable budgets after dropping mismatches:", len(ok))
    print("Band counts after dropping mismatches:", dict(sorted(Counter(r["budget_band_adj"] for r in ok).items())))
    print("Flagged films (title year | TMDB year | nominal budget | name):")
    for r in sorted(bad, key=lambda r: -int(r["year_gap"]))[:80]:
        print("   %s | %s | %s | %s" % (r["year"], r["as_of_date"][:4], r["budget_nominal"] or "-", r["movie_name"]))
    print("Saved:", path)
    print("Updated:", dict_path)


if __name__ == "__main__":
    main()
