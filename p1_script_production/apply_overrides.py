"""Mark budgets a human found suspicious, and add a simple 'use for the budget estimator' column.

Reads outputs/budget_overrides.csv (movie_name, reason). Matching films get
flag_budget_suspect = 1. use_for_budget_estimator = 1 only when the budget is usable and
not suspect. Safe to run again.
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
    ov_path = os.path.join(out_dir, "budget_overrides.csv")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ov = {}
    with open(ov_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ov[r["movie_name"].strip()] = r["reason"].strip()
    names = {r["movie_name"] for r in rows}
    for k in ov:
        if k not in names:
            print("WARNING: no film named exactly:", k)
    for r in rows:
        why = ov.get(r["movie_name"], "")
        r["flag_budget_suspect"] = 1 if why else 0
        r["budget_suspect_reason"] = why
        r["use_for_budget_estimator"] = 1 if (r["flag_budget_usable"] == "1" and not why) else 0
    new_cols = ["flag_budget_suspect", "budget_suspect_reason", "use_for_budget_estimator"]
    fields = [c for c in rows[0].keys() if c not in new_cols and c != "budget_table_version"]
    i = fields.index("flag_year_mismatch") + 1
    fields[i:i] = new_cols
    fields.append("budget_table_version")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    dict_path = os.path.join(out_dir, "feature_dictionary_p1.csv")
    with open(dict_path, encoding="utf-8") as f:
        d = list(csv.DictReader(f))
    have = {r["name"] for r in d}
    for n_, g_, de_, s_ in [
        ("flag_budget_suspect", "flag", "1 if a person reviewed the TMDB budget and judged it implausible (see budget_suspect_reason)", "manual review"),
        ("budget_suspect_reason", "flag", "Why the budget was marked suspect", "manual review"),
        ("use_for_budget_estimator", "flag", "1 if the budget is usable and not suspect: the rows to train the budget estimator on", "rule"),
    ]:
        if n_ not in have:
            d.append({"name": n_, "group": g_, "definition": de_, "source": s_,
                      "pre_release": "no", "owner": "P1", "version": VERSION})
    with open(dict_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "group", "definition", "source", "pre_release", "owner", "version"])
        w.writeheader()
        w.writerows(d)

    use = [r for r in rows if str(r["use_for_budget_estimator"]) == "1"]
    print("Overrides read:", len(ov))
    print("Marked suspect:", sum(r["flag_budget_suspect"] for r in rows))
    print("Rows to train the budget estimator on:", len(use))
    print("   by year split:", dict(Counter(r["year_split"] for r in use)))
    print("   by adjusted band:", dict(sorted(Counter(r["budget_band_adj"] for r in use).items())))
    print("Saved:", path)


if __name__ == "__main__":
    main()
