"""List budgets that look too high or too low for their decade, for a human to review.

Nothing is changed. It compares each film's adjusted budget with the median adjusted
budget of films from the same decade, and prints the biggest outliers on both sides.
"""
import csv
import os
import statistics
import sys
from collections import defaultdict


def main():
    sys.stdout.reconfigure(errors="replace")
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "outputs", "p1_budget_table_v1.csv")
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["flag_budget_usable"] == "1" and r["flag_year_mismatch"] == "0"]
    by_dec = defaultdict(list)
    for r in rows:
        by_dec[int(r["year"]) // 10 * 10].append(float(r["budget_adj"]))
    med = {d: statistics.median(v) for d, v in by_dec.items()}
    for r in rows:
        d = int(r["year"]) // 10 * 10
        r["ratio"] = float(r["budget_adj"]) / med[d]
        r["dec_med"] = med[d]
    rows.sort(key=lambda r: -r["ratio"])

    def show(r):
        print("   %5.1fx | %s | nominal $%s | adjusted $%s | decade median $%.1fM | gross $%s" % (
            r["ratio"], r["movie_name"], format(int(r["budget_nominal"]), ","),
            format(int(r["budget_adj"]), ","), r["dec_med"] / 1e6,
            format(int(r["gross_nominal"]), ",") if r["gross_nominal"] else "-"))

    print("Films checked:", len(rows))
    print("30 budgets that are the BIGGEST multiple of their decade's median:")
    for r in rows[:30]:
        show(r)
    print("15 budgets that are the SMALLEST fraction of their decade's median:")
    for r in rows[-15:]:
        show(r)


if __name__ == "__main__":
    main()
