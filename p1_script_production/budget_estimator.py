"""P1: budget estimator. Maps the 10 driver scores to a budget band and a budget range (curve).

Run from the p1_script_production folder:
    python budget_estimator.py --name pilot200
Reads : outputs/p1_driver_scores_v1_<name>.csv and outputs/p1_budget_table_v1.csv
Writes: outputs/p1_budget_estimates_v1_<name>.csv (out-of-fold estimates), outputs/budget_model_v1.json
Only TRAIN-split scripts (films through 2017) are used. Validation and test films are never touched here.
Needs numpy:  python -m pip install numpy
"""
import argparse
import csv
import json
import math
from collections import Counter

import numpy as np

ESTIMATOR_VERSION = "p1_budget_estimator_v1"
DRIVERS = ["shoot_days", "cast_size", "locations", "exteriors_nights", "sets",
           "stunts", "vfx", "special_conditions", "costume_makeup", "music"]
BANDS = ["micro", "low", "mid", "high", "tentpole"]
EDGES = [1e6, 1e7, 4e7, 1e8]          # micro <1M, low 1-10M, mid 10-40M, high 40-100M, tentpole 100M+
LOG_EDGES = [math.log10(e) for e in EDGES]


def band_of(budget):
    return sum(1 for e in EDGES if budget >= e)          # 0..4


def norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def band_probs(mu, sigma):
    cdf = [0.0] + [norm_cdf((e - mu) / sigma) for e in LOG_EDGES] + [1.0]
    return [cdf[i + 1] - cdf[i] for i in range(5)]


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    ym = y.mean()
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - ym))
    return {"mu": mu, "sd": sd, "w": w, "ym": ym}


def ridge_predict(m, X):
    return ((X - m["mu"]) / m["sd"]) @ m["w"] + m["ym"]


def ranks(a):
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pilot200")
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--splits", default="train")
    args = ap.parse_args()

    bud = {}
    with open("outputs/p1_budget_table_v1.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bud[r["imdb_id"]] = r
    drv = list(csv.DictReader(open("outputs/p1_driver_scores_v1_%s.csv" % args.name, encoding="utf-8")))

    split_counts = Counter(b["year_split"] for b in bud.values())
    print("year_split values in the budget table:", dict(split_counts))
    allowed = set(s.strip() for s in args.splits.split(","))
    rows, why = [], Counter()
    for d in drv:
        b = bud.get(d["imdb_id"])
        if b is None:
            why["not in budget table"] += 1
        elif str(b["use_for_budget_estimator"]).strip() != "1":
            why["budget not usable (flag)"] += 1
        elif b["year_split"] not in allowed:
            why["not in split " + "/".join(sorted(allowed)) + " (%s)" % b["year_split"]] += 1
        elif not b["budget_adj"] or float(b["budget_adj"]) <= 0:
            why["no budget number"] += 1
        else:
            rows.append((d, b))
    print("Scripts scored by drivers: %d | used to build the estimator: %d" % (len(drv), len(rows)))
    for k, v in why.items():
        print("   left out - %s: %d" % (k, v))
    if len(rows) < 30:
        print("Too few scripts to fit a model. Stopping.")
        return

    X = np.array([[float(d["score_" + k]) for k in DRIVERS] for d, _ in rows])
    budget = np.array([float(b["budget_adj"]) for _, b in rows])
    y = np.log10(budget)
    actual_band = np.array([band_of(v) for v in budget])
    n = len(rows)
    print("\nActual bands in the data:", {BANDS[i]: int((actual_band == i).sum()) for i in range(5)})

    rng = np.random.RandomState(args.seed)
    fold = rng.permutation(n) % args.folds
    oof = np.zeros(n)
    for k in range(args.folds):
        tr, te = fold != k, fold == k
        m = ridge_fit(X[tr], y[tr], args.alpha)
        oof[te] = ridge_predict(m, X[te])
    resid = y - oof
    sigma = float(resid.std())
    pred_band = np.array([band_of(10 ** v) for v in oof])
    acc = float((pred_band == actual_band).mean())
    within1 = float((abs(pred_band - actual_band) <= 1).mean())
    maj = Counter(actual_band.tolist()).most_common(1)[0][0]
    base_acc = float((actual_band == maj).mean())
    base_w1 = float((abs(actual_band - maj) <= 1).mean())
    r2 = 1 - float((resid ** 2).sum() / ((y - y.mean()) ** 2).sum())
    rho = float(np.corrcoef(ranks(oof), ranks(y))[0, 1])
    print("\nHow well it does on scripts it was NOT trained on (%d-fold check):" % args.folds)
    print("   right band exactly:        %.1f%%   (just guessing '%s' every time: %.1f%%)" % (100 * acc, BANDS[maj], 100 * base_acc))
    print("   right band or one away:    %.1f%%   (guessing '%s': %.1f%%)" % (100 * within1, BANDS[maj], 100 * base_w1))
    print("   typical miss on budget:    about %.1fx too high or too low (about 2 in 3 guesses land within %.1fx of the true budget)" % (
        10 ** float(np.abs(resid).mean()), 10 ** sigma))
    print("   R-squared (log budget):    %.2f   | rank correlation: %.2f   (0 = no signal, 1 = perfect)" % (r2, rho))

    probs = np.array([band_probs(v, sigma) for v in oof])
    print("\nCalibration check (does it say 30% when it is right 30% of the time?)")
    print("   %-9s %-22s %s" % ("band", "predicted share (sum)", "actual share"))
    for i in range(5):
        print("   %-9s %5.1f%%                  %5.1f%%" % (BANDS[i], 100 * probs[:, i].mean(), 100 * (actual_band == i).mean()))

    final = ridge_fit(X, y, args.alpha)
    print("\nWhich drivers matter (standardized weight on log budget; bigger = more):")
    for k, w in sorted(zip(DRIVERS, final["w"]), key=lambda t: -abs(t[1])):
        print("   %-19s %+.3f" % (k, w))

    out = "outputs/p1_budget_estimates_v1_%s.csv" % args.name
    z = 1.2816
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["imdb_id", "as_of_date", "movie_name", "year", "year_split", "actual_budget_adj", "actual_band",
                    "est_log10_budget", "est_budget_p10", "est_budget_p50", "est_budget_p90", "est_band"] +
                   ["prob_" + b for b in BANDS] + ["estimate_type", "estimator_version"])
        for i, (d, b) in enumerate(rows):
            w.writerow([d["imdb_id"], d["as_of_date"], d["movie_name"], d["year"], d["year_split"],
                        round(budget[i]), BANDS[actual_band[i]], round(oof[i], 4),
                        round(10 ** (oof[i] - z * sigma)), round(10 ** oof[i]), round(10 ** (oof[i] + z * sigma)),
                        BANDS[pred_band[i]]] + [round(p, 4) for p in probs[i]] + ["out_of_fold", ESTIMATOR_VERSION])
    json.dump({"version": ESTIMATOR_VERSION, "drivers": DRIVERS, "alpha": args.alpha, "sigma_log10": sigma,
               "mu": final["mu"].tolist(), "sd": final["sd"].tolist(), "w": final["w"].tolist(), "ym": float(final["ym"]),
               "trained_on": n, "splits": sorted(allowed), "edges": EDGES}, open("outputs/budget_model_v1.json", "w"), indent=1)
    print("\nSaved:", out)
    print("Saved: outputs/budget_model_v1.json")


main()
