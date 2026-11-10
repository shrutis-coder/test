"""P1: roll scene tags up into the 10 budget driver scores (1 to 5) per script.

Run from the p1_script_production folder:
    python driver_scores.py --name pilot200
Reads : ../../raw_data/tagger/scene_tags_<name>.csv  and  outputs/p1_production_features_v1.csv
Writes: outputs/p1_driver_scores_v1_<name>.csv, outputs/driver_cutoffs_v1.json,
        outputs/feature_dictionary_p1_drivers_v1.csv
"""
import argparse
import csv
import json
import os
from collections import defaultdict

DRIVER_VERSION = "p1_drivers_v1"
CONDS = ["water", "weather", "fire_explosions", "vehicles", "aerial", "animals", "children",
         "firearms", "period_setting", "gore_or_prosthetics", "remote_location", "wire_or_rig"]
# driver name -> (plain-English meaning, how the raw number is computed)
DRIVERS = {
    "shoot_days": ("Rough size of the shoot", "scene_count + 2 x number of heavy scenes (stunts, vfx or set_complexity averaging 1.5 or more)"),
    "cast_size": ("How many different speaking characters", "unique_speaking_characters from the parser"),
    "locations": ("How many different locations", "unique_locations from the parser"),
    "exteriors_nights": ("Share of outdoor and night shooting", "ext_share + night_share from the parser"),
    "sets": ("How elaborate the sets are", "average set_complexity per scene (tagger)"),
    "stunts": ("How much stunt work", "average stunts score per scene (tagger)"),
    "vfx": ("How much visual effects", "average vfx score per scene (tagger)"),
    "special_conditions": ("Water, weather, fire, animals, etc.", "average number of special conditions per scene (tagger)"),
    "costume_makeup": ("Costume and makeup burden", "average costume_makeup score per scene (tagger)"),
    "music": ("Music on screen", "average music score per scene (tagger)"),
}


def num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def to_score(value, cuts):
    return 1 + sum(1 for c in cuts if value > c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pilot200")
    ap.add_argument("--tags", default=None)
    ap.add_argument("--features", default="outputs/p1_production_features_v1.csv")
    ap.add_argument("--cutoffs", default="outputs/driver_cutoffs_v1.json")
    ap.add_argument("--refit", action="store_true", help="recompute the 1-to-5 cut points from this set")
    args = ap.parse_args()
    tags_path = args.tags or "../../raw_data/tagger/scene_tags_%s.csv" % args.name
    out_path = "outputs/p1_driver_scores_v1_%s.csv" % args.name

    feats = {}
    with open(args.features, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            feats[r["imdb_id"]] = r

    per = defaultdict(list)
    with open(tags_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            per[r["imdb_id"]].append(r)

    rows = []
    skipped = []
    for imdb, scenes in per.items():
        ft = feats.get(imdb)
        if ft is None:
            skipped.append(imdb)
            continue
        n = len(scenes)

        def mean(col):
            return sum(num(s[col]) for s in scenes) / n
        heavy = sum(1 for s in scenes if max(num(s["stunts"]), num(s["vfx"]), num(s["set_complexity"])) >= 1.5)
        scene_count = int(num(ft["scene_count"], n))
        raw = {
            "shoot_days": scene_count + 2 * heavy,
            "cast_size": num(ft["speaking_characters"]),
            "locations": num(ft["unique_locations"]),
            "exteriors_nights": num(ft["ext_share"]) + num(ft["night_share"]),
            "sets": mean("set_complexity"),
            "stunts": mean("stunts"),
            "vfx": mean("vfx"),
            "special_conditions": sum(sum(num(s["cond_" + c]) for c in CONDS) for s in scenes) / n,
            "costume_makeup": mean("costume_makeup"),
            "music": mean("music"),
        }
        rows.append({
            "imdb_id": imdb, "as_of_date": ft.get("as_of_date", ""), "movie_name": ft.get("movie_name", ""),
            "year": ft.get("year", ""), "year_split": ft.get("year_split", ""),
            "_raw": raw, "n_scenes_tagged": n, "scene_count": scene_count,
            "tag_coverage": round(n / scene_count, 3) if scene_count else 0,
            "min_runs_used": min(int(num(s["runs_used"])) for s in scenes),
            "flag_low_confidence": ft.get("flag_low_confidence", ""), "parse_mode": ft.get("parse_mode", ""),
            "prompt_version": scenes[0].get("prompt_version", ""), "tagger_version": scenes[0].get("tagger_version", ""),
            "model": scenes[0].get("model", ""),
        })

    if os.path.exists(args.cutoffs) and not args.refit:
        cutoffs = json.load(open(args.cutoffs))
        print("Using saved cut points from", args.cutoffs)
    else:
        cutoffs = {}
        for d in DRIVERS:
            vals = sorted(r["_raw"][d] for r in rows)
            cutoffs[d] = [percentile(vals, p) for p in (0.2, 0.4, 0.6, 0.8)]
        json.dump(cutoffs, open(args.cutoffs, "w"), indent=1)
        print("Computed new cut points from this set and saved them to", args.cutoffs)

    out_rows = []
    for r in sorted(rows, key=lambda x: x["imdb_id"]):
        o = {k: v for k, v in r.items() if k != "_raw"}
        for d in DRIVERS:
            o["raw_" + d] = round(r["_raw"][d], 4)
            o["score_" + d] = to_score(r["_raw"][d], cutoffs[d])
        o["driver_version"] = DRIVER_VERSION
        out_rows.append(o)
    if not out_rows:
        print("No scripts to write. Check the file names.")
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    dpath = "outputs/feature_dictionary_p1_drivers_v1.csv"
    with open(dpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "meaning", "how_computed", "range", "source", "version"])
        for d, (meaning, how) in DRIVERS.items():
            w.writerow(["score_" + d, meaning, "1 to 5 by comparing raw_%s with saved cut points (20th/40th/60th/80th percentiles)" % d, "1-5", "scene tags + parser", DRIVER_VERSION])
            w.writerow(["raw_" + d, meaning + " (raw number)", how, "number >= 0", "scene tags + parser", DRIVER_VERSION])
        for col, meaning in [("n_scenes_tagged", "scenes with tags"), ("tag_coverage", "scenes tagged / scene_count; low means missing tags"),
                             ("min_runs_used", "fewest runs averaged for any scene (3 is full)")]:
            w.writerow([col, meaning, "from tagger output", "", "scene tags", DRIVER_VERSION])

    print("\nScripts scored: %d | missing from features file (skipped): %d" % (len(out_rows), len(skipped)))
    low_cov = [o for o in out_rows if o["tag_coverage"] < 0.95]
    print("Scripts with under 95%% of scenes tagged: %d" % len(low_cov))
    print("\nHow many scripts got each score (1 to 5):")
    for d in DRIVERS:
        counts = [sum(1 for o in out_rows if o["score_" + d] == k) for k in range(1, 6)]
        print("   %-19s %s" % (d, "  ".join("%d:%3d" % (k + 1, c) for k, c in enumerate(counts))))
    print("\nSaved:", os.path.abspath(out_path))
    print("Saved:", os.path.abspath(dpath))


main()
