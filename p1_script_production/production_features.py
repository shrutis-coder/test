"""Production features v1 (P1, rules only, no LLM).

One row per script: genre, length, scene and location counts, INT/EXT and day/night
shares, a rough language-based rating guess, and flags for scripts that parsed badly.
The 10 budget driver scores are NOT here yet - they come from the Scene tagger later.
"""
import csv
import glob
import json
import os
import re
import statistics
import sys

FEATURES_VERSION = "production_features_v1"
WORDS_PER_PAGE = 200  # rule of thumb: about 200 words per screenplay page, 1 page = 1 minute

STRONG = re.compile(r"\bfuck\w*", re.I)
OTHER = re.compile(r"\b(shit\w*|bitch\w*|asshole\w*|cock\w*|pussy|pussies|cunt\w*|dick\w*|bastard\w*|motherfuck\w*|whore\w*|slut\w*)\b", re.I)
MILD = re.compile(r"\b(damn\w*|hell|ass|asses|crap\w*|goddamn\w*|piss\w*)\b", re.I)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def rating_guess(strong, other, mild):
    if strong >= 2:
        return "R"
    if strong == 1 or other >= 3:
        return "PG-13"
    if other >= 1 or mild >= 3:
        return "PG"
    return "G"


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main():
    sys.stdout.reconfigure(errors="replace")
    here = os.path.dirname(os.path.abspath(__file__))
    raw = os.path.join(here, "..", "..", "raw_data")
    out_dir = os.path.join(here, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    parsed_path = os.path.join(raw, "scene_parser_scripts.csv")
    if not os.path.exists(parsed_path):
        print("scene_parser_scripts.csv not found. Run scene_parser.py first.")
        return
    with open(parsed_path, encoding="utf-8") as f:
        parsed = {r["imdb_id"]: r for r in csv.DictReader(f)}

    # words and language counts from the real script text (tags removed)
    text_feats = {}
    with open(os.path.join(raw, "moviesum_clean.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            t = re.sub(r"<[^>]+>", " ", d["script"])
            words = len(t.split())
            strong, other, mild = len(STRONG.findall(t)), len(OTHER.findall(t)), len(MILD.findall(t))
            text_feats[d["imdb_id"]] = (words, strong, other, mild)

    cands = [p for p in glob.glob(os.path.join(raw, "*.csv")) if "tmdb" in os.path.basename(p).lower()]
    if not cands:
        print("No TMDB csv found in raw_data.")
        return
    tmdb_path = max(cands, key=os.path.getsize)
    print("Using:", tmdb_path)
    csv.field_size_limit(10**9)
    tm = {}
    with open(tmdb_path, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for c in ("imdb_id", "genres", "runtime", "revenue"):
            if c not in rd.fieldnames:
                print("TMDB file is missing column:", c)
                return
        for row in rd:
            i = row["imdb_id"]
            if i in parsed:
                old = tm.get(i)
                if old is None or num(row["revenue"]) > num(old["revenue"]):
                    tm[i] = row

    all_genres = sorted({g.strip() for r in tm.values() for g in (r["genres"] or "").split(",") if g.strip()})
    genre_cols = ["genre_" + slug(g) for g in all_genres]

    rows = []
    for iid, p in parsed.items():
        words, strong, other, mild = text_feats[iid]
        t = tm.get(iid)
        gl = [g.strip() for g in (t["genres"] if t else "").split(",") if g.strip()]
        n = int(p["scene_count"])
        nloc = int(p["unique_locations"])
        known_tod = int(p["day_count"]) + int(p["night_count"])
        row = {
            "imdb_id": iid,
            "as_of_date": p["as_of_date"],
            "movie_name": p["movie_name"],
            "year": p["year"],
            "year_split": p["year_split"],
            "genres": ", ".join(gl),
            "genre_primary": gl[0] if gl else "",
            "script_words_clean": words,
            "est_pages": round(words / WORDS_PER_PAGE, 1),
            "est_runtime_min": round(words / WORDS_PER_PAGE, 1),
            "scene_count": n,
            "int_share": p["int_share"],
            "ext_share": p["ext_share"],
            "night_share": p["night_share"],
            "unknown_time_share": round(int(p["unknown_time_count"]) / n, 4) if n else "",
            "unique_locations": nloc,
            "scenes_per_location": round(n / nloc, 2) if nloc else "",
            "speaking_characters": p["unique_speaking_characters"],
            "dialogue_lines": p["dialogue_lines"],
            "description_words": p["description_words"],
            "profanity_strong_per_1k": round(1000.0 * strong / max(1, words), 3),
            "profanity_other_per_1k": round(1000.0 * other / max(1, words), 3),
            "profanity_mild_per_1k": round(1000.0 * mild / max(1, words), 3),
            "est_rating_language": rating_guess(strong, other, mild),
            "parse_mode": p["parse_mode"],
            "flag_low_confidence": 1 if (p["parse_mode"] != "strict" or n < 30) else 0,
            "flag_genre_missing": 0 if gl else 1,
            "features_version": FEATURES_VERSION,
        }
        for g, col in zip(all_genres, genre_cols):
            row[col] = 1 if g in gl else 0
        rows.append(row)

    out_path = os.path.join(out_dir, "p1_production_features_v1.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # feature dictionary (everyone's new features go in the shared dictionary before use)
    D = []

    def add(name, group, definition, source, pre="yes"):
        D.append({"name": name, "group": group, "definition": definition, "source": source,
                  "pre_release": pre, "owner": "P1", "version": FEATURES_VERSION})

    add("imdb_id", "key", "IMDb ID, the join key for every table", "MovieSum")
    add("as_of_date", "key", "Release date of the film; features must only use data before it", "TMDB release_date")
    add("movie_name", "key", "Title with year, as in MovieSum", "MovieSum")
    add("year", "key", "Release year parsed from the MovieSum title", "MovieSum")
    add("year_split", "key", "train through 2017, validate 2018-2019, skip 2020-2021, test 2022 on", "rule on year")
    add("genres", "production", "Genre list for the film (stands in for a script-based genre guess)", "TMDB genres", "yes (proxy)")
    add("genre_primary", "production", "First genre listed", "TMDB genres", "yes (proxy)")
    add("script_words_clean", "production", "Words in the script with markup tags removed", "script text")
    add("est_pages", "production", "script words / %d (rule of thumb)" % WORDS_PER_PAGE, "script text")
    add("est_runtime_min", "production", "Estimated runtime in minutes: 1 page = 1 minute", "script text")
    add("scene_count", "production", "Number of scene headings found by the Scene parser", "scene_parser_v2")
    add("int_share", "production", "Share of scenes inside (INT/EXT counts half each)", "scene_parser_v2")
    add("ext_share", "production", "Share of scenes outside", "scene_parser_v2")
    add("night_share", "production", "Night scenes / (day + night scenes); blank if no times known", "scene_parser_v2")
    add("unknown_time_share", "production", "Share of scenes whose time of day could not be read", "scene_parser_v2")
    add("unique_locations", "production", "Number of different location names in scene headings", "scene_parser_v2")
    add("scenes_per_location", "production", "scene_count / unique_locations; high means few repeated sets", "scene_parser_v2")
    add("speaking_characters", "production", "Number of different speaking characters", "scene_parser_v2")
    add("dialogue_lines", "production", "Number of dialogue blocks", "scene_parser_v2")
    add("description_words", "production", "Words in scene descriptions (action text)", "scene_parser_v2")
    add("profanity_strong_per_1k", "production", "F-word forms per 1,000 words", "script text")
    add("profanity_other_per_1k", "production", "Other strong swear words per 1,000 words", "script text")
    add("profanity_mild_per_1k", "production", "Mild swear words (damn, hell, ass...) per 1,000 words", "script text")
    add("est_rating_language", "production", "Rough G/PG/PG-13/R guess from language only, NOT validated", "script text")
    add("parse_mode", "flag", "strict, loose, text_fallback or no_headings: how the scenes were found", "scene_parser_v2")
    add("flag_low_confidence", "flag", "1 if parse_mode is not strict or fewer than 30 scenes were found", "scene_parser_v2")
    add("flag_genre_missing", "flag", "1 if no TMDB genre was found", "TMDB")
    for col in genre_cols:
        add(col, "production", "1 if the film has this genre (multi-hot)", "TMDB genres", "yes (proxy)")
    add("features_version", "key", "Version of the feature code that made the row", "this file")
    dict_path = os.path.join(out_dir, "feature_dictionary_p1.csv")
    with open(dict_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(D[0].keys()))
        w.writeheader()
        w.writerows(D)
    missing = [c for c in rows[0].keys() if c not in {d["name"] for d in D}]
    if missing:
        print("WARNING: not in dictionary:", missing)

    # report
    print("Rows written:", len(rows), "| columns:", len(rows[0]))
    print("Saved:", out_path)
    print("Saved:", dict_path)
    print("Genre columns:", len(genre_cols), "| scripts with no genre:", sum(r["flag_genre_missing"] for r in rows))
    print("Low-confidence scripts (flag):", sum(r["flag_low_confidence"] for r in rows))
    from collections import Counter
    print("Rating guess counts:", dict(Counter(r["est_rating_language"] for r in rows)))
    pairs = [(r["est_runtime_min"], num(tm[r["imdb_id"]]["runtime"])) for r in rows
             if r["imdb_id"] in tm and 60 <= num(tm[r["imdb_id"]]["runtime"]) <= 240]
    if len(pairs) > 10:
        a = [x for x, _ in pairs]
        b = [y for _, y in pairs]
        print("Runtime check on {} films: correlation {:.2f} | median real minutes per estimated minute {:.2f}".format(
            len(pairs), statistics.correlation(a, b), statistics.median(y / x for x, y in pairs if x)))
    ws = sorted(r["script_words_clean"] for r in rows)
    print("Script words (cleaned): min {} | median {} | max {}".format(ws[0], ws[len(ws) // 2], ws[-1]))
    print("First 3 rows:")
    for r in rows[:3]:
        print("  ", {k: r[k] for k in list(r)[:12]})


if __name__ == "__main__":
    main()
