import csv
import json
import os
from collections import Counter

here = os.path.dirname(os.path.abspath(__file__))
raw = os.path.join(here, "..", "..", "raw_data")

best = {}
all_rows = []
with open(os.path.join(raw, "moviesum_index.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        r["word_count"] = int(r["word_count"])
        all_rows.append(r)
        cur = best.get(r["imdb_id"])
        if cur is None or r["word_count"] > cur["word_count"]:
            best[r["imdb_id"]] = r

keep = {(r["imdb_id"], r["movie_name"]) for r in best.values()}
removed = [r["movie_name"] for r in all_rows if (r["imdb_id"], r["movie_name"]) not in keep]


def year_split(y):
    if y == "":
        return "unknown"
    y = int(y)
    if y <= 2017:
        return "train"
    if y <= 2019:
        return "validate"
    if y <= 2021:
        return "skip"
    return "test"


for r in best.values():
    r["year_split"] = year_split(r["year"])

out_csv = os.path.join(raw, "moviesum_clean.csv")
fields = ["imdb_id", "movie_name", "year", "year_split", "scene_tags", "word_count"]
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(best.values())

written = set()
out_jsonl = os.path.join(raw, "moviesum_clean.jsonl")
with open(out_jsonl, "w", encoding="utf-8") as out:
    for split in ["train", "val", "test"]:
        with open(os.path.join(raw, split + ".jsonl"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                key = (d["imdb_id"], d["movie_name"])
                if key in keep and key not in written:
                    written.add(key)
                    r = best[d["imdb_id"]]
                    out.write(json.dumps({
                        "imdb_id": d["imdb_id"],
                        "movie_name": d["movie_name"],
                        "year": r["year"],
                        "year_split": r["year_split"],
                        "script": d["script"],
                    }) + "\n")

print("Scripts kept:", len(written))
print("Duplicates removed:", len(removed))
for name in removed:
    print("   removed:", name)
print("Split counts:", dict(Counter(r["year_split"] for r in best.values())))
print("Saved:", out_csv)
print("Saved:", out_jsonl)
