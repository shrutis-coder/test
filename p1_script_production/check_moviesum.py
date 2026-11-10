import csv
import json
import os
import re
from collections import Counter

here = os.path.dirname(os.path.abspath(__file__))
raw = os.path.join(here, "..", "..", "raw_data")

rows = []
keys_seen = set()
for split in ["train", "val", "test"]:
    path = os.path.join(raw, split + ".jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            keys_seen.update(d.keys())
            name = d.get("movie_name", "") or ""
            m = re.search(r"_(\d{4})$", name)
            script = d.get("script", "") or ""
            rows.append({
                "imdb_id": d.get("imdb_id", "") or "",
                "movie_name": name,
                "year": m.group(1) if m else "",
                "split": split,
                "scene_tags": script.count("<scene>"),
                "word_count": len(script.split()),
            })

out_path = os.path.join(raw, "moviesum_index.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

ids = [r["imdb_id"] for r in rows]
bad_ids = [i for i in ids if not re.match(r"^tt\d+$", i)]
dupes = [i for i, c in Counter(ids).items() if c > 1]
no_year = [r["movie_name"] for r in rows if not r["year"]]
short = [r["movie_name"] for r in rows if r["word_count"] < 1000]
years = sorted(int(r["year"]) for r in rows if r["year"])

print("Columns in file:", sorted(keys_seen))
print("Total scripts:", len(rows))
print("Bad or missing IMDb IDs:", len(bad_ids))
print("Duplicate IMDb IDs:", len(dupes))
print("Names with no year:", len(no_year))
print("Scripts under 1,000 words:", len(short))
if years:
    print("Release years:", years[0], "to", years[-1])
print("Saved table to:", out_path)
