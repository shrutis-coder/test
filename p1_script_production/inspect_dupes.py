import csv
import os
from collections import defaultdict

here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "..", "..", "raw_data", "moviesum_index.csv")

groups = defaultdict(list)
with open(path, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        groups[r["imdb_id"]].append(r)

for imdb_id, rs in groups.items():
    if len(rs) > 1:
        print(imdb_id)
        for r in rs:
            print("    ", r["movie_name"], "|", r["split"], "|", r["word_count"], "words")
