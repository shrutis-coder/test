import json
import os
import re
from collections import Counter

here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "..", "..", "raw_data", "moviesum_clean.jsonl")

tags = Counter()
slug = re.compile(r"^\s*(INT|EXT|I/E|INT\./EXT|EST)\b", re.I)
n_sd = n_slug = 0
odd = []
sample = None
with open(path, encoding="utf-8") as f:
    for n, line in enumerate(f):
        d = json.loads(line)
        s = d["script"]
        tags.update(re.findall(r"</?([A-Za-z_]+)>", s))
        for sd in re.findall(r"<stage_direction>(.*?)</stage_direction>", s, re.S):
            n_sd += 1
            if slug.match(sd):
                n_slug += 1
            elif len(odd) < 10:
                odd.append(re.sub(r"\s+", " ", sd)[:80])
        if n == 10:
            sample = d
        if n >= 49:
            break

print("Tag counts over 50 scripts:", dict(tags))
print("stage_direction blocks:", n_sd, "| that look like sluglines:", n_slug)
print("Examples that do NOT look like sluglines:")
for o in odd:
    print("   ", o)
print("---- sample:", sample["movie_name"])
print(re.sub(r"\s+", " ", sample["script"])[:1500])
