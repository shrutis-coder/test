import json
import os
import random
import re
import sys
from collections import Counter

from scene_parser import parse_script, clean

sys.stdout.reconfigure(errors="replace")
here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "..", "..", "raw_data", "moviesum_clean.jsonl")

zero = []
unk_heads = []
unk_time_texts = Counter()
nonhead = []
variant = re.compile(r"\b(INT|EXT|INTERIOR|EXTERIOR)\b", re.I)

with open(path, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        sc = parse_script(d["script"])
        if not any(s["is_heading"] for s in sc):
            zero.append((d, sc))
        for s in sc:
            if s["is_heading"] and s["day_night"] == "unknown":
                unk_heads.append(s["heading"])
                unk_time_texts[s["time_text"].upper()] += 1
            elif (not s["is_heading"]) and s["heading"] and variant.search(s["heading"]):
                nonhead.append(s["heading"])

print("=" * 80)
print("A) SCRIPTS WITH ZERO HEADINGS:", len(zero), "- first 6, first 4 scene blocks of each")
for d, sc in zero[:6]:
    tags = len(re.findall(r"<stage_direction>", d["script"]))
    print("\n##", d["movie_name"], "| scene blocks:", len(sc), "| stage_direction tags:", tags)
    for s in sc[:4]:
        print("   block", s["scene_no"], "| heading:", (s["heading"] or "(none)")[:70],
              "| text:", clean(re.sub(r"<[^>]+>", " ", s["text"]))[:110])

print("\n" + "=" * 80)
print("B) HEADINGS WITH UNKNOWN TIME OF DAY:", len(unk_heads))
print("   with NO time text at all:", unk_time_texts[""])
print("   most common time texts that were NOT understood:")
for k, v in unk_time_texts.most_common(30):
    if k:
        print("     ", v, "x", k[:50])
random.Random(5).shuffle(unk_heads)
print("   20 random examples:")
for h in unk_heads[:20]:
    print("     ", h[:90])

print("\n" + "=" * 80)
print("C) NON-HEADINGS THAT CONTAIN INT/EXT/INTERIOR/EXTERIOR:", len(nonhead))
random.Random(9).shuffle(nonhead)
for h in nonhead[:40]:
    print("     ", h[:90])
