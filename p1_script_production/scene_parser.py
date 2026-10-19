"""Scene parser (P1, rules only, no LLM).

Splits MovieSum scripts into scenes and counts INT/EXT and day/night.
Other files (like the Scene tagger) can do:  from scene_parser import parse_script
"""
import csv
import json
import os
import random
import re
from collections import Counter

PARSER_VERSION = "scene_parser_v1"

SLUG = re.compile(
    r"^(?:\d+[A-Z]{0,2}[.)]?\s+)?"
    r"(INT\.?\s*/\s*EXT\.?|EXT\.?\s*/\s*INT\.?|I/E\.?|INT\.?|EXT\.?|EST\.?)"
    r"(?=[\s.\-]|$)\s*[.\-]?\s*(.*)$",
    re.I,
)
SPLIT = re.compile(r"\s+[-\u2013\u2014]+\s+|\s*(?:-{2,}|[\u2013\u2014]+)\s*")
NIGHT_W = {"NIGHT", "NIGHTTIME", "EVENING", "DUSK", "MIDNIGHT", "SUNSET", "TWILIGHT"}
DAY_W = {"DAY", "DAYTIME", "MORNING", "AFTERNOON", "DAWN", "NOON", "SUNRISE", "MIDDAY", "DAYBREAK"}
INHERIT_W = {"CONTINUOUS", "SAME", "LATER", "MOMENTS", "CONT'D"}


def clean(s):
    return re.sub(r"\s+", " ", s).strip()


def kind_of(tag):
    t = re.sub(r"[^A-Z/]", "", tag.upper())
    if "/" in t:
        return "MIXED"
    if t.startswith("INT") or t == "IE":
        return "INT"
    return "EXT"


def day_night_of(time_text):
    toks = re.findall(r"[A-Z']+", time_text.upper())
    for t in toks:
        if t in NIGHT_W:
            return "night", False
        if t in DAY_W:
            return "day", False
    if any(t in INHERIT_W for t in toks):
        return "inherit", True
    return "unknown", False


def parse_script(text):
    scenes = []
    prev = "unknown"
    for n, raw in enumerate(re.findall(r"<scene>(.*?)</scene>", text, re.S), start=1):
        sd = re.search(r"<stage_direction>(.*?)</stage_direction>", raw, re.S)
        heading = clean(sd.group(1)) if sd else ""
        m = SLUG.match(heading) if heading else None
        speakers = set()
        for c in re.findall(r"<character>(.*?)</character>", raw, re.S):
            c = clean(re.sub(r"\(.*?\)", "", c)).upper()
            if c:
                speakers.add(c)
        desc = " ".join(re.findall(r"<scene_description>(.*?)</scene_description>", raw, re.S))
        s = {
            "scene_no": n,
            "is_heading": 1 if m else 0,
            "heading": heading,
            "int_ext": "",
            "location": "",
            "time_text": "",
            "day_night": "",
            "inherited": 0,
            "n_speakers": len(speakers),
            "speakers": speakers,
            "dialogue_lines": len(re.findall(r"<dialogue>", raw)),
            "desc_words": len(desc.split()),
            "text": raw,
        }
        if m:
            s["int_ext"] = kind_of(m.group(1))
            rest = m.group(2)
            parts = [p for p in SPLIT.split(rest) if p.strip()]
            first = parts[0] if parts else ""
            later = " ".join(parts[1:])
            if not later:
                words = first.split()
                if len(words) > 1:
                    last = words[-1]
                    if day_night_of(last)[0] != "unknown":
                        later = last
                        first = " ".join(words[:-1])
            if not later and len(first.split()) == 1 and day_night_of(first)[0] != "unknown":
                later, first = first, ""
            s["location"] = clean(re.sub(r"\(.*?\)", "", first)).upper()
            s["time_text"] = clean(later)
            dn, inh = day_night_of(later)
            if dn == "inherit":
                dn, inh = prev, True
            s["day_night"] = dn
            s["inherited"] = 1 if inh else 0
            prev = dn
        scenes.append(s)
    return scenes


def summarize(scenes, script_text):
    h = [s for s in scenes if s["is_heading"]]
    n = len(h)
    kinds = Counter(s["int_ext"] for s in h)
    dn = Counter(s["day_night"] for s in h)
    all_speakers = set()
    for s in scenes:
        all_speakers |= s["speakers"]
    known = dn["day"] + dn["night"]
    return {
        "scene_count": n,
        "non_heading_blocks": len(scenes) - n,
        "int_count": kinds["INT"],
        "ext_count": kinds["EXT"],
        "int_ext_mixed_count": kinds["MIXED"],
        "int_share": round((kinds["INT"] + 0.5 * kinds["MIXED"]) / n, 4) if n else "",
        "ext_share": round((kinds["EXT"] + 0.5 * kinds["MIXED"]) / n, 4) if n else "",
        "day_count": dn["day"],
        "night_count": dn["night"],
        "unknown_time_count": dn["unknown"],
        "night_share": round(dn["night"] / known, 4) if known else "",
        "unique_locations": len({s["location"] for s in h if s["location"]}),
        "unique_speaking_characters": len(all_speakers),
        "dialogue_lines": sum(s["dialogue_lines"] for s in scenes),
        "description_words": sum(s["desc_words"] for s in scenes),
        "script_words": len(script_text.split()),
    }


def plain_text(script):
    t = re.sub(r"</(stage_direction|scene_description|character|dialogue|parenthetical)>", "\n", script)
    t = t.replace("<scene>", "\n\n--------------- NEW SCENE BLOCK ---------------\n")
    t = re.sub(r"<[^>]+>", "", t)
    return re.sub(r"\n{3,}", "\n\n", t)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    raw = os.path.join(here, "..", "..", "raw_data")
    as_of = {}
    p = os.path.join(raw, "moviesum_with_outcomes.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                as_of[r["imdb_id"]] = r["as_of_date"]
    else:
        print("NOTE: moviesum_with_outcomes.csv not found, so as_of_date will be blank.")

    scene_rows, script_rows, store = [], [], {}
    non_heading = Counter()
    with open(os.path.join(raw, "moviesum_clean.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            iid = d["imdb_id"]
            sc = parse_script(d["script"])
            for s in sc:
                if not s["is_heading"] and s["heading"]:
                    non_heading[s["heading"].upper()[:40]] += 1
                scene_rows.append({
                    "imdb_id": iid, "as_of_date": as_of.get(iid, ""),
                    "scene_no": s["scene_no"], "is_heading": s["is_heading"],
                    "heading": s["heading"], "int_ext": s["int_ext"],
                    "location": s["location"], "time_text": s["time_text"],
                    "day_night": s["day_night"], "inherited": s["inherited"],
                    "n_speakers": s["n_speakers"], "dialogue_lines": s["dialogue_lines"],
                    "desc_words": s["desc_words"], "parser_version": PARSER_VERSION,
                })
            row = {"imdb_id": iid, "as_of_date": as_of.get(iid, ""),
                   "movie_name": d["movie_name"], "year": d["year"],
                   "year_split": d["year_split"]}
            row.update(summarize(sc, d["script"]))
            row["parser_version"] = PARSER_VERSION
            script_rows.append(row)
            store[iid] = (d, sc)

    def write(name, rows):
        path = os.path.join(raw, name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return path

    print("Scripts parsed:", len(script_rows))
    print("Scenes found:", len(scene_rows))
    print("Saved:", write("scene_parser_scenes.csv", scene_rows))
    print("Saved:", write("scene_parser_scripts.csv", script_rows))

    heads = [r for r in scene_rows if r["is_heading"]]
    unk = sum(1 for r in heads if r["day_night"] == "unknown")
    print("Heading scenes:", len(heads), "| time of day unknown:", unk,
          "({:.1f}%)".format(100.0 * unk / max(1, len(heads))))
    zero = [r["movie_name"] for r in script_rows if r["scene_count"] == 0]
    print("Scripts with 0 heading scenes:", len(zero), zero[:10])
    few = [r["movie_name"] for r in script_rows if 0 < r["scene_count"] < 30]
    print("Scripts with under 30 heading scenes:", len(few), few[:10])
    print("Most common stage directions that are NOT scene headings:")
    for k, v in non_heading.most_common(15):
        print("   ", v, "x", k)

    # 20 test scripts; 5 of them get a hand-check sheet
    ids = sorted(store.keys())
    pick = random.Random(7).sample(ids, min(20, len(ids)))
    write("parser_test_20.csv", [r for r in script_rows if r["imdb_id"] in set(pick)])
    hc_dir = os.path.join(raw, "handcheck")
    os.makedirs(hc_dir, exist_ok=True)
    rnd = random.Random(11)
    lines = []
    for iid in pick[:5]:
        d, sc = store[iid]
        safe = re.sub(r"[^A-Za-z0-9]+", "_", d["movie_name"])
        with open(os.path.join(hc_dir, safe + ".txt"), "w", encoding="utf-8") as f:
            f.write(plain_text(d["script"]))
        summ = next(r for r in script_rows if r["imdb_id"] == iid)
        lines.append("=" * 100)
        lines.append("{}  ({})   script file: handcheck/{}.txt".format(d["movie_name"], iid, safe))
        lines.append("heading scenes {} | INT {} | EXT {} | INT/EXT {} | day {} | night {} | unknown {}".format(
            summ["scene_count"], summ["int_count"], summ["ext_count"], summ["int_ext_mixed_count"],
            summ["day_count"], summ["night_count"], summ["unknown_time_count"]))
        heads_here = [s for s in sc if s["is_heading"]]
        chosen = heads_here[:15] + rnd.sample(heads_here[15:], min(10, max(0, len(heads_here) - 15)))
        lines.append("--- 25 scenes to check: scene# | INT/EXT | day/night (* = copied from earlier scene) | place | heading as written")
        for s in chosen:
            lines.append("#{:<4}| {:<7}| {:<8}| {:<28}| {}".format(
                s["scene_no"], s["int_ext"], s["day_night"] + ("*" if s["inherited"] else ""),
                s["location"][:27], s["heading"][:70]))
        odd = [s for s in sc if not s["is_heading"]][:6]
        lines.append("--- first blocks that were NOT counted as scenes (is that right?):")
        for s in odd:
            lines.append("#{:<4}| {}".format(s["scene_no"], (s["heading"] or "(no heading) " + clean(s["text"])[:60])[:90]))
        lines.append("")
    hp = os.path.join(raw, "parser_handcheck.txt")
    with open(hp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Saved:", hp)
    print("Saved: parser_test_20.csv and the handcheck folder")


if __name__ == "__main__":
    main()
