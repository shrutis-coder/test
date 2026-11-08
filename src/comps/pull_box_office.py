"""P5 Comps: opening weekend and domestic gross from Wikipedia, for the legs label.

legs = domestic (US + Canada) total gross / domestic opening weekend. A film
with good word of mouth keeps earning after opening (legs ~3-4); a front-loaded
one doesn't (~2). Box Office Mojo has these numbers but its terms forbid
scraping, so this reads them from the "Box office" section of each film's
English Wikipedia article, which almost always says, e.g.:

    "Get Out grossed $176 million in the United States and Canada ..."
    "It went on to open for $33.4 million, finishing first at the box office."

Steps:
  1. Look up the English Wikipedia article for every labeled film (Wikidata, by IMDb ID).
  2. Read each article's wikitext from the Wikipedia API (50 per request).
  3. In the Box office section, find the first domestic-total and opening-weekend
     sentence. US dollars only; skips projections ("expected to", "tracking")
     and sentences about other markets ("in China", "worldwide", "overseas").
  4. legs = domestic / opening, kept only if between 1.3 and 40 (see MIN_LEGS).

Run from the repo root (after clean_and_label.py has made the labeled table):

    python src/comps/pull_box_office.py

Saves raw/comps/wikipedia_box_office.csv (with the source sentences, so any
number can be checked). clean_and_label.py then fills the legs column from it.
"""

import argparse
import json
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pull_wikipedia import WDQS, TruncatedResponse, clean_value, fetch_wikitext, http  # noqa: E402

TITLES_BY_ID = """
SELECT ?imdb ?title WHERE {
  VALUES ?imdb { %s }
  ?f wdt:P345 ?imdb .
  ?article schema:about ?f ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?title .
}
"""

MONEY = r"(?:US)?\$\s?(\d[\d,]*(?:\.\d+)?)\s*(million|billion)?"
SCALE = {None: 1, "million": 1e6, "billion": 1e9}
US_PLACE = (r"(?:the\s+)?(?:United States and Canada|U\.?S\.? and Canada|US and Canada|North America|"
            r"United States|U\.S\.|US)\b")
DOMESTIC = [
    # "grossed $176 million in the United States and Canada" / "earned $40 million domestically"
    re.compile(r"\b(?:grossed|earned|made|took in|grossing|earning|making|taking in|collected)\s+"
               r"(?:a total of\s+|over\s+|more than\s+|about\s+|nearly\s+|almost\s+|roughly\s+)?"
               + MONEY + r"\s+(?:domestically\b|(?:in|from)\s+" + US_PLACE + r")", re.I),
    # "earned $40 million at the North American box office", "at the US and Canadian box office"
    re.compile(r"\b(?:grossed|earned|made|took in|grossing|earning|gross)\s+(?:of\s+|over\s+|more than\s+|about\s+)?"
               + MONEY + r"\s+at the\s+(?:domestic|North American|American|"
               r"U\.?S\.?(?: and Canadian)?|US(?: and Canadian)?)\s+box office", re.I),
    # "a domestic total of $40 million", "domestic gross of $40 million"
    re.compile(r"\bdomestic (?:total|gross|box office(?: total)?)\s+of\s+" + MONEY, re.I),
    # "... worldwide, including $40 million in North America" / "$1.4 million from the US and Canada"
    re.compile(MONEY + r"\s+(?:domestically\b|(?:in|from)\s+" + US_PLACE + r")", re.I),
]
# Filler between a verb and the amount: "with a total gross of", "to an estimated", "collected a total of".
CONNECT = (r"(?:a\s+|an\s+)?(?:total\s+|estimated\s+|reported\s+|combined\s+)?"
           r"(?:gross\s+|take\s+|haul\s+|earnings\s+|tally\s+)?(?:of\s+)?"
           r"(?:about\s+|over\s+|around\s+|approximately\s+|nearly\s+|roughly\s+)?")
# Tried in order, most specific first, so "grossed $108,617 for the weekend and went on to gross
# $22.5 million" yields the weekend figure, not the total.
OPENING = [
    # "$258,000 in six theaters during its opening weekend"
    re.compile(MONEY + r"[^.$]{0,50}?\b(?:in|during|over|from|on|for)\s+(?:its|the)\s+(?:\w+\s+){0,3}"
               r"(?:opening|first|debut)\s+weekend", re.I),
    # "opened on 1,062 screens and grossed $4,022,891 for the weekend"
    re.compile(MONEY + r"\s+(?:for|over)\s+(?:the|that)\s+weekend", re.I),
    # "In its opening weekend, the film grossed $9.1 million", "For its first weekend, it collected a total of $21.6 million"
    re.compile(r"\b(?:opening|first|debut)\s+weekend\b[^.$]{0,90}?\b(?:grossed|earned|made|took in|grossing|earning|"
               r"making|taking in|collected|collecting|with|of)\s+" + CONNECT + MONEY, re.I),
    # "went on to open for $33.4 million", "debuted at No. 1 ... with a total gross of $26.4 million",
    # "ended up debuting to $21.2 million" ("to gross $X" is a later total, so it's excluded)
    re.compile(r"\b(?:open(?:ed|s|ing)?|debut(?:ed|s|ing)?)\b[^.$]{0,120}?\b"
               r"(?:to(?!\s+gross)|for|with|grossing|earning|making|taking in|collecting)\s+" + CONNECT + MONEY, re.I),
]
NOT_WEEKEND = re.compile(r"^[^.]{0,40}?\b(?:first day|opening day|on (?:Friday|Thursday|Saturday)|previews|"
                         r"opening night|midnight)", re.I)
PROJECTION = re.compile(r"projected|expected to|tracking|forecast|predicted|estimated to|anticipated|"
                        r"was set to|was going to|initially", re.I)
OTHER_MARKET = re.compile(r"worldwide|global|overseas|international|other territories|other countries|"
                          r"\bin (?:China|Japan|South Korea|Korea|India|the United Kingdom|the UK|France|Germany|"
                          r"Italy|Spain|Australia|Mexico|Brazil|Russia|Hong Kong|Taiwan)\b", re.I)
US_MARKER = re.compile(US_PLACE + r"|domestic", re.I)
SECTION = re.compile(r"^(={2,4})\s*(?:Box[ -]office|Commercial performance|Commercial)\b.*?\1\s*$", re.I | re.M)
FALLBACK_SECTION = re.compile(r"^(={2,4})\s*(?:Release|Reception|Theatrical release|Theatrical run)\b.*?\1\s*$",
                              re.I | re.M)
# No real film earns only its opening weekend (front-loaded wide releases end near 1.5-2x), so
# legs under 1.3 means the same figure was read as both opening and total.
MIN_LEGS, MAX_LEGS = 1.3, 40.0
MAX_DOLLARS = 2e9   # above any film's domestic gross; catches typos like "$7,943,778 million"


def to_dollars(num, unit):
    return float(num.replace(",", "")) * SCALE[unit.lower() if unit else None]


def _section(wikitext, pattern):
    m = pattern.search(wikitext or "")
    if not m:
        return None
    level = len(m.group(1))
    rest = wikitext[m.end():]
    nxt = re.search(r"^={2,%d}[^=].*?={2,%d}\s*$" % (level, level), rest, re.M)
    return rest[:nxt.start()] if nxt else rest


def box_office_section(wikitext):
    """The Box office section; if the article has none, its Release / Reception section."""
    return _section(wikitext, SECTION) or _section(wikitext, FALLBACK_SECTION)


def first_money(patterns, sentence, check_after=None):
    for pat in patterns:
        for m in pat.finditer(sentence):
            if check_after and check_after.search(sentence[m.end():]):
                continue
            return to_dollars(m.group(1), m.group(2))
    return None


def parse_section(section):
    """(domestic_gross, opening_weekend, domestic_sentence, opening_sentence) from section wikitext."""
    if not section:
        return None, None, None, None
    text = clean_value(section.replace("\n", " "))
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text)
    domestic = opening = dom_s = open_s = None
    for s in sentences:
        if PROJECTION.search(s):
            continue
        is_opening = re.search(r"\bopen|\bdebut|first weekend", s, re.I)
        if domestic is None:
            # The verbless "$X in North America" form only counts outside opening-weekend sentences.
            pats = DOMESTIC if not is_opening else DOMESTIC[:3]
            value = first_money(pats, s)
            if value is not None:
                domestic, dom_s = value, s
        if opening is None and is_opening:
            if OTHER_MARKET.search(s) and not US_MARKER.search(s):
                continue
            value = first_money(OPENING, s, check_after=NOT_WEEKEND)
            if value is not None:
                opening, open_s = value, s
        if domestic is not None and opening is not None:
            break
    return domestic, opening, dom_s, open_s


def parse_box_office(wikitext):
    return parse_section(box_office_section(wikitext))


def article_titles(imdb_ids, batch=300):
    rows = []
    ids = sorted(set(imdb_ids))
    for i in range(0, len(ids), batch):
        values = " ".join(f'"{x}"' for x in ids[i:i + batch])
        for attempt in range(3):
            try:
                res = http(WDQS, {"query": TITLES_BY_ID % values}, accept="application/sparql-results+json")
                break
            except TruncatedResponse:
                time.sleep(10)
        else:
            print(f"  skipped {batch} IDs after repeated Wikidata timeouts")
            continue
        rows += [(b["imdb"]["value"], b["title"]["value"]) for b in res["results"]["bindings"]]
        done = min(i + batch, len(ids))
        if done % 3000 < batch or done == len(ids):
            print(f"  Wikidata: {done:,}/{len(ids):,} IDs looked up")
        time.sleep(1)
    return pd.DataFrame(rows, columns=["imdb_id", "wiki_title"]).drop_duplicates("imdb_id")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labeled", default="clean/comps/comps_labeled.csv")
    p.add_argument("--out", default="raw/comps/wikipedia_box_office.csv")
    p.add_argument("--cache", default="raw/comps/wikipedia_box_office_sections.jsonl",
                   help="saved Box office sections, so the parsing can be redone without downloading")
    p.add_argument("--from-cache", action="store_true", help="reparse the saved sections; no network")
    args = p.parse_args(argv)

    if args.from_cache:
        rows = [json.loads(line) for line in open(args.cache)]
        titles = pd.DataFrame(rows)[["imdb_id", "wiki_title"]]
        sections = {r["wiki_title"]: r["section"] for r in rows}
        print(f"Reparsing {len(titles):,} saved sections from {args.cache}")
    else:
        ids = pd.read_csv(args.labeled, usecols=["imdb_id"])["imdb_id"].dropna().unique()
        print(f"Looking up Wikipedia articles for {len(ids):,} labeled films...")
        titles = article_titles(ids)
        print(f"{len(titles):,} have an English Wikipedia article. Reading them...")
        texts = fetch_wikitext(titles["wiki_title"].tolist())
        sections = {t: box_office_section(texts.get(t, "")) for t in titles["wiki_title"]}
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        with open(args.cache, "w") as fh:
            for imdb_id, t in zip(titles["imdb_id"], titles["wiki_title"]):
                fh.write(json.dumps({"imdb_id": imdb_id, "wiki_title": t, "section": sections.get(t)}) + "\n")
        print(f"Saved the sections to {args.cache} (rerun with --from-cache to reparse without downloading)")

    parsed = titles["wiki_title"].map(lambda t: parse_section(sections.get(t)))
    titles[["domestic_gross", "opening_weekend", "domestic_sentence", "opening_sentence"]] = \
        pd.DataFrame(parsed.tolist(), index=titles.index)
    for col in ("domestic_gross", "opening_weekend"):
        titles.loc[titles[col] > MAX_DOLLARS, col] = float("nan")
    titles["legs"] = titles["domestic_gross"] / titles["opening_weekend"]
    bad = ~titles["legs"].between(MIN_LEGS, MAX_LEGS)
    titles.loc[bad, "legs"] = float("nan")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    titles.to_csv(args.out, index=False)
    print(f"\nSaved {len(titles):,} films to {args.out}")
    print(f"  domestic gross found: {int(titles['domestic_gross'].notna().sum()):,}, "
          f"opening weekend: {int(titles['opening_weekend'].notna().sum()):,}, "
          f"usable legs (between {MIN_LEGS:g} and {MAX_LEGS:g}): {int(titles['legs'].notna().sum()):,}")
    print(f"  median legs: {titles['legs'].median():.2f}")


if __name__ == "__main__":
    main()
