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
  4. legs = domestic / opening, kept only if between 1 and 40.

Run from the repo root (after clean_and_label.py has made the labeled table):

    python src/comps/pull_box_office.py

Saves raw/comps/wikipedia_box_office.csv (with the source sentences, so any
number can be checked). clean_and_label.py then fills the legs column from it.
"""

import argparse
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
DOMESTIC = re.compile(
    r"\b(?:grossed|earned|made|took in|grossing|earning)\s+(?:a total of\s+|over\s+|more than\s+|about\s+)?"
    + MONEY + r"\s+(?:domestically|in (?:the )?(?:United States and Canada|U\.?S\.? and Canada|North America|"
    r"United States|US|U\.S\.)\b)", re.I)
OPENING = [
    re.compile(r"\b(?:open(?:ed|s)?|debut(?:ed|s)?)\s+(?:to|for|with)\s+(?:a\s+)?(?:total of\s+)?" + MONEY, re.I),
    re.compile(MONEY + r"\s+(?:in|during|over|from)\s+its\s+(?:domestic\s+)?opening\s+weekend", re.I),
    re.compile(r"\bopening weekend\s+(?:of|total of|gross of|haul of|take of)\s+" + MONEY, re.I),
]
SKIP_SENTENCE = re.compile(
    r"projected|expected to|tracking|forecast|predicted|estimated to|anticipated|"
    r"worldwide|global|overseas|international|other territories|opening day|first day|"
    r"\bin (?:China|Japan|South Korea|Korea|India|the United Kingdom|the UK|France|Germany|Italy|"
    r"Spain|Australia|Mexico|Brazil|Russia)\b", re.I)
SECTION = re.compile(r"^(={2,4})\s*(?:Box[ -]office|Commercial performance|Commercial)\b.*?\1\s*$", re.I | re.M)
MIN_LEGS, MAX_LEGS = 1.0, 40.0


def to_dollars(num, unit):
    return float(num.replace(",", "")) * SCALE[unit.lower() if unit else None]


def box_office_section(wikitext):
    m = SECTION.search(wikitext or "")
    if not m:
        return None
    level = len(m.group(1))
    rest = wikitext[m.end():]
    nxt = re.search(r"^={2,%d}[^=].*?={2,%d}\s*$" % (level, level), rest, re.M)
    return rest[:nxt.start()] if nxt else rest


def parse_box_office(wikitext):
    """(domestic_gross, opening_weekend, domestic_sentence, opening_sentence) from an article."""
    section = box_office_section(wikitext)
    if not section:
        return None, None, None, None
    text = clean_value(section.replace("\n", " "))
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text)
    domestic = opening = dom_s = open_s = None
    for s in sentences:
        if domestic is None:
            m = DOMESTIC.search(s)
            if m and not re.search(r"projected|expected to|tracking", s, re.I):
                domestic, dom_s = to_dollars(m.group(1), m.group(2)), s
        if opening is None and not SKIP_SENTENCE.search(s):
            for pat in OPENING:
                m = pat.search(s)
                if m:
                    opening, open_s = to_dollars(m.group(1), m.group(2)), s
                    break
        if domestic is not None and opening is not None:
            break
    return domestic, opening, dom_s, open_s


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
    args = p.parse_args(argv)

    ids = pd.read_csv(args.labeled, usecols=["imdb_id"])["imdb_id"].dropna().unique()
    print(f"Looking up Wikipedia articles for {len(ids):,} labeled films...")
    titles = article_titles(ids)
    print(f"{len(titles):,} have an English Wikipedia article. Reading them...")
    texts = fetch_wikitext(titles["wiki_title"].tolist())

    parsed = titles["wiki_title"].map(lambda t: parse_box_office(texts.get(t, "")))
    titles[["domestic_gross", "opening_weekend", "domestic_sentence", "opening_sentence"]] = \
        pd.DataFrame(parsed.tolist(), index=titles.index)
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
