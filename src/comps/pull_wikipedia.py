"""P5 Comps: read budget and box office from Wikipedia film info boxes.

For recent films TMDB often has a budget but no gross (or the reverse), and
Wikidata only stores a fraction of what Wikipedia shows. Wikipedia's film info
box ("| budget =" and "| gross =") usually has both, even for mid-size films.

Steps:
  1. From the TMDB file, pick released films in the chosen years that are
     missing a budget or a gross (the candidates).
  2. Ask Wikidata which of them have an English Wikipedia article (by IMDb ID).
  3. Fetch those articles' wikitext from the Wikipedia API, 50 per request.
  4. Parse the info box: US dollars only, "$15-20 million" ranges keep the top,
     a gross labelled only as domestic/North America is skipped.

Usage (from the repo root):

    python src/comps/pull_wikipedia.py

Saves raw/comps/wikipedia_financials.csv; clean_and_label.py merges it with
--wikipedia. Wikipedia text is CC BY-SA and its API is meant for programs.
"""

import argparse
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

UA = "FilmCrowdGreenlightClassProject/1.0 (UC Berkeley student project; python urllib)"
WDQS = "https://query.wikidata.org/sparql"
WP_API = "https://en.wikipedia.org/w/api.php"
FILM_TYPES = "wd:Q11424 wd:Q202866 wd:Q24869 wd:Q93204"   # film, animated, feature, documentary

TITLE_QUERY = """
SELECT ?imdb ?title WHERE {
  ?f wdt:P31 %(type)s ; wdt:P577 ?d .
  FILTER(?d >= "%(start)sT00:00:00Z"^^xsd:dateTime && ?d < "%(end)sT00:00:00Z"^^xsd:dateTime)
  ?f wdt:P345 ?imdb .
  ?article schema:about ?f ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?title .
}
"""


def ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


CTX = ssl_context()


class TruncatedResponse(Exception):
    """The server sent a cut-off (unparseable) response, usually a query timeout."""


def http(url, data=None, accept="application/json", retries=4):
    body = urllib.parse.urlencode(data).encode() if data else None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers={"User-Agent": UA, "Accept": accept})
            with urllib.request.urlopen(req, timeout=120, context=CTX) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Wikidata's query service cuts the response off when a query hits its time limit.
                raise TruncatedResponse(url)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(20 * attempt)
                continue
            sys.exit(f"Request failed ({e.code}) for {url}: {e.read().decode('utf-8', 'replace')[:300]}")
        except urllib.error.URLError as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                sys.exit("Python can't verify HTTPS certificates. Run:  python3 -m pip install certifi")
            if attempt == retries:
                raise
            time.sleep(10 * attempt)


# ---- Info box parsing -------------------------------------------------------
REF = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
COMMENT = re.compile(r"<!--.*?-->", re.S)
NOTE_TEMPLATES = re.compile(r"\{\{\s*(efn|refn|sfn|r|cn|citation needed|ref label|NoteTag)\b[^{}]*\}\}", re.I)
USD_TEMPLATE = re.compile(r"\{\{\s*(?:US\$|USD)\s*\|\s*([^|{}]+)[^{}]*\}\}", re.I)
NBSP = re.compile(r"&nbsp;|\{\{\s*nbsp\s*\}\}| ", re.I)
LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")
ANY_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
TAG = re.compile(r"<[^>]+>")
# "$15 million", "US$1.2 billion", "$15–20 million", "$4,500,000". "A$", "CA$" etc. don't match.
AMOUNT = re.compile(r"(?<![A-Za-z])(?:US)?\$\s?(\d[\d,]*(?:\.\d+)?)(?:\s?[–—-]\s?\$?(\d[\d,]*(?:\.\d+)?))?\s*(million|billion)?",
                    re.I)
SCALE = {None: 1, "million": 1e6, "billion": 1e9}
DOMESTIC_ONLY = re.compile(r"domestic|north america|united states|\bUS\b|\bU\.S\.", re.I)


def infobox_field(wikitext, field):
    """Raw text of '| field = ...' inside the film info box (first line only)."""
    start = re.search(r"\{\{\s*Infobox film", wikitext, re.I)
    if not start:
        return None
    m = re.search(r"^\s*\|\s*" + field + r"\s*=(.*)$", wikitext[start.start():], re.M | re.I)
    return m.group(1).strip() if m else None


def clean_value(raw):
    text = COMMENT.sub("", raw)
    text = REF.sub("", text)
    text = NOTE_TEMPLATES.sub("", text)
    text = NBSP.sub(" ", text)
    text = USD_TEMPLATE.sub(lambda m: "$" + m.group(1), text)
    text = LINK.sub(r"\1", text)
    text = ANY_TEMPLATE.sub(" ", text)
    text = TAG.sub(" ", text)
    return " ".join(text.split())


def parse_usd(raw, is_gross=False):
    """Largest US-dollar amount in an info box value, or None."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = clean_value(raw)
    if is_gross and DOMESTIC_ONLY.search(text) and not re.search(r"worldwide|global|total", text, re.I):
        return None
    best = None
    for low, high, unit in AMOUNT.findall(text):
        unit = unit.lower() if unit else None
        for num in (low, high):
            if num:
                value = float(num.replace(",", "")) * SCALE[unit]
                best = value if best is None else max(best, value)
    return best


# ---- Main -------------------------------------------------------------------
def candidates(big_path, years):
    cols = ["imdb_id", "release_date", "budget", "revenue", "status"]
    df = pd.read_csv(big_path, usecols=cols, low_memory=False)
    year = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    b = pd.to_numeric(df["budget"], errors="coerce")
    r = pd.to_numeric(df["revenue"], errors="coerce")
    missing = ~((b >= 1000) & (r >= 1000))
    keep = year.between(*years) & (df["status"] == "Released") & missing \
        & df["imdb_id"].astype(str).str.match(r"^tt\d+$")
    return set(df.loc[keep, "imdb_id"])


def query_range(film_type, start, end):
    """Titles for one film type released in [start, end). Splits the range if Wikidata times out."""
    try:
        res = http(WDQS, {"query": TITLE_QUERY % {"type": film_type, "start": start.date(), "end": end.date()}},
                   accept="application/sparql-results+json")
        time.sleep(2)
        return [(b["imdb"]["value"], b["title"]["value"]) for b in res["results"]["bindings"]]
    except TruncatedResponse:
        if (end - start).days <= 31:
            print(f"    Wikidata kept timing out for {film_type} {start.date()}; skipping that month.")
            return []
        mid = start + (end - start) / 2
        mid = pd.Timestamp(mid.year, mid.month, 1)
        if mid <= start:
            mid = start + pd.DateOffset(months=1)
        print(f"    timed out on {film_type} {start.date()}..{end.date()}; splitting")
        time.sleep(5)
        return query_range(film_type, start, mid) + query_range(film_type, mid, end)


def article_titles(years):
    rows = []
    # One year either side: festival vs wide release dates differ between sources.
    for y in range(years[0] - 1, years[1] + 2):
        got = []
        for film_type in FILM_TYPES.split():
            got += query_range(film_type, pd.Timestamp(y, 1, 1), pd.Timestamp(y + 1, 1, 1))
        print(f"  Wikidata {y}: {len(set(got)):,} films with an English Wikipedia article")
        rows.extend(got)
    return pd.DataFrame(rows, columns=["imdb_id", "wiki_title"]).drop_duplicates("imdb_id")


def fetch_wikitext(titles):
    texts = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        params = {
            "action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
            "redirects": 1, "format": "json", "formatversion": 2, "maxlag": 5,
            "titles": "|".join(batch),
        }
        try:
            res = http(WP_API, params)
        except TruncatedResponse:
            time.sleep(10)
            try:
                res = http(WP_API, params)
            except TruncatedResponse:
                print(f"  skipped a batch of {len(batch)} articles (bad response twice)")
                continue
        if "error" in res:
            time.sleep(10)
            continue
        redirect = {r["from"]: r["to"] for r in res.get("query", {}).get("redirects", [])}
        normal = {n["from"]: n["to"] for n in res.get("query", {}).get("normalized", [])}
        pages = {p["title"]: p for p in res.get("query", {}).get("pages", [])}
        for t in batch:
            final = redirect.get(normal.get(t, t), normal.get(t, t))
            page = pages.get(final)
            if page and "revisions" in page:
                texts[t] = page["revisions"][0]["slots"]["main"]["content"]
        done = min(i + 50, len(titles))
        if done % 1000 < 50 or done == len(titles):
            print(f"  Wikipedia: {done:,}/{len(titles):,} articles read")
        time.sleep(1)
    return texts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--big", default="raw/comps/TMDB_movie_dataset_v11.csv", help="TMDB dataset CSV (to pick candidate films)")
    p.add_argument("--from-year", type=int, default=2022)
    p.add_argument("--to-year", type=int, default=2025)
    p.add_argument("--out", default="raw/comps/wikipedia_financials.csv")
    args = p.parse_args(argv)
    years = (args.from_year, args.to_year)

    cand = candidates(args.big, years)
    print(f"{len(cand):,} TMDB films from {years[0]}-{years[1]} are missing a budget or gross.")
    print("Finding their Wikipedia articles via Wikidata...")
    titles = article_titles(years)
    titles = titles[titles["imdb_id"].isin(cand)]
    print(f"{len(titles):,} of the candidates have an English Wikipedia article.")
    if titles.empty:
        sys.exit("No articles found; paste this output to Claude.")

    texts = fetch_wikitext(titles["wiki_title"].tolist())
    titles["budget_raw"] = titles["wiki_title"].map(lambda t: infobox_field(texts.get(t, ""), "budget"))
    titles["gross_raw"] = titles["wiki_title"].map(lambda t: infobox_field(texts.get(t, ""), "gross"))
    titles["budget_usd"] = titles["budget_raw"].map(parse_usd)
    titles["gross_usd"] = titles["gross_raw"].map(lambda s: parse_usd(s, is_gross=True))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    titles.to_csv(args.out, index=False)
    both = titles["budget_usd"].notna() & titles["gross_usd"].notna()
    print(f"\nSaved {len(titles):,} films to {args.out}")
    print(f"  US-dollar budget: {int(titles['budget_usd'].notna().sum()):,}, "
          f"gross: {int(titles['gross_usd'].notna().sum()):,}, both: {int(both.sum()):,}")


if __name__ == "__main__":
    main()
