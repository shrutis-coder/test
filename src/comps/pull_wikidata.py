"""P5 Comps: pull film budgets and box office from Wikidata.

Wikidata is the open database behind Wikipedia's info boxes. Many films there
have a budget (property P2130, "cost"), a box office total (P2142) and an IMDb
ID (P345), so the result joins straight onto our table by imdb_id.

Why: TMDB rarely records budgets for recent films, so the 2022-2025 test split
is short of the plan's 400 minimum. Wikipedia editors fill these in from trade
press, so Wikidata often has them.

Usage (from the repo root):

    python src/comps/pull_wikidata.py

Saves raw/comps/wikidata_financials.csv. clean_and_label.py merges it with
--wikidata. Wikidata is CC0 (public domain), and its query service is meant
for programs, so there's no scraping or terms-of-use issue.

Rules:
  * Only amounts in US dollars are kept.
  * Box office: a figure marked "worldwide" wins; otherwise a figure with no
    region is used (gross_scope = "unspecified"); figures for one country
    only (e.g. United States) are ignored.
  * Several values (e.g. a budget range) -> the largest is kept, which keeps
    ROI on the cautious side.
  * Deprecated statements are skipped.
"""

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {
    # Wikimedia asks automated clients to identify themselves.
    "User-Agent": "FilmCrowdGreenlightClassProject/1.0 (UC Berkeley student project; python urllib)",
    "Accept": "application/sparql-results+json",
}
USD = "http://www.wikidata.org/entity/Q4917"
WORLDWIDE = "http://www.wikidata.org/entity/Q13780930"

QUERY = """
SELECT ?film ?imdb ?amount ?region WHERE {
  hint:Query hint:optimizer "None" .
  ?st psv:%(prop)s ?v .
  ?v wikibase:quantityUnit wd:Q4917 ;
     wikibase:quantityAmount ?amount .
  ?film p:%(prop)s ?st ;
        wdt:P345 ?imdb .
  ?st wikibase:rank ?rank .
  FILTER(?rank != wikibase:DeprecatedRank)
  FILTER(STRSTARTS(?imdb, "tt"))
  OPTIONAL { ?st pq:P1001 ?region }
}
"""


def ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def run_query(query, retries=4):
    data = urllib.parse.urlencode({"query": query}).encode()
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(ENDPOINT, data=data, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120, context=ssl_context()) as resp:
                return json.loads(resp.read().decode("utf-8"))["results"]["bindings"]
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                wait = 30 * attempt
                print(f"  Wikidata busy ({e.code}); retrying in {wait}s...")
                time.sleep(wait)
                continue
            sys.exit(f"Wikidata query failed ({e.code}): {body}")
        except urllib.error.URLError as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                sys.exit("Python can't verify HTTPS certificates. Run:  python3 -m pip install certifi")
            if attempt == retries:
                raise
            time.sleep(15 * attempt)


def to_frame(bindings):
    rows = [{
        "wikidata_id": b["film"]["value"].rsplit("/", 1)[-1],
        "imdb_id": b["imdb"]["value"].strip(),
        "amount": float(b["amount"]["value"]),
        "region": b.get("region", {}).get("value"),
    } for b in bindings]
    return pd.DataFrame(rows, columns=["wikidata_id", "imdb_id", "amount", "region"])


def summarize(budget_rows, gross_rows):
    """One row per imdb_id: budget_usd, gross_usd, gross_scope."""
    b = budget_rows[budget_rows["region"].isna() | (budget_rows["region"] == WORLDWIDE)]
    budget = b.groupby("imdb_id").agg(budget_usd=("amount", "max"), wikidata_id=("wikidata_id", "first"))

    ww = gross_rows[gross_rows["region"] == WORLDWIDE].groupby("imdb_id")["amount"].max()
    unspec = gross_rows[gross_rows["region"].isna()].groupby("imdb_id")["amount"].max()
    gross = pd.DataFrame({"gross_usd": ww, "gross_scope": "worldwide"})
    extra = unspec[~unspec.index.isin(ww.index)]
    gross = pd.concat([gross, pd.DataFrame({"gross_usd": extra, "gross_scope": "unspecified"})])
    gross.index.name = "imdb_id"
    gross_ids = gross_rows.groupby("imdb_id")["wikidata_id"].first()

    out = budget.join(gross, how="outer")
    out["wikidata_id"] = out["wikidata_id"].fillna(gross_ids)
    return out.reset_index()[["imdb_id", "wikidata_id", "budget_usd", "gross_usd", "gross_scope"]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="raw/comps/wikidata_financials.csv")
    args = p.parse_args(argv)

    print("Querying Wikidata for budgets (P2130)... this can take up to a minute.")
    budget_rows = to_frame(run_query(QUERY % {"prop": "P2130"}))
    print(f"  {len(budget_rows):,} US-dollar budget statements")
    time.sleep(5)
    print("Querying Wikidata for box office (P2142)...")
    gross_rows = to_frame(run_query(QUERY % {"prop": "P2142"}))
    print(f"  {len(gross_rows):,} US-dollar box office statements")

    if budget_rows.empty and gross_rows.empty:
        sys.exit("Wikidata returned no rows. The query may need adjusting; paste this output to Claude.")

    out = summarize(budget_rows, gross_rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    both = out["budget_usd"].notna() & out["gross_usd"].notna()
    print(f"\nSaved {len(out):,} films to {args.out}")
    print(f"  with both budget and box office: {int(both.sum()):,}")
    print(f"  box office marked worldwide: {int((out['gross_scope'] == 'worldwide').sum()):,}, "
          f"no region given: {int((out['gross_scope'] == 'unspecified').sum()):,}")


if __name__ == "__main__":
    main()
