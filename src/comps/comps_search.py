"""P5 Comps, box 3: Comps search.

For every film, finds its 5-10 closest comparable films and summarizes how
they did. These features feed the comps expert; the comp list itself feeds the
investor view's "five closest comps".

Rules (team plan, "Comps search"):
  * Nearest by synopsis embedding (cosine similarity).
  * Same budget band (micro / low / mid / high / tentpole, inflation-adjusted).
  * Same genre: the comp must include the film's main (first-listed) genre.
    Most films have 2-3 genres and Drama alone covers half the data, so "any
    shared genre" would be almost no filter, and "identical genre list" would
    leave most films with no comps.
  * Before the as-of date: the comp was released at least --min-gap-days
    (default 180) before the film's as_of_date, so its gross was already known.
    This is what stops comps from leaking the future.
  * Never itself.
  * Not older than --max-age-years (default 20) before the as-of date. Old comps
    predicted worse; a 20-year cap raised the comps expert's out-of-fold AUC
    from 0.590 to 0.597 and lowered its log loss.

Run from the repo root (after embed_synopses.py):

    python src/comps/comps_search.py

Reads:   clean/comps/comps_labeled.csv
         features/comps/synopsis_embeddings.npz
Writes:  features/comps/comps_features.csv   one row per film (imdb_id, as_of_date, features)
         features/comps/comps_list.csv       one row per (film, comp), ranked
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

K_MAX = 10
K_MIN = 5
# "Close" comp threshold depends on the embedding model's similarity scale.
CLOSE_THRESHOLD = {"sentence-transformers/all-MiniLM-L6-v2": 0.5, "tfidf-svd-256": 0.3}

FEATURE_COLUMNS = [
    "imdb_id", "as_of_date", "split", "n_candidates", "n_comps",
    "comp_log_roi_mean", "comp_log_roi_std", "comp_roi_median",
    "comp_breakeven_rate", "comp_breakout_rate",
    "closest_similarity", "mean_similarity", "n_close_comps", "comp_recency_years",
    "embedding_model",
]


def primary_genre(genres):
    return genres.split(",")[0].strip() if isinstance(genres, str) and genres.strip() else None


def load(labeled_path, emb_path):
    df = pd.read_csv(labeled_path, low_memory=False)
    emb = np.load(emb_path, allow_pickle=False)
    vec_ids = emb["imdb_id"].astype(str)
    vectors = emb["vectors"].astype(np.float32)
    model = str(emb["model"])
    pos = pd.Series(np.arange(len(vec_ids)), index=vec_ids)
    df["vec_row"] = df["imdb_id"].map(pos)
    df["as_of"] = pd.to_datetime(df["as_of_date"])
    df["release"] = pd.to_datetime(df["release_date"])
    df["main_genre"] = df["genres"].map(primary_genre)
    df["genre_set"] = df["genres"].fillna("").map(lambda g: {x.strip() for x in g.split(",") if x.strip()})
    return df, vectors, model


def search_band(band_df, vectors, k_max, min_gap_days, max_age_years=None, min_votes=0):
    """Top-k comps for every film in one budget band. Returns (rows of comp indices, sims, candidate counts)."""
    films = band_df.reset_index(drop=True)
    has_vec = films["vec_row"].notna().to_numpy()
    V = np.zeros((len(films), vectors.shape[1]), dtype=np.float32)
    V[has_vec] = vectors[films.loc[has_vec, "vec_row"].astype(int)]

    sims = V @ V.T                                              # cosine (vectors are normalized)
    as_of = films["as_of"].to_numpy("datetime64[D]").astype(np.int64)
    released = films["release"].to_numpy("datetime64[D]").astype(np.int64)
    ok = released[None, :] <= (as_of[:, None] - min_gap_days)   # comp j known before film i's as-of date
    if max_age_years:
        ok &= released[None, :] >= (as_of[:, None] - int(max_age_years * 365.25))   # not too old
    if min_votes:
        votes = films["vote_count"].fillna(0).to_numpy()
        ok &= (votes >= min_votes)[None, :]                     # comp is reasonably well known

    genres = sorted({g for s in films["genre_set"] for g in s})
    gidx = {g: n for n, g in enumerate(genres)}
    G = np.zeros((len(films), len(genres)), dtype=bool)
    for r, s in enumerate(films["genre_set"]):
        G[r, [gidx[g] for g in s]] = True
    main = films["main_genre"].map(gidx)
    has_main = main.notna().to_numpy()
    ok &= has_main[:, None]
    ok[has_main] &= G[:, main[has_main].astype(int)].T          # comp j includes film i's main genre

    ok &= has_vec[:, None] & has_vec[None, :]
    np.fill_diagonal(ok, False)                                 # never itself
    same_id = films["imdb_id"].to_numpy()
    ok &= same_id[:, None] != same_id[None, :]

    sims = np.where(ok, sims, -np.inf)
    n_cand = ok.sum(axis=1)
    k = min(k_max, max(len(films) - 1, 1))
    top = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    top_sims = np.take_along_axis(sims, top, axis=1)
    order = np.argsort(-top_sims, axis=1)
    top = np.take_along_axis(top, order, axis=1)
    top_sims = np.take_along_axis(top_sims, order, axis=1)
    return films, top, top_sims, n_cand


def summarize(films, top, top_sims, n_cand, close, model):
    feats, comps = [], []
    log_roi = films["log_roi"].to_numpy()
    roi = films["roi_multiple"].to_numpy()
    be = films["breakeven"].to_numpy()
    bo = films["breakout"].to_numpy()
    rel = films["release"].to_numpy("datetime64[D]")
    for i in range(len(films)):
        valid = np.isfinite(top_sims[i])
        idx, sim = top[i][valid], top_sims[i][valid]
        n = len(idx)
        row = {"imdb_id": films.at[i, "imdb_id"], "as_of_date": films.at[i, "as_of_date"],
               "split": films.at[i, "split"], "n_candidates": int(n_cand[i]), "n_comps": n,
               "embedding_model": model}
        if n:
            years = (films.at[i, "as_of"].to_datetime64().astype("datetime64[D]") - rel[idx]).astype(int) / 365.25
            row.update({
                "comp_log_roi_mean": float(log_roi[idx].mean()),
                "comp_log_roi_std": float(log_roi[idx].std()) if n >= 2 else np.nan,
                "comp_roi_median": float(np.median(roi[idx])),
                "comp_breakeven_rate": float(be[idx].mean()),
                "comp_breakout_rate": float(bo[idx].mean()),
                "closest_similarity": float(sim[0]),
                "mean_similarity": float(sim.mean()),
                "n_close_comps": int((sim >= close).sum()),
                "comp_recency_years": float(np.median(years)),
            })
            for rank, (j, s) in enumerate(zip(idx, sim), start=1):
                comps.append({
                    "imdb_id": films.at[i, "imdb_id"], "title": films.at[i, "title"],
                    "as_of_date": films.at[i, "as_of_date"], "rank": rank,
                    "comp_imdb_id": films.at[j, "imdb_id"], "comp_title": films.at[j, "title"],
                    "comp_release_date": films.at[j, "release_date"],
                    "comp_roi_multiple": round(float(roi[j]), 3), "similarity": round(float(s), 4),
                })
        feats.append(row)
    return feats, comps


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labeled", default="clean/comps/comps_labeled.csv")
    p.add_argument("--embeddings", default="features/comps/synopsis_embeddings.npz")
    p.add_argument("--out-dir", default="features/comps")
    p.add_argument("--k", type=int, default=K_MAX, help="comps per film (plan: 5 to 10)")
    p.add_argument("--min-gap-days", type=int, default=180,
                   help="a comp must be released this many days before the film's as_of_date")
    # 20 years: chosen on train out-of-fold accuracy of the comps expert (Nov 7, 2026). Caps of 10-25
    # years all beat no cap (AUC ~0.60 vs 0.590, lower log loss); 20 keeps 97.6% of films at 5+ comps.
    p.add_argument("--max-age-years", type=float, default=20,
                   help="only comps released within this many years of the as-of date (0 = no limit)")
    p.add_argument("--min-votes", type=int, default=0, help="only comps with at least this many TMDB votes")
    p.add_argument("--close", type=float, help="similarity counted as a close comp (default depends on model)")
    p.add_argument("--show", nargs="*", default=["Get Out", "Paranormal Activity", "Inception", "Fight Club"],
                   help="titles whose comps to print as a sanity check")
    args = p.parse_args(argv)
    if not os.path.exists(args.embeddings):
        sys.exit(f"{args.embeddings} not found. Run src/comps/embed_synopses.py first.")

    df, vectors, model = load(args.labeled, args.embeddings)
    close = args.close if args.close is not None else CLOSE_THRESHOLD.get(model, 0.5)
    print(f"{len(df):,} films, {int(df['vec_row'].notna().sum()):,} with a synopsis vector ({model}).")
    extra = (f", within {args.max_age_years:g} years" if args.max_age_years else "") + \
            (f", >= {args.min_votes} TMDB votes" if args.min_votes else "")
    print(f"Rules: same budget band, includes main genre, released >= {args.min_gap_days} days before as_of_date"
          f"{extra}, top {args.k}. Close comp = similarity >= {close}.")

    feats, comps = [], []
    for band, band_df in df.groupby("budget_band"):
        films, top, top_sims, n_cand = search_band(band_df, vectors, args.k, args.min_gap_days,
                                                   args.max_age_years, args.min_votes)
        f, c = summarize(films, top, top_sims, n_cand, close, model)
        feats += f
        comps += c
        print(f"  {band:<9} {len(films):>5,} films")

    feats = pd.DataFrame(feats).reindex(columns=FEATURE_COLUMNS)
    comps = pd.DataFrame(comps)
    os.makedirs(args.out_dir, exist_ok=True)
    feats.to_csv(os.path.join(args.out_dir, "comps_features.csv"), index=False)
    comps.to_csv(os.path.join(args.out_dir, "comps_list.csv"), index=False)

    print(f"\nWrote {len(feats):,} rows to {args.out_dir}/comps_features.csv "
          f"and {len(comps):,} comps to {args.out_dir}/comps_list.csv")
    nc = feats["n_comps"]
    print("\nComps per film:")
    print(f"  {K_MIN}+ comps (plan minimum): {int((nc >= K_MIN).sum()):,}   "
          f"1-{K_MIN - 1}: {int(nc.between(1, K_MIN - 1).sum()):,}   none: {int((nc == 0).sum()):,}")
    print("  by split, share with 5+ comps:")
    print((feats.assign(ok=nc >= K_MIN).groupby("split")["ok"].mean().round(3)).to_string())

    for title in args.show:
        hits = comps[comps["title"].str.lower() == title.lower()]
        if hits.empty:
            continue
        film = df[df["imdb_id"] == hits["imdb_id"].iloc[0]].iloc[0]
        print(f"\n{film['title']} ({film['release_year']}, {film['budget_band']}, {film['genres']}; "
              f"actual {film['roi_multiple']:.1f}x)")
        for _, r in hits.head(5).iterrows():
            print(f"  {r['rank']:>2}. {r['comp_title']} ({str(r['comp_release_date'])[:4]})  "
                  f"sim {r['similarity']:.2f}  {r['comp_roi_multiple']:.1f}x")


if __name__ == "__main__":
    main()
