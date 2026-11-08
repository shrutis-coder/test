"""P5 Comps, box 4: Comps expert.

Turns each film's comp features into the 0-100 comps sub-score the combiner uses.

  * Target: breakeven (gross >= 2.5x budget). Sub-score = 100 x P(breakeven), so
    the combiner can take it as a logit (team plan, "sub-scores as logits").
  * Inputs: comp features only (from comps_search.py). Budget and genre go to the
    combiner directly, so the expert doesn't double-count them.
  * Model: scikit-learn HistGradientBoostingClassifier (handles missing values,
    e.g. films with no comps; installs on every Python version we use).
  * Out-of-fold: train films (<= 2017) are scored by a model that never saw them
    (5-fold). Every other film is scored by one model fit on all train films.
    This is what lets P6 train the combiner on these scores without leakage.
  * Evaluation on train (out-of-fold) and validate only. Test films get scores,
    but their accuracy is NOT measured here: the plan opens the test set once, at the end.

Run from the repo root (after comps_search.py):

    python src/comps/comps_expert.py

Reads:   clean/comps/comps_labeled.csv, features/comps/comps_features.csv
Writes:  features/comps/comps_expert_scores.csv   imdb_id, as_of_date, split, comps_score (0-100), comps_logit, fold
         models/experts/comps_expert.pkl           final model + feature list + version
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

VERSION = "comps_expert_v1"
SEED = 0
N_FOLDS = 5
FEATURES = [
    "n_candidates", "n_comps",
    "comp_log_roi_mean", "comp_log_roi_std", "comp_roi_median",
    "comp_breakeven_rate", "comp_breakout_rate",
    "closest_similarity", "mean_similarity", "n_close_comps", "comp_recency_years",
]
TARGET = "breakeven"
PARAMS = dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=15, min_samples_leaf=40,
              l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
              n_iter_no_change=30, random_state=SEED)


def make_model():
    return HistGradientBoostingClassifier(**PARAMS)


def clip_prob(p):
    return np.clip(p, 1e-4, 1 - 1e-4)


def metrics(y, p, name):
    p = clip_prob(p)
    return {"set": name, "films": len(y), "breakeven_rate": round(float(np.mean(y)), 3),
            "auc": round(roc_auc_score(y, p), 3), "log_loss": round(log_loss(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labeled", default="clean/comps/comps_labeled.csv")
    p.add_argument("--features", default="features/comps/comps_features.csv")
    p.add_argument("--scores-out", default="features/comps/comps_expert_scores.csv")
    p.add_argument("--model-out", default="models/experts/comps_expert.pkl")
    args = p.parse_args(argv)

    lab = pd.read_csv(args.labeled, usecols=["imdb_id", TARGET], low_memory=False)
    feats = pd.read_csv(args.features)
    df = feats.merge(lab, on="imdb_id", how="left")
    X, y = df[FEATURES], df[TARGET].astype(int)
    train = (df["split"] == "train").to_numpy()
    val = (df["split"] == "validate").to_numpy()

    # Out-of-fold scores for train films.
    oof = np.full(len(df), np.nan)
    fold = np.full(len(df), -1)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train_idx = np.flatnonzero(train)
    for k, (fit, hold) in enumerate(skf.split(train_idx, y.iloc[train_idx])):
        m = make_model().fit(X.iloc[train_idx[fit]], y.iloc[train_idx[fit]])
        oof[train_idx[hold]] = m.predict_proba(X.iloc[train_idx[hold]])[:, 1]
        fold[train_idx[hold]] = k
        print(f"  fold {k + 1}/{N_FOLDS}: {m.n_iter_} trees")

    # One model on all train films scores everything else (validate, test, skip, excluded).
    final = make_model().fit(X[train], y[train])
    prob = np.where(train, oof, final.predict_proba(X)[:, 1])
    prob = clip_prob(prob)

    out = df[["imdb_id", "as_of_date", "split"]].copy()
    out["comps_score"] = (100 * prob).round(2)
    out["comps_logit"] = np.log(prob / (1 - prob)).round(4)
    out["fold"] = fold
    out["model_version"] = VERSION
    os.makedirs(os.path.dirname(args.scores_out), exist_ok=True)
    out.to_csv(args.scores_out, index=False)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    with open(args.model_out, "wb") as fh:
        pickle.dump({"model": final, "features": FEATURES, "target": TARGET,
                     "version": VERSION, "params": PARAMS}, fh)

    # Evaluation: train out-of-fold and validate only. Test stays sealed.
    rows = [metrics(y[train], prob[train], "train (out-of-fold)"),
            metrics(y[val], prob[val], "validate")]
    base = y[train].mean()
    rows.append({**metrics(y[val], np.full(val.sum(), base), "validate: base rate only"), "auc": 0.5})
    simple = df["comp_breakeven_rate"].fillna(base).to_numpy()
    rows.append(metrics(y[val], simple[val], "validate: comp breakeven rate alone"))
    report = pd.DataFrame(rows)
    print(f"\nWrote {len(out):,} scores to {args.scores_out} and the model to {args.model_out}")
    print("\n" + report.to_string(index=False))

    print("\nScore buckets on validate (do higher scores mean more breakevens?):")
    v = out[val].assign(breakeven=y[val].to_numpy())
    print(v.groupby(pd.qcut(v["comps_score"], 4), observed=True)["breakeven"]
          .agg(films="size", breakeven_rate="mean").round(3).to_string())

    imp = permutation_importance(final, X[val], y[val], scoring="roc_auc", n_repeats=10, random_state=SEED)
    print("\nWhat the model leans on (drop in validate AUC when a feature is shuffled):")
    print(pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False).round(4).to_string())

    with open(os.path.splitext(args.model_out)[0] + "_report.json", "w") as fh:
        json.dump({"version": VERSION, "metrics": rows, "params": PARAMS, "features": FEATURES}, fh, indent=2)
    print(f"\nTest films ({int((df['split'] == 'test').sum())}) were scored but not evaluated: "
          "the plan opens the test set once, at the end.")


if __name__ == "__main__":
    main()
