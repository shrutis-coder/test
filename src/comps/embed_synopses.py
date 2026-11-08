"""P5 Comps, box 2: Embedding model.

Turns every film synopsis (TMDB overview) into a vector so the comps search can
find films with similar stories.

Model: sentence-transformers/all-MiniLM-L6-v2 (384 numbers per film, normalized
so a dot product is cosine similarity). Small, fast on a laptop, and a standard
baseline for sentence similarity. Script embeddings (P1/P2) can be added later
with the same model so scripts and synopses live in one space.

Run from the repo root:

    python -m pip install sentence-transformers      # once
    python src/comps/embed_synopses.py

The first run downloads the model (~90 MB) from Hugging Face.

Reads:   clean/comps/comps_labeled.csv
Writes:  features/comps/synopsis_embeddings.npz
         (imdb_id, vectors, model name; films without a synopsis are left out)

--model tfidf uses TF-IDF + SVD instead (no download, weaker on meaning).
It exists for testing where the model can't be downloaded.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def embed_minilm(texts):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("sentence-transformers isn't installed. Run:  python3 -m pip install sentence-transformers")
    model = SentenceTransformer(MINILM)
    print(f"Embedding {len(texts):,} synopses with {MINILM} on {model.device}...")
    return model.encode(texts, batch_size=64, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True).astype(np.float32)


def embed_tfidf(texts, dims=256):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    print(f"Embedding {len(texts):,} synopses with TF-IDF + SVD ({dims} dims)...")
    tfidf = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True).fit_transform(texts)
    vecs = TruncatedSVD(n_components=dims, random_state=0).fit_transform(tfidf)
    return normalize(vecs).astype(np.float32)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labeled", default="clean/comps/comps_labeled.csv")
    p.add_argument("--out", default="features/comps/synopsis_embeddings.npz")
    p.add_argument("--model", choices=["minilm", "tfidf"], default="minilm")
    args = p.parse_args(argv)

    df = pd.read_csv(args.labeled, usecols=["imdb_id", "overview"], low_memory=False)
    has_text = df["overview"].fillna("").str.strip().str.len() > 0
    df = df[has_text]
    texts = df["overview"].str.strip().tolist()

    start = time.time()
    vecs = embed_minilm(texts) if args.model == "minilm" else embed_tfidf(texts)
    model_name = MINILM if args.model == "minilm" else "tfidf-svd-256"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, imdb_id=df["imdb_id"].to_numpy(dtype=str), vectors=vecs, model=model_name)
    print(f"Saved {len(vecs):,} vectors ({vecs.shape[1]} dims, {model_name}) to {args.out} "
          f"in {time.time() - start:.0f}s. Skipped {int((~has_text).sum())} films with no synopsis.")


if __name__ == "__main__":
    main()
