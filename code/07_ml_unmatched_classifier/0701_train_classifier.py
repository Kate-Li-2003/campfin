"""
0701_train_classifier.py

Train classifiers that predict level1_category, level2_category, and
naics_code (where labeled) from an entity name. Used by 0702 to fill in
classifications for entities that didn't match masterfile / keyword /
EDD / non-company-individual sources.

Pipeline:
  1. Load running_list.csv (training set: name + level1/level2/naics).
  2. Encode names with sentence-transformers (all-MiniLM-L6-v2).
     Embeddings are cached to disk keyed by (model_name, name list hash).
  3. For each target column, train a multinomial Logistic Regression on
     the embeddings. Held-out accuracy reported via an 80/20 stratified
     split.
  4. Refit each classifier on the full dataset and save with joblib.

Outputs (under data/07_output_ml_classification/models/):
  - level1_clf.joblib, level2_clf.joblib, naics_clf.joblib
  - encoder_name.txt          (sentence-transformers model name)
  - embeddings_cache.npz      (training embeddings, keyed by name hash)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = REPO_ROOT / "data/03_input/masterfile/running_list.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "data/07_output_ml_classification/models"
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
TARGETS = ["level1_category", "level2_category", "naics_code"]
MIN_CLASS_SUPPORT = 5  # drop classes with fewer than this many examples


def _hash_names(names: list[str], model_name: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(str(len(names)).encode())
    for n in names:
        h.update(n.encode())
        h.update(b"\0")
    return h.hexdigest()


def encode_names(names: list[str], model_name: str, cache_path: Path) -> np.ndarray:
    key = _hash_names(names, model_name)
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        if str(z["key"]) == key:
            print(f"  Loaded cached embeddings: {cache_path.name}  shape={z['emb'].shape}")
            return z["emb"]
        print("  Cached embeddings stale; recomputing.")

    # Lazy import — heavy dependency.
    from sentence_transformers import SentenceTransformer

    print(f"  Loading encoder: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"  Encoding {len(names):,} names ...")
    emb = model.encode(
        names,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, emb=emb, key=np.array(key))
    print(f"  Cached to: {cache_path}")
    return emb


def train_one(emb: np.ndarray, y: pd.Series, target: str) -> LogisticRegression | None:
    mask = y.notna() & (y.astype(str).str.strip() != "")
    counts = y[mask].value_counts()
    keep_classes = counts[counts >= MIN_CLASS_SUPPORT].index
    mask = mask & y.isin(keep_classes)

    n = int(mask.sum())
    if n < 100:
        print(f"  [{target}] too few labeled rows ({n}); skipping.")
        return None

    X = emb[mask.values]
    y_ = y[mask].astype(str).values

    print(f"\n  [{target}]")
    print(
        f"    labeled rows: {n:,}   classes (>= {MIN_CLASS_SUPPORT} ex.): "
        f"{len(keep_classes):,}   dropped rare: "
        f"{int((counts < MIN_CLASS_SUPPORT).sum()):,}"
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_, test_size=0.2, stratify=y_, random_state=42
    )
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    proba = clf.predict_proba(X_te)
    top3 = top_k_accuracy_score(y_te, proba, k=min(3, proba.shape[1]), labels=clf.classes_)
    print(f"    held-out accuracy: top1={acc:.3f}  top3={top3:.3f}")

    # Refit on full dataset for final model.
    final = LogisticRegression(max_iter=1000, solver="lbfgs")
    final.fit(X, y_)
    return final


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--encoder", type=str, default=DEFAULT_ENCODER)
    args = p.parse_args(argv)

    print(f"Training data: {args.train}")
    df = pd.read_csv(args.train)
    df = df[df["name"].notna()].copy()
    df["name"] = df["name"].astype(str).str.strip()
    df = df[df["name"] != ""]

    # NAICS may come in as float (e.g. 524210.0) or string ("44-45");
    # normalize to a clean categorical string label.
    if "naics_code" in df.columns:
        def _clean_naics(v):
            if pd.isna(v):
                return pd.NA
            s = str(v).strip()
            if s.endswith(".0"):
                s = s[:-2]
            return s if s else pd.NA
        df["naics_code"] = df["naics_code"].map(_clean_naics)

    names = df["name"].tolist()
    print(f"  rows: {len(df):,}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "embeddings_cache.npz"
    emb = encode_names(names, args.encoder, cache_path)

    (args.out_dir / "encoder_name.txt").write_text(args.encoder)

    for target in TARGETS:
        if target not in df.columns:
            continue
        clf = train_one(emb, df[target], target)
        if clf is None:
            continue
        out = args.out_dir / f"{target}_clf.joblib"
        joblib.dump(clf, out)
        print(f"    saved: {out.name}  ({len(clf.classes_)} classes)")

    print(f"\nDone. Models written to: {args.out_dir}")


if __name__ == "__main__":
    main()
