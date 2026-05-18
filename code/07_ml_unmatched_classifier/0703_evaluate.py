"""
0703_evaluate.py

Empirical accuracy of the 0701-trained classifiers, reported on two
disjoint datasets:

  1. TRAIN  — running_list.csv (the data the classifiers were fit on).
              An overfit-sanity check; expect high numbers.
  2. OOD    — entities in 05_output classified CSVs whose data_source_1
              is one of {masterfile, keyword match, manual, edd}. These
              are real-race entities with a label assigned by a non-ML
              source; the ML model has never seen them at training time
              (modulo any name collisions with running_list, which we
              filter out for a clean comparison).

For each target column (level1_category, level2_category, naics_code)
we report top-1 and top-3 accuracy on rows that have a ground-truth
label for that target. OOD numbers are also broken down by data_source_1.

Run after 0701:
  python code/07_ml_unmatched_classifier/0703_evaluate.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = REPO_ROOT / "data/03_input/masterfile/running_list.csv"
DEFAULT_MODEL_DIR = REPO_ROOT / "data/07_output_ml_classification/models"
DEFAULT_OOD_INPUTS = [
    REPO_ROOT / "output/05_output/donors_classified_with_manual.csv",
    REPO_ROOT / "output/05_output/lt_governor_race_2026_over_10k_classified.csv",
    REPO_ROOT / "output/05_output/insurance_commissioner_race_2026_over_10k_classified.csv",
]
DEFAULT_OUT = REPO_ROOT / "data/07_output_ml_classification/eval_report.csv"
TARGETS = ["level1_category", "level2_category", "naics_code"]
ELIGIBLE_SOURCES = {"masterfile", "keyword match", "manual", "edd"}


def _clean_naics(v):
    if pd.isna(v):
        return pd.NA
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s if s else pd.NA


def _normalize_str(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), pd.NA).astype("string").str.strip()


def _load_encoder(model_dir: Path):
    from sentence_transformers import SentenceTransformer
    name = (model_dir / "encoder_name.txt").read_text().strip()
    print(f"  encoder: {name}")
    return SentenceTransformer(name)


def _load_classifiers(model_dir: Path) -> dict:
    out = {}
    for t in TARGETS:
        p = model_dir / f"{t}_clf.joblib"
        if p.exists():
            out[t] = joblib.load(p)
    return out


def _topk_correct(clf, X: np.ndarray, y_true: np.ndarray, k: int) -> np.ndarray:
    proba = clf.predict_proba(X)
    # Indices of the top-k predicted class per row, in descending probability.
    topk_idx = np.argsort(-proba, axis=1)[:, :k]
    topk_labels = np.asarray(clf.classes_, dtype=object)[topk_idx]
    y_true_arr = np.asarray(y_true, dtype=object).reshape(-1, 1)
    return (topk_labels == y_true_arr).any(axis=1)


def _encode(encoder, names: list[str]) -> np.ndarray:
    return encoder.encode(
        names,
        batch_size=128,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def evaluate_block(
    df: pd.DataFrame, name_col: str, encoder, clfs: dict, label_prefix: str
) -> list[dict]:
    """Compute top-1 / top-3 accuracy per target on rows of `df` where the
    ground-truth label is non-null. Returns one dict per (target)."""
    df = df[df[name_col].notna()].copy()
    df[name_col] = df[name_col].astype(str).str.strip()
    df = df[df[name_col] != ""]
    if df.empty:
        return []

    if "naics_code" in df.columns:
        df["naics_code"] = df["naics_code"].map(_clean_naics)
    for t in ("level1_category", "level2_category"):
        if t in df.columns:
            df[t] = _normalize_str(df[t])

    emb = _encode(encoder, df[name_col].tolist())
    rows = []
    for t, clf in clfs.items():
        if t not in df.columns:
            continue
        truth = df[t]
        mask = truth.notna() & truth.astype(str).isin(set(map(str, clf.classes_)))
        n_total = int(truth.notna().sum())
        n_in_vocab = int(mask.sum())
        if n_in_vocab == 0:
            rows.append({
                "block": label_prefix, "target": t,
                "n_with_truth": n_total, "n_in_classifier_vocab": 0,
                "top1": np.nan, "top3": np.nan,
            })
            continue
        y_true = truth[mask].astype(str).to_numpy(dtype=object)
        X = emb[mask.values]
        k1 = _topk_correct(clf, X, y_true, k=1)
        k3 = _topk_correct(clf, X, y_true, k=min(3, len(clf.classes_)))
        rows.append({
            "block": label_prefix, "target": t,
            "n_with_truth": n_total, "n_in_classifier_vocab": n_in_vocab,
            "top1": float(k1.mean()), "top3": float(k3.mean()),
        })
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--ood-inputs", type=Path, nargs="*", default=DEFAULT_OOD_INPUTS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    encoder = _load_encoder(args.model_dir)
    clfs = _load_classifiers(args.model_dir)
    if not clfs:
        raise SystemExit(f"No classifiers under {args.model_dir}. Run 0701 first.")
    print(f"  loaded classifiers: {list(clfs.keys())}")

    # ----- TRAIN block -----
    print("\n=== TRAIN: running_list.csv (overfit sanity check) ===")
    train_df = pd.read_csv(args.train)
    train_rows = evaluate_block(train_df, "name", encoder, clfs, "train(all)")
    for r in train_rows:
        _print_row(r)

    # Cache normalized training-name set for OOD overlap filtering.
    train_names = set(
        train_df["name"].dropna().astype(str).str.strip().str.lower().tolist()
    )

    # ----- OOD blocks -----
    print("\n=== OOD: real-race entities (per-file, per-source) ===")
    ood_rows = []
    aggregated_ood = []  # for the pooled report

    for path in args.ood_inputs:
        if not path.exists():
            print(f"  skip (not found): {path}")
            continue
        df = pd.read_csv(path)
        if "data_source_1" not in df.columns or "employer" not in df.columns:
            print(f"  skip (no data_source_1/employer): {path.name}")
            continue
        df = df[df["data_source_1"].isin(ELIGIBLE_SOURCES)].copy()
        if df.empty:
            continue

        # Drop rows whose name is literally in the training set — those
        # would be guaranteed lookups and inflate OOD accuracy.
        emp_lower = df["employer"].astype(str).str.strip().str.lower()
        df = df.loc[~emp_lower.isin(train_names)].copy()
        if df.empty:
            print(f"  {path.name}: all rows overlap training set; nothing to eval")
            continue

        print(f"\n  --- {path.name} ({len(df)} OOD rows after train-overlap filter) ---")
        per_file = evaluate_block(df, "employer", encoder, clfs, f"ood:{path.stem}")
        for r in per_file:
            _print_row(r)
        ood_rows.extend(per_file)

        # Per-source within the file.
        for src in sorted(df["data_source_1"].unique()):
            sub = df[df["data_source_1"] == src]
            if sub.empty:
                continue
            print(f"\n    by source = {src}  (n={len(sub)})")
            per_src = evaluate_block(
                sub, "employer", encoder, clfs, f"ood:{path.stem}|{src}"
            )
            for r in per_src:
                _print_row(r, indent="      ")
            ood_rows.extend(per_src)

        aggregated_ood.append(df)

    if aggregated_ood:
        pooled = pd.concat(aggregated_ood, ignore_index=True)
        print(f"\n  --- POOLED OOD across all files (n={len(pooled)}) ---")
        pooled_rows = evaluate_block(pooled, "employer", encoder, clfs, "ood:pooled")
        for r in pooled_rows:
            _print_row(r)
        ood_rows.extend(pooled_rows)

    # ----- Write report -----
    report = pd.DataFrame(train_rows + ood_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    print(f"\nWrote: {args.out}")


def _print_row(r: dict, indent: str = "  ") -> None:
    t1 = "  n/a" if pd.isna(r["top1"]) else f"{r['top1']:.3f}"
    t3 = "  n/a" if pd.isna(r["top3"]) else f"{r['top3']:.3f}"
    print(
        f"{indent}{r['target']:<18s} "
        f"n_truth={r['n_with_truth']:>5,}  "
        f"n_in_vocab={r['n_in_classifier_vocab']:>5,}  "
        f"top1={t1}  top3={t3}"
    )


if __name__ == "__main__":
    main()
