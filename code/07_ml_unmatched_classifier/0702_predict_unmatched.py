"""
0702_predict_unmatched.py

Apply the classifiers trained by 0701 to entities in 05_output classified
CSVs that have data_source_1 still null (unmatched by masterfile,
keyword, EDD, or non-company-individual sources).

For each unmatched entity, predict level1_category, level2_category, and
naics_code (where supported), along with model confidence (max softmax
probability). Writes an augmented per-race CSV.

The predicted-rows get data_source_1 = "ml prediction" when their level1
confidence is at or above --threshold (default 0.30); below that they
remain flagged unmatched, but the predictions are still written so the
analyst can inspect borderline cases.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "data/07_output_ml_classification/models"
DEFAULT_INPUTS = [
    REPO_ROOT / "output/05_output/lt_governor_race_2026_over_10k_classified.csv",
    REPO_ROOT / "output/05_output/insurance_commissioner_race_2026_over_10k_classified.csv",
]
DEFAULT_OUT_DIR = REPO_ROOT / "data/07_output_ml_classification"
DEFAULT_THRESHOLD = 0.30

TARGETS = ["level1_category", "level2_category", "naics_code"]


def _load_encoder(model_dir: Path):
    name = (model_dir / "encoder_name.txt").read_text().strip()
    from sentence_transformers import SentenceTransformer
    print(f"  Encoder: {name}")
    return SentenceTransformer(name)


def _load_classifiers(model_dir: Path) -> dict:
    clfs = {}
    for t in TARGETS:
        p = model_dir / f"{t}_clf.joblib"
        if p.exists():
            clfs[t] = joblib.load(p)
            print(f"  Loaded {p.name} ({len(clfs[t].classes_)} classes)")
    if not clfs:
        raise SystemExit(f"No classifiers found under {model_dir}. Run 0701 first.")
    return clfs


def _predict(clf, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    proba = clf.predict_proba(X)
    idx = proba.argmax(axis=1)
    return clf.classes_[idx], proba[np.arange(len(idx)), idx]


def predict_for_file(
    input_path: Path,
    out_dir: Path,
    encoder,
    clfs: dict,
    threshold: float,
) -> dict:
    df = pd.read_csv(input_path)
    mask = df["data_source_1"].isna()
    n_unmatched = int(mask.sum())

    print(f"\n{input_path.name}")
    print(f"  rows total:      {len(df):>5,}")
    print(f"  rows unmatched:  {n_unmatched:>5,}")

    pred_cols = {
        f"ml_{t}": pd.Series([pd.NA] * len(df), dtype="object") for t in clfs
    } | {f"ml_{t}_conf": pd.Series([np.nan] * len(df), dtype="float64") for t in clfs}

    if n_unmatched:
        names = df.loc[mask, "employer"].astype(str).tolist()
        emb = encoder.encode(
            names,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        for t, clf in clfs.items():
            labels, conf = _predict(clf, emb)
            pred_cols[f"ml_{t}"].loc[mask] = labels
            pred_cols[f"ml_{t}_conf"].loc[mask] = conf

    out = pd.concat([df, pd.DataFrame(pred_cols)], axis=1)

    # Promote ML predictions above threshold to filled-in classification.
    if "level1_category" in clfs and n_unmatched:
        conf_l1 = out["ml_level1_category_conf"]
        promote = mask & (conf_l1 >= threshold)
        n_promote = int(promote.sum())
        for t in clfs:
            # Cast target column to object so string predictions (incl.
            # NAICS strings like "31-33") can be written into what may
            # have started as float64.
            if t in out.columns:
                out[t] = out[t].astype(object)
            out.loc[promote, t] = out.loc[promote, f"ml_{t}"]
        out.loc[promote, "data_source_1"] = "ml prediction"

        print(
            f"  predicted (any conf):   {n_unmatched:>5,}   "
            f"promoted (level1 conf >= {threshold:.2f}): {n_promote:,}"
        )
        if n_promote:
            promo_df = out.loc[promote].copy()
            promo_df["amount_total"] = promo_df.get("amount_total", pd.Series(0.0))
            print("\n  --- promoted predictions ---")
            print(
                promo_df[
                    [
                        "employer",
                        "n_donors",
                        "ml_level1_category",
                        "ml_level1_category_conf",
                        "ml_level2_category",
                        "ml_level2_category_conf",
                        "ml_naics_code",
                        "ml_naics_code_conf",
                    ]
                ]
                .to_string(index=False, max_colwidth=40)
            )
        # Also show low-confidence rows so the analyst can eyeball them.
        low = mask & ~promote
        if low.any():
            print(f"\n  --- still unmatched (level1 conf < {threshold:.2f}) ---")
            print(
                out.loc[low, [
                    "employer", "n_donors",
                    "ml_level1_category", "ml_level1_category_conf",
                ]].to_string(index=False, max_colwidth=40)
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_path.stem}_with_ml.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  wrote: {out_path.relative_to(REPO_ROOT)}")
    return {"file": input_path.name, "unmatched": n_unmatched, "promoted": int(promote.sum()) if n_unmatched else 0}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--inputs", type=Path, nargs="*", default=DEFAULT_INPUTS)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = p.parse_args(argv)

    encoder = _load_encoder(args.model_dir)
    clfs = _load_classifiers(args.model_dir)

    results = []
    for inp in args.inputs:
        if not inp.exists():
            print(f"  skip (not found): {inp}")
            continue
        results.append(
            predict_for_file(inp, args.out_dir, encoder, clfs, args.threshold)
        )

    print("\n" + "=" * 70)
    print("RECAP")
    print("=" * 70)
    print(f"  {'file':55s} {'unmatched':>10} {'promoted':>10}")
    for r in results:
        print(f"  {r['file']:55s} {r['unmatched']:>10,} {r['promoted']:>10,}")


if __name__ == "__main__":
    main()
