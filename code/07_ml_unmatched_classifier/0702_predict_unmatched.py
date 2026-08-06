"""
0702_predict_unmatched.py

Apply the classifiers trained by 0701 to entities in 05_output classified
CSVs that have data_source_1 still null (unmatched by masterfile,
keyword, EDD, or non-company-individual sources).

For each unmatched entity, predict level1_category, level2_category, and
naics_code (where supported), along with model confidence (max softmax
probability).

The predicted-rows get data_source_1 = "ml prediction" when their level1
confidence is at or above --threshold (default 0.30); below that they
remain flagged unmatched, but the predictions are still written so the
analyst can inspect borderline cases.

Classification is computed once per unique entity (efficient — incl. live
EDD lookups upstream). The final `<stem>_with_ml.csv` is then written at the
DONATION level: the entity classification is broadcast back onto each
qualifying (>= --amount-min) donation, so the output has one row per
original donation rather than one row per employer/entity. This requires the
raw transactional file (derived as <raw-dir>/<stem>.csv or pinned via
--raw-inputs); if it is unavailable the entity-level result is written
instead. --amount-min must match the value used in the 0504 run that
produced the input.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Sibling modules (this dir is digit-prefixed, so make it importable).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyword_priors as kp  # noqa: E402
import text_features as tf  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "data/07_output_ml_classification/models"
DEFAULT_INPUTS = [
    REPO_ROOT / "output/05_output/lt_governor_race_2026_over_10k_classified.csv",
    REPO_ROOT / "output/05_output/insurance_commissioner_race_2026_over_10k_classified.csv",
]
DEFAULT_OUT_DIR = REPO_ROOT / "data/07_output_ml_classification"
# Raw transactional donation files (one row per donation). Used to expand the
# entity-level classification back to donation level. The matching raw file for
# an `output/05_output/<stem>_classified.csv` input is looked up here as
# <raw-dir>/<stem>.csv unless --raw-inputs overrides it.
DEFAULT_RAW_DIR = REPO_ROOT / "data/04_output_latest_data_pulls"
# Must match the --amount-min used when the 05 file was produced (0504), so the
# donations re-derived here are exactly the ones that were classified.
DEFAULT_AMOUNT_MIN = 10000.0
DEFAULT_THRESHOLD = 0.30

# Entity-level aggregate columns that are meaningless on a per-donation row and
# are dropped when expanding to donation level.
_ENTITY_AGG_COLS = {"n_donors", "amount_total"}


def _load_0504():
    """Dynamically load 0504 (digit-prefixed filename → not importable) to
    reuse its donation-level entity-assignment logic, so the donation→entity
    mapping stays defined in exactly one place."""
    path = (
        REPO_ROOT
        / "code/05_candidate_industry_affiliations/0504_classify_other_races.py"
    )
    spec = importlib.util.spec_from_file_location("other_races_classifier", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

TARGETS = ["level1_category", "level2_category", "naics_code"]


def _load_subcode_resolution():
    """Load the shared sub-code resolver (code/03_aggregating_data/
    subcode_resolution.py) — maps parent-level NAICS to the custom scheme
    (see CUSTOM_CODES.md). Digit-prefixed dir -> load by path."""
    path = REPO_ROOT / "code/03_aggregating_data/subcode_resolution.py"
    spec = importlib.util.spec_from_file_location("subcode_resolution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_0501():
    """Load 0501 to reuse apply_custom_label_overrides (the authoritative
    custom-code regex pass; see CUSTOM_CODES.md)."""
    path = (
        REPO_ROOT
        / "code/05_candidate_industry_affiliations/0501_classify_donors_with_keywords.py"
    )
    spec = importlib.util.spec_from_file_location("keywords_classifier", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _text_format(model_dir: Path) -> str:
    """Embed-text format the loaded models were trained with. Models saved
    before the v2 change carry no marker -> legacy 'v1'."""
    marker = model_dir / "text_format.txt"
    return marker.read_text().strip() if marker.exists() else "v1"


def build_embed_text_legacy(employer, occupation) -> str:
    """v1 (pre-text_features) format: employer-led, no junk handling.
    Kept so old models keep receiving the text format they were trained on."""
    emp = tf.clean_field(employer)
    occ = tf.clean_occupation(occupation)
    if emp and occ:
        return f"{emp}; occupation: {occ}"
    if emp:
        return emp
    if occ:
        return f"occupation: {occ}"
    return ""


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


def expand_to_donation_level(
    entity_out: pd.DataFrame, raw_path: Path, amount_min: float
) -> pd.DataFrame:
    """Broadcast the finalized entity-level classification back onto the
    individual donations, so the output is indexed by donation (one row per
    qualifying >= amount_min donation) instead of by entity.

    The entity classification was computed once per unique entity (efficient,
    incl. live EDD lookups). Here we re-derive the exact donation→entity
    mapping with 0504.assign_entities and left-join the entity columns onto
    each donation. Entity-level aggregate columns (n_donors, amount_total) are
    dropped; entity_kind / occupation_norm come from the per-donation frame.

    amount_min MUST match the value used to produce `entity_out` (the 0504
    run); the row count is cross-checked against entity_out['n_donors'].sum()
    and a mismatch is reported (almost always an amount_min mismatch).
    """
    mod = _load_0504()
    raw = mod.kw._load_input(raw_path)
    donations = mod.assign_entities(raw, amount_min).reset_index(drop=True)

    # Classification/ML columns to broadcast: everything from the entity-level
    # result except the aggregates and the columns the donation frame already
    # carries per-row. 'employer' is kept as the join key.
    drop = _ENTITY_AGG_COLS | {"entity_kind", "occupation_norm"}
    class_cols = [c for c in entity_out.columns if c not in drop]

    merged = donations.merge(
        entity_out[class_cols],
        left_on="entity",
        right_on="employer",
        how="left",
    ).drop(columns="entity")

    # Order: original donation columns first, then the classification block.
    raw_cols = [c for c in donations.columns if c != "entity"]
    appended = [c for c in merged.columns if c not in raw_cols]
    merged = merged[raw_cols + appended]

    n_expected = int(entity_out["n_donors"].sum()) if "n_donors" in entity_out else None
    if n_expected is not None and len(merged) != n_expected:
        print(
            f"  WARNING: donation rows ({len(merged):,}) != sum of entity "
            f"n_donors ({n_expected:,}). Check that --amount-min "
            f"({amount_min:g}) matches the 0504 run that produced this file."
        )
    return merged


def predict_for_file(
    input_path: Path,
    out_dir: Path,
    encoder,
    clfs: dict,
    threshold: float,
    raw_path: Path | None = None,
    amount_min: float = DEFAULT_AMOUNT_MIN,
    text_format: str = "v1",
    prior_threshold: float = kp.DEFAULT_PRIOR_THRESHOLD,
) -> dict:
    df = pd.read_csv(input_path)
    mask = df["data_source_1"].isna()
    n_unmatched = int(mask.sum())

    print(f"\n{input_path.name}")
    print(f"  rows total:      {len(df):>5,}")
    print(f"  rows unmatched:  {n_unmatched:>5,}")
    print(f"  embed-text format: {text_format}")

    pred_cols = {
        f"ml_{t}": pd.Series([pd.NA] * len(df), dtype="object") for t in clfs
    } | {f"ml_{t}_conf": pd.Series([np.nan] * len(df), dtype="float64") for t in clfs}

    if n_unmatched:
        occ_series = (
            df.loc[mask, "occupation_norm"]
            if "occupation_norm" in df.columns
            else pd.Series([""] * n_unmatched, index=df.index[mask])
        )
        kind_series = (
            df.loc[mask, "entity_kind"]
            if "entity_kind" in df.columns
            else pd.Series([""] * n_unmatched, index=df.index[mask])
        )
        if text_format == "v2":
            names = [
                tf.build_embed_text(e, o, entity_kind=k)
                for e, o, k in zip(df.loc[mask, "employer"], occ_series, kind_series)
            ]
        else:
            names = [
                build_embed_text_legacy(e, o)
                for e, o in zip(df.loc[mask, "employer"], occ_series)
            ]
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

    # ---- post-ML layers (see CUSTOM_CODES.md) ----
    if n_unmatched:
        # (1) High-signal keyword priors: override ML on the previously
        # unmatched rows where naics confidence < prior_threshold.
        out = kp.apply_keyword_priors(out, mask, threshold=prior_threshold)
    # (2) Shared sub-code resolver (replaces apply_custom_label_overrides;
    # see CUSTOM_CODES.md). The regex pass runs inside the resolver with
    # the same later-rows-win semantics, so ML never final-stamps a
    # political entity. Partitioned parents (52/56/77) are mapped to
    # terminal sub-codes, the occupation gate assigns 100, and results
    # land in custom_code / custom_label / resolution_method.
    sr = _load_subcode_resolution()
    schema = sr.load_schema()
    occ_col = "occupation" if "occupation" in out.columns else None
    out = sr.resolve_frame(out, schema, name_col="employer", occ_col=occ_col,
                           parent_col="naics_code", encoder=encoder)
    newly = out["custom_code"].notna() & out["data_source_1"].isna()
    out.loc[newly, "data_source_1"] = "custom resolver"
    # (3) 99 fallback: anything still uncoded is Unknown/Uncategorized
    # (per guidance). ml_* columns are preserved for inspection, and 99
    # rows should be re-attempted on future contributions.
    still = out["custom_code"].isna()
    if still.any():
        out.loc[still, "custom_code"] = "99"
        out.loc[still, "custom_label"] = "Uncategorized"
        out.loc[still, "data_source_1"] = out.loc[still, "data_source_1"].where(
            out.loc[still, "data_source_1"].notna(), "uncategorized fallback")
        print(f"  99 fallback: {int(still.sum()):,} rows -> Uncategorized")

    # Expand the entity-level result back to donation level (one row per
    # qualifying donation) when the raw transactional file is available and
    # this is an org-aware (0504) classification.
    n_donations = None
    if raw_path is not None and raw_path.exists() and "entity_kind" in out.columns:
        final = expand_to_donation_level(out, raw_path, amount_min)
        n_donations = len(final)
        print(
            f"  expanded to donation level: {len(out):,} entities -> "
            f"{n_donations:,} donations (raw: {raw_path.name})"
        )
    else:
        final = out
        if raw_path is not None and not raw_path.exists():
            print(
                f"  NOTE: raw file not found ({raw_path}); writing entity-level "
                f"output. Pass --raw-inputs/--raw-dir to expand to donation level."
            )
        elif "entity_kind" not in out.columns:
            print(
                "  NOTE: no entity_kind column (not a 0504 output); writing "
                "entity-level output."
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / f"{input_path.stem}_with_ml.csv").resolve()
    final.to_csv(out_path, index=False)
    try:
        shown = out_path.relative_to(REPO_ROOT)
    except ValueError:
        shown = out_path
    print(f"\n  wrote: {shown}  ({len(final):,} rows)")
    return {
        "file": input_path.name,
        "unmatched": n_unmatched,
        "promoted": int(promote.sum()) if n_unmatched else 0,
        "donations": n_donations,
    }


def _raw_path_for(input_path: Path, raw_dir: Path) -> Path:
    """Map an `output/05_output/<stem>_classified.csv` input back to its raw
    transactional file `<raw_dir>/<stem>.csv`."""
    stem = input_path.stem
    if stem.endswith("_classified"):
        stem = stem[: -len("_classified")]
    return raw_dir / f"{stem}.csv"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--inputs", type=Path, nargs="*", default=DEFAULT_INPUTS)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument(
        "--prior-threshold",
        type=float,
        default=kp.DEFAULT_PRIOR_THRESHOLD,
        help="keyword priors override ML when naics confidence < this",
    )
    # Donation-level expansion. --raw-inputs (parallel to --inputs) pins the
    # raw transactional file per input; otherwise each is derived as
    # <raw-dir>/<stem>.csv. --amount-min must match the 0504 run.
    p.add_argument("--raw-inputs", type=Path, nargs="*", default=None)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--amount-min", type=float, default=DEFAULT_AMOUNT_MIN)
    args = p.parse_args(argv)

    if args.raw_inputs is not None and len(args.raw_inputs) != len(args.inputs):
        raise SystemExit(
            f"--raw-inputs ({len(args.raw_inputs)}) must have one entry per "
            f"--inputs ({len(args.inputs)})."
        )

    encoder = _load_encoder(args.model_dir)
    clfs = _load_classifiers(args.model_dir)
    text_format = _text_format(args.model_dir)

    results = []
    for i, inp in enumerate(args.inputs):
        if not inp.exists():
            print(f"  skip (not found): {inp}")
            continue
        raw_path = (
            args.raw_inputs[i]
            if args.raw_inputs is not None
            else _raw_path_for(inp, args.raw_dir)
        )
        results.append(
            predict_for_file(
                inp,
                args.out_dir,
                encoder,
                clfs,
                args.threshold,
                raw_path=raw_path,
                amount_min=args.amount_min,
                text_format=text_format,
                prior_threshold=args.prior_threshold,
            )
        )

    print("\n" + "=" * 70)
    print("RECAP")
    print("=" * 70)
    print(f"  {'file':55s} {'unmatched':>10} {'promoted':>10} {'donations':>10}")
    for r in results:
        dn = f"{r['donations']:,}" if r.get("donations") is not None else "-"
        print(f"  {r['file']:55s} {r['unmatched']:>10,} {r['promoted']:>10,} {dn:>10}")


if __name__ == "__main__":
    main()
