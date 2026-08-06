"""Classify a raw race-donations export under the CUSTOM scheme.

PROCEDURAL script. End-to-end path for one PowerSearch/CalAccess race
export (e.g. governor 2026), using the two-masterfile design:

  1. Derive entities (individual -> employer; org/committee -> own name),
     aggregate donation totals, keep entities >= --amount-min (default 5k).
  2. Masterfile #2 lookup (running_list_custom.csv, built by 0306) on
     normalized entity name -> terminal custom codes directly.
  3. Unmatched entities -> ML at PARENT level (existing naics_code_clf,
     unchanged) with keyword-prior override below --prior-threshold,
     then the shared sub-code resolver (subcode_resolution.py) maps the
     parent prediction into the custom scheme (regex pass is authoritative
     and runs last, so ML can never final-stamp a political entity).
  4. Occupation gate (100 Retired/Homemaker/Student) and 99 fallback.

Outputs: <stem>_entities_custom.csv (one row per entity, with provenance)
and <stem>_donations_custom.csv (every qualifying donation stamped with
its entity's classification).

Usage:
    python 0705_classify_race_custom.py INPUT.csv [--amount-min 5000]
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import keyword_priors as kp   # noqa: E402
import text_features as tf    # noqa: E402

# subcode_resolution lives in 03_aggregating_data (digit-leading dir names
# aren't importable; load by path, same pattern 0702 uses for 0501).
_spec = importlib.util.spec_from_file_location(
    "subcode_resolution",
    REPO_ROOT / "code/03_aggregating_data/subcode_resolution.py",
)
sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sr)

MODELS_DIR = REPO_ROOT / "data/07_output_ml_classification/models"
DEFAULT_MF2 = REPO_ROOT / "data/03_input/masterfile/running_list_custom.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "data/07_output_ml_classification"
ML_MIN_CONF = 0.50   # promote ML parent prediction at/above this


def normalize_name(s: pd.Series) -> pd.Series:
    """Same normalization used to build the running list (0301/0305)."""
    return (
        s.fillna("").astype(str).str.upper()
        .str.replace(r"[^A-Z0-9 ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _col(df: pd.DataFrame, *cands: str) -> str | None:
    def key(c):
        return re.sub(r"[\s._]+", " ", str(c)).strip().lower()
    norm = {key(c): c for c in df.columns}
    for c in cands:
        if key(c) in norm:
            return norm[key(c)]
    return None


def aggregate_entities(df: pd.DataFrame, amount_min: float) -> pd.DataFrame:
    """One row per (entity, occupation) with total amount and donor count."""
    name_c = _col(df, "Contributor Name")
    emp_c = _col(df, "Contributor Employer")
    occ_c = _col(df, "Contributor Occupation")
    amt_c = _col(df, "Amount")
    if not (name_c and amt_c):
        raise SystemExit("Input needs 'Contributor Name' and 'Amount' columns.")

    w = pd.DataFrame({
        "contributor": df[name_c].map(tf.clean_field),
        "employer": df[emp_c].map(tf.clean_field) if emp_c else "",
        "occupation": df[occ_c].map(tf.clean_field) if occ_c else "",
        "amount": pd.to_numeric(df[amt_c], errors="coerce").fillna(0.0),
    })
    # Per-donation threshold, matching 0401_filter_10k_donations.py: only
    # contributions >= amount_min are classified (inclusive). Small
    # donations are dropped BEFORE aggregation, so entity totals below
    # reflect qualifying contributions only.
    n_all = len(w)
    w = w[w["amount"] >= amount_min]
    print(f"  donations >= ${amount_min:,.0f}: {len(w):,} of {n_all:,}")

    # Individual = "Last, First"-shaped name without org tokens.
    is_ind = w["contributor"].str.match(r"^[^,]+,\s*\S+") & ~w[
        "contributor"].str.contains(tf._ORG_TOKEN_RE, na=False)
    w["entity_kind"] = np.where(is_ind, "individual", "org")
    junk_emp = [tf.is_junk_employer(e, c) for e, c in zip(w["employer"], w["contributor"])]
    w["entity"] = np.where(
        is_ind & w["employer"].ne("") & ~pd.Series(junk_emp, index=w.index),
        w["employer"], w["contributor"],
    )

    keep = (
        w.groupby(["entity", "entity_kind", "occupation"], dropna=False)
        .agg(amount_total=("amount", "sum"), n_donations=("amount", "size"),
             employer=("employer", "first"), contributor=("contributor", "first"))
        .reset_index()
    )
    keep["name_norm"] = normalize_name(keep["entity"])
    print(f"  qualifying entities: {len(keep):,}")
    return keep


def match_masterfile2(ents: pd.DataFrame, mf2_path: Path) -> pd.DataFrame:
    mf2 = pd.read_csv(mf2_path, dtype=str).drop_duplicates("name_norm", keep="first")
    lut = mf2.set_index("name_norm")
    hit = ents["name_norm"].isin(lut.index) & ents["name_norm"].ne("")
    for col in ("custom_code", "custom_label", "resolution_method"):
        ents.loc[hit, col] = ents.loc[hit, "name_norm"].map(lut[col])
    ents.loc[hit, "data_source"] = "masterfile2"
    print(f"  masterfile #2 matches: {int(hit.sum()):,} / {len(ents):,}")
    return ents


def ml_parent_predict(ents: pd.DataFrame, mask: pd.Series, encoder,
                      prior_threshold: float) -> pd.DataFrame:
    """Parent-level ML (existing model, unchanged) + keyword priors for the
    still-unmatched rows. Writes ml_parent_code / ml_parent_conf."""
    clf = joblib.load(MODELS_DIR / "naics_code_clf.joblib")
    texts = [
        tf.build_embed_text(e, o, entity_kind=k, contributor_name=c)
        for e, o, k, c in zip(
            ents.loc[mask, "entity"], ents.loc[mask, "occupation"],
            ents.loc[mask, "entity_kind"], ents.loc[mask, "contributor"])
    ]
    emb = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    proba = clf.predict_proba(emb)
    idx = ents.index[mask]
    ents.loc[idx, "ml_parent_code"] = clf.classes_[proba.argmax(axis=1)]
    ents.loc[idx, "ml_parent_conf"] = proba.max(axis=1)

    # Keyword priors override low-confidence ML (same rule as 0702).
    for i in idx:
        conf = ents.at[i, "ml_parent_conf"]
        if pd.isna(conf) or conf < prior_threshold:
            prior = kp.prior_for(ents.at[i, "employer"], ents.at[i, "occupation"])
            if prior:
                ents.at[i, "ml_parent_code"] = prior
                ents.at[i, "prior_fired"] = prior
    promoted = mask & (ents["ml_parent_conf"].ge(ML_MIN_CONF)
                       | ents["prior_fired"].notna())
    print(f"  ML/prior parent codes assigned: {int(promoted.sum()):,} "
          f"(of {int(mask.sum()):,} unmatched)")
    return ents


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input", type=Path)
    p.add_argument("--mf2", type=Path, default=DEFAULT_MF2)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--amount-min", type=float, default=5000.0)
    p.add_argument("--prior-threshold", type=float, default=kp.DEFAULT_PRIOR_THRESHOLD)
    args = p.parse_args(argv)

    print("=" * 70)
    print(f"CLASSIFY RACE (custom scheme): {args.input.name}")
    print("=" * 70)

    raw = pd.read_csv(args.input, dtype=str)
    ents = aggregate_entities(raw, args.amount_min)
    for col in ("custom_code", "custom_label", "resolution_method",
                "data_source", "ml_parent_code", "prior_fired"):
        ents[col] = pd.NA
    ents["ml_parent_conf"] = np.nan

    # (2) Masterfile #2 lookup.
    ents = match_masterfile2(ents, args.mf2)

    # (3) ML at parent level for the rest, then the shared resolver.
    schema = sr.load_schema()
    encoder = sr.load_encoder()
    unmatched = ents["data_source"].isna()
    if unmatched.any():
        ents = ml_parent_predict(ents, unmatched, encoder, args.prior_threshold)
        sub = ents.loc[unmatched].copy()
        promoted = sub["ml_parent_conf"].ge(ML_MIN_CONF) | sub["prior_fired"].notna()
        sub["parent_for_resolver"] = sub["ml_parent_code"].where(promoted)
        resolved = sr.resolve_frame(
            sub, schema, name_col="entity", occ_col="occupation",
            parent_col="parent_for_resolver", encoder=encoder,
        )
        for col in ("custom_code", "custom_label", "resolution_method"):
            ents.loc[resolved.index, col] = resolved[col]
        ents.loc[resolved.index, "resolution_confidence"] = resolved["resolution_confidence"]
        ents.loc[unmatched & ents["custom_code"].notna(), "data_source"] = "ml+resolver"

    # (4) 99 fallback.
    still = ents["custom_code"].isna()
    labels = dict(zip(schema["naics_code"], schema["naics_label"]))
    ents.loc[still, ["custom_code", "custom_label"]] = ["99", labels["99"]]
    ents.loc[still, "data_source"] = "uncategorized fallback"
    print(f"  99 fallback: {int(still.sum()):,} entities")

    print("\n  classification by custom code ($ total):")
    rec = ents.groupby(["custom_code", "custom_label"])["amount_total"].sum()
    print((rec.sort_values(ascending=False) / 1e6).round(2).to_string())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ent_path = args.out_dir / f"{args.input.stem}_entities_custom.csv"
    ents.to_csv(ent_path, index=False)
    print(f"\n  Wrote: {ent_path}  ({len(ents):,} entities)")

    # Donation-level: stamp each qualifying donation with its entity's code.
    name_c = _col(raw, "Contributor Name")
    emp_c = _col(raw, "Contributor Employer")
    occ_c = _col(raw, "Contributor Occupation")
    raw2 = raw.copy()
    # Same per-donation threshold as aggregation (0401 semantics).
    raw2 = raw2[pd.to_numeric(raw2[_col(raw, "Amount")], errors="coerce")
                .fillna(0.0) >= args.amount_min]
    raw2["_contrib"] = raw2[name_c].map(tf.clean_field)
    raw2["_emp"] = raw2[emp_c].map(tf.clean_field) if emp_c else ""
    raw2["_occ"] = raw2[occ_c].map(tf.clean_field) if occ_c else ""
    is_ind = raw2["_contrib"].str.match(r"^[^,]+,\s*\S+") & ~raw2[
        "_contrib"].str.contains(tf._ORG_TOKEN_RE, na=False)
    junk = [tf.is_junk_employer(e, c) for e, c in zip(raw2["_emp"], raw2["_contrib"])]
    raw2["_entity"] = np.where(is_ind & raw2["_emp"].ne("") & ~pd.Series(junk),
                               raw2["_emp"], raw2["_contrib"])
    raw2["_key"] = normalize_name(raw2["_entity"]) + "||" + raw2["_occ"].str.upper()
    ents["_key"] = ents["name_norm"] + "||" + ents["occupation"].str.upper()
    stamp = ents.set_index("_key")[
        ["custom_code", "custom_label", "data_source", "resolution_method"]]
    don = raw2.join(stamp, on="_key").drop(
        columns=["_contrib", "_emp", "_occ", "_entity", "_key"])
    don = don[don["custom_code"].notna()]
    don_path = args.out_dir / f"{args.input.stem}_donations_custom.csv"
    don.to_csv(don_path, index=False)
    print(f"  Wrote: {don_path}  ({len(don):,} donations)")


if __name__ == "__main__":
    main()
