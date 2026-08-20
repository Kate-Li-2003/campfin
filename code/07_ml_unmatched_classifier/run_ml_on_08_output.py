"""run_ml_on_08_output.py

Runs ML inference on all entities from the 08 alternative pipeline and writes
per-entity ML predictions that the 10-pipeline Rmd joins for consensus
resolution with LLM output.

Reads:
  classification_input_combined.csv  (all above-$5k contributions, direct + PAC)

Writes:
  data/07_output_ml_classification/08_entities_with_ml.csv
  Columns: entity_id, ml_naics_code, ml_naics_code_conf,
           ml_level2_category, ml_level2_category_conf, prior_naics_code
  One row per unique classification unit (unique org name or unique
  employer+occupation pair); entity_id aliases are kept separate.

Usage
-----
    python run_ml_on_08_output.py
    python run_ml_on_08_output.py --model-dir data/07_output_ml_classification/models
    python run_ml_on_08_output.py --input path/to/classification_input_combined.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd 

# Imports from the same 07 directory (not modified)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import text_features as tf
import keyword_priors as kp

# Paths ---------------------------------------------------------------------

CODE_ROOT  = _HERE.parent
DATA_ROOT  = CODE_ROOT.parent / "data"

DEFAULT_INPUT     = CODE_ROOT / "08_alternative_pipeline" / "08_outputs" / "classification_input_combined.csv"
DEFAULT_MODEL_DIR = DATA_ROOT / "07_output_ml_classification" / "models"
DEFAULT_OUT       = DATA_ROOT / "07_output_ml_classification" / "08_entities_with_ml.csv"

# ---------------------------------------------------------------------------
# Code translation: keyword prior old-scheme -> new custom sector scheme
# keyword_priors.py fires old NAICS-style codes; translate via Google Sheets.
# ---------------------------------------------------------------------------

_SUBSECTOR_MAP_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "11QHvNJsdtMlc1YKo_iNvMB_Jfn5Ui-iYdlWhFYjSm9g"
    "/export?format=csv&gid=701123226"
)


def _load_prior_translation(url: str = _SUBSECTOR_MAP_URL) -> dict[str, str]:
    """Load subsector→sector map from Google Sheets (same sheet as build_ml_training_data)."""
    df = pd.read_csv(url, dtype=str).fillna("")
    return {
        row["subsector"].strip(): row["sector"].strip()
        for _, row in df.iterrows()
        if row["subsector"].strip() and row["sector"].strip()
    }


def _translate_prior_code(code: str | None, prior_map: dict[str, str]) -> str | None:
    if not code:
        return None
    c = str(code).strip()
    return prior_map.get(c, c)  # pass through if already a current sector code


# Load model -------------------------------------------------------------

def _load_models(model_dir: Path) -> dict:
    encoder_name = (model_dir / "encoder_name.txt").read_text().strip()
    text_fmt = "v2"
    fmt_file = model_dir / "text_format.txt"
    if fmt_file.exists():
        text_fmt = fmt_file.read_text().strip()

    print(f"  encoder : {encoder_name}")
    print(f"  format  : {text_fmt}")

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(encoder_name)

    clf_naics = joblib.load(model_dir / "naics_code_clf.joblib")
    clf_l1    = joblib.load(model_dir / "level1_category_clf.joblib")
    clf_l2    = joblib.load(model_dir / "level2_category_clf.joblib")

    return {
        "encoder":    encoder,
        "text_fmt":   text_fmt,
        "clf_naics":  clf_naics,
        "clf_l1":     clf_l1,
        "clf_l2":     clf_l2,
    }


# Embed text ----------------------------------------------------

def _build_texts(df: pd.DataFrame, text_fmt: str) -> pd.Series:
    """Build embed strings using text_features.build_embed_text() (format v2).

    Mirrors 0705's entity-derivation: for individuals with a valid employer,
    classify by employer; for orgs (and individuals with no valid employer),
    classify by Contributor.Name. The resolved entity name is passed as the
    'employer' parameter to build_embed_text, same as 0705 does.
    """
    texts = []
    for _, row in df.iterrows():
        employer  = str(row.get("employer", "") or "")
        occ       = str(row.get("occupation", "") or "")
        kind      = str(row.get("entity_type", "") or "")
        cname     = str(row.get("Contributor.Name", "") or "")
        # Orgs have no employer column in the 08 output — use Contributor.Name,
        # consistent with 0705 which falls back to contributor name for non-individuals.
        entity = employer if kind.lower() == "individual" else cname
        if text_fmt == "v2":
            texts.append(tf.build_embed_text(entity, occ, kind, cname))
        else:
            # v1 fallback: plain occupation text
            texts.append(tf.occupation_train_text(occ))
    return pd.Series(texts, index=df.index)


# Predictions ----------------------------------------------------

def _run_inference(entities: pd.DataFrame, models: dict, prior_map: dict) -> pd.DataFrame:
    encoder   = models["encoder"]
    text_fmt  = models["text_fmt"]
    clf_naics = models["clf_naics"]
    clf_l2    = models["clf_l2"]

    texts = _build_texts(entities, text_fmt)
    empty_mask = texts.str.strip() == ""
    if empty_mask.all():
        print("  WARNING: all embed texts are empty — check employer/occupation columns")

    print(f"  Encoding {len(entities):,} entities ({empty_mask.sum():,} empty texts)...")
    embeddings = encoder.encode(texts.tolist(), batch_size=256, show_progress_bar=True)

    print("  Predicting naics_code...")
    naics_pred = clf_naics.predict(embeddings)
    naics_prob = clf_naics.predict_proba(embeddings)
    naics_conf = naics_prob.max(axis=1)

    print("  Predicting level2_category...")
    l2_pred = clf_l2.predict(embeddings)
    l2_prob = clf_l2.predict_proba(embeddings)
    l2_conf = l2_prob.max(axis=1)

    print("  Applying keyword priors...")
    prior_codes_raw = [
        kp.prior_for(
            str(row.get("employer", "") or ""),
            str(row.get("occupation", "") or ""),
        )
        for _, row in entities.iterrows()
    ]
    prior_codes = [_translate_prior_code(c, prior_map) for c in prior_codes_raw]

    out = pd.DataFrame({
        "entity_id":               entities["entity_id"].values,
        "entity_type":             entities["entity_type"].values,
        "Contributor.Name":        entities["Contributor.Name"].values,
        "employer":                entities["employer"].values,   # standardized_employer_name
        "occupation":              entities["occupation"].values, # standardized_occupation
        "ml_naics_code":           naics_pred,
        "ml_naics_code_conf":      naics_conf.round(4),
        "ml_level2_category":      l2_pred,
        "ml_level2_category_conf": l2_conf.round(4),
        "prior_naics_code":        prior_codes,
    })

    # Override low-confidence ML predictions with keyword prior where it fired
    prior_fired = out["prior_naics_code"].notna()
    low_conf    = out["ml_naics_code_conf"] < kp.DEFAULT_PRIOR_THRESHOLD
    override    = prior_fired & low_conf
    if override.any():
        out.loc[override, "ml_naics_code"] = out.loc[override, "prior_naics_code"]
        print(f"  Keyword prior overrode ML for {override.sum():,} low-confidence entities")

    return out


# Main ----------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run ML inference on 08 pipeline entities.")
    p.add_argument("--input",     type=Path, default=DEFAULT_INPUT,
                   help="classification_input_combined.csv from 08 pipeline")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                   help="Directory containing trained model artifacts")
    p.add_argument("--out",       type=Path, default=DEFAULT_OUT,
                   help="Output path for ML predictions CSV")
    args = p.parse_args(argv)

    # Load prior translation map from Google Sheets
    print("Loading prior code translation map...")
    try:
        prior_map = _load_prior_translation()
        print(f"  {len(prior_map)} subsector→sector entries")
    except Exception as exc:
        print(f"  WARNING: could not load prior translation ({exc}); codes passed through unchanged")
        prior_map = {}

    # Load models
    print(f"Loading models from {args.model_dir} ...")
    models = _load_models(args.model_dir)

    # Read input file
    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        return
    all_rows = pd.read_csv(args.input, dtype=str).fillna("")
    print(f"  {len(all_rows):,} rows from {args.input.name}")

    # Normalise column names: combined file uses standardized_employer_name /
    # standardized_occupation; the rest of this script expects employer / occupation.
    all_rows = all_rows.rename(columns={
        "standardized_employer_name": "employer",
        "standardized_occupation":    "occupation",
    })

    # One prediction per unique classification unit:
    #   orgs       → unique Contributor.Name
    #   individuals → unique (employer, occupation) pair
    # Multiple name-variants for the same entity_id are kept separate because
    # aliases have not yet been confirmed; conflicts are resolved downstream.
    is_ind = all_rows["entity_type"].str.lower() == "individual"
    all_rows["_dedup_key"] = np.where(
        is_ind,
        all_rows["employer"].str.strip() + "|" + all_rows["occupation"].str.strip(),
        all_rows["Contributor.Name"].str.strip(),
    )
    entities = (
        all_rows
        .drop_duplicates(subset=["entity_type", "_dedup_key"], keep="first")
        .drop(columns=["_dedup_key"])
        .copy()
    )
    print(f"\n{len(entities):,} unique classification units (from {len(all_rows):,} total rows)")

    # Run inference
    print("\nRunning ML inference...")
    results = _run_inference(entities, models, prior_map)

    # Summary
    print(f"\nCode distribution (ml_naics_code):")
    print(results["ml_naics_code"].value_counts().to_string())
    n_prior = results["prior_naics_code"].notna().sum()
    print(f"\nKeyword prior fired for {n_prior:,} / {len(results):,} entities")

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    print(f"\nWrote: {args.out}")
    print("\nNext step: join 08_entities_with_ml.csv to classified_contributors.csv")
    print("by entity_id in the 10-pipeline assign_*_final_classification.Rmd files.")


if __name__ == "__main__":
    main()
