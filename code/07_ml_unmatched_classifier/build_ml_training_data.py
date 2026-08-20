"""build_ml_training_data.py

Assembles a training file from the three data sources
produced by the 08 alternative pipeline, translating all NAICS codes to the
current custom sector scheme. Pass the output to 0701_train_classifier.py
via --train.

Sources (priority for deduplication, highest first):
  C  08_inputs/already_classified_contributions.csv
    - these are contributions that already went through the entire classification pipeline
    - only rows where code_final_source is "John", a keyword/regex match, or an
       identity override (not including LLM or ML contributions that are less certain)
    - codes 99 and 100 are excluded from all sources
  B  data/03_input/masterfile/running_list_opensecrets_alt.csv  (OpenSecrets)
  A  data/03_input/masterfile/running_list_alt.csv  (H1B + EDD)

NAICS code maps are loaded from Google Sheets at runtime. 

Usage
-----
    python build_ml_training_data.py
    python build_ml_training_data.py --out path/to/running_list_combined.csv

Then retrain (no changes to 0701):
    python code/07_ml_unmatched_classifier/0701_train_classifier.py \\
        --train data/03_input/masterfile/running_list_combined_YYYY-MM-DD.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# text_features is in the same package directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
import text_features as tf  # noqa: E402


# Paths ---------------------------------------------------------------------------

CODE_ROOT = Path(__file__).resolve().parent.parent           # campfin/code/
DATA_ROOT = CODE_ROOT.parent / "data"                        # campfin/data/

DEFAULT_RUNNING_LIST   = DATA_ROOT / "03_input" / "masterfile" / "running_list_alt.csv"
DEFAULT_OPENSECRETS    = DATA_ROOT / "03_input" / "masterfile" / "running_list_opensecrets_alt.csv"
DEFAULT_PRE_CLASSIFIED = CODE_ROOT / "08_alternative_pipeline" / "08_inputs" / "already_classified_contributions.csv"
DEFAULT_OUT = DATA_ROOT / "03_input" / "masterfile" / f"running_list_combined_{date.today().isoformat()}.csv"

OUTPUT_COLS = [
    "name", "name_norm", "level1_category", "level2_category",
    "level3_category", "naics_code", "naics_label", "source",
]

# Load normalize_name from 0805 via importlib (digit-prefix module) ---------------

_spec = importlib.util.spec_from_file_location(
    "classify_contributors_0805",
    CODE_ROOT / "08_alternative_pipeline" / "0805_classify_contributors.py",
)
_m0805 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _m0805
_spec.loader.exec_module(_m0805)

normalize_name = _m0805.normalize_name   # pd.Series -> pd.Series

# Google Sheet URLs ---------------------------------------------------------------

# 4/6-digit NAICS code -> custom sector (used FIRST before truncating to 2 digits)
NAICS_DETAIL_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1gHfG8iSJn-IZez3hs8DpeUNQmQyT0pq7YW9YQ4E_sm4"
    "/export?format=csv&gid=0"
)

# Subsector (old/2-digit code) -> sector (new custom code)
SUBSECTOR_MAP_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "11QHvNJsdtMlc1YKo_iNvMB_Jfn5Ui-iYdlWhFYjSm9g"
    "/export?format=csv&gid=701123226"
)

# Sector code -> sector_description (human-readable label)
SECTOR_LABELS_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "11QHvNJsdtMlc1YKo_iNvMB_Jfn5Ui-iYdlWhFYjSm9g"
    "/export?format=csv&gid=605391596"
)

# OpenSecrets level3_category -> custom naics code
OS_LEVEL3_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1wiBdiEzOG_XxlNqCXLedMoConOU6WQsRZbrhTFlYywA"
    "/export?format=csv&gid=2010624656"
)


# Reference map loaders -----------------------------------------------------------

def _load_naics_detail_map(url: str = NAICS_DETAIL_URL) -> dict[str, str]:
    """4/6-digit NAICS code -> custom sector code (Sheet 1).
    Applied before truncating to 2 digits."""
    df = pd.read_csv(url, dtype=str).fillna("")
    return {
        row["naics_code"].strip(): row["sector"].strip()
        for _, row in df.iterrows()
        if row["naics_code"].strip() and row["sector"].strip()
    }


def _load_subsector_map(url: str = SUBSECTOR_MAP_URL) -> dict[str, str]:
    """Subsector (old/2-digit code) -> sector (new custom code).
    Applied after truncating to 2 digits when no detail-map match exists."""
    df = pd.read_csv(url, dtype=str).fillna("")
    return {
        row["subsector"].strip(): row["sector"].strip()
        for _, row in df.iterrows()
        if row["subsector"].strip() and row["sector"].strip()
    }


def _load_code_labels(url: str = SECTOR_LABELS_URL) -> dict[str, str]:
    """Custom sector code -> sector description.
    Deduplicates on sector since codes repeat when subsectors are listed."""
    df = pd.read_csv(url, dtype=str).fillna("")
    return (
        df[df["sector"].str.strip() != ""]
        .drop_duplicates(subset="sector", keep="first")
        .set_index("sector")["sector_description"]
        .to_dict()
    )


def _load_os_level3_crosswalk(url: str = OS_LEVEL3_URL) -> dict[str, str]:
    """OpenSecrets level3_category -> custom naics code."""
    df = pd.read_csv(url, dtype=str).fillna("")
    return {
        row["Open Secrets Level 3 Categories"].strip(): row["custom_naics"].strip()
        for _, row in df.iterrows()
        if row["Open Secrets Level 3 Categories"].strip() and row["custom_naics"].strip()
    }


# Helpers -------------------------------------------------------------------------

"""
This catches self-employed employer strings that is_junk_employer() misses 
Handles: "SELF / WINNINGRESULTS", "SELF - NO SEPARATE BUSINESS NAME",
          "(SELF-EMPLOYED), ALBERT LASSAGA", "FULLMER, SELF EMPLOYED-JAMES",
          "KRIEGER, SELF: KAITLYN J."
"""

_SELF_EMPLOYER_RE = re.compile(
    r"(?:^\s*\(?\s*|,\s*)\bself\b\s*(?:employ\w*|[\-/:,]|$)",
    re.IGNORECASE,
)

# Placeholder employer values not caught by is_junk_employer (e.g. "BLANK BLANK", "N.A.")
_PLACEHOLDER_EMPLOYER_RE = re.compile(
    r"^\s*(?:(?:blank\s*)+|n\.?\s*a\.?)\s*$",
    re.IGNORECASE,
)

# "RETIRED" / "RETIRE" as an occupation — not in text_features.JUNK_OCCUPATIONS
_RETIREMENT_OCC_RE = re.compile(r"^\s*retired?\s*$", re.IGNORECASE)


def _is_unusable_individual(employer: str, occupation: str) -> bool:
    """True when an individual contributor row should be dropped from training.

    Covers cases that text_features.is_junk_employer / JUNK_OCCUPATIONS miss:
    - Retirement occupation (no industry signal)
    - Self-employed employer variants not anchored at string start
    - Placeholder employer values like "BLANK" or "N.A."
    """
    if _RETIREMENT_OCC_RE.match(occupation.strip()):
        return True
    if _SELF_EMPLOYER_RE.search(employer):
        return True
    if _PLACEHOLDER_EMPLOYER_RE.match(employer.strip()):
        return True
    return False


def _parent_code(code: str) -> str:
    """Truncate a 4/6-digit NAICS code to its 2-digit parent group."""
    code = str(code).strip()
    if re.match(r"^\d{3,6}$", code):
        p = code[:2]
        if p in ("31", "32", "33"):
            return "31-33"
        if p in ("44", "45"):
            return "44-45"
        if p in ("48", "49"):
            return "48-49"
        return p
    return code


def _translate(
    raw: str,
    detail_map: dict[str, str],
    subsector_map: dict[str, str],
    code_labels: dict[str, str],
) -> str:
    """Translate a raw NAICS or subsector code to the current custom sector scheme.

    Order:
      0. Already a known sector code -> pass through unchanged
      1. Direct lookup in 4/6-digit detail map (e.g. 221114 -> 20)
      2. Direct lookup in subsector map (handles alphanumeric codes like 22a, 52a)
      3. Truncate to 2-digit parent, then look up in subsector map (e.g. 6211 -> 62 -> 60)
    """
    raw = str(raw).strip() if pd.notna(raw) else ""
    if not raw:
        return ""
    if raw in code_labels:
        return raw
    if raw in detail_map:
        return detail_map[raw]
    if raw in subsector_map:
        return subsector_map[raw]
    parent = _parent_code(raw)
    return subsector_map.get(parent, "")


def _apply_scheme_labels(df: pd.DataFrame, code_labels: dict[str, str]) -> pd.DataFrame:
    """Fill naics_label from the sector labels map. Does NOT touch
    level1/level2/level3."""
    df = df.copy()
    df["naics_label"] = df["naics_code"].map(code_labels).fillna(df.get("naics_label", ""))
    return df


_EXCLUDE_CODES = {"99", "100", ""}

# code_final_source patterns to INCLUDE from already_classified_contributions.csv:
# only want to include the ones we're very confident in for training
#   "John*"       manually verified by John
#   "*regex*"     keyword/regex match (e.g. "masterfile2: regex") - not sure about including these
#   "*keyword*"   keyword match (0805 style)
#   "*identity*"  identity-based override (0805 style)
#   "*override*"  identity-based override variant
def _keep_source(s: pd.Series) -> pd.Series:
    sl = s.str.lower()
    return (
        sl.str.startswith("john")
        | sl.str.contains("regex",    na=False)
        | sl.str.contains("keyword",  na=False)
        | sl.str.contains("identity", na=False)
        | sl.str.contains("override", na=False)
    )


# Source loaders ------------------------------------------------------------------

def _load_source_a(
    path: Path,
    detail_map: dict[str, str],
    subsector_map: dict[str, str],
    code_labels: dict[str, str],
) -> pd.DataFrame:
    """H1B + EDD running list.

    Translation order for each raw NAICS code:
      1. Exact match in 4/6-digit detail map
      2. Truncate to 2-digit parent, then look up in subsector map
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    df["naics_code"] = df["naics_code"].apply(
        lambda c: _translate(c, detail_map, subsector_map, code_labels)
    )
    df = df[~df["naics_code"].isin(_EXCLUDE_CODES)].copy()
    df = _apply_scheme_labels(df, code_labels)
    df["name_norm"] = normalize_name(df["name"].astype(str))
    if "level3_category" not in df.columns:
        df["level3_category"] = ""
    return df[OUTPUT_COLS].copy()


def _load_source_b(
    path: Path,
    level3_map: dict[str, str],
    code_labels: dict[str, str],
) -> pd.DataFrame:
    """OpenSecrets running list.

    naics_code is derived solely from level3_category via the crosswalk sheet.
    level1/level2/level3 are preserved as is.

    Organizations: the entity name is the training signal (same as source A).
    Individuals: the employer + occupation is the training signal, formatted
    with build_embed_text so training matches the serving-side text format.
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    df["naics_code"] = df["level3_category"].map(level3_map).fillna("")
    if "level3_category" not in df.columns:
        df["level3_category"] = ""
    # Source B can have empty naics_code (no level3 match) — keep those since they
    # still carry level1/level2 signal. Exclude only 99 and 100.
    df = df[~df["naics_code"].isin({"99", "100"})].copy()

    is_ind = df["entity_type"].str.lower() == "individual"

    org_df = df[~is_ind].copy()
    org_df["name_norm"] = normalize_name(org_df["name"].astype(str))

    ind_df = df[is_ind].copy()
    # Drop individuals with retirement occupations, self-employed employer strings,
    # or placeholder employers before building embed text.
    ind_keep = [
        not _is_unusable_individual(e, o)
        for e, o in zip(ind_df["employer"], ind_df["occupation"])
    ]
    ind_df = ind_df[ind_keep].copy()
    ind_df["name"] = [
        tf.build_embed_text(e, o, entity_kind="individual")
        for e, o in zip(ind_df["employer"], ind_df["occupation"])
    ]
    ind_df["name_norm"] = normalize_name(ind_df["employer"].astype(str))
    ind_df = ind_df[ind_df["name"] != ""]

    df = pd.concat([org_df, ind_df], ignore_index=True)
    df = _apply_scheme_labels(df, code_labels)
    return df[OUTPUT_COLS].copy()


def _load_source_c(
    path: Path,
    detail_map: dict[str, str],
    subsector_map: dict[str, str],
    code_labels: dict[str, str],
) -> pd.DataFrame:
    """Pre-classified contributions.

    Only rows manually verified by John, matched by keyword/regex, or from
    identity overrides. Codes 99 and 100 are excluded.
    code_final values are translated through the same subsector/detail maps as
    source A (many are old subsector codes like 52a, 77a, 71, etc.).
    level1/level2/level3 are left empty.

    Individuals (have a valid Contributor.Employer) are encoded with
    build_embed_text in v2 format — "occupation: X; employer: Y" — matching
    the serving-side text format used by run_ml_on_08_output.py.
    Organizations (no valid employer) use Contributor.Name directly.
    Rows where build_embed_text returns "" (junk employer + no occupation,
    or employer is the contributor's own name with no occupation) are dropped.
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df[
        _keep_source(df["code_final_source"])
        & ~df["code_final"].isin(_EXCLUDE_CODES)
    ].copy()
    df["naics_code"] = df["code_final"].apply(
        lambda c: _translate(c, detail_map, subsector_map, code_labels)
    )
    df = df[~df["naics_code"].isin(_EXCLUDE_CODES)].copy()

    # build_embed_text below uses contributor_name so
    # the remaining edge case (employer = own name) is handled there.
    is_ind = ~df["Contributor.Employer"].apply(tf.is_junk_employer)

    ind_df = df[is_ind].copy()
    ind_keep = [
        not _is_unusable_individual(e, o)
        for e, o in zip(ind_df["Contributor.Employer"], ind_df["Contributor.Occupation"])
    ]
    ind_df = ind_df[ind_keep].copy()
    ind_df["name"] = [
        tf.build_embed_text(e, o, entity_kind="individual", contributor_name=cn)
        for e, o, cn in zip(
            ind_df["Contributor.Employer"],
            ind_df["Contributor.Occupation"],
            ind_df["Contributor.Name"],
        )
    ]
    ind_df["name_norm"] = normalize_name(ind_df["Contributor.Employer"].astype(str))
    ind_df = ind_df[ind_df["name"] != ""]

    org_df = df[~is_ind].copy()
    org_df["name"] = org_df["Contributor.Name"]
    org_df["name_norm"] = normalize_name(org_df["Contributor.Name"].astype(str))

    df = pd.concat([ind_df, org_df], ignore_index=True)
    df["source"]          = "pre_classified"
    df["level1_category"] = ""
    df["level2_category"] = ""
    df["level3_category"] = ""
    df["naics_label"]     = ""
    df = _apply_scheme_labels(df, code_labels)
    return df[OUTPUT_COLS].copy()


# Main ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Assemble ML training data from 08 pipeline sources.")
    p.add_argument("--running-list",   type=Path, default=DEFAULT_RUNNING_LIST,
                   help="H1B+EDD running list (running_list_alt.csv)")
    p.add_argument("--opensecrets",    type=Path, default=DEFAULT_OPENSECRETS,
                   help="OpenSecrets running list (running_list_opensecrets_alt.csv)")
    p.add_argument("--pre-classified", type=Path, default=DEFAULT_PRE_CLASSIFIED,
                   help="Pre-classified contributions (already_classified_contributions.csv)")
    p.add_argument("--out",            type=Path, default=DEFAULT_OUT,
                   help="Output path for combined training CSV (must not already exist)")
    args = p.parse_args(argv)

    if args.out.exists():
        p.error(
            f"Output file already exists: {args.out}\n"
            "Rename or delete it first, or pass a different --out path."
        )

    print("Loading reference maps from Google Sheets...")
    try:
        detail_map = _load_naics_detail_map()
        print(f"  4/6-digit NAICS detail map: {len(detail_map)} entries")
    except Exception as exc:
        print(f"  WARNING: could not load NAICS detail map ({exc}); skipping step 1 of translation")
        detail_map = {}

    try:
        subsector_map = _load_subsector_map()
        print(f"  Subsector -> sector map: {len(subsector_map)} entries")
    except Exception as exc:
        print(f"  WARNING: could not load subsector map ({exc}); 2-digit codes may not translate")
        subsector_map = {}

    try:
        code_labels = _load_code_labels()
        print(f"  Sector labels: {len(code_labels)} codes")
    except Exception as exc:
        print(f"  WARNING: could not load sector labels ({exc}); naics_label will be empty")
        code_labels = {}

    try:
        level3_map = _load_os_level3_crosswalk()
        print(f"  OpenSecrets level3 crosswalk: {len(level3_map)} entries")
    except Exception as exc:
        print(f"  WARNING: could not load OS level3 crosswalk ({exc}); naics_code will be empty for source B")
        level3_map = {}

    print(f"\nSource A: {args.running_list}")
    src_a = _load_source_a(args.running_list, detail_map, subsector_map, code_labels)
    print(f"  {len(src_a):,} rows, {src_a['naics_code'].nunique()} distinct codes")

    print(f"\nSource B: {args.opensecrets}")
    src_b = _load_source_b(args.opensecrets, level3_map, code_labels)
    n_no_code = (src_b["naics_code"] == "").sum()
    print(f"  {len(src_b):,} rows, {n_no_code:,} without naics_code (no level3 match)")

    print(f"\nSource C: {args.pre_classified}")
    src_c = _load_source_c(args.pre_classified, detail_map, subsector_map, code_labels)
    print(f"  {len(src_c):,} rows (John/keyword/identity only; codes 99 and 100 excluded)")

    # Combine: A < B < C (last kept in dedup, so C has highest priority)
    combined = pd.concat([src_a, src_b, src_c], ignore_index=True)
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=["name_norm"], keep="last")
    print(f"\nCombined: {n_before:,} → {len(combined):,} rows after dedup by name_norm")

    print("\nRows per new-scheme code:")
    print(combined["naics_code"].value_counts().to_string())

    unrecognized = [c for c in combined["naics_code"].unique() if c and c not in code_labels]
    if unrecognized:
        print(f"\nWARNING: {len(unrecognized)} unrecognized codes still present: {sorted(unrecognized)[:10]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"\nWrote: {args.out}")
    print("\nNext step:")
    print(f"  python code/07_ml_unmatched_classifier/0701_train_classifier.py \\")
    print(f"      --train {args.out}")


if __name__ == "__main__":
    main()
