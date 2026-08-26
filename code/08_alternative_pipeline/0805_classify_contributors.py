"""
0805_classify_contributors.py

Alternative classification pipeline for pre-aggregated contributor data.
Operates directly on classification_input.csv (output of entity-resolution
stage 0802_entity_resolution_simplified).

  Part 0:  Pre-classified direct match — TEMPORARY (may incorporate into 0802 instead)
        - pre-classified data are the contributors that have already been classified in this pipeline
        - Organizations: match based on Contributor.Name alone
        - Individuals:  match based on Contributor.Name AND Contributor.Employer
  Part 0b: Employer lookup from pre-classified data 
        - Individuals only: employer vs any org name or individual employer in already_classified
        - Takes priority over running-list and keyword matches
  Part 1: Running-list match (H1B + EDD employer NAICS codes)
        - Organizations: Contributor.Name matched against running_list_alt.csv
        - Individuals:   employer matched against running_list_alt.csv
                            (skipped for non-company occupations)
  Part 2: EDD live lookup for still-unmatched contributors
        - Organizations: name queried against CA EDD employer database
        - Individuals:   employer queried (skipped for non-company occupations)
  Part 3: Column-specific keyword match
        - there are three different keyword lists: one to match keywords in org names,
                one to match keywords in employer names and one to match keywords in occupation names
        - company_keywords  -> contributor name  (organizations only)
        - employer_keywords -> employer        (individuals)
        - occupation_keywords -> occupation    (individuals)
  Normalization: Translate NAICS codes to custom schema
        - any code still on the old subcode_resolution.py scheme (from
           Parts 0/0b's already_classified_contributions.csv or Part 3's keyword
           sheets) is translated to the current sector scheme via the same
           old-to-new map used by Part 4, before Part 4 runs.
  Part 4: Identity-based overrides (runs last, wins over everything above)
        - Raw NAICS codes from Part 1/2 are translated from official NAICS
             to the custom sector scheme via build_naics_crosswalk() (see
             NAICS_SECTOR_DESC_URL / NAICS_CODE_OVERRIDE_URL / NAICS_OLD_TO_NEW_URL)
        - Employer/occupation regexes (custom_naics_labels_updated.csv) flag PACs,
             unions, associations, tribes, government agencies, and retired/
             homemaker/student individuals -- same "PACs win last" priority as
             the legacy 05_candidate_industry_affiliations pipeline

Usage
-----
    python 0803_classify_contributors.py
    python 0803_classify_contributors.py --input path/to/classification_input.csv
    python 0803_classify_contributors.py --no-edd   # skip EDD live lookups

Updates needed
-----
- may need to remove part 0 if entity resolution eventually deals w/ matching to pre-classified
- make sure no self-employed string end up in the employer lookup
- build in step to make sure there are no conflicts between employers in the employer lookup
- Some employer/occupation regexes need to be updated - maybe just integrate with keywords because it does the same thing essentially
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUTS_DIR  = Path(__file__).parent / "08_inputs"
OUTPUTS_DIR = Path(__file__).parent / "08_outputs"
EDD_CACHE_PATH = REPO_ROOT / "code/03_aggregating_data/.edd_naics_cache.json"

DEFAULT_INPUT = OUTPUTS_DIR / "classification_input.csv"
DEFAULT_RUNNING_LIST = REPO_ROOT / "data/03_input/masterfile/running_list_alt.csv"
DEFAULT_OUT = OUTPUTS_DIR / "classified_contributors.csv"
DEFAULT_PRE_CLASSIFIED = INPUTS_DIR / "already_classified_contributions.csv"

_KEYWORD_SHEET_ID = "1WN3KQt9S3Xn5mT2kZxinQ5OYhgCmldA-Q8d5VDXoEg0"
KEYWORD_SHEET_URLS = {
    "employer_keywords":   f"https://docs.google.com/spreadsheets/d/{_KEYWORD_SHEET_ID}/export?format=csv&gid=1277477692",
    "company_keywords":    f"https://docs.google.com/spreadsheets/d/{_KEYWORD_SHEET_ID}/export?format=csv&gid=20437707",
    "occupation_keywords": f"https://docs.google.com/spreadsheets/d/{_KEYWORD_SHEET_ID}/export?format=csv&gid=171257935",
}

"""
NAICS CROSSWALK OVERVIEW

- NAICS_SECTOR_DESC_URL:  current sector/subsector descriptions (labels only)
- NAICS_CODE_OVERRIDE_URL: Maps 4 and 6-digit NAICS codes to custom categories (e.g. renewables, tech,
                             defense, media, etc.)
- NAICS_OLD_TO_NEW_URL: fallback: old subcode_resolution.py subsector -> current
                        sector, keyed by the  2-digit NAICS parent
"""

_NAICS_SHEET_ID = "11QHvNJsdtMlc1YKo_iNvMB_Jfn5Ui-iYdlWhFYjSm9g"
NAICS_SECTOR_DESC_URL = f"https://docs.google.com/spreadsheets/d/{_NAICS_SHEET_ID}/export?format=csv&gid=605391596"
NAICS_OLD_TO_NEW_URL  = f"https://docs.google.com/spreadsheets/d/{_NAICS_SHEET_ID}/export?format=csv&gid=701123226"
_NAICS_OVERRIDE_SHEET_ID = "1gHfG8iSJn-IZez3hs8DpeUNQmQyT0pq7YW9YQ4E_sm4"
NAICS_CODE_OVERRIDE_URL = f"https://docs.google.com/spreadsheets/d/{_NAICS_OVERRIDE_SHEET_ID}/export?format=csv&gid=0"

# employer/occupation regex overrides (PACs, unions, retired, etc.)
# same file 03_aggregating_data/subcode_resolution.py reads
# This uses the OLD subsector codes (e.g. 52a/52b, 56a/56b, 77a/77b, 91); NAICS_OLD_TO_NEW_URL translates maps to current schema

CUSTOM_NAICS_LABELS_PATH = REPO_ROOT / "data/03_input/masterfile/custom_naics_labels_updated.csv"

_NAICS_RANGE_MAP = {"31": "31-33", "32": "31-33", "33": "31-33",
                     "44": "44-45", "45": "44-45", "48": "48-49", "49": "48-49"}

def normalize_naics_parent(code) -> str:
    """'531210' -> '53', '33' -> '31-33', '52.0' -> '52'; '' if empty."""
    if pd.isna(code):
        return ""
    s = str(code).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if not s or s in ("31-33", "44-45", "48-49"):
        return s
    if re.fullmatch(r"\d{3,6}", s):
        s = s[:2]
    return _NAICS_RANGE_MAP.get(s, s)


def load_old_to_new_map(old_to_new_url: str = NAICS_OLD_TO_NEW_URL) -> dict:
    """{old subcode_resolution.py subsector -> current-schema sector}, e.g.
    '91' -> '90', '52a' -> '52'. Also collapses old "a"/"b" subsector suffixes
    that agree on the same sector into a 2-digit default (52a/52b both
    -> 52)."""
    old_to_new = pd.read_csv(old_to_new_url, dtype=str).dropna(subset=["subsector", "sector"])
    default_map = {row.subsector.strip(): row.sector.strip() for row in old_to_new.itertuples(index=False)}
    base_targets: dict[str, set] = {}
    for subsector, sector in default_map.items():
        base = re.sub(r"[a-z]$", "", subsector)
        base_targets.setdefault(base, set()).add(sector)
    for base, targets in base_targets.items():
        if base not in default_map and len(targets) == 1:
            default_map[base] = next(iter(targets))
    return default_map


def load_sector_labels(sector_desc_url: str = NAICS_SECTOR_DESC_URL) -> dict:
    """{current sector code -> sector_description}, from NAICS_SECTOR_DESC_URL (Sheet 1)."""
    labels = pd.read_csv(sector_desc_url, dtype=str).dropna(subset=["sector"])
    return {row.sector.strip(): row.sector_description.strip() for row in labels.itertuples(index=False)}


def build_naics_crosswalk(
    sector_desc_url: str = NAICS_SECTOR_DESC_URL,
    override_url: str = NAICS_CODE_OVERRIDE_URL,
    old_to_new_map: dict | None = None,
):
    """Function mapping a raw EDD/H1B NAICS code to custom codes.

    Resolution order:
      1. Longest matching prefix in the code-override sheet
      2. Fall back to the 2-digit official sector's current-scheme default
         (old_to_new_map, see load_old_to_new_map()).
      3. Unresolved -> (pd.NA, pd.NA).
    """
    overrides = pd.read_csv(override_url, dtype=str)
    overrides = overrides.dropna(subset=["naics_code", "sector"])
    override_list = sorted(
        (
            (row.naics_code.strip(), row.sector.strip())
            for row in overrides.itertuples(index=False)
        ),
        key=lambda kv: -len(kv[0]),
    )

    default_map = old_to_new_map if old_to_new_map is not None else load_old_to_new_map()
    label_map = load_sector_labels(sector_desc_url)

    def crosswalk(raw_code):
        if pd.isna(raw_code) or not str(raw_code).strip():
            return pd.NA, pd.NA
        raw = str(raw_code).strip()
        if raw.endswith(".0"):
            raw = raw[:-2]
        for prefix, sector in override_list:
            if raw.startswith(prefix):
                return sector, label_map.get(sector, pd.NA)
        parent = normalize_naics_parent(raw)
        sector = default_map.get(parent)
        if sector:
            return sector, label_map.get(sector, pd.NA)
        return pd.NA, pd.NA

    print(f"  NAICS crosswalk: {len(override_list):,} code overrides, "
          f"{len(default_map):,} sector defaults, {len(label_map):,} labels")
    return crosswalk


def _resolve_partition_parents(
    df: pd.DataFrame,
    partition_schema: pd.DataFrame,
    old_to_new_map: dict,
    label_map: dict,
) -> pd.DataFrame:
    """For NAICS codes that map to DIFFERENT custom sectors (e.g. '56': 56a->50, 56b->20), 
    disambiguate using employer name_regex from custom_naics_labels_updated.csv. Rows that
    match no regex default to the first child (file order) — so '56'
    without a keyword match -> 56a -> 50. Parents where all children agree on the
    same sector (e.g. 52: 52a/52b→52; 77: 77a/77b→77) are skipped."""
    df = df.copy()

    org_mask = df["entity_type"] == "organization"
    emp_text = pd.Series("", index=df.index, dtype=object)
    emp_text[org_mask]  = df.loc[org_mask, "Contributor.Name"].fillna("")
    emp_text[~org_mask] = df.loc[~org_mask, "employer"].fillna("")

    any_printed = False

    for parent in partition_schema["parent_code"].dropna().unique():
        children = partition_schema[partition_schema["parent_code"] == parent]

        # only disambiguate parents whose children disagree on target sector
        child_sectors = {
            old_to_new_map.get(str(r.naics_code).strip(), str(r.naics_code).strip())
            for r in children.itertuples(index=False)
        }
        if len(child_sectors) <= 1:
            continue

        parent_rows = df["naics_code"].astype(str).str.strip() == str(parent).strip()
        if not parent_rows.any():
            continue

        if not any_printed:
            print()
            print("=" * 70)
            print("Resolving ambiguous partition-parent codes via name_regex")
            print("=" * 70)
            any_printed = True

        unresolved = parent_rows.copy()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="This pattern is interpreted as a regular expression",
                category=UserWarning,
            )
            for child in children.itertuples(index=False):
                regex = getattr(child, "name_regex", None)
                if not regex or pd.isna(regex):
                    continue
                hit = unresolved & emp_text.str.contains(str(regex), case=False, regex=True, na=False)
                n = int(hit.sum())
                child_sector = old_to_new_map.get(str(child.naics_code).strip(), str(child.naics_code).strip())
                df.loc[hit, "naics_code"] = child_sector
                df.loc[hit, "naics_label"] = label_map.get(child_sector, child.naics_label)
                unresolved.loc[hit] = False
                if n:
                    print(f"  {parent} → {child.naics_code} ({child_sector}): {n:,} matched by name")

        # default: first child for rows matching no regex (e.g. bare '56' → 56a → 50)
        if unresolved.any():
            first = children.iloc[0]
            default_sector = old_to_new_map.get(str(first.naics_code).strip(), str(first.naics_code).strip())
            df.loc[unresolved, "naics_code"] = default_sector
            df.loc[unresolved, "naics_label"] = label_map.get(default_sector, first.naics_label)
            print(f"  {parent} → {first.naics_code} ({default_sector}): {int(unresolved.sum()):,} defaulted (no regex match)")

    return df


def apply_naics_crosswalk(
    df: pd.DataFrame,
    crosswalk,
    partition_schema: pd.DataFrame | None = None,
    old_to_new_map: dict | None = None,
    label_map: dict | None = None,
) -> pd.DataFrame:
    """Apply `crosswalk` to every row's raw naics_code, overwriting naics_code/naics_label
    in place for rows that get a resolved custom code.

    If partition_schema is provided, rows that remain on an ambiguous bare partition-parent
    code (e.g. '56': 56a→50, 56b→20) are disambiguated via employer name_regex from
    custom_naics_labels.csv, defaulting to the first child for rows with no regex match."""
    df = df.copy()
    resolved = df["naics_code"].map(crosswalk)
    new_code  = resolved.map(lambda t: t[0])
    new_label = resolved.map(lambda t: t[1])
    has_new = new_code.notna()
    df.loc[has_new, "naics_code"]  = new_code[has_new]
    df.loc[has_new, "naics_label"] = new_label[has_new]
    if partition_schema is not None and old_to_new_map is not None:
        df = _resolve_partition_parents(df, partition_schema, old_to_new_map, label_map or {})
    return df

# input column name mapping
INPUT_COLS = {
    "employer":   "processed_employer_name",
    "occupation": "processed_occupation",
    "city":       "standardized_city",
    "state":      "standardized_state",
    "zip":        "zip_code_processed",
}

NON_COMPANY_OCCUPATIONS = {
    "retired", "self-employed", "self employed",
    "not employed", "unemployed", "homemaker", "none",
    "self",  
}

OUT_COLS = [
    "classification_unit_id", "entity_id", "entity_type",
    "Contributor.Name", "employer", "occupation",
    "city", "state", "total_amount",
    "naics_code", "naics_label",
    "level1_category", "level2_category", "level3_category",
    "data_source_1", "data_source_2", "matched_keyword",
]


# ---------- helpers ----------

def normalize_name(s: pd.Series) -> pd.Series:
    return (
        s.fillna("").astype(str).str.upper()
        .str.replace(r"[^A-Z0-9 ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


# mirrors standardize_occupation_employer() from standardization_helpers.R.
# applied to already_classified_contributions.csv raw fields so they match the processed names in the new contributions data
_STD_OCC_EMP_SUBS = [
    (r"^N/A$",                        "NONE"),
    (r"^NA$",                         "NONE"),
    (r"^N A$",                        "NONE"),
    (r"^BLANK$",                      "UNKNOWN"),
    (r"^NONE OF YOUR BUSINESS$",      "UNKNOWN"),
    (r"^PREFER NOT TO DISCLOSE$",     "UNKNOWN"),
    (r"^NOT EMPLOYED \(RETIRED\)$",   "RETIRED"),
    (r"^NONE \(RETIRED\)$",           "RETIRED"),
    (r"^NOT EMPLOYED-RETIRED$",       "RETIRED"),
    (r"^NONE-RETIRED$",               "RETIRED"),
    (r"^RETIRED NONE$",               "RETIRED"),
    (r"^RETIRED NOT EMPLOYED",        "RETIRED"),
    (r"^NOT EMPLOYED$",               "NONE"),
    (r"^UNEMPLOYED$",                 "NONE"),
    (r"^NO$",                         "NONE"),
    (r"^NOT EMPOYED$",                "NONE"),
    (r"^NOT EMLOYED$",                "NONE"),
    (r"^NOT-EMPLOYED$",               "NONE"),
    (r"^NOT RMPLOYED$",               "NONE"),
    (r"^NOT EMPLOYYED$",              "NONE"),
    (r"^A, N /$",                     "NONE"),
    (r"^INFORMATION REQUESTED$",      "UNKNOWN"),
    (r"INFORMATION REQUESTED-?\s*",   ""),
    (r"SELF-EMPLOYED",                "SELF EMPLOYED"),
]

def _standardize_occ_emp(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.upper().str.strip()
    for pattern, repl in _STD_OCC_EMP_SUBS:
        s = s.str.replace(pattern, repl, regex=True)
    return s.str.strip()


def _is_non_company_occ(occ: str) -> bool:
    return str(occ).strip().lower() in NON_COMPANY_OCCUPATIONS


# normalize NON_COMPANY_OCCUPATIONS
NON_COMPANY_EMPLOYER_NORM = set(normalize_name(pd.Series(sorted(NON_COMPANY_OCCUPATIONS))))


# part 0: pre-classified match ------------------------------------------------------------

def match_pre_classified(
    df: pd.DataFrame, pre_classified_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (Temporary?) direct match against already_classified_contributions.csv.
    Runs first.

    Matching:
      Organizations — Contributor.Name (normalized) vs Contributor.Name
      Individuals   — (Contributor.Name, employer) vs (Contributor.Name, Contributor.Employer)

    Returns (pre_classified_df, unclassified_df). pre_classified_df has
    naics_code / naics_label / data_source_1 / data_source_2 filled.
    unclassified_df passes through to subsequent steps unchanged.
    """
    pre = pd.read_csv(pre_classified_path, dtype=str)
    # Match on raw Contributor.Name / Contributor.Employer so join is stable regardless of standardization
    pre["_pre_name"] = normalize_name(pre["Contributor.Name"])
    # normalize employer: uses raw Contributor.Employer from both sides, then collapses "not employed" variants
    # so they match 
    pre["_pre_emp"]  = normalize_name(
        _standardize_occ_emp(pre["Contributor.Employer"]).replace({"NONE": "", "UNKNOWN": ""})
    )

    df = df.copy()
    df["_name_norm"] = normalize_name(df["Contributor.Name"])
    _emp_src = "Contributor.Employer" if "Contributor.Employer" in df.columns else "employer"
    df["_emp_norm"]  = normalize_name(
        _standardize_occ_emp(df[_emp_src]).replace({"NONE": "", "UNKNOWN": ""})
    )

    org_mask = df["entity_type"] == "organization"
    ind_mask  = ~org_mask

    pre_cols = ["code_final", "code_final_description", "code_final_source"]
    if "revised_category" in pre.columns:
        pre_cols = pre_cols + ["revised_category"]

    # orgs: name match -----------------------------------------------
    org_pre = pre.drop_duplicates("_pre_name")[["_pre_name"] + pre_cols]
    org_merged = (
        df[org_mask]
        .merge(org_pre, left_on="_name_norm", right_on="_pre_name", how="left")
        .drop(columns="_pre_name")
    )
    org_matched   = org_merged[org_merged["code_final"].notna()]
    org_unmatched = org_merged[org_merged["code_final"].isna()].drop(columns=pre_cols, errors="ignore")

    # individuals: direct name + employer match -----------------------
    ind_pre = (pre.drop_duplicates(subset=["_pre_name", "_pre_emp"])
                  [["_pre_name", "_pre_emp"] + pre_cols])
    ind_merged = (
        df[ind_mask]
        .merge(ind_pre, left_on=["_name_norm", "_emp_norm"],
               right_on=["_pre_name", "_pre_emp"], how="left")
        .drop(columns=["_pre_name", "_pre_emp"])
    )
    ind_direct    = ind_merged[ind_merged["code_final"].notna()]
    ind_remaining = ind_merged[ind_merged["code_final"].isna()].drop(columns=pre_cols)

    # assemble -----------------------------------------------------
    pre_df = pd.concat([org_matched, ind_direct], ignore_index=True)
    unclassified_df = pd.concat([org_unmatched, ind_remaining], ignore_index=True)

    for col in ["_name_norm", "_emp_norm"]:
        pre_df          = pre_df.drop(columns=col, errors="ignore")
        unclassified_df = unclassified_df.drop(columns=col, errors="ignore")

    # Megadonors → 100a, Major Donors → 100b
    if "revised_category" in pre_df.columns:
        rev = pre_df["revised_category"].fillna("").str.strip()
        pre_df.loc[rev == "Megadonors",   "code_final"]             = "100a"
        pre_df.loc[rev == "Megadonors",   "code_final_description"] = "Megadonors"
        pre_df.loc[rev == "Major Donors", "code_final"]             = "100b"
        pre_df.loc[rev == "Major Donors", "code_final_description"] = "Major Donors"

    pre_df["naics_code"]      = pre_df["code_final"]
    pre_df["naics_label"]     = pre_df["code_final_description"]
    pre_df["level1_category"] = pd.NA
    pre_df["level2_category"] = pd.NA
    pre_df["level3_category"] = pd.NA
    pre_df["data_source_1"]   = "pre_classified"
    pre_df["data_source_2"]   = pre_df["code_final_source"]
    pre_df["matched_keyword"] = pd.NA
    pre_df = pre_df.drop(columns=[c for c in pre_cols if c in pre_df.columns])

    print()
    print("=" * 70)
    print("PART 0 — pre-classified direct match (temporary)")
    print("=" * 70)
    n_org_total = int(org_mask.sum())
    n_ind_total = int(ind_mask.sum())
    for label, n, total in [
        ("org  name_match",    len(org_matched), n_org_total),
        ("ind  name+employer", len(ind_direct),  n_ind_total),
    ]:
        print(f"  {label:26s}: {n:>4,} / {total:>4,} ({n / max(total, 1) * 100:.1f}%)")
    n_total = len(pre_df)
    print(f"  {'total':26s}: {n_total:>4,} / {len(df):>4,} ({n_total / max(len(df), 1) * 100:.1f}%)")

    return pre_df, unclassified_df

# part 1: employer lookup match ----------------------------------------------------------------------

# want to make sure no employers related to self-employed get in here

def _build_employer_lookup(pre_classified_path: Path) -> pd.DataFrame:
    """Build employer_norm -> classification lookup from already_classified_contributions.csv.
    Combines names (as employer keys) and individual employer fields."""
    pre = pd.read_csv(pre_classified_path, dtype=str)
    pre["_pre_name"] = normalize_name(pre["Contributor.Name"])
    pre["_pre_emp"]  = normalize_name(
        _standardize_occ_emp(pre["Contributor.Employer"]).replace({"": "NONE"})
    )
    pre_cols = ["code_final", "code_final_description", "code_final_source"]
    # "UNKNOWN" catches values like "N/A"/"BLANK" after standardization.
    emp_usable = ~pre["_pre_emp"].isin(NON_COMPANY_EMPLOYER_NORM | {"UNKNOWN"}) & (pre["_pre_emp"] != "")
    return pd.concat([
        pre[["_pre_name"] + pre_cols].rename(columns={"_pre_name": "_emp_key"}),
        pre.loc[emp_usable, ["_pre_emp"] + pre_cols].rename(columns={"_pre_emp": "_emp_key"}),
    ]).drop_duplicates(subset="_emp_key", keep="first")


def match_employer_lookup(
    df: pd.DataFrame, pre_classified_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classify unmatched individuals whose employer matches any org name
    or individual employer in already_classified_contributions.csv.
    
    Returns (employer_matched_df, unmatched_df).
    """
    emp_lookup = _build_employer_lookup(pre_classified_path)
    pre_cols = ["code_final", "code_final_description", "code_final_source"]

    df = df.copy()
    df["_emp_norm"] = normalize_name(df["employer"].fillna(""))

    ind_mask = df["entity_type"] == "individual"
    has_emp  = (df["_emp_norm"] != "") & ~df["_emp_norm"].isin(NON_COMPANY_EMPLOYER_NORM)

    matchable   = df[ind_mask & has_emp]
    not_matchable = df[~(ind_mask & has_emp)]

    merged = (
        matchable
        .merge(emp_lookup, left_on="_emp_norm", right_on="_emp_key", how="left")
        .drop(columns="_emp_key")
    )
    emp_matched  = merged[merged["code_final"].notna()]
    emp_remaining = merged[merged["code_final"].isna()].drop(columns=pre_cols, errors="ignore")

    unmatched_df = pd.concat([emp_remaining, not_matchable], ignore_index=True)
    for col in ["_emp_norm"]:
        emp_matched  = emp_matched.drop(columns=col, errors="ignore")
        unmatched_df = unmatched_df.drop(columns=col, errors="ignore")

    emp_df = emp_matched.copy()
    emp_df["naics_code"]      = emp_df["code_final"]
    emp_df["naics_label"]     = emp_df["code_final_description"]
    emp_df["level1_category"] = pd.NA
    emp_df["level2_category"] = pd.NA
    emp_df["level3_category"] = pd.NA
    emp_df["data_source_1"]   = "employer_lookup"
    emp_df["data_source_2"]   = emp_df["code_final_source"]
    emp_df["matched_keyword"] = pd.NA
    emp_df = emp_df.drop(columns=pre_cols)

    print()
    print("=" * 70)
    print("PART 0b — employer lookup from pre-classified data")
    print("=" * 70)
    n_ind_total = int((df["entity_type"] == "individual").sum())
    n_matched   = len(emp_df)
    print(f"  {'ind  employer_lookup':26s}: {n_matched:>4,} / {n_ind_total:>4,} ({n_matched / max(n_ind_total, 1) * 100:.1f}%)")

    return emp_df, unmatched_df


# part 2: running-list match (H1B + EDD) --------------------------------------------------------------------------

def match_running_list(
    df: pd.DataFrame,
    running_list_path: Path,
    naics_crosswalk=None,
    partition_schema: pd.DataFrame | None = None,
    old_to_new_map: dict | None = None,
    label_map: dict | None = None,
) -> pd.DataFrame:
    rl = pd.read_csv(running_list_path, dtype=str)
    rl = rl.drop_duplicates("name_norm", keep="first")
    keep = ["name_norm", "name", "naics_code", "naics_label",
            "level1_category", "level2_category", "level3_category", "source"]
    rl_use = rl[[c for c in keep if c in rl.columns]].rename(columns={"name": "match_name"})

    df = df.copy()
    df["contributor_name_norm"] = normalize_name(df["Contributor.Name"])
    df["employer_norm"] = normalize_name(df["employer"])

    # Orgs: match on Contributor.Name
    org_mask = df["entity_type"] == "organization"
    orgs = df[org_mask].merge(
        rl_use, left_on="contributor_name_norm", right_on="name_norm", how="left"
    ).drop(columns="name_norm")

    # Individuals: match on employer, skip non-company occupations
    ind = df[~org_mask].copy()
    ind["_skip"] = ind["occupation"].apply(_is_non_company_occ) | ind["employer_norm"].isin(["", "NONE"])
    ind_matchable = ind[~ind["_skip"]].merge(
        rl_use, left_on="employer_norm", right_on="name_norm", how="left"
    ).drop(columns="name_norm")
    ind_skipped = ind[ind["_skip"]].copy()
    for col in ["match_name", "naics_code", "naics_label", "level1_category",
                "level2_category", "level3_category", "source"]:
        if col not in ind_skipped.columns:
            ind_skipped[col] = pd.NA

    individuals = pd.concat([ind_matchable, ind_skipped], ignore_index=True)
    individuals = individuals.drop(columns="_skip", errors="ignore")

    result = pd.concat([orgs, individuals], ignore_index=True)
    result["data_source_1"] = pd.NA
    result["data_source_2"] = pd.NA
    result["matched_keyword"] = pd.NA

    matched = result["match_name"].notna()
    result.loc[matched, "data_source_1"] = "running_list"
    result.loc[matched, "data_source_2"] = result.loc[matched, "source"]
    result = result.drop(columns=["source", "match_name"], errors="ignore")

    if naics_crosswalk is not None:
        result = apply_naics_crosswalk(result, naics_crosswalk,
                                       partition_schema=partition_schema,
                                       old_to_new_map=old_to_new_map,
                                       label_map=label_map)

    n_orgs_matched = int((result["entity_type"] == "organization").sum() and
                         result.loc[result["entity_type"] == "organization", "data_source_1"].notna().sum())
    print()
    print("=" * 70)
    print("PART 1 — running-list match (H1B + EDD)")
    print("=" * 70)
    _print_match_stats(result, "organization")
    _print_match_stats(result, "individual")

    return result


def _print_match_stats(df: pd.DataFrame, entity_type: str) -> None:
    sub = df[df["entity_type"] == entity_type]
    n = len(sub)
    n_matched = int(sub["data_source_1"].notna().sum())
    print(f"  {entity_type:14s}: {n_matched:>4,} / {n:>4,} matched "
          f"({n_matched / max(n, 1) * 100:.1f}%)")


# part 2: EDD live lookup --------------------------------------------------------------------------------

def match_edd(
    df: pd.DataFrame,
    delay: float = 1.0,
    naics_crosswalk=None,
    partition_schema: pd.DataFrame | None = None,
    old_to_new_map: dict | None = None,
    label_map: dict | None = None,
) -> pd.DataFrame:
    """Query the CA EDD employer database for unmatched contributors.

    Organizations are queried by name; individuals by employer
    (non-company occupations are skipped).
    """
    sys.path.insert(0, str(REPO_ROOT / "code" / "03_aggregating_data"))
    from edd_naics_lookup import EDDClient  # noqa: E402

    unmatched = df["data_source_1"].isna()
    if not unmatched.any():
        print()
        print("=" * 70)
        print("PART 2 — EDD live lookup")
        print("=" * 70)
        print("  Nothing to look up.")
        return df

    df = df.copy()

    # determine query string
    org_mask = (df["entity_type"] == "organization") & unmatched
    non_company = df["occupation"].apply(_is_non_company_occ)
    ind_mask = (df["entity_type"] == "individual") & unmatched & ~non_company

    df["_edd_query"] = pd.NA
    df.loc[org_mask, "_edd_query"] = df.loc[org_mask, "Contributor.Name"]
    df.loc[ind_mask, "_edd_query"] = df.loc[ind_mask, "employer"]

    # drop rows
    df.loc[df["_edd_query"].fillna("").str.strip() == "", "_edd_query"] = pd.NA

    to_query = df.dropna(subset=["_edd_query"]).drop_duplicates("_edd_query")
    n_queries = len(to_query)

    print()
    print("=" * 70)
    print("PART 2 — EDD live lookup")
    print("=" * 70)
    print(f"  Unmatched orgs:        {int(org_mask.sum()):>4,}")
    print(f"  Unmatched individuals: {int(ind_mask.sum()):>4,}")
    print(f"  Unique queries:       {n_queries:>4,}")
    print(f"  (cache: {EDD_CACHE_PATH}  delay: {delay:.1f}s/req)")
    print()

    client = EDDClient(delay=delay, cache_path=EDD_CACHE_PATH)
    edd_results: dict[str, object] = {}
    for i, row in enumerate(to_query.itertuples(), 1):
        q = str(row._edd_query)
        r = client.lookup(q)
        edd_results[q] = r
        print(
            f"  [{i:>4}/{n_queries}] {q[:50]:50s} -> "
            f"{r.status:11s} {r.naics_code or '':6s} {r.match_name or ''}",
            file=sys.stderr,
        )

    n_hit = 0
    for idx in df.index[df["_edd_query"].notna()]:
        q = str(df.at[idx, "_edd_query"])
        r = edd_results.get(q)
        if r and r.status == "ok" and r.naics_code:
            df.at[idx, "naics_code"] = r.naics_code
            df.at[idx, "naics_label"] = r.naics_description
            df.at[idx, "data_source_1"] = "edd"
            df.at[idx, "data_source_2"] = "edd_live"
            n_hit += 1

    print()
    print(f"  EDD matches:                 {n_hit:>4,} / {n_queries:>4,} "
          f"({n_hit / max(n_queries, 1) * 100:.1f}%)")

    df = df.drop(columns="_edd_query")
    if naics_crosswalk is not None:
        edd_rows = df["data_source_1"] == "edd"
        df.loc[edd_rows] = apply_naics_crosswalk(df.loc[edd_rows], naics_crosswalk,
                                                  partition_schema=partition_schema,
                                                  old_to_new_map=old_to_new_map,
                                                  label_map=label_map)
    return df


# part 3: keyword match ------------------------------------------------------------------------

def load_keywords(path: Path) -> dict[str, pd.DataFrame]:
    """Load keywords and split by source_sheet into groups."""
    df = pd.read_csv(path)
    if "level3_category" not in df.columns:
        df["level3_category"] = pd.NA
    if "source_sheet" not in df.columns:
        df["source_sheet"] = "employer_keywords"

    groups: dict[str, list[dict]] = {
        "employer_keywords": [],
        "company_keywords": [],
        "occupation_keywords": [],
    }
    for _, r in df.iterrows():
        sheet = str(r.get("source_sheet", "employer_keywords")).strip()
        if sheet not in groups:
            continue
        for part in re.split(r"\s*[,/]\s*", str(r["keywords"])):
            kw = part.strip().lower()
            if kw:
                groups[sheet].append({
                    "keyword": kw,
                    "level1_category": r.get("level1_category"),
                    "level2_category": r.get("level2_category"),
                    "level3_category": r.get("level3_category"),
                })

    result = {}
    for sheet, rows in groups.items():
        if rows:
            kw_df = (
                pd.DataFrame(rows)
                .drop_duplicates(subset="keyword", keep="first")
                .sort_values("keyword", key=lambda s: s.str.len(), ascending=False)
                .reset_index(drop=True)
            )
        else:
            kw_df = pd.DataFrame(columns=["keyword", "level1_category", "level2_category", "level3_category"])
        result[sheet] = kw_df
        print(f"  Loaded {len(kw_df):>4,} {sheet}")

    return result


def _blank_to_na(val):
    s = str(val).strip()
    return s if s and s.lower() != "nan" else pd.NA


def load_keywords_from_sheets() -> dict[str, pd.DataFrame]:
    """Load keywrods from Google Sheets. Uses custom_naics_updated as the NAICS code."""

    result = {}
    for sheet_name, url in KEYWORD_SHEET_URLS.items():
        df = pd.read_csv(url, dtype=str)
        rows = []
        for _, r in df.iterrows():
            kw_cell = str(r.get("keywords", "")).strip()
            if not kw_cell or kw_cell.lower() == "nan":
                continue
            for part in re.split(r"\s*[,/]\s*", kw_cell):
                kw = part.strip().lower()
                if kw:
                    rows.append({
                        "keyword":         kw,
                        "naics_code":      _blank_to_na(r.get("custom_naics_updated")),
                        "level1_category": _blank_to_na(r.get("level1_category")),
                        "level2_category": _blank_to_na(r.get("level2_category")),
                        "level3_category": _blank_to_na(r.get("level3_category")),
                    })
        if rows:
            kw_df = (
                pd.DataFrame(rows)
                .drop_duplicates(subset="keyword", keep="first")
                .sort_values("keyword", key=lambda s: s.str.len(), ascending=False)
                .reset_index(drop=True)
            )
        else:
            kw_df = pd.DataFrame(columns=["keyword", "naics_code", "level1_category", "level2_category", "level3_category"])
        result[sheet_name] = kw_df
        print(f"  Loaded {len(kw_df):>4,} {sheet_name} (Google Sheets)")
    return result


def _compile_patterns(kw_df: pd.DataFrame) -> list[tuple]:
    return [
        (re.compile(rf"\b{re.escape(row.keyword)}\b", re.IGNORECASE), row)
        for row in kw_df.itertuples(index=False)
    ]


def _apply_keyword_match(df: pd.DataFrame, indices, field: str,
                         patterns: list[tuple], source_tag: str) -> int:
    n_matched = 0
    for idx in indices:
        field_val = str(df.at[idx, field]).lower()
        for pat, row in patterns:
            if pat.search(field_val):
                df.at[idx, "level1_category"] = row.level1_category
                df.at[idx, "level2_category"] = row.level2_category
                df.at[idx, "level3_category"] = row.level3_category
                df.at[idx, "matched_keyword"] = row.keyword
                df.at[idx, "data_source_1"] = "keyword match"
                df.at[idx, "data_source_2"] = source_tag
                if hasattr(row, "naics_code") and pd.notna(row.naics_code):
                    df.at[idx, "naics_code"] = row.naics_code
                n_matched += 1
                break
    return n_matched


def match_keywords(df: pd.DataFrame, keywords_path: Path | None = None) -> pd.DataFrame:
    print()
    print("=" * 70)
    print("PART 2 — column-specific keyword match")
    print("=" * 70)

    if keywords_path is not None:
        kw_groups = load_keywords(keywords_path)
    else:
        kw_groups = load_keywords_from_sheets()
    print()

    company_patterns = _compile_patterns(kw_groups["company_keywords"])
    employer_patterns = _compile_patterns(kw_groups["employer_keywords"])
    occupation_patterns = _compile_patterns(kw_groups["occupation_keywords"])

    unmatched = df["data_source_1"].isna()

    # company_keywords -> Contributor.Name (orgs only)
    org_unmatched = df.index[unmatched & (df["entity_type"] == "organization")].tolist()
    n_company = _apply_keyword_match(df, org_unmatched, "Contributor.Name",
                                     company_patterns, "company_keywords")
    print(f"  company_keywords  (Contributor.Name, orgs): {n_company:>4,} matched")

    # employer_keywords -> employer (any entity with an employer)
    unmatched = df["data_source_1"].isna()
    has_employer = ~df["employer_norm"].isin(["", "NONE"]) & df["employer"].notna()
    emp_unmatched = df.index[unmatched & has_employer].tolist()
    n_employer = _apply_keyword_match(df, emp_unmatched, "employer",
                                      employer_patterns, "employer_keywords")
    print(f"  employer_keywords (employer field):          {n_employer:>4,} matched")

    # occupation_keywords -> occupation (individuals only)
    unmatched = df["data_source_1"].isna()
    ind_unmatched = df.index[unmatched & (df["entity_type"] == "individual")].tolist()
    n_occ = _apply_keyword_match(df, ind_unmatched, "occupation",
                                  occupation_patterns, "occupation_keywords")
    print(f"  occupation_keywords (occupation field):      {n_occ:>4,} matched")

    return df


#  normalize codes ----------------------------------------------

def normalize_custom_codes(
    df: pd.DataFrame, old_to_new_map: dict, label_map: dict,
    code_col: str = "naics_code", label_col: str = "naics_label",
) -> pd.DataFrame:
    """Translate any code that's still in the old subcode_resolution.py scheme
    (52a, 71, 91, ...) to the current sector scheme, via the old_to_new_map.

    Runs once across the whole combined dataframe so it catches all of
    those regardless of which part produced the code.
    """
    df = df.copy()
    codes = df[code_col]
    translated = codes.map(lambda c: old_to_new_map.get(str(c).strip(), c) if pd.notna(c) else c)
    changed = translated.astype(str) != codes.astype(str)
    n = int(changed.sum())
    if n:
        df[code_col] = translated
        new_label = df.loc[changed, code_col].map(label_map)
        df.loc[changed, label_col] = new_label.where(new_label.notna(), df.loc[changed, label_col])
    print()
    print("=" * 70)
    print("Normalizing old-scheme codes (pre_classified/employer_lookup/keyword-match)")
    print("=" * 70)
    print(f"  Translated {n:,} / {len(df):,} rows to the current sector scheme")
    return df


# part 4: identity-based overrides (PACs, unions, retired, etc.) -----------------------------------

def load_identity_overrides(
    path: Path = CUSTOM_NAICS_LABELS_PATH, old_to_new_map: dict | None = None
) -> pd.DataFrame:
    """Load employer/occupation regex rules from custom_naics_labels_updated.csv, translate
    each naics code to current schema"""
    schema = pd.read_csv(path, dtype=str)
    schema = schema[schema["name_regex"].notna()].copy()
    if old_to_new_map:
        schema["naics_code"] = schema["naics_code"].map(lambda c: old_to_new_map.get(c, c))
    return schema


def apply_identity_overrides(df: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    """identity-based regexes (PACs, unions, associations, tribes,
    government agencies, retired/homemaker/student individuals, etc.) win over
    whatever running-list/EDD/keyword matching produced 
    Employer rules run in file order (later rows overwrite earlier); the occupation rule (100) runs last.
    """
    df = df.copy()
    org_mask = df["entity_type"] == "organization"
    emp_text = pd.Series("", index=df.index, dtype=object)
    emp_text[org_mask]  = df.loc[org_mask, "Contributor.Name"].fillna("")
    emp_text[~org_mask] = df.loc[~org_mask, "employer"].fillna("")
    occ_text = df["occupation"].fillna("").astype(str)
    # pre_classified manually reviewed rows are locked
    not_pre_classified = df["data_source_1"] != "pre_classified"

    print()
    print("=" * 70)
    print("PART 4 — identity-based overrides (PACs, unions, retired, etc.)")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="This pattern is interpreted as a regular expression",
            category=UserWarning,
        )

        for r in schema[schema["applies_to"] == "employer"].itertuples(index=False):
            hit = not_pre_classified & emp_text.astype(str).str.contains(r.name_regex, case=False, regex=True, na=False)
            n = int(hit.sum())
            if not n:
                continue
            df.loc[hit, "naics_code"]    = r.naics_code
            df.loc[hit, "naics_label"]   = r.naics_label
            df.loc[hit, "data_source_1"] = "identity_override"
            df.loc[hit, "data_source_2"] = f"employer_regex:{r.naics_code}"
            print(f"  employer-rule {r.naics_code:>4s} ({r.naics_label}): {n:,} matched")

        for r in schema[schema["applies_to"] == "occupation"].itertuples(index=False):
            hit = not_pre_classified & occ_text.str.contains(r.name_regex, case=False, regex=True, na=False)
            n = int(hit.sum())
            if not n:
                continue
            df.loc[hit, "naics_code"]    = r.naics_code
            df.loc[hit, "naics_label"]   = r.naics_label
            df.loc[hit, "data_source_1"] = "identity_override"
            df.loc[hit, "data_source_2"] = f"occupation_regex:{r.naics_code}"
            print(f"  occupation-rule {r.naics_code:>4s} ({r.naics_label}): {n:,} matched")

    # Individuals with employer="NONE" or blank —> assign code 100.
    none_emp = not_pre_classified & (~org_mask) & df["employer"].fillna("").str.strip().str.upper().isin(["", "NONE"])
    n_none = int(none_emp.sum())
    if n_none:
        df.loc[none_emp, "naics_code"]    = "100"
        df.loc[none_emp, "naics_label"]   = "Retired/Homemaker/Student/Unemployed"
        df.loc[none_emp, "data_source_1"] = "identity_override"
        df.loc[none_emp, "data_source_2"] = "employer_none:100"
        print(f"  employer-NONE override 100 (Retired/Homemaker/Student/Unemployed): {n_none:,} matched")

    return df


# summary --------------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    n = len(df)
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Total contributors: {n:,}\n")
    src = df["data_source_1"].fillna("unmatched")
    for entity_type in ("organization", "individual"):
        sub = df[df["entity_type"] == entity_type]
        sub_src = sub["data_source_1"].fillna("unmatched")
        print(f"  {entity_type} ({len(sub):,} total):")
        for s in ("pre_classified", "employer_lookup", "running_list", "edd", "keyword match", "identity_override", "unmatched"):
            count = int((sub_src == s).sum())
            if count:
                print(f"    {s:14s}: {count:>4,} ({count / max(len(sub), 1) * 100:>5.1f}%)")
        print()



# main --------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--running-list", type=Path, default=DEFAULT_RUNNING_LIST)
    p.add_argument("--keywords", type=Path, default=None,
                   help="Local keyword CSV to override Google Sheets fetch")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--pre-classified", type=Path, default=DEFAULT_PRE_CLASSIFIED)
    p.add_argument("--no-edd", action="store_true", help="Skip EDD live lookups (Part 2)")
    p.add_argument("--edd-delay", type=float, default=1.0, help="Seconds between EDD requests")
    p.add_argument("--no-naics-crosswalk", action="store_true",
                   help="Skip translating running-list/EDD raw NAICS codes to the custom sector "
                        "scheme, AND skip normalizing already-custom old-scheme codes (from "
                        "pre_classified/employer_lookup/keyword-match) to the current scheme")
    p.add_argument("--no-identity-overrides", action="store_true",
                   help="Skip the final PAC/union/association/retired identity-override pass")
    p.add_argument("--custom-naics-labels", type=Path, default=CUSTOM_NAICS_LABELS_PATH,
                   help="Local editorial regex taxonomy for Part 4 identity overrides")
    args = p.parse_args(argv)

    print(f"Input: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    rename_map = {src: dst for dst, src in INPUT_COLS.items() if src in df.columns and src != dst}
    if rename_map:
        df = df.rename(columns=rename_map)
    print(f"  rows={len(df):,}  entity_types={df['entity_type'].value_counts().to_dict()}")

    naics_crosswalk = None
    old_to_new_map = None
    label_map = None
    partition_schema = None
    if not args.no_naics_crosswalk:
        old_to_new_map = load_old_to_new_map()
        naics_crosswalk = build_naics_crosswalk(old_to_new_map=old_to_new_map)
        label_map = load_sector_labels()
        raw_labels = pd.read_csv(args.custom_naics_labels, dtype=str)
        partition_schema = raw_labels[raw_labels["kind"] == "partition_child"].copy()

    pre_df = pd.DataFrame()
    emp_df = pd.DataFrame()

    if args.pre_classified.exists():
        pre_df, df = match_pre_classified(df, args.pre_classified)
        emp_df, df = match_employer_lookup(df, args.pre_classified)
    else:
        print("\nPart 0 / employer lookup skipped: --pre-classified file not found")

    df = match_running_list(df, args.running_list, naics_crosswalk=naics_crosswalk,
                            partition_schema=partition_schema, old_to_new_map=old_to_new_map,
                            label_map=label_map)
    if not args.no_edd:
        df = match_edd(df, delay=args.edd_delay, naics_crosswalk=naics_crosswalk,
                       partition_schema=partition_schema, old_to_new_map=old_to_new_map,
                       label_map=label_map)
    df = match_keywords(df, args.keywords)

    parts = [p for p in [pre_df, emp_df, df] if not (isinstance(p, pd.DataFrame) and p.empty)]
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else df

    if old_to_new_map:
        df = normalize_custom_codes(df, old_to_new_map, label_map or load_sector_labels())

    if not args.no_identity_overrides:
        identity_schema = load_identity_overrides(args.custom_naics_labels, old_to_new_map)
        df = apply_identity_overrides(df, identity_schema)

    # label unitemized contributions
    unitemized_mask = df["Contributor.Name"].fillna("").str.strip().str.lower() == "unitemized contributions"
    if unitemized_mask.any():
        df.loc[unitemized_mask, "naics_code"]    = "101"
        df.loc[unitemized_mask, "naics_label"]   = "Unitemized Contributions"
        df.loc[unitemized_mask, "data_source_1"] = "pre_classified"
        df.loc[unitemized_mask, "data_source_2"] = "unitemized_contributions"
        print(f"\nUnitemized Contributions: {int(unitemized_mask.sum()):,} rows → code 101")

    print_summary(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in OUT_COLS if c in df.columns] + [
        c for c in df.columns if c not in OUT_COLS
    ]
    df[cols].to_csv(args.out, index=False)
    print(f"\nWrote: {args.out}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
