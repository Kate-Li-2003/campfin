"""
0803_classify_contributors.py

Alternative classification pipeline for pre-aggregated contributor data.
Operates directly on classification_input.csv (output of the entity-resolution
stage).

  Part 0:  Pre-classified direct match — TEMPORARY
           - Organizations: canonical_name vs Contributor.Name
           - Individuals:   (canonical_name, employer) vs (Contributor.Name, Contributor.Employer)
  Part 0b: Employer lookup from pre-classified data — PERMANENT
           - Individuals only: employer vs any org name or individual employer in already_classified
           - Takes priority over running-list and keyword matches
  Part 1: Running-list match (H1B + EDD employer -> NAICS)
           - Organizations: canonical_name matched against running_list_alt.csv
           - Individuals:   employer matched against running_list_alt.csv
                            (skipped for non-company occupations)
  Part 2: EDD live lookup for still-unmatched contributors
           - Organizations: canonical_name queried against CA EDD employer database
           - Individuals:   employer queried (skipped for non-company occupations)
  Part 3: Column-specific keyword match
           - company_keywords  -> contributor name  (organizations only)
           - employer_keywords -> employer        (individuals)
           - occupation_keywords -> occupation    (individuals)

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
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUTS_DIR  = Path(__file__).parent / "08_inputs"
OUTPUTS_DIR = Path(__file__).parent / "08_outputs"
EDD_CACHE_PATH = REPO_ROOT / "code/03_aggregating_data/.edd_naics_cache.json"

DEFAULT_INPUT = OUTPUTS_DIR / "classification_input.csv"
DEFAULT_RUNNING_LIST = REPO_ROOT / "data/03_input/masterfile/running_list_alt.csv"
DEFAULT_KEYWORDS = (
    REPO_ROOT
    / "data/03_input/training data (manual classifications)/Keywords_Manually_Collected.csv"
)
DEFAULT_OUT = OUTPUTS_DIR / "classified_contributors.csv"
DEFAULT_PRE_CLASSIFIED = INPUTS_DIR / "already_classified_contributions.csv"

NON_COMPANY_OCCUPATIONS = {
    "retired", "self-employed", "self employed",
    "not employed", "unemployed", "homemaker", "none",
}

OUT_COLS = [
    "classification_unit_id", "entity_id", "entity_type",
    "canonical_name", "employer", "occupation",
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


def _is_non_company_occ(occ: str) -> bool:
    return str(occ).strip().lower() in NON_COMPANY_OCCUPATIONS


# ---------- part 0: pre-classified match ----------

def match_pre_classified(
    df: pd.DataFrame, pre_classified_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporary direct match against already_classified_contributions.csv.
    Runs before employer lookup, running-list, and keyword passes.

    Matching:
      Organizations — canonical_name (normalized) vs Contributor.Name
      Individuals   — (canonical_name, employer) vs (Contributor.Name, Contributor.Employer)

    Returns (pre_classified_df, unclassified_df). pre_classified_df has
    naics_code / naics_label / data_source_1 / data_source_2 filled.
    unclassified_df passes through to subsequent steps unchanged.
    """
    pre = pd.read_csv(pre_classified_path, dtype=str)
    pre["_pre_name"] = normalize_name(pre["Contributor.Name"])
    pre["_pre_emp"]  = normalize_name(pre["Contributor.Employer"].fillna(""))

    df = df.copy()
    df["_name_norm"] = normalize_name(df["canonical_name"])
    df["_emp_norm"]  = normalize_name(df["employer"].fillna(""))

    org_mask = df["entity_type"] == "organization"
    ind_mask  = ~org_mask

    pre_cols = ["code_final", "code_final_description", "code_final_source"]

    # --- orgs: name match ---
    org_pre = pre.drop_duplicates("_pre_name")[["_pre_name"] + pre_cols]
    org_merged = (
        df[org_mask]
        .merge(org_pre, left_on="_name_norm", right_on="_pre_name", how="left")
        .drop(columns="_pre_name")
    )
    org_matched   = org_merged[org_merged["code_final"].notna()]
    org_unmatched = org_merged[org_merged["code_final"].isna()].drop(columns=pre_cols)

    # --- individuals: direct name + employer match ---
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

    # --- assemble ---
    pre_df = pd.concat([org_matched, ind_direct], ignore_index=True)
    unclassified_df = pd.concat([org_unmatched, ind_remaining], ignore_index=True)

    for col in ["_name_norm", "_emp_norm"]:
        pre_df          = pre_df.drop(columns=col, errors="ignore")
        unclassified_df = unclassified_df.drop(columns=col, errors="ignore")

    pre_df["naics_code"]      = pre_df["code_final"]
    pre_df["naics_label"]     = pre_df["code_final_description"]
    pre_df["level1_category"] = pd.NA
    pre_df["level2_category"] = pd.NA
    pre_df["level3_category"] = pd.NA
    pre_df["data_source_1"]   = "pre_classified"
    pre_df["data_source_2"]   = pre_df["code_final_source"]
    pre_df["matched_keyword"] = pd.NA
    pre_df = pre_df.drop(columns=pre_cols)

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

# ---------- part 1: employer lookup match ----------

# want to make sure no employers related to self-employed get in here

def _build_employer_lookup(pre_classified_path: Path) -> pd.DataFrame:
    """Build employer_norm → classification lookup from already_classified_contributions.csv.
    Combines org canonical names (as employer keys) and individual employer fields."""
    pre = pd.read_csv(pre_classified_path, dtype=str)
    pre["_pre_name"] = normalize_name(pre["Contributor.Name"])
    pre["_pre_emp"]  = normalize_name(pre["Contributor.Employer"].fillna(""))
    pre_cols = ["code_final", "code_final_description", "code_final_source"]
    return pd.concat([
        pre[["_pre_name"] + pre_cols].rename(columns={"_pre_name": "_emp_key"}),
        pre.loc[pre["_pre_emp"] != "", ["_pre_emp"] + pre_cols].rename(columns={"_pre_emp": "_emp_key"}),
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
    has_emp  = df["_emp_norm"] != ""

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


# ---------- part 2: running-list match ----------

def match_running_list(df: pd.DataFrame, running_list_path: Path) -> pd.DataFrame:
    rl = pd.read_csv(running_list_path, dtype=str)
    rl = rl.drop_duplicates("name_norm", keep="first")
    keep = ["name_norm", "name", "naics_code", "naics_label",
            "level1_category", "level2_category", "level3_category", "source"]
    rl_use = rl[[c for c in keep if c in rl.columns]].rename(columns={"name": "match_name"})

    df = df.copy()
    df["canonical_name_norm"] = normalize_name(df["canonical_name"])
    df["employer_norm"] = normalize_name(df["employer"])

    # Orgs: match on canonical_name
    org_mask = df["entity_type"] == "organization"
    orgs = df[org_mask].merge(
        rl_use, left_on="canonical_name_norm", right_on="name_norm", how="left"
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


# ---------- part 2: EDD live lookup ----------

def match_edd(df: pd.DataFrame, delay: float = 1.0) -> pd.DataFrame:
    """Query the CA EDD employer database for still-unmatched contributors.

    Organizations are queried by canonical_name; individuals by employer
    (non-company occupations are skipped). Results fill naics_code,
    naics_label, data_source_1, and data_source_2 in-place.
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

    # Determine query string: canonical_name for orgs, employer for individuals.
    org_mask = (df["entity_type"] == "organization") & unmatched
    non_company = df["occupation"].apply(_is_non_company_occ)
    ind_mask = (df["entity_type"] == "individual") & unmatched & ~non_company

    df["_edd_query"] = pd.NA
    df.loc[org_mask, "_edd_query"] = df.loc[org_mask, "canonical_name"]
    df.loc[ind_mask, "_edd_query"] = df.loc[ind_mask, "employer"]

    # Drop rows with no usable query string.
    df.loc[df["_edd_query"].fillna("").str.strip() == "", "_edd_query"] = pd.NA

    to_query = df.dropna(subset=["_edd_query"]).drop_duplicates("_edd_query")
    n_queries = len(to_query)

    print()
    print("=" * 70)
    print("PART 2 — EDD live lookup")
    print("=" * 70)
    print(f"  Still-unmatched orgs:        {int(org_mask.sum()):>4,}")
    print(f"  Still-unmatched individuals: {int(ind_mask.sum()):>4,}")
    print(f"  Unique queries to run:       {n_queries:>4,}")
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
    return df


# ---------- part 3: column-specific keyword match ----------

def load_keywords(path: Path) -> dict[str, pd.DataFrame]:
    """Load keywords and split by source_sheet into named groups."""
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
                n_matched += 1
                break
    return n_matched


def match_keywords(df: pd.DataFrame, keywords_path: Path) -> pd.DataFrame:
    print()
    print("=" * 70)
    print("PART 2 — column-specific keyword match")
    print("=" * 70)

    kw_groups = load_keywords(keywords_path)
    print()

    company_patterns = _compile_patterns(kw_groups["company_keywords"])
    employer_patterns = _compile_patterns(kw_groups["employer_keywords"])
    occupation_patterns = _compile_patterns(kw_groups["occupation_keywords"])

    unmatched = df["data_source_1"].isna()

    # company_keywords -> canonical_name (orgs only)
    org_unmatched = df.index[unmatched & (df["entity_type"] == "organization")].tolist()
    n_company = _apply_keyword_match(df, org_unmatched, "canonical_name",
                                     company_patterns, "company_keywords")
    print(f"  company_keywords  (canonical_name, orgs):   {n_company:>4,} matched")

    # employer_keywords -> employer (any entity with a real employer)
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


# ---------- summary ----------

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
        for s in ("pre_classified", "employer_lookup", "running_list", "edd", "keyword match", "unmatched"):
            count = int((sub_src == s).sum())
            if count:
                print(f"    {s:14s}: {count:>4,} ({count / max(len(sub), 1) * 100:>5.1f}%)")
        print()



# ---------- main ----------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--running-list", type=Path, default=DEFAULT_RUNNING_LIST)
    p.add_argument("--keywords", type=Path, default=DEFAULT_KEYWORDS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--pre-classified", type=Path, default=DEFAULT_PRE_CLASSIFIED)
    p.add_argument("--no-edd", action="store_true", help="Skip EDD live lookups (Part 2)")
    p.add_argument("--edd-delay", type=float, default=1.0, help="Seconds between EDD requests")
    args = p.parse_args(argv)

    print(f"Input: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  rows={len(df):,}  entity_types={df['entity_type'].value_counts().to_dict()}")

    pre_df = pd.DataFrame()
    emp_df = pd.DataFrame()

    if args.pre_classified.exists():
        pre_df, df = match_pre_classified(df, args.pre_classified)
        emp_df, df = match_employer_lookup(df, args.pre_classified)
    else:
        print("\nPart 0 / employer lookup skipped: --pre-classified file not found")

    df = match_running_list(df, args.running_list)
    if not args.no_edd:
        df = match_edd(df, delay=args.edd_delay)
    df = match_keywords(df, args.keywords)

    parts = [p for p in [pre_df, emp_df, df] if not (isinstance(p, pd.DataFrame) and p.empty)]
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else df

    print_summary(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in OUT_COLS if c in df.columns] + [
        c for c in df.columns if c not in OUT_COLS
    ]
    df[cols].to_csv(args.out, index=False)
    print(f"\nWrote: {args.out}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
