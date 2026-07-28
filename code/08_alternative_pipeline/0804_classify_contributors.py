"""
0803_classify_contributors.py

Alternative classification pipeline for pre-aggregated contributor data.
Operates directly on classification_input.csv (output of the entity-resolution
stage).

  Part 1: Running-list match (H1B + EDD employer -> NAICS)
           - Organizations: canonical_name matched against running_list_alt.csv
           - Individuals:   employer matched against running_list_alt.csv
                            (skipped for non-company occupations)
  Part 2: Column-specific keyword match
           - company_keywords  -> contributor name  (organizations only)
           - employer_keywords -> employer        (individuals)
           - occupation_keywords -> occupation    (individuals)

Usage
-----
    python 0803_classify_contributors.py
    python 0803_classify_contributors.py --input path/to/classification_input.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUTS_DIR = Path(__file__).parent / "08_inputs"

DEFAULT_INPUT = INPUTS_DIR / "classification_input.csv"
DEFAULT_RUNNING_LIST = REPO_ROOT / "data/03_input/masterfile/running_list_alt.csv"
DEFAULT_KEYWORDS = (
    REPO_ROOT
    / "data/03_input/training data (manual classifications)/Keywords_Manually_Collected.csv"
)
DEFAULT_OUT = INPUTS_DIR / "classified_contributors.csv"

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


# ---------- part 1: running-list match ----------

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


# ---------- part 2: column-specific keyword match ----------

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
        for s in ("running_list", "keyword match", "unmatched"):
            count = int((sub_src == s).sum())
            print(f"    {s:14s}: {count:>4,} ({count / max(len(sub), 1) * 100:>5.1f}%)")
        print()


# ---------- main ----------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--running-list", type=Path, default=DEFAULT_RUNNING_LIST)
    p.add_argument("--keywords", type=Path, default=DEFAULT_KEYWORDS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    print(f"Input: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  rows={len(df):,}  entity_types={df['entity_type'].value_counts().to_dict()}")

    df = match_running_list(df, args.running_list)
    df = match_keywords(df, args.keywords)
    print_summary(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in OUT_COLS if c in df.columns] + [
        c for c in df.columns if c not in OUT_COLS
    ]
    df[cols].to_csv(args.out, index=False)
    print(f"\nWrote: {args.out}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
