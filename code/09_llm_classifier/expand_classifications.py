"""Expand deduped LLM classification output back to one row per raw contribution.

0901_web_search.py dedupes classification_input.csv down to unique search_key
units *before* anything gets searched or saved -- for every group of raw
contribution rows sharing the same contributor identity, only one
representative row's uuid survives into the batch/cache files that 0902
classifies. So a "full" classification output (classification_full_*.csv) has
one row per unique search_key, not one row per raw contribution -- most uuids
in classification_input.csv are never directly represented.

This module re-joins each unit's classification back onto *every* raw
contribution row that shares its search_key, via the same recipe
0901_web_search.py uses to compute search_key in the first place. The result
has one row per uuid in classification_input.csv, matching the granularity
assign_final_classification.Rmd expects for its uuid-based join against
rule/os.

Usage
-----
    python expand_classifications.py --llm 09_outputs/classification_full_full_2026-08-13.csv \
                                      --llm 09_outputs/classification_full_web_cache_full_2026-08-13.csv \
                                      --out 09_outputs/classification_full_expanded_2026-08-13.csv

Or import expand_to_full_coverage() directly (used by 0902's save cell so every
run produces an already-expanded file automatically).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import COLUMNS, DEFAULT_INPUT_PATH
from standardization_helpers import (
    normalize_for_key, normalize_employer_for_key, lightly_process_employer,
    standardize_occupation_employer, NOT_EMPLOYED,
)

_NOT_EMPLOYED = set(NOT_EMPLOYED)


def _is_not_employed(employer: str, occupation: str) -> bool:
    e = (employer or "").strip().lower()
    o = (occupation or "").strip().lower()
    return (not e or e in _NOT_EMPLOYED) and (not o or o in _NOT_EMPLOYED)


def _make_key(name_raw, employer_raw, employer_processed, occ, city="", state="", entity_type="individual"):
    key_name = normalize_for_key(name_raw)
    key_employer = normalize_employer_for_key(employer_raw)
    if entity_type == "individual" and _is_not_employed(employer_processed, occ):
        return (key_name, key_employer, normalize_for_key(city), normalize_for_key(state))
    return (key_name, key_employer, "", "")


def compute_search_keys(classification_input_path: str = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """Recompute search_key for every raw row of classification_input.csv --
    same recipe 0901_web_search.py uses -- and return just [uuid, search_key]
    (one row per raw contribution, no deduping)."""
    df = pd.read_csv(classification_input_path, dtype=str).fillna("")
    name_raw_col, emp_raw_col = COLUMNS["name_raw"], COLUMNS["employer_raw"]
    employer_col, occ_col = COLUMNS["employer"], COLUMNS["occupation"]
    city_col, state_col, entity_type_col = COLUMNS.get("city"), COLUMNS.get("state"), COLUMNS["entity_type"]

    cities = df[city_col] if city_col in df.columns else pd.Series("", index=df.index)
    states = df[state_col] if state_col in df.columns else pd.Series("", index=df.index)

    search_employer = [
        lightly_process_employer(std) if (std.strip() or etype == "organization")
        else standardize_occupation_employer((raw or "").strip().upper())
        for std, raw, etype in zip(df[employer_col], df[emp_raw_col], df[entity_type_col])
    ]
    df["search_key"] = [
        "|".join(_make_key(nr, er, ep, o, c, s, et))
        for nr, er, ep, o, c, s, et in zip(
            df[name_raw_col], df[emp_raw_col], search_employer, df[occ_col], cities, states, df[entity_type_col]
        )
    ]
    return df[["uuid", "search_key"]]


def expand_to_full_coverage(
    llm_paths: list[str],
    classification_input_path: str = DEFAULT_INPUT_PATH,
    classification_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return one row per raw contribution uuid in classification_input.csv,
    with each unit's classification (from whichever llm_paths file covers its
    search_key) broadcast to every uuid sharing that unit.

    llm_paths: one or more classification_full_*.csv files (must have search_key).
    classification_cols: which classification columns to bring over; defaults to
    the standard set (naics_code_llm, naics_confidence, open_secrets_category,
    open_secrets_confidence, naics_reasoning, industry_summary, confidence,
    is_prominent, prominence_reason) -- only columns actually present are used.
    """
    default_cols = [
        "naics_code_llm", "naics_description", "naics_confidence",
        "open_secrets_category", "open_secrets_confidence", "naics_reasoning",
        "industry_summary", "confidence", "is_prominent", "prominence_reason",
    ]
    classification_cols = classification_cols or default_cols

    frames = [pd.read_csv(p, dtype=str) for p in llm_paths]
    combined = pd.concat(frames, ignore_index=True)
    dupes = combined["search_key"].duplicated().sum()
    if dupes:
        print(f"  WARNING: {dupes:,} search_key(s) appear in more than one input file -- "
              f"keeping the first occurrence")
        combined = combined.drop_duplicates(subset="search_key", keep="first")

    keep_cols = ["search_key"] + [c for c in classification_cols if c in combined.columns]
    combined = combined[keep_cols]

    bridge = compute_search_keys(classification_input_path)
    expanded = bridge.merge(combined, on="search_key", how="left")

    n_total = len(expanded)
    n_classified = expanded[keep_cols[1]].notna().sum() if len(keep_cols) > 1 else 0
    print(f"  Expanded {len(frames)} file(s) covering {combined['search_key'].nunique():,} units "
          f"-> {n_total:,} raw rows ({n_classified:,} classified, "
          f"{n_total - n_classified:,} still unclassified)")
    return expanded


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--llm", action="append", required=True, dest="llm_paths",
                    help="classification_full_*.csv file (repeatable)")
    p.add_argument("--input", default=DEFAULT_INPUT_PATH, help="classification_input.csv path")
    p.add_argument("--out", required=True, help="Output path")
    args = p.parse_args(argv)

    expanded = expand_to_full_coverage(args.llm_paths, args.input)
    expanded.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}  ({len(expanded):,} rows)")


if __name__ == "__main__":
    main()
