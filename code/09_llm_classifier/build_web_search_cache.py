"""Consolidate disparate web search output files into a single web_search_cache.csv.

Run once to seed the cache from prior outputs, then web_search.py keeps it updated.

Usage:
    python build_web_search_cache.py
    python build_web_search_cache.py --inputs path1.csv path2.csv
    python build_web_search_cache.py --output 09_outputs/web_search_cache.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from standardization_helpers import (
    normalize_for_key, normalize_employer_for_key, NOT_EMPLOYED,
    standardize_city, standardize_state, standardize_occupation_employer,
)


_NOT_EMPLOYED = NOT_EMPLOYED  # imported from standardization_helpers


def _is_not_employed(employer: str, occupation: str) -> bool:
    e = (employer or "").strip().lower()
    o = (occupation or "").strip().lower()
    return (not e or e in _NOT_EMPLOYED) and (not o or o in _NOT_EMPLOYED)


def _make_search_key(name_raw: str, employer_raw: str, employer_processed: str,
                     occ: str, city: str = "", state: str = "") -> str:
    key_name     = normalize_for_key(name_raw)
    key_employer = normalize_employer_for_key(employer_raw)  # maps "None"/NA variants → ""
    if _is_not_employed(employer_processed, occ):
        # Format mirrors "|".join(tuple) in 0901_web_search.py: name|employer|city|state
        return f"{key_name}|{key_employer}|{normalize_for_key(city)}|{normalize_for_key(state)}"
    return f"{key_name}|{key_employer}||"


CACHE_COLS = [
    "search_key",
    "Contributor.Name", "Contributor.Employer",
    "standardized_name", "standardized_employer_name",
    "standardized_city", "Contributor.State",
    "industry_summary", "urls", "confidence", "is_prominent", "prominence_reason",
]

# All known prior output files, relative to this script's directory.
# Files listed later win on duplicate keys (assumed more recent / higher quality).
DEFAULT_INPUTS = [
    "../10_assign_classifications/10_inputs/web_search_output_100_200_v2.csv",
    "../10_assign_classifications/10_inputs/web_search_output_200_400.csv",
    "../10_assign_classifications/10_inputs/web_search_output_400_600.csv",
    "../10_assign_classifications/10_inputs/web_search_output_100_600_1000.csv",
    "../10_assign_classifications/10_inputs/web_search_output_new_rows_071326.csv",
    "../10_assign_classifications/10_inputs/prominent_individual_results_071326.csv",
]

DEFAULT_OUTPUT = "09_outputs/web_search_cache.csv"


def _build_key_for_row(row) -> str:
    if "search_key" in row.index and row["search_key"]:
        return row["search_key"]

    # Raw names — used by normalize_for_key() inside _make_search_key
    name_raw     = row.get("Contributor.Name", "") or row.get("standardized_name", "") or ""
    employer_raw = row.get("Contributor.Employer", "") or row.get("standardized_employer_name", "") or ""

    # Processed employer for _is_not_employed() check
    if "processed_employer_name" in row.index:
        employer_processed = row.get("processed_employer_name", "") or ""
    else:
        employer_processed = standardize_occupation_employer(
            row.get("standardized_employer_name", "") or ""
        )

    occ = row.get("processed_occupation", "") or ""

    if "standardized_city" in row.index:
        city = standardize_city(row.get("standardized_city", "") or "")  # re-apply in case upstream output was imperfect
    elif "Contributor.City" in row.index:
        city = standardize_city(row.get("Contributor.City", "") or "")
    else:
        city = ""

    if "standardized_state" in row.index:
        state = row.get("standardized_state", "") or ""
    elif "Contributor.State" in row.index:
        state = standardize_state(row.get("Contributor.State", "") or "")
    else:
        state = ""

    return _make_search_key(name_raw, employer_raw, employer_processed, occ, city, state)


def load_file(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  Not found, skipping: {path.name}")
        return None

    df = pd.read_csv(path)
    str_cols = df.select_dtypes(include=["object", "str"]).columns
    df[str_cols] = df[str_cols].fillna("")

    df["search_key"] = df.apply(_build_key_for_row, axis=1)

    present = [c for c in CACHE_COLS if c in df.columns]
    df = df[present].copy()

    # Fill any missing columns (old files lack city/state, is_prominent, etc.)
    for col, default in [
        ("Contributor.Name", ""), ("Contributor.Employer", ""),
        ("standardized_city", ""), ("Contributor.State", ""),
        ("is_prominent", False), ("prominence_reason", ""),
        ("industry_summary", ""), ("urls", ""), ("confidence", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    df = df[CACHE_COLS]
    print(f"  Loaded {len(df):,} rows from {path.name}")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS, metavar="CSV",
                   help="Web search output files to consolidate.")
    p.add_argument("--output", default=DEFAULT_OUTPUT, metavar="CSV",
                   help=f"Output path (default: {DEFAULT_OUTPUT}).")
    args = p.parse_args()

    root = Path(__file__).resolve().parent

    frames = []
    for raw in args.inputs:
        path = Path(raw) if Path(raw).is_absolute() else root / raw
        df = load_file(path)
        if df is not None:
            frames.append(df)

    if not frames:
        print("No files loaded — nothing to write.")
        return

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    # Later files win on duplicates (assumed to be more recent / higher quality)
    combined = combined.drop_duplicates(subset=["search_key"], keep="last")
    print(f"\nRows before dedup: {before:,}  |  after: {len(combined):,}")

    out_path = Path(args.output) if Path(args.output).is_absolute() else root / args.output
    out_path.parent.mkdir(exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Wrote {len(combined):,} rows to {out_path}")


if __name__ == "__main__":
    main()
