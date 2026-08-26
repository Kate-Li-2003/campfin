"""
0802_build_running_list.py

Build the alternative running list: H-1B and EDD employers ONLY and their raw NAICS codes.
Mirrors what 0301/0303_build_running_list.py does, but excludes OpenSecrets data and manually labeled data. 
-------
  - H1BEmployer_match.csv : loaded via load_h1b() from 0301_build_running_list.py
  - running_list.csv      : existing masterfile; rows with source == "edd" reused as-is

Output
------
  - data/03_input/masterfile/running_list_alt.csv
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pandas as pd

# strip common suffixes from normalized names
_CORP_SUFFIX_RE = re.compile(
    r"\s+(?:INC|INCORPORATED|LLC|LLP|LP|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED)\s*$"
)

_NAICS_RANGE_MAP = {"31": "31-33", "32": "31-33", "33": "31-33",
                    "44": "44-45", "45": "44-45", "48": "48-49", "49": "48-49"}

def _naics_parent(code) -> str:
    """Map raw NAICS code to its 2-digit sector for conflict comparison.
    e.g. '334413' → '31-33', '523210' → '52'. Custom codes (≤2 digits) pass through."""
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

REPO_ROOT = Path(__file__).resolve().parents[2]
MASTERFILE_DIR = REPO_ROOT / "data" / "03_input" / "masterfile"

OLD_RUNNING_LIST_PATH = MASTERFILE_DIR / "running_list.csv"
OUT_PATH = MASTERFILE_DIR / "running_list_alt.csv"

# HIGHER number = preferred source
# Prefer EDD codes are more precise (6-digit NAICS) and easier to map to custom codes.
SOURCE_PRIORITY = {
    "edd": 2,
    "h1b_employer": 1,
}


def load_h1b_df() -> pd.DataFrame:
    """Reuse load_h1b() from 0301_build_running_list.py as-is.
       H1B_PATH is patched here after import.
    """
    sys.path.insert(0, str(REPO_ROOT / "code" / "03_aggregating_data"))
    rl0301 = importlib.import_module("0301_build_running_list")
    rl0301.H1B_PATH = MASTERFILE_DIR / "H1BEmployer_match.csv"

    h1b_df = rl0301.load_h1b(state_filter=None)
    h1b_df["entity_type"] = "organization"
    return h1b_df


def load_edd_df() -> pd.DataFrame:
    old_rl = pd.read_csv(OLD_RUNNING_LIST_PATH, dtype=str)
    edd_df = old_rl[old_rl["source"] == "edd"].copy()
    edd_df["entity_type"] = "organization"
    return edd_df


def build_running_list() -> pd.DataFrame:
    h1b_df = load_h1b_df()
    edd_df = load_edd_df()

    combined = pd.concat([h1b_df, edd_df], ignore_index=True)
    combined["_priority"] = combined["source"].map(SOURCE_PRIORITY).fillna(0)
    running = (
        combined.sort_values("_priority", ascending=False, kind="stable")
        .drop_duplicates(subset="name_norm", keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )


    # drop entities for which EDD and H1B had different NAICS codes
    # e.g. Intuit vs Intuit Inc. had different codes between the sources
    running["_stripped"] = (
        running["name_norm"]
        .str.replace(_CORP_SUFFIX_RE, "", regex=True) # strip suffixes
        .str.strip()
    )

    # normalize codes before comparison
    running["_parent_code"] = running["naics_code"].map(_naics_parent)
    has_code = (running["_parent_code"] != "") & (running["_stripped"] != "")
    code_by_base = running[has_code].groupby("_stripped")["_parent_code"].nunique()
    conflict_bases = set(code_by_base[code_by_base > 1].index)
    if conflict_bases:
        conflict_mask = running["_stripped"].isin(conflict_bases)
        conflict_df = (
            running[conflict_mask]
            [["_stripped", "name", "name_norm", "naics_code", "_parent_code", "source"]]
            .sort_values(["_stripped", "_parent_code"])
        )
        n_conflict = int(conflict_mask.sum())
        print(f"\nConflict removal — {len(conflict_bases)} base-name group(s) with mismatched codes:")
        print(f"  dropping {n_conflict} entries (neither can be trusted for auto-classification)")
        print(conflict_df.to_string(index=False))

        conflict_out = Path(__file__).parent / "08_outputs" / "running_list_conflicts.csv"
        conflict_out.parent.mkdir(parents=True, exist_ok=True)
        conflict_df.to_csv(conflict_out, index=False)
        print(f"  conflict details written to: {conflict_out}")

        running = running[~conflict_mask].copy()
    else:
        print("\nConflict detection: no base-name conflicts found.")

    return running.drop(columns=["_stripped", "_parent_code"]).reset_index(drop=True)


def main() -> None:
    running = build_running_list()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    running.to_csv(OUT_PATH, index=False)

    print(f"Wrote alternative running list: {OUT_PATH}")
    print(f"Rows: {len(running):,}")
    print("Entity type breakdown:")
    print(running["entity_type"].value_counts().to_string())
    print("Source breakdown:")
    print(running["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
