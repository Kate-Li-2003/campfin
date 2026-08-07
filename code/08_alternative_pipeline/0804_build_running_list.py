"""
0802_build_running_list.py

Build the alternative running list: H-1B and EDD employer -> NAICS matches.
Mirrors what 0301/0303_build_running_list.py do for the original
running_list.csv, but excludes OpenSecrets data and manually labeled data. 
-------
  - H1BEmployer_match.csv : loaded via load_h1b() from 0301_build_running_list.py
  - running_list.csv      : existing masterfile; rows with source == "edd" reused as-is

Output
------
  - data/03_input/masterfile/running_list_alt.csv
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MASTERFILE_DIR = REPO_ROOT / "data" / "03_input" / "masterfile"

OLD_RUNNING_LIST_PATH = MASTERFILE_DIR / "running_list.csv"
OUT_PATH = MASTERFILE_DIR / "running_list_alt.csv"

SOURCE_PRIORITY = {
    "h1b_employer": 2,
    "edd": 1,
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
    return (
        combined.sort_values("_priority", ascending=False, kind="stable")
        .drop_duplicates(subset="name_norm", keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )


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
