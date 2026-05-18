"""
build_running_list.py

Aggregate name -> industry-classification reference datasets into a single
"running list" used to label donor names for CA campaign finance analysis
(2026 midterms / governor's race).

Sources merged (vertically stacked, one row per unique normalized name):
  - opensecrets_matches.csv : 3-level OpenSecrets industry hierarchy
  - H1BEmployer_match.csv   : USCIS H-1B employer file w/ NAICS sector labels

Output:
  - data/03_input/masterfile/running_list.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MASTERFILE_DIR = REPO_ROOT / "data" / "03_input" / "masterfile"

OPENSECRETS_PATH = MASTERFILE_DIR / "opensecrets_matches.csv"
H1B_PATH = MASTERFILE_DIR / "H1BEmployer_match.csv"
OUT_PATH = MASTERFILE_DIR / "running_list.csv"

# Unified schema. Both sources are conformed to these columns; fields a
# source doesn't carry stay NaN (e.g. NAICS for OpenSecrets, level2/3 for H-1B).
RUNNING_LIST_COLS = [
    "name",
    "name_norm",
    "level1_category",
    "level2_category",
    "level3_category",
    "naics_code",
    "naics_label",
    "source",
]


def normalize_name(s: pd.Series) -> pd.Series:
    """Mirror the normalization used in opensecrets_matches.name_norm so the
    two sources can be deduped on a common key: uppercase, strip everything
    that isn't alphanumeric or space, collapse whitespace.
    """
    return (
        s.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9 ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def load_opensecrets() -> pd.DataFrame:
    df = pd.read_csv(OPENSECRETS_PATH)
    df["naics_code"] = pd.NA
    df["naics_label"] = pd.NA
    df["source"] = "opensecrets"
    return df[RUNNING_LIST_COLS]


def load_h1b(state_filter: str | None = None) -> pd.DataFrame:
    """Load USCIS H-1B file and conform to the running-list schema.

    state_filter: e.g. "CA" to keep only CA-headquartered petitioners. Default
    keeps the national list — donors to CA campaigns can come from anywhere.
    """
    # File is UTF-16 tab-separated with trailing whitespace in some headers.
    df = pd.read_csv(H1B_PATH, encoding="utf-16", sep="\t")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Employer (Petitioner) Name"]).copy()

    if state_filter:
        df = df[df["Petitioner State"].str.strip() == state_filter].copy()

    # NAICS arrives as "54 - Professional, Scientific, and Technical Services".
    naics_split = (
        df["Industry (NAICS) Code"].astype(str).str.split(" - ", n=1, expand=True)
    )
    df["naics_code"] = naics_split[0].str.strip()
    df["naics_label"] = (
        naics_split[1].str.strip() if naics_split.shape[1] > 1 else pd.NA
    )

    df = df.rename(columns={"Employer (Petitioner) Name": "name"})
    df["name_norm"] = normalize_name(df["name"])
    df = df[df["name_norm"] != ""]

    # Multiple H-1B rows per employer (per fiscal year, per filing). Collapse
    # to one row per employer; pick the modal NAICS in case sectors disagree.
    def _mode_or_na(s: pd.Series):
        s = s.dropna()
        return s.mode().iat[0] if not s.empty else pd.NA

    df = df.groupby("name_norm", as_index=False).agg(
        name=("name", "first"),
        naics_code=("naics_code", _mode_or_na),
        naics_label=("naics_label", _mode_or_na),
    )

    df["level1_category"] = pd.NA
    df["level2_category"] = pd.NA
    df["level3_category"] = pd.NA
    df["source"] = "h1b_employer"
    return df[RUNNING_LIST_COLS]


def build_running_list(h1b_state_filter: str | None = None) -> pd.DataFrame:
    os_df = load_opensecrets()
    h1b_df = load_h1b(state_filter=h1b_state_filter)

    combined = pd.concat([os_df, h1b_df], ignore_index=True)

    # Same employer can appear in both sources. Prefer OpenSecrets since it
    # carries the richer 3-level taxonomy; collapse on normalized name.
    combined["_priority"] = (combined["source"] == "opensecrets").astype(int)
    combined = (
        combined.sort_values("_priority", ascending=False, kind="stable")
        .drop_duplicates(subset="name_norm", keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )
    return combined


def main() -> None:
    running = build_running_list(h1b_state_filter=None)
    running.to_csv(OUT_PATH, index=False)

    print(f"Wrote running list: {OUT_PATH}")
    print(f"Rows: {len(running):,}")
    print("Source breakdown:")
    print(running["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
