"""
0806_extend_running_list_from_edd.py

Promote EDD-derived classifications from classified_contributors.csv into
running_list_alt.csv so subsequent 0804 runs hit the running list directly
instead of re-querying EDD.

Rules:
  - Only rows where data_source_1 == "edd" are considered.
  - NAICS 999990 ("Unknown") is excluded.
  - Existing name_norm values in running_list_alt are NOT overwritten
    (H-1B / prior EDD entries take precedence).
  - Re-running is idempotent: previously-added EDD rows are skipped.

The query string that produced the EDD hit is used as `name` (and the
basis for `name_norm`):
  - Organizations: canonical_name
  - Individuals:   employer

Usage
-----
    python 0806_extend_running_list_from_edd.py
    python 0806_extend_running_list_from_edd.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLASSIFIED = Path(__file__).parent / "08_outputs" / "classified_contributors.csv"
DEFAULT_RUNNING_LIST = REPO_ROOT / "data/03_input/masterfile/running_list_alt.csv"

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
    return (
        s.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9 ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--classified", type=Path, default=DEFAULT_CLASSIFIED)
    p.add_argument("--running-list", type=Path, default=DEFAULT_RUNNING_LIST)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing running_list_alt.csv",
    )
    args = p.parse_args(argv)

    classified = pd.read_csv(args.classified, dtype=str)
    rl = pd.read_csv(args.running_list, dtype=str)

    print(f"Loaded {len(classified):,} classified rows and {len(rl):,} running_list_alt rows")
    print(f"  current source mix: {rl['source'].value_counts().to_dict()}")

    # Filter to EDD-classified rows with a usable NAICS code.
    edd = classified[
        (classified["data_source_1"] == "edd")
        & classified["naics_code"].notna()
        & (classified["naics_code"].astype(str).str.strip() != "")
        & (classified["naics_code"].astype(str) != "999990")
    ].copy()

    n_unknown = (
        (classified["data_source_1"] == "edd")
        & (classified["naics_code"].astype(str) == "999990")
    ).sum()

    print(f"\nEDD-classified candidates from classified file:")
    print(f"  with valid NAICS:                  {len(edd):>5,}")
    print(f"  excluded (NAICS 999990 'Unknown'):  {n_unknown:>5,}")

    # Determine the query string: canonical_name for orgs, employer for individuals.
    edd["_name"] = pd.NA
    org_mask = edd["entity_type"] == "organization"
    ind_mask = ~org_mask
    edd.loc[org_mask, "_name"] = edd.loc[org_mask, "canonical_name"].astype(str)
    edd.loc[ind_mask, "_name"] = edd.loc[ind_mask, "employer"].astype(str)
    edd = edd[edd["_name"].notna() & (edd["_name"].str.strip() != "")]

    new = pd.DataFrame({
        "name": edd["_name"].astype(str),
        "name_norm": normalize_name(edd["_name"]),
        "level1_category": pd.NA,
        "level2_category": pd.NA,
        "level3_category": pd.NA,
        "naics_code": edd["naics_code"].astype(str),
        "naics_label": edd["naics_label"].astype(object),
        "source": "edd",
    })
    new = new[new["name_norm"] != ""]
    new = new.drop_duplicates("name_norm", keep="first")

    existing = set(rl["name_norm"].dropna().astype(str))
    collisions = new["name_norm"].isin(existing).sum()
    new = new[~new["name_norm"].isin(existing)]

    print(f"\nMerge plan:")
    print(f"  rows after dedupe:               {len(new) + collisions:>5,}")
    print(f"  collisions (name_norm in rl):    {collisions:>5,}  (skipped)")
    print(f"  net new rows to append:          {len(new):>5,}")

    if len(new) == 0:
        print("\nNothing new to add. running_list_alt.csv is unchanged.")
        return

    out = pd.concat([rl[RUNNING_LIST_COLS], new[RUNNING_LIST_COLS]], ignore_index=True)

    if args.dry_run:
        print("\n--dry-run: not writing. Sample of new rows:")
        print(new.head(10).to_string(index=False))
        return

    out.to_csv(args.running_list, index=False)
    print(f"\nWrote: {args.running_list}")
    print(f"  total rows now:     {len(out):>6,}")
    print(f"  source breakdown:   {out['source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
