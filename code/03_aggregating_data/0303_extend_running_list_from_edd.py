"""
0303_extend_running_list_from_edd.py

Promote EDD-derived classifications from donors_classified.csv into the
running_list.csv reference, so subsequent classification runs hit the
running list directly instead of re-querying EDD.

Rules:
  - Only rows where match_source == "edd" are considered.
  - NAICS 999990 ("Unknown") is excluded — those are matches on employer
    name but with no useful industry signal.
  - Existing name_norm values in running_list are NOT overwritten (the
    OpenSecrets / H-1B taxonomies take precedence).
  - Re-running is idempotent: previously-added EDD rows are skipped.

The donor-side employer string is used as `name` (and the basis for
`name_norm`), so future donor strings normalizing to the same key get
matched without an EDD round-trip.

Usage
-----
    python 0303_extend_running_list_from_edd.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DONORS = REPO_ROOT / "output/03_output/masterfile_classified_donors.csv"
DEFAULT_RUNNING_LIST = REPO_ROOT / "data/03_input/masterfile/running_list.csv"

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


def _format_naics(code: float | int | str) -> str:
    """Render naics_code as a clean integer string. donors_classified.csv
    stores codes as float (because of NaN columns), so 541110.0 needs to
    become "541110" before joining the running_list (where naics_code is
    stored as a string).
    """
    if pd.isna(code):
        return ""
    if isinstance(code, float):
        return str(int(code))
    return str(code)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--donors", type=Path, default=DEFAULT_DONORS)
    p.add_argument("--running-list", type=Path, default=DEFAULT_RUNNING_LIST)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing running_list.csv",
    )
    args = p.parse_args(argv)

    donors = pd.read_csv(args.donors)
    rl = pd.read_csv(args.running_list)

    print(f"Loaded {len(donors):,} donor rows and {len(rl):,} running_list rows")
    print(f"  current source mix: {rl['source'].value_counts().to_dict()}")

    # 1. Filter to EDD-classified rows with a usable NAICS code.
    edd = donors[
        (donors["match_source"] == "edd")
        & donors["naics_code"].notna()
        & (donors["naics_code"] != 999990)  # EDD "Unknown" — drop
    ].copy()
    n_unknown = (
        (donors["match_source"] == "edd") & (donors["naics_code"] == 999990)
    ).sum()
    print(f"\nEDD-classified candidates from donors file:")
    print(f"  with valid NAICS:                {len(edd):>5,}")
    print(f"  excluded (NAICS 999990 'Unknown'): {n_unknown:>5,}")

    # 2. Build new rows in running_list schema.
    new = pd.DataFrame(
        {
            "name": edd["employer"].astype(str),
            "name_norm": edd["employer_norm"].astype(str),
            "level1_category": pd.NA,
            "level2_category": pd.NA,
            "level3_category": pd.NA,
            "naics_code": edd["naics_code"].map(_format_naics),
            # naics_label was already overwritten with the long-form EDD
            # description in 0302 step 3, so it's authoritative here.
            "naics_label": edd["naics_label"].astype(object),
            "source": "edd",
        }
    )
    new = new[new["name_norm"] != ""]
    new = new.drop_duplicates("name_norm", keep="first")

    # 3. Skip any name_norm already in running_list (don't overwrite
    #    OpenSecrets / H-1B / prior EDD entries).
    existing = set(rl["name_norm"].dropna().astype(str))
    collisions = new["name_norm"].isin(existing).sum()
    new = new[~new["name_norm"].isin(existing)]

    print(f"\nMerge plan:")
    print(f"  rows after dedupe:               {len(new) + collisions:>5,}")
    print(f"  collisions (name_norm in rl):    {collisions:>5,}  (skipped)")
    print(f"  net new rows to append:          {len(new):>5,}")

    if len(new) == 0:
        print("\nNothing new to add. running_list.csv is unchanged.")
        return

    # 4. Append + write.
    out = pd.concat(
        [rl[RUNNING_LIST_COLS], new[RUNNING_LIST_COLS]], ignore_index=True
    )

    if args.dry_run:
        print("\n--dry-run: not writing. Sample of new rows:")
        print(new.head(10).to_string(index=False))
        return

    out.to_csv(args.running_list, index=False)
    print(f"\nWrote: {args.running_list}")
    print(f"  total rows now:                  {len(out):>6,}")
    print(f"  source breakdown:                {out['source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
