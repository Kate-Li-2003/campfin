"""
0304_map_opensecrets_naics_crosswalk.py

Fill in NAICS codes for the OpenSecrets-sourced rows of running_list.csv
using the OpenSecrets-level3 -> NAICS-sector crosswalk.

Why
---
running_list.csv is the training set for the 07 ML classifier. OpenSecrets
rows (~40k, the bulk of the data) arrive with NO naics_code: 0301 loads them
with naics_code = NA because the OpenSecrets taxonomy carries no NAICS. As a
result the naics_code model trains almost entirely on the ~22k H-1B rows and
its NAICS predictions are weak / biased.

This step joins each OpenSecrets row's `level3_category` to the crosswalk
(`Open Secrets Level 3 Categories` -> 2-digit NAICS sector + sector label),
populating naics_code / naics_label so the NAICS target has ~3x the labeled
data and is aligned with the OpenSecrets industry breakdown. The 2-digit
sector granularity matches the existing H-1B rows exactly (same code strings
like "54", "31-33", and same sector descriptions), so the classes merge
cleanly rather than fragmenting the label space.

Rules
-----
  - Only rows where source == "opensecrets" are touched.
  - Only rows whose naics_code is still empty are filled (idempotent;
    H-1B / EDD rows and any already-filled rows are left untouched).
  - Codes are cleaned to match the H-1B label space: trailing annotation
    characters ("72*" -> "72", "44-45?" -> "44-45") are stripped.
  - Labels are cleaned likewise (" (not in source list)" dropped so e.g.
    "Accommodation and Food Services (not in source list)" merges with the
    H-1B "Accommodation and Food Services").
  - Crosswalk entries with no NAICS code (the non-industry buckets such as
    "Unitemized (small) contributions") leave naics_code empty.

The same crosswalk join is the right thing to also do inside
0301_build_running_list.load_opensecrets(); this script exists so the live
running_list.csv can be enriched in place without a full rebuild (rebuilding
would drop EDD rows whose upstream donor file has since shrunk).

Usage
-----
    python 0304_map_opensecrets_naics_crosswalk.py            # write in place
    python 0304_map_opensecrets_naics_crosswalk.py --dry-run  # preview only
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNING_LIST = REPO_ROOT / "data/03_input/masterfile/running_list.csv"
DEFAULT_CROSSWALK = (
    REPO_ROOT / "data/03_input/masterfile/NAICS_OpenSecrets_Crosswalk_Mapped.xlsx"
)

# Crosswalk column names (NAICS_OpenSecrets_Crosswalk_Mapped.xlsx / Sheet1).
CW_KEY = "Open Secrets Level 3 Categories"
CW_CODE = "2-Digit NAICS Codes"
CW_LABEL = "NAICS Sector Description"


def _clean_code(v) -> str:
    """Normalize a crosswalk NAICS code to the string form used in the
    running list. Strips annotation marks ("72*", "44-45?") but keeps the
    hyphenated sector ranges ("31-33", "44-45", "48-49") intact."""
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if s.endswith(".0"):  # guard against codes read back as floats
        s = s[:-2]
    s = re.sub(r"[*?]", "", s).strip()
    return "" if s.lower() in ("", "nan") else s


def _clean_label(v) -> str:
    """Drop the editorial "(not in source list)" annotation so the label
    matches the H-1B sector description for the same code."""
    if pd.isna(v):
        return ""
    s = str(v).strip()
    s = s.replace(" (not in source list)", "").strip()
    return "" if s.lower() == "nan" else s


def load_crosswalk(path: Path) -> pd.DataFrame:
    cw = pd.read_excel(path)
    cw.columns = [c.strip() for c in cw.columns]
    missing = {CW_KEY, CW_CODE, CW_LABEL} - set(cw.columns)
    if missing:
        raise SystemExit(f"Crosswalk {path.name} missing columns: {missing}")

    cw = cw.assign(
        _key=cw[CW_KEY].astype(str).str.strip(),
        _code=cw[CW_CODE].map(_clean_code),
        _label=cw[CW_LABEL].map(_clean_label),
    )
    # Keep only entries that actually carry a NAICS code; drop blank/dup keys.
    cw = cw[(cw["_key"] != "") & (cw["_code"] != "")]
    cw = cw.drop_duplicates("_key", keep="first")
    return cw[["_key", "_code", "_label"]]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--running-list", type=Path, default=DEFAULT_RUNNING_LIST)
    p.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing running_list.csv",
    )
    args = p.parse_args(argv)

    rl = pd.read_csv(args.running_list, dtype=str)
    cw = load_crosswalk(args.crosswalk)
    code_map = dict(zip(cw["_key"], cw["_code"]))
    label_map = dict(zip(cw["_key"], cw["_label"]))

    print(f"Loaded {len(rl):,} running_list rows; crosswalk has {len(cw):,} "
          f"mapped categories.")
    print(f"  source mix: {rl['source'].value_counts().to_dict()}")

    is_os = rl["source"] == "opensecrets"
    empty_naics = rl["naics_code"].isna() | (rl["naics_code"].astype(str).str.strip() == "")
    target = is_os & empty_naics

    key = rl["level3_category"].astype(str).str.strip()
    mapped_code = key.map(code_map)
    mapped_label = key.map(label_map)

    to_fill = target & mapped_code.notna()
    no_map = target & mapped_code.isna()

    print(f"\nOpenSecrets rows needing NAICS:    {int(target.sum()):>6,}")
    print(f"  filled from crosswalk:           {int(to_fill.sum()):>6,}")
    print(f"  left empty (no usable mapping):  {int(no_map.sum()):>6,}")
    if no_map.any():
        cats = key[no_map].value_counts()
        print("    unmapped level3 categories:")
        for cat, n in cats.items():
            print(f"      {n:>5,}  {cat!r}")

    rl.loc[to_fill, "naics_code"] = mapped_code[to_fill]
    rl.loc[to_fill, "naics_label"] = mapped_label[to_fill]

    print("\nNAICS coverage after fill, by source:")
    has = rl["naics_code"].notna() & (rl["naics_code"].astype(str).str.strip() != "")
    print(
        rl.assign(has_naics=has)
        .groupby("source")["has_naics"].agg(["sum", "count"])
        .to_string()
    )

    if args.dry_run:
        print("\n--dry-run: not writing. Sample of newly filled rows:")
        cols = ["name", "level3_category", "naics_code", "naics_label"]
        print(rl.loc[to_fill, cols].head(12).to_string(index=False))
        return

    if to_fill.any():
        backup = args.running_list.with_suffix(args.running_list.suffix + ".pre_naics_bak")
        if not backup.exists():
            shutil.copy2(args.running_list, backup)
            print(f"\nBacked up original to: {backup.name}")
        rl.to_csv(args.running_list, index=False)
        print(f"Wrote: {args.running_list}")
    else:
        print("\nNothing to fill; running_list.csv unchanged.")


if __name__ == "__main__":
    main()
