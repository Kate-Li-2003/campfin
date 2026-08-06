"""Build Masterfile #2 (custom classification scheme) from Masterfile #1.

PROCEDURAL script. Masterfile #1 (running_list.csv) holds real NAICS
codes from static sources (OpenSecrets/H1B via 0301, EDD via 0302/0303,
crosswalk via 0304, manual real-NAICS rows via 0305). This script derives
Masterfile #2 (running_list_custom.csv) by running every entity through
the sub-code resolver (subcode_resolution.py) against the editorial
taxonomy in custom_naics_labels.csv, then applying manual custom-coded
overrides (manual_custom_overrides.csv, written by 0305) last.

Idempotent and cheap: when editorial preferences change, edit
custom_naics_labels.csv and rerun THIS script only — nothing upstream
(EDD caches, crosswalk, ML training) is touched.

Usage:
    python 0306_build_custom_masterfile.py [--no-encoder] [--out PATH]

--no-encoder skips the embedding fallback (partitioned sectors 52/56/77
with no regex hit are marked `unresolved-partition` for later re-run).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import subcode_resolution as sr

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MF1 = REPO_ROOT / "data/03_input/masterfile/running_list.csv"
DEFAULT_MF2 = REPO_ROOT / "data/03_input/masterfile/running_list_custom.csv"
MANUAL_OVERRIDES = REPO_ROOT / "data/03_input/masterfile/manual_custom_overrides.csv"


def apply_manual_overrides(mf2: pd.DataFrame, path: Path = MANUAL_OVERRIDES) -> pd.DataFrame:
    """Manual custom-coded rows (from 0305) beat automation. Matched on
    name_norm; stamps custom_code/custom_label, method 'manual'."""
    if not path.exists():
        return mf2
    manual = pd.read_csv(path, dtype=str).drop_duplicates("name_norm", keep="last")
    lut = manual.set_index("name_norm")
    hit = mf2["name_norm"].isin(lut.index)
    for col in ("custom_code", "custom_label"):
        if col in lut.columns:
            mf2.loc[hit, col] = mf2.loc[hit, "name_norm"].map(lut[col])
    mf2.loc[hit, "resolution_method"] = "manual"
    mf2.loc[hit, "resolution_confidence"] = 1.0
    print(f"  manual overrides applied: {int(hit.sum()):,} rows")
    return mf2


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mf1", type=Path, default=DEFAULT_MF1)
    p.add_argument("--out", type=Path, default=DEFAULT_MF2)
    p.add_argument("--schema", type=Path, default=sr.DEFAULT_SCHEMA)
    p.add_argument("--no-encoder", action="store_true")
    args = p.parse_args(argv)

    print("=" * 70)
    print("BUILD MASTERFILE #2 (custom classification scheme)")
    print("=" * 70)

    mf1 = pd.read_csv(args.mf1, dtype=str)
    schema = sr.load_schema(args.schema)
    print(f"  masterfile #1 rows: {len(mf1):,}   schema codes: {len(schema)}")

    encoder = None
    if not args.no_encoder:
        print("  loading sentence encoder for partition fallback ...")
        encoder = sr.load_encoder()

    mf2 = sr.resolve_frame(
        mf1, schema, name_col="name", occ_col=None,
        parent_col="naics_code", encoder=encoder,
    )
    mf2 = apply_manual_overrides(mf2)

    print()
    print("  resolution methods:")
    print(mf2["resolution_method"].value_counts().to_string())
    print()
    print("  top custom codes:")
    print(mf2["custom_code"].value_counts().head(25).to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mf2.to_csv(args.out, index=False)
    print(f"\n  Wrote: {args.out}  ({len(mf2):,} rows)")


if __name__ == "__main__":
    main()
