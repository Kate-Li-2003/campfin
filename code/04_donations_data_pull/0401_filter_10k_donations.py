"""
0401_filter_10k_donations.py

Filter the Lt. Governor and Insurance Commissioner 2026 donation files to
only donations over $10,000 (from any donor type — PAC, organization, or
individual). Output feeds 05_candidate_industry_affiliations classification.

Usage:
  python code/04_donations_data_pull/0401_filter_10k_donations.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUTS = [
    REPO_ROOT / "data/lt_governor_race_2026.csv",
    REPO_ROOT / "data/insurance_commissioner_race_2026.csv",
]
DEFAULT_OUT_DIR = REPO_ROOT / "data/04_output_latest_data_pulls/0401_races_10kfilters"
AMOUNT_MIN = 10000


def filter_file(input_path: Path, out_dir: Path, amount_min: float) -> dict:
    df = pd.read_csv(input_path)
    n_total = len(df)
    total_amount = float(df["Amount"].sum())

    kept = df[df["Amount"] > amount_min].copy()
    n_kept = len(kept)
    kept_amount = float(kept["Amount"].sum())

    out_path = out_dir / f"{input_path.stem}_over_{int(amount_min/1000)}k.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_path, index=False)

    print(f"\n{input_path.name}")
    print(f"  input rows:      {n_total:>7,}   ${total_amount:>15,.0f}")
    print(
        f"  >${int(amount_min):,} rows:    {n_kept:>7,}   ${kept_amount:>15,.0f}   "
        f"({n_kept / max(n_total, 1) * 100:.2f}% of rows, "
        f"{kept_amount / max(total_amount, 1) * 100:.2f}% of $)"
    )
    print(f"  wrote:           {out_path.relative_to(REPO_ROOT)}")
    return {"input": input_path.name, "kept": n_kept, "kept_amount": kept_amount}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--inputs", type=Path, nargs="*", default=DEFAULT_INPUTS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--amount-min", type=float, default=AMOUNT_MIN)
    args = p.parse_args(argv)

    for inp in args.inputs:
        filter_file(inp, args.out_dir, args.amount_min)


if __name__ == "__main__":
    main()
