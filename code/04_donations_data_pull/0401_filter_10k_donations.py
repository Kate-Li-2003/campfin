"""
0401_filter_10k_donations.py

Filter one or more donation files (Power Search format — e.g. the output of
pull_calaccess.py, or a manual Power Search export) down to contributions at or
above a dollar threshold, from any donor type (PAC, organization, individual).
Every contribution that clears the threshold is kept regardless of type — no
ballot-measure / candidate / transaction-type restriction is applied. Output
feeds 05_candidate_industry_affiliations classification.

The threshold is set with --amount-min (default 10,000) and the comparison is
inclusive, so "$5k+" includes contributions of exactly $5,000.

Usage:
  # Defaults: Lt. Governor + Insurance Commissioner, $10,000+
  python code/04_donations_data_pull/0401_filter_10k_donations.py

  # Governor's race, $5,000+
  python code/04_donations_data_pull/0401_filter_10k_donations.py \
      --inputs data/01CalAccess_CampaignFinance_Data/governor_race_2026.csv \
      --amount-min 5000
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
AMOUNT_MIN = 10000


def _threshold_label(amount_min: float) -> str:
    """'10k' for 10000, '5k' for 5000, otherwise the plain integer amount."""
    return f"{int(amount_min / 1000)}k" if amount_min >= 1000 else str(int(amount_min))


def default_out_dir(amount_min: float) -> Path:
    """Threshold-aware output dir (e.g. .../0401_races_5kfilters). The 10k
    default resolves to the original .../0401_races_10kfilters path."""
    return REPO_ROOT / "data/04_output_latest_data_pulls" / f"0401_races_{_threshold_label(amount_min)}filters"


def filter_file(input_path: Path, out_dir: Path, amount_min: float) -> dict:
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    n_total = len(df)

    # Coerce Amount → numeric. Blank / non-numeric values (e.g. the search-
    # criteria footer rows Power Search appends) become NaN and are excluded by
    # the >= comparison, so footers are dropped for free.
    amount = pd.to_numeric(df["Amount"], errors="coerce")
    total_amount = float(amount.sum())

    keep = amount >= amount_min  # inclusive: "$Nk+" includes exactly $N,000
    kept = df[keep].copy()
    n_kept = len(kept)
    kept_amount = float(amount[keep].sum())

    out_path = out_dir / f"{input_path.stem}_over_{_threshold_label(amount_min)}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_path, index=False)

    print(f"\n{input_path.name}")
    print(f"  input rows:        {n_total:>8,}   ${total_amount:>15,.0f}")
    print(
        f"  >= ${int(amount_min):,} rows:   {n_kept:>8,}   ${kept_amount:>15,.0f}   "
        f"({n_kept / max(n_total, 1) * 100:.2f}% of rows, "
        f"{kept_amount / max(total_amount, 1) * 100:.2f}% of $)"
    )
    try:
        shown = out_path.relative_to(REPO_ROOT)
    except ValueError:
        shown = out_path  # --out-dir outside the repo
    print(f"  wrote:             {shown}")
    return {"input": input_path.name, "kept": n_kept, "kept_amount": kept_amount}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--inputs", type=Path, nargs="*", default=DEFAULT_INPUTS)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output dir. Defaults to a threshold-aware folder under "
                        "data/04_output_latest_data_pulls/.")
    p.add_argument("--amount-min", type=float, default=AMOUNT_MIN)
    args = p.parse_args(argv)

    out_dir = args.out_dir or default_out_dir(args.amount_min)
    for inp in args.inputs:
        if not inp.exists():
            raise SystemExit(
                f"Input not found: {inp}\n"
                f"Run pull_calaccess.py first to generate it, or pass --inputs."
            )
        filter_file(inp, out_dir, args.amount_min)


if __name__ == "__main__":
    main()
