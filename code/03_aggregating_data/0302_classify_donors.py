"""
0302_classify_donors.py

Multi-stage donor -> industry classification pipeline.

Step 1  Filter a transactional donor file to qualifying high-value,
        company-affiliated donations and aggregate to (employer, n_donors).
        Auto-skipped when the input is already in aggregated form.
Step 2  Match aggregated employers against the running_list reference
        (built by 0301_build_running_list.py).
Step 3  For employers still unmatched, fall back to live EDD NAICS lookups
        (code/edd_naics_lookup.py).

Each step prints its own counts / dollar value / percentage figures.

Usage
-----
    # demo: pre-aggregated input, step 1 auto-skipped
    python 0302_classify_donors.py

    # transactional input — step 1 runs, then 2 + 3
    python 0302_classify_donors.py \\
        --input ../../data/01CalAccess_CampaignFinance_Data/governor_race_2026-04-27.csv

Options:
    --input PATH        Donor file (csv or xlsx). Default: the demo file.
    --out PATH          Where to write the enriched output csv.
    --running-list PATH Reference file from step 0301.
    --amount-min N      Step-1 amount threshold (default: 10000).
    --no-edd            Skip step 3 (live EDD lookups).
    --delay SEC         Throttle between EDD requests (default: 1.0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

from edd_naics_lookup import EDDClient  # noqa: E402

# Default to the latest transactional gov-race pull. Update the filename
# below when a newer CalAccess pull lands in data/04_output_latest_data_pulls/.
DEFAULT_INPUT = REPO_ROOT / "data/04_output_latest_data_pulls/governor_race_2026-04-27.csv"
DEFAULT_RUNNING_LIST = REPO_ROOT / "data/03_input/masterfile/running_list.csv"
DEFAULT_OUT = REPO_ROOT / "output/03_output/masterfile_classified_donors.csv"

# Occupations that disqualify a donation from being "company-affiliated"
# under step-1 criteria. Compared case-insensitively after .strip().
NON_COMPANY_OCCUPATIONS = {
    "retired",
    "self-employed",
    "self employed",
    "not employed",
    "unemployed",
    "homemaker",
}

# Schema markers used to auto-detect the input.
TRANSACTIONAL_MARKERS = {"Amount", "Contributor Employer", "Contributor Occupation", "Contributor ID"}
AGGREGATED_MARKERS = {"employer", "n_donors"}


# ---------- helpers ----------

def normalize_name(s: pd.Series) -> pd.Series:
    """Match opensecrets `name_norm` style: uppercase, drop non-alphanumerics
    (keeping spaces), collapse whitespace. The same normalization used to
    build running_list.csv, so post-normalization names join on equality.
    """
    return (
        s.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9 ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def load_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def detect_format(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if TRANSACTIONAL_MARKERS.issubset(cols):
        return "transactional"
    if AGGREGATED_MARKERS.issubset(cols):
        return "aggregated"
    raise ValueError(
        f"Cannot detect input format. Need either {sorted(TRANSACTIONAL_MARKERS)} "
        f"or {sorted(AGGREGATED_MARKERS)}; got {sorted(cols)}"
    )


# ---------- step 1 ----------

def step1_filter_and_aggregate(df: pd.DataFrame, amount_min: float) -> pd.DataFrame:
    """Apply step-1 filters and aggregate to one row per employer.

    Qualifying donation:
      - Amount > amount_min
      - Contributor ID is null  (no PAC affiliation)
      - Occupation not in NON_COMPANY_OCCUPATIONS
      - Contributor Employer is non-blank (otherwise unclassifiable)
    """
    n_total = len(df)
    total_amount = float(df["Amount"].sum())

    occ = df["Contributor Occupation"].fillna("").str.lower().str.strip()
    employer = df["Contributor Employer"].astype(str).str.strip()
    qual = (
        (df["Amount"] > amount_min)
        & df["Contributor ID"].isna()
        & ~occ.isin(NON_COMPANY_OCCUPATIONS)
        & df["Contributor Employer"].notna()
        & (employer != "")
    )
    filtered = df.loc[qual]
    n_qual = len(filtered)
    qual_amount = float(filtered["Amount"].sum())

    print()
    print("=" * 70)
    print("STEP 1 — filter transactional donations")
    print("=" * 70)
    print(f"  Total donations in input:         {n_total:>10,}   ${total_amount:>15,.0f}")
    print(
        f"  Qualifying (>${int(amount_min):,}, no PAC ID, "
        f"company-affiliated occupation, employer present):"
    )
    print(
        f"    rows:  {n_qual:>10,}   "
        f"({n_qual / max(n_total, 1) * 100:>5.2f}% of input rows)"
    )
    print(
        f"    $ val: ${qual_amount:>14,.0f}   "
        f"({qual_amount / max(total_amount, 1) * 100:>5.2f}% of input $)"
    )

    aggregated = (
        filtered.groupby("Contributor Employer", as_index=False)
        .size()
        .rename(columns={"Contributor Employer": "employer", "size": "n_donors"})
        .sort_values("n_donors", ascending=False)
        .reset_index(drop=True)
    )
    print(f"  Unique employers (after aggregation): {len(aggregated):,}")
    return aggregated


# ---------- step 2 ----------

def step2_match_running_list(employers_df: pd.DataFrame, running_list_path: Path) -> pd.DataFrame:
    rl = pd.read_csv(running_list_path)
    keep_cols = [
        "name_norm",
        "name",
        "level1_category",
        "level2_category",
        "level3_category",
        "naics_code",
        "naics_label",
        "source",
    ]
    rl = rl[[c for c in keep_cols if c in rl.columns]].rename(
        columns={"name": "match_name", "source": "match_running_list_source"}
    )
    # If the same name_norm appears in both opensecrets and h1b, build_running_list
    # already deduped — but assert defensively before we left-join.
    rl = rl.drop_duplicates("name_norm", keep="first")

    out = employers_df.copy()
    out["employer_norm"] = normalize_name(out["employer"])
    merged = out.merge(rl, left_on="employer_norm", right_on="name_norm", how="left").drop(
        columns="name_norm"
    )

    matched = merged["match_name"].notna()
    n_emp = len(merged)
    n_match = int(matched.sum())
    n_donations = int(merged["n_donors"].sum())
    n_matched_donations = int(merged.loc[matched, "n_donors"].sum())

    print()
    print("=" * 70)
    print("STEP 2 — match against running_list.csv")
    print("=" * 70)
    print(f"  Reference rows in running_list:   {len(rl):>10,}")
    print(
        f"  Employers matched:                {n_match:>10,}   "
        f"({n_match / max(n_emp, 1) * 100:>5.2f}% of {n_emp:,} employers)"
    )
    print(
        f"  Donations matched:                {n_matched_donations:>10,}   "
        f"({n_matched_donations / max(n_donations, 1) * 100:>5.2f}% of {n_donations:,} donations)"
    )

    merged["match_source"] = pd.NA
    merged.loc[matched, "match_source"] = "running_list"
    return merged


# ---------- step 3 ----------

def step3_edd_lookup(merged: pd.DataFrame, delay: float) -> pd.DataFrame:
    print()
    print("=" * 70)
    print("STEP 3 — EDD lookup for unmatched employers")
    print("=" * 70)

    unmatched_mask = merged["match_name"].isna()
    unmatched = merged.loc[unmatched_mask]
    if unmatched.empty:
        print("  Nothing to look up.")
        return merged

    # Dedupe by normalized name so equivalent raw spellings hit EDD once.
    unique_norms = unmatched.drop_duplicates("employer_norm")
    n_total_unmatched = len(unmatched)
    n_unique = len(unique_norms)
    print(f"  Unmatched employer rows:          {n_total_unmatched:>10,}")
    print(f"  Distinct normalized names to query: {n_unique:>10,}")
    print(f"  (using EDD client; ~{delay:.1f}s/request, on-disk cache active)")
    print()

    client = EDDClient(delay=delay)
    edd_by_norm: dict[str, dict] = {}
    for i, row in enumerate(unique_norms.itertuples(), 1):
        r = client.lookup(row.employer)
        edd_by_norm[row.employer_norm] = {
            "edd_match_name": r.match_name,
            "edd_match_type": r.match_type,
            "edd_naics_code": r.naics_code,
            "edd_naics_description": r.naics_description,
            "edd_status": r.status,
        }
        print(
            f"  [{i:>4}/{n_unique}] {str(row.employer)[:50]:50s} -> "
            f"{r.status:11s} {r.naics_code or '':6s} {r.match_name or ''}",
            file=sys.stderr,
        )

    for col in ("edd_match_name", "edd_match_type", "edd_naics_code", "edd_naics_description", "edd_status"):
        merged[col] = merged["employer_norm"].map(lambda k: edd_by_norm.get(k, {}).get(col))

    edd_hit = merged["edd_status"] == "ok"
    # Promote EDD hits into the canonical match columns so downstream consumers
    # don't have to reason about which source filled them in.
    merged.loc[edd_hit, "match_name"] = merged.loc[edd_hit, "edd_match_name"]
    merged.loc[edd_hit, "naics_code"] = merged.loc[edd_hit, "edd_naics_code"]
    merged.loc[edd_hit, "naics_label"] = merged.loc[edd_hit, "edd_naics_description"]
    merged.loc[edd_hit, "match_source"] = "edd"

    n_emp_total = len(merged)
    n_donations = int(merged["n_donors"].sum())
    n_edd_emp = int(edd_hit.sum())
    n_edd_donations = int(merged.loc[edd_hit, "n_donors"].sum())
    print()
    print(
        f"  EDD employer matches:             {n_edd_emp:>10,}   "
        f"({n_edd_emp / max(n_total_unmatched, 1) * 100:>5.2f}% of {n_total_unmatched:,} unmatched, "
        f"{n_edd_emp / max(n_emp_total, 1) * 100:.2f}% of all employers)"
    )
    print(
        f"  EDD donation matches:             {n_edd_donations:>10,}   "
        f"({n_edd_donations / max(n_donations, 1) * 100:>5.2f}% of all donations)"
    )
    return merged


# ---------- summary ----------

def print_final_summary(merged: pd.DataFrame) -> None:
    n_emp = len(merged)
    n_don = int(merged["n_donors"].sum())
    src = merged["match_source"].fillna("unmatched")
    by_emp = src.value_counts()
    by_don = merged.groupby(src)["n_donors"].sum()

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Employers: {n_emp:,}    Donations: {n_don:,}")
    for source in ("running_list", "edd", "unmatched"):
        e = int(by_emp.get(source, 0))
        d = int(by_don.get(source, 0))
        print(
            f"  {source:14s} "
            f"employers: {e:>5,} ({e / max(n_emp, 1) * 100:>5.2f}%)   "
            f"donations: {d:>5,} ({d / max(n_don, 1) * 100:>5.2f}%)"
        )


# ---------- main ----------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--running-list", type=Path, default=DEFAULT_RUNNING_LIST)
    p.add_argument("--amount-min", type=float, default=10000)
    p.add_argument("--no-edd", action="store_true")
    p.add_argument("--delay", type=float, default=1.0)
    args = p.parse_args(argv)

    df = load_input(args.input)
    fmt = detect_format(df)
    print(f"Input: {args.input}")
    print(f"  rows={len(df):,}   format={fmt}")

    if fmt == "transactional":
        employers = step1_filter_and_aggregate(df, amount_min=args.amount_min)
    else:
        employers = df[["employer", "n_donors"]].copy()
        print()
        print("=" * 70)
        print("STEP 1 — skipped (input is already aggregated)")
        print("=" * 70)
        print(f"  Input rows treated as employers: {len(employers):,}")
        print(f"  Total donations represented:     {int(employers['n_donors'].sum()):,}")

    merged = step2_match_running_list(employers, args.running_list)
    if not args.no_edd:
        merged = step3_edd_lookup(merged, delay=args.delay)
    print_final_summary(merged)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"\nWrote: {args.out}  ({len(merged):,} rows)")


if __name__ == "__main__":
    main()
