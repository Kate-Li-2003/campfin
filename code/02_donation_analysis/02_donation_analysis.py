"""
02_donation_analysis.py
=======================
Visualizes and analyzes CA Gubernatorial Race 2026 campaign contributions.

Steps
-----
1.  Load the governor_race CSV and classify each row as:
      - Individual   : has a Contributor Occupation entry
      - Organization : no Contributor ID AND no Contributor Occupation
      - PAC/Committee: has a Contributor ID

2.  Histograms of donation magnitudes for each donor type (log-scaled x-axis).

3.  Bucket individual donors into four tiers:
      High Spender  : donation ≥ CA individual contribution limit (~$36,400)
      Medium Spender: $10,000 ≤ donation < limit
      Lower Spender :    $200 ≤ donation < $10,000
      Everyday      :           donation < $200

4.  Match individual donor names in each bucket against the
    ContributionOpenSecretsCategories CSV. Report match rates and
    the industry breakdown of matched donors.

Usage
-----
  python 02_donation_analysis.py

Outputs (written to ../data/02_output/)
  figures/01_donation_distributions.png
  figures/02_individual_buckets.png
  figures/03_opensecrets_match_rates.png
  figures/04_matched_industry_by_bucket.png
  opensecrets_match_summary.csv
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script mode
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── CONFIGURATION ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent   # campaign finance categorization/
DATA_DIR = BASE_DIR / "data"

GOV_RACE_PATH = (
    DATA_DIR / "01CalAccess_CampaignFinance_Data" / "governor_race_2026.csv"
)
OPENSECRETS_PATH = DATA_DIR / "ContributionOpenSecretsCategories (1).csv"

OUTPUT_DIR = DATA_DIR / "02_output"
FIG_DIR    = OUTPUT_DIR / "figures"

# CA FPPC individual contribution limit for 2025-2026 statewide races (Governor)
# Source: FPPC Regulation 18545.7 — adjust if updated before the election
CA_INDIVIDUAL_LIMIT = 36_400

# Bucket thresholds (in USD)
BUCKET_LIMITS = {
    "Everyday (<$200)":         (0,       200),
    "Lower ($200–$10K)":        (200,   10_000),
    "Medium ($10K–limit)":      (10_000, CA_INDIVIDUAL_LIMIT),
    "High (≥ limit)":           (CA_INDIVIDUAL_LIMIT, float("inf")),
}

BUCKET_ORDER = list(BUCKET_LIMITS.keys())  # low → high for charts

# Palette
COLORS = {
    "Individual":    "#4C72B0",
    "Organization":  "#55A868",
    "PAC/Committee": "#C44E52",
}
BUCKET_COLORS = ["#74B9E1", "#55A868", "#E8A838", "#C44E52"]   # everyday → high

# ── HELPERS ──────────────────────────────────────────────────────────────────

def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.family": "sans-serif",
    })


def _clean_amount(series: pd.Series) -> pd.Series:
    """Strip $ / commas and coerce to float."""
    s = series.astype(str).str.replace(r"[$,]", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in df (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalise_name(name: str) -> str:
    """
    Lightweight normalisation for name matching:
    upper-case, strip punctuation, collapse whitespace.
    """
    if not isinstance(name, str):
        return ""
    n = name.upper()
    n = re.sub(r"[^A-Z0-9 ]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


# ── 1.  LOAD & CLASSIFY ──────────────────────────────────────────────────────

def load_and_classify(path: Path) -> pd.DataFrame:
    """
    Load the governor race CSV and add a `donor_type` column
    ('Individual', 'Organization', 'PAC/Committee').

    Classification rules (per Kate's domain knowledge):
      - Individual   : Contributor Occupation is non-empty
      - PAC/Committee: has a Contributor ID (even if Occupation is blank)
      - Organization : neither of the above
    """
    print(f"Loading {path.name} …")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    print(f"  {len(df):,} rows × {df.shape[1]} columns loaded.")
    print(f"  Columns: {list(df.columns)}\n")

    # ── locate key columns ────────────────────────────────────────────────
    col_name = _detect_column(df, ["Contributor Name", "CTRIB_NAML", "contributor_name"])
    col_occ  = _detect_column(df, ["Contributor Occupation", "CTRIB_OCC", "contributor_occupation"])
    col_id   = _detect_column(df, [
        "Contributor ID", "CMTE_ID", "FILER_ID", "contributor_id", "Committee ID",
    ])
    col_amt  = _detect_column(df, ["Amount", "AMOUNT", "amount", "TRAN_AMT1"])

    if col_amt is None:
        raise ValueError("Cannot find an Amount column in the CSV.")
    if col_name is None:
        raise ValueError("Cannot find a Contributor Name column in the CSV.")

    # ── clean amount ──────────────────────────────────────────────────────
    df["amount_clean"] = _clean_amount(df[col_amt])

    # ── build boolean masks ───────────────────────────────────────────────
    # Occupation present  → individual
    has_occ = (
        df[col_occ].notna() & df[col_occ].str.strip().ne("") & df[col_occ].str.lower().ne("nan")
        if col_occ else pd.Series(False, index=df.index)
    )

    # Contributor ID present → PAC / committee
    has_id  = (
        df[col_id].notna() & df[col_id].str.strip().ne("") & df[col_id].str.lower().ne("nan")
        if col_id else pd.Series(False, index=df.index)
    )

    # Assign donor_type (PAC check before individual check for safety)
    conditions = [has_id, has_occ]
    choices    = ["PAC/Committee", "Individual"]
    df["donor_type"] = np.select(conditions, choices, default="Organization")

    # ── store canonical column references ─────────────────────────────────
    df["_name"] = df[col_name].fillna("").str.strip()
    df["_occ"]  = df[col_occ].fillna("").str.strip() if col_occ else ""

    # Summary
    counts = df["donor_type"].value_counts()
    print("Donor type breakdown:")
    for dt, n in counts.items():
        pct = n / len(df) * 100
        print(f"  {dt:<18} {n:>7,}  ({pct:5.1f}%)")
    print()

    return df


# ── 2.  DISTRIBUTION HISTOGRAMS ──────────────────────────────────────────────

def plot_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    """
    One row of histograms, one panel per donor type.
    x-axis is log-scaled so small and large donations are both visible.
    """
    donor_types = ["Individual", "Organization", "PAC/Committee"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(
        "Distribution of Contribution Amounts by Donor Type\n(CA Gubernatorial Race 2026)",
        fontsize=14, y=1.01,
    )

    for ax, dt in zip(axes, donor_types):
        sub = df.loc[df["donor_type"] == dt, "amount_clean"].dropna()
        sub = sub[sub > 0]          # log-scale needs positive values

        if sub.empty:
            ax.set_title(f"{dt}\n(no data)")
            ax.set_visible(False)
            continue

        # Log-spaced bins from $1 to just above the max
        log_min = max(0, np.floor(np.log10(sub.min())))
        log_max = np.ceil(np.log10(sub.max()))
        bins = np.logspace(log_min, log_max, 40)

        ax.hist(sub, bins=bins, color=COLORS[dt], edgecolor="white",
                linewidth=0.5, alpha=0.85)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x:,.0f}" if x >= 1 else f"${x:.2f}"
        ))

        # Annotation: median & count
        med = sub.median()
        ax.axvline(med, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(
            med * 1.15, ax.get_ylim()[1] * 0.9,
            f"Median\n${med:,.0f}",
            fontsize=8, color="black", va="top",
        )

        ax.set_title(f"{dt}\n(n = {len(sub):,})", fontsize=12)
        ax.set_xlabel("Donation Amount (USD, log scale)")
        ax.set_ylabel("# of Contributions")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    path = out_dir / "01_donation_distributions.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")


# ── 3.  INDIVIDUAL DONOR BUCKETS ─────────────────────────────────────────────

def assign_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `bucket` column to individual-donor rows."""
    indiv = df[df["donor_type"] == "Individual"].copy()

    def _bucket(amt: float) -> str:
        for label, (lo, hi) in BUCKET_LIMITS.items():
            if lo <= amt < hi:
                return label
        return BUCKET_ORDER[-1]   # catch edge case at exact ceiling

    indiv["bucket"] = indiv["amount_clean"].apply(
        lambda x: _bucket(x) if pd.notna(x) else "Everyday (<$200)"
    )
    return indiv


def plot_buckets(indiv: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: count and total dollars contributed per bucket."""
    bucket_stats = (
        indiv.groupby("bucket", observed=True)["amount_clean"]
        .agg(count="count", total="sum")
        .reindex(BUCKET_ORDER)
        .fillna(0)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Individual Donors by Spending Tier\n(CA Gubernatorial Race 2026)",
        fontsize=14,
    )

    x = np.arange(len(BUCKET_ORDER))

    # Panel A: # of donors
    bars1 = ax1.bar(x, bucket_stats["count"], color=BUCKET_COLORS, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [textwrap.fill(b, 14) for b in BUCKET_ORDER], fontsize=9
    )
    ax1.set_ylabel("Number of Contributions")
    ax1.set_title("Count of Individual Contributions")
    for bar, val in zip(bars1, bucket_stats["count"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                 f"{val:,.0f}", ha="center", va="bottom", fontsize=8)

    # Panel B: total dollars
    bars2 = ax2.bar(x, bucket_stats["total"] / 1e6, color=BUCKET_COLORS, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [textwrap.fill(b, 14) for b in BUCKET_ORDER], fontsize=9
    )
    ax2.set_ylabel("Total Contributions ($ millions)")
    ax2.set_title("Total Dollars per Tier")
    for bar, val in zip(bars2, bucket_stats["total"] / 1e6):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"${val:.2f}M", ha="center", va="bottom", fontsize=8)

    # Reference line for CA individual limit
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Summary note below the figure
    fig.text(
        0.5, -0.04,
        f"High Spender threshold: CA FPPC individual limit ≈ ${CA_INDIVIDUAL_LIMIT:,} "
        f"(2025-26 statewide race). "
        f"Adjust CA_INDIVIDUAL_LIMIT in the script if the limit changes.",
        ha="center", fontsize=8, style="italic", color="gray",
    )

    fig.tight_layout()
    path = out_dir / "02_individual_buckets.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")

    # Print summary table
    print("\nIndividual Donor Bucket Summary:")
    print(f"  {'Bucket':<30}  {'Count':>8}  {'Total $':>14}  {'Avg Donation':>14}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*14}  {'-'*14}")
    for b in BUCKET_ORDER:
        n   = int(bucket_stats.loc[b, "count"])
        tot = bucket_stats.loc[b, "total"]
        avg = tot / n if n else 0
        print(f"  {b:<30}  {n:>8,}  ${tot:>13,.2f}  ${avg:>13,.2f}")
    print()
    return


# ── 4.  OPENSECRETS MATCHING ─────────────────────────────────────────────────

def load_opensecrets(path: Path) -> pd.DataFrame:
    """Load and normalise the OpenSecrets categories lookup."""
    print(f"Loading {path.name} …")
    os_df = pd.read_csv(path, dtype=str, low_memory=False)
    os_df.columns = [c.strip() for c in os_df.columns]
    os_df["name_norm"] = os_df["name"].apply(normalise_name)
    print(f"  {len(os_df):,} entries ({(os_df['level1_category'] != 'Uncoded').sum():,} "
          f"with a non-Uncoded industry).\n")
    return os_df


def match_to_opensecrets(indiv: pd.DataFrame, os_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt exact (normalised) name matching between individual donor names
    and OpenSecrets names.  Adds columns:
        os_matched        bool
        os_level1         str  (level1_category from OpenSecrets, or NaN)
        os_level2         str
        os_level3         str
    """
    # Build lookup dict: normalised name → first matching row
    os_lookup = (
        os_df.drop_duplicates("name_norm")
        .set_index("name_norm")[["level1_category", "level2_category", "level3_category"]]
        .to_dict(orient="index")
    )

    indiv = indiv.copy()
    indiv["name_norm"] = indiv["_name"].apply(normalise_name)

    # Vectorised look-up
    indiv["os_matched"] = indiv["name_norm"].isin(os_lookup)
    indiv["os_level1"]  = indiv["name_norm"].map(
        {k: v["level1_category"] for k, v in os_lookup.items()}
    )
    indiv["os_level2"]  = indiv["name_norm"].map(
        {k: v["level2_category"] for k, v in os_lookup.items()}
    )
    indiv["os_level3"]  = indiv["name_norm"].map(
        {k: v["level3_category"] for k, v in os_lookup.items()}
    )

    # Count of contributors (not donations) per bucket
    print("OpenSecrets match summary per bucket:")
    print(f"  {'Bucket':<30}  {'Total':>8}  {'Matched':>8}  {'Match %':>8}  "
          f"{'Has Industry':>13}  {'Industry %':>11}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*13}  {'-'*11}")
    for b in BUCKET_ORDER:
        sub = indiv[indiv["bucket"] == b]
        total   = len(sub)
        matched = sub["os_matched"].sum()
        has_ind = (
            sub.loc[sub["os_matched"], "os_level1"]
            .ne("Uncoded").sum()
        )
        pct_m = matched / total * 100 if total else 0
        pct_i = has_ind / total * 100 if total else 0
        print(f"  {b:<30}  {total:>8,}  {matched:>8,}  {pct_m:>7.1f}%  "
              f"{has_ind:>13,}  {pct_i:>10.1f}%")
    print()

    return indiv


def plot_match_rates(indiv: pd.DataFrame, out_dir: Path) -> None:
    """Stacked bar: matched vs. unmatched by bucket."""
    rows = []
    for b in BUCKET_ORDER:
        sub = indiv[indiv["bucket"] == b]
        total   = len(sub)
        matched_ind  = sub.loc[sub["os_matched"], "os_level1"].ne("Uncoded").sum()
        matched_unc  = (sub["os_matched"] & sub["os_level1"].eq("Uncoded")).sum()
        unmatched    = (~sub["os_matched"]).sum()
        rows.append({
            "bucket": b,
            "Has Industry": matched_ind,
            "In OpenSecrets (Uncoded)": matched_unc,
            "Not in OpenSecrets": unmatched,
            "total": total,
        })
    stats = pd.DataFrame(rows).set_index("bucket")

    fig, ax = plt.subplots(figsize=(11, 5.5))

    stack_cols = ["Has Industry", "In OpenSecrets (Uncoded)", "Not in OpenSecrets"]
    stack_colors = ["#4C72B0", "#74B9E1", "#CCCCCC"]
    bottom = np.zeros(len(BUCKET_ORDER))

    for col, color in zip(stack_cols, stack_colors):
        vals = stats.reindex(BUCKET_ORDER)[col].fillna(0).values
        bars = ax.bar(BUCKET_ORDER, vals, bottom=bottom, color=color,
                      edgecolor="white", linewidth=0.5, label=col)
        for bar, v in zip(bars, vals):
            if v > 0:
                yc = bar.get_y() + bar.get_height() / 2
                if bar.get_height() > stats["total"].max() * 0.03:
                    ax.text(bar.get_x() + bar.get_width() / 2, yc,
                            f"{int(v):,}", ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
        bottom += vals

    ax.set_xlabel("Individual Donor Tier")
    ax.set_ylabel("Number of Contributions")
    ax.set_title(
        "OpenSecrets Match Rate by Individual Donor Tier\n"
        "(CA Gubernatorial Race 2026)",
        fontsize=13,
    )
    ax.set_xticklabels(
        [textwrap.fill(b, 14) for b in BUCKET_ORDER], fontsize=9
    )
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = out_dir / "03_opensecrets_match_rates.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")


def plot_industry_by_bucket(indiv: pd.DataFrame, out_dir: Path) -> None:
    """
    For donors with an identified industry (level1 ≠ 'Uncoded'),
    show a stacked horizontal bar of sector breakdown per tier.
    Only donors matched AND with a non-Uncoded level1 are included.
    """
    matched = indiv[
        indiv["os_matched"] & indiv["os_level1"].ne("Uncoded")
    ].copy()

    if matched.empty:
        print("  No matched donors with industry classifications — skipping plot 04.")
        return

    # Pivot: bucket × level1_category → counts
    pivot = (
        matched.groupby(["bucket", "os_level1"])
        .size()
        .unstack(fill_value=0)
        .reindex(BUCKET_ORDER)
        .fillna(0)
    )

    # Keep only top N sectors by total count for readability
    TOP_N = 10
    top_sectors = pivot.sum().nlargest(TOP_N).index.tolist()
    other = pivot.drop(columns=top_sectors, errors="ignore").sum(axis=1)
    pivot = pivot[top_sectors].copy()
    if (other > 0).any():
        pivot["Other"] = other
        top_sectors.append("Other")

    # Convert to row percentages (within-bucket share)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).fillna(0) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.get_cmap("tab20", len(top_sectors))
    colors = [cmap(i) for i in range(len(top_sectors))]

    bottom = np.zeros(len(BUCKET_ORDER))
    for col, color in zip(top_sectors, colors):
        vals = pivot_pct[col].values if col in pivot_pct.columns else np.zeros(len(BUCKET_ORDER))
        ax.bar(BUCKET_ORDER, vals, bottom=bottom, color=color,
               edgecolor="white", linewidth=0.4, label=col)
        bottom += vals

    ax.set_xlabel("Individual Donor Tier")
    ax.set_ylabel("Share of Industry-Matched Donors (%)")
    ax.set_title(
        "Industry Sector Breakdown of OpenSecrets-Matched Donors\n"
        "by Individual Donor Tier (CA Gubernatorial Race 2026)",
        fontsize=13,
    )
    ax.set_xticklabels(
        [textwrap.fill(b, 14) for b in BUCKET_ORDER], fontsize=9
    )
    ax.set_ylim(0, 105)
    ax.legend(
        loc="upper right", bbox_to_anchor=(1.22, 1), frameon=False, fontsize=8
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate with absolute count
    count_by_bucket = matched.groupby("bucket").size().reindex(BUCKET_ORDER).fillna(0)
    for i, b in enumerate(BUCKET_ORDER):
        n = int(count_by_bucket[b])
        ax.text(i, 102, f"n={n:,}", ha="center", va="bottom", fontsize=8, color="gray")

    fig.tight_layout()
    path = out_dir / "04_matched_industry_by_bucket.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")


def save_match_csv(indiv: pd.DataFrame, out_dir: Path) -> None:
    """Save a CSV summary: one row per bucket with match statistics."""
    rows = []
    for b in BUCKET_ORDER:
        sub = indiv[indiv["bucket"] == b]
        total    = len(sub)
        total_usd = sub["amount_clean"].sum()
        matched  = sub["os_matched"].sum()
        with_ind = sub.loc[sub["os_matched"], "os_level1"].ne("Uncoded").sum()
        rows.append({
            "bucket":                  b,
            "n_contributions":         total,
            "total_amount_usd":        round(total_usd, 2),
            "n_matched_opensecrets":   int(matched),
            "pct_matched":             round(matched / total * 100, 2) if total else 0,
            "n_with_industry":         int(with_ind),
            "pct_with_industry":       round(with_ind / total * 100, 2) if total else 0,
        })
    out = pd.DataFrame(rows)
    path = out_dir / "opensecrets_match_summary.csv"
    out.to_csv(path, index=False)
    print(f"  Saved → {path.name}")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_style()

    # Make output dirs
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & classify ────────────────────────────────────────────────
    if not GOV_RACE_PATH.exists():
        sys.exit(
            f"\nERROR: Governor race file not found:\n  {GOV_RACE_PATH}\n"
            "Update GOV_RACE_PATH at the top of this script."
        )
    df = load_and_classify(GOV_RACE_PATH)

    # ── 2. Distribution histograms ────────────────────────────────────────
    print("Generating distribution histograms …")
    plot_distributions(df, FIG_DIR)

    # ── 3. Individual donor buckets ───────────────────────────────────────
    print("Bucketing individual donors …")
    indiv = assign_buckets(df)
    plot_buckets(indiv, FIG_DIR)

    # ── 4. OpenSecrets matching ───────────────────────────────────────────
    if not OPENSECRETS_PATH.exists():
        print(
            f"\nWARNING: OpenSecrets file not found:\n  {OPENSECRETS_PATH}\n"
            "Skipping name-matching section."
        )
        return

    os_df = load_opensecrets(OPENSECRETS_PATH)
    indiv = match_to_opensecrets(indiv, os_df)

    print("Generating OpenSecrets match visualizations …")
    plot_match_rates(indiv, FIG_DIR)
    plot_industry_by_bucket(indiv, FIG_DIR)
    save_match_csv(indiv, OUTPUT_DIR)

    print("\nDone. All outputs written to:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
