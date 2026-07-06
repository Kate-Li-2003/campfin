"""
0503_candidate_industry_visualizations.py

For each of the top-10 candidates in the 2026 CA Governor's race
(by total contributions received), produce two views of how their
>$10K donations break down by industry:

  1. A 2x5 grid of pie charts — one per candidate (canonical view).
  2. A stacked horizontal bar chart — same data, but oriented for
     side-by-side comparison across candidates.

Inputs:
  - data/04_output_latest_data_pulls/governor_race_2026.csv
        Per-donation transactional file. We use Recipient Name + Recipient
        Committee to attribute each row to a candidate, then filter to
        Amount > $10K.
  - output/05_output/donors_classified_with_manual.csv
        Per-employer industry classification produced by 0502.

Output (in output/05_output/figures/):
  - candidate_industry_pies.png
  - candidate_industry_stacked_bar.png
  - candidate_industry_summary.csv  (the numbers behind the charts)

Caveats this script handles:
  * Some rows have Recipient Name = "NAN, NAN" but a populated committee.
    For those we parse the candidate surname out of "<SURNAME> FOR
    GOVERNOR ..." in the committee field.
  * Opposition / issue-PAC rows ("NO ON STEYER", etc.) are filtered out
    so their dollars don't get inflated into the targeted candidate.
  * The pie shows only *classified* >$10K dollars — the share that's
    Unclassified appears in the subtitle so coverage stays transparent
    without dominating Steyer's all-self-funded pie.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DONATIONS = REPO_ROOT / "data/04_output_latest_data_pulls/governor_race_2026.csv"
DEFAULT_CLASSIFIED = REPO_ROOT / "output/05_output/donors_classified_with_manual.csv"
DEFAULT_FIGURES_DIR = REPO_ROOT / "output/05_output/figures"

AMOUNT_MIN = 10000
TOP_N = 10
SMALL_SLICE_PCT = 3.0  # below this, fold into "Other"

OPPOSITION_RE = re.compile(r"\b(NO ON|OPPOSE|OPPOSING|AGAINST|RESILIENT.*AFFORDABLE)\b")

# Employer values that look like real names but are actually placeholders
# typed by donors. The classification pipeline treats these as Unclassified
# at viz time so a junk match upstream (e.g. "Candidate" -> 922160) doesn't
# pollute the pies. The post-normalization (uppercase, alphanumeric+space)
# form is what gets compared.
PLACEHOLDER_EMPLOYERS = {
    "CANDIDATE",
    "SELF",
    "SELF EMPLOYED",
    "SELF EMPLOYEED",
    "NONE",
    "N A",
    "NA",
    "REQUESTED",
    "REQUESTING",
    "INFORMATION REQUESTED",
    "RETIRED",
    "NOT EMPLOYED",
    "UNEMPLOYED",
    "HOMEMAKER",
}

# Map NAICS 2-digit sectors -> a coarser, journalist-friendly bucket label.
NAICS2_BUCKET = {
    "11": "Agriculture",
    "21": "Mining / Oil & Gas",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "32": "Manufacturing",
    "33": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail",
    "45": "Retail",
    "48": "Transportation",
    "49": "Transportation",
    "51": "Information / Media / Tech",
    "52": "Finance & Insurance",
    "53": "Real Estate",
    "54": "Professional Services",
    "55": "Mgmt of Companies",
    "56": "Admin / Support Services",
    "61": "Education",
    "62": "Health Care",
    "71": "Arts & Entertainment",
    "72": "Hospitality / Food",
    "81": "Other Services / Civic",
    "92": "Government",
}

# Map OpenSecrets-style level1 categories -> the same bucket scheme so
# the keyword/OS rows align with the NAICS-derived ones.
L1_TO_BUCKET = {
    "Health": "Health Care",
    "Communications & Electronics": "Information / Media / Tech",
    "Finance, Insurance & Real Estate": "Finance & Insurance",
    "Government Agencies/Education/Other": "Education",
    "Lawyers & Lobbyists": "Professional Services",
    "General Business": "General Business",
    "Ideology/Single Issue": "Other Services / Civic",
    "Transportation": "Transportation",
    "Agriculture": "Agriculture",
    "Energy & Natural Resources": "Mining / Oil & Gas",
    "Construction": "Construction",
    "Defense": "Manufacturing",
    "Labor": "Other Services / Civic",
    "Party": "Other Services / Civic",
}


# ---------- helpers ----------

def normalize_name(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9 ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def parse_candidate(row: pd.Series) -> str | None:
    """Surname-key for the candidate this donation is to. Returns None
    for opposition / non-candidate committees (so they get filtered out)."""
    committee = str(row.get("Recipient Committee") or "").upper()
    if OPPOSITION_RE.search(committee):
        return None
    name = row.get("Recipient Name")
    if pd.notna(name) and name != "NAN, NAN":
        return str(name).split(",")[0].strip().upper()
    m = re.match(r"^([A-Za-z][A-Za-z\.\-]+)\s+FOR\s+GOVERNOR", committee)
    return m.group(1).upper() if m else None


def industry_bucket(level1: object, naics_code: object) -> str:
    """Single industry label per employer. Prefers OS-style level1 (mapped
    to the same coarse buckets), falls back to NAICS-2 sector."""
    if pd.notna(level1):
        v = str(level1).strip()
        if v:
            return L1_TO_BUCKET.get(v, v)
    if pd.notna(naics_code):
        code = str(naics_code).strip()
        if code.endswith(".0"):
            code = code[:-2]
        # Strip non-digits, take first two
        digits = "".join(c for c in code if c.isdigit())
        if len(digits) >= 2:
            return NAICS2_BUCKET.get(digits[:2], "Other")
    return "Unclassified"


# ---------- core build ----------

def _dedupe_nan_recipients(don: pd.DataFrame) -> pd.DataFrame:
    """Some donations are booked twice in the source file: once with a
    proper Recipient Name (e.g. "CLOOBECK, STEPHEN J.") and once with
    Recipient Name = "NAN, NAN" but the same committee/date/amount/
    contributor. Drop the NAN copies that match a proper row.
    """
    proper = don[don["Recipient Name"] != "NAN, NAN"]
    nan = don[don["Recipient Name"] == "NAN, NAN"]
    if nan.empty:
        return don

    # Contributor names in NAN rows often have ", nan" suffix; strip it
    # for matching. Stable on a copy so we don't mutate `don`.
    proper_keys = set(
        zip(
            proper["Start Date"],
            proper["Amount"],
            proper["Contributor Name"].fillna("").astype(str),
        )
    )
    nan = nan.copy()
    nan["_contrib_clean"] = (
        nan["Contributor Name"].fillna("").astype(str).str.replace(r",\s*nan$", "", regex=True)
    )
    is_dup = [
        (sd, amt, contrib) in proper_keys
        for sd, amt, contrib in zip(nan["Start Date"], nan["Amount"], nan["_contrib_clean"])
    ]
    n_dup = sum(is_dup)
    if n_dup:
        nan = nan.loc[[not d for d in is_dup]]
        print(f"  deduped {n_dup:,} NAN/NAN rows that mirrored proper-name rows")
    nan = nan.drop(columns=["_contrib_clean"])
    return pd.concat([proper, nan], ignore_index=True)


def build_industry_table(
    donations_path: Path, classified_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per_donation_df, per_candidate_industry_summary_df).

    per_donation_df: each big donation joined with its employer's industry
                    bucket and the candidate it went to.
    per_candidate_industry_summary_df: long-form (candidate, industry,
                    total$, n) for the top-N candidates.
    """
    don = pd.read_csv(donations_path)
    cls = pd.read_csv(classified_path)

    cls["industry"] = cls.apply(
        lambda r: industry_bucket(r.get("level1_category"), r.get("naics_code")), axis=1
    )
    industry_lookup = cls.set_index("employer_norm")["industry"].to_dict()

    don = _dedupe_nan_recipients(don)
    don["candidate"] = don.apply(parse_candidate, axis=1)
    don = don[don["candidate"].notna()].copy()

    # Top-N candidates by total $ (any size).
    totals = don.groupby("candidate", as_index=False)["Amount"].sum().sort_values(
        "Amount", ascending=False
    )
    top = totals.head(TOP_N)["candidate"].tolist()

    big = don[(don["Amount"] > AMOUNT_MIN) & don["candidate"].isin(top)].copy()
    big["employer_norm"] = normalize_name(big["Contributor Employer"])
    # Don't trust running_list lookups for placeholder employer strings —
    # those are donor-typed placeholders, not real companies.
    is_placeholder = big["employer_norm"].isin(PLACEHOLDER_EMPLOYERS)
    big["industry"] = big["employer_norm"].map(industry_lookup).fillna("Unclassified")
    big.loc[is_placeholder, "industry"] = "Unclassified"
    n_placeholder = int(is_placeholder.sum())
    if n_placeholder:
        print(f"  reset {n_placeholder} placeholder-employer rows to Unclassified")

    summary = (
        big.groupby(["candidate", "industry"], as_index=False)
        .agg(total=("Amount", "sum"), n=("Amount", "count"))
    )

    # Preserve the candidate ordering by total $.
    cand_order = (
        big.groupby("candidate")["Amount"].sum().sort_values(ascending=False).index.tolist()
    )
    summary["candidate"] = pd.Categorical(summary["candidate"], categories=cand_order, ordered=True)
    summary = summary.sort_values(["candidate", "total"], ascending=[True, False]).reset_index(drop=True)
    return big, summary


# ---------- visualizations ----------

def _consolidate_for_pie(by_industry: pd.Series, total: float) -> pd.Series:
    """Fold any slice <SMALL_SLICE_PCT% into a single 'Other' slice; also
    drops 'Unclassified' since the pies show only attributable $."""
    if "Unclassified" in by_industry.index:
        by_industry = by_industry.drop("Unclassified")
    if by_industry.sum() == 0:
        return by_industry
    pct = by_industry / by_industry.sum() * 100
    big = by_industry[pct >= SMALL_SLICE_PCT]
    small = by_industry[pct < SMALL_SLICE_PCT]
    if len(small):
        big = pd.concat([big, pd.Series({"Other": small.sum()})])
    return big.sort_values(ascending=False)


def render_pie_grid(big: pd.DataFrame, out_path: Path) -> None:
    candidates = (
        big.groupby("candidate")["Amount"].sum().sort_values(ascending=False).index.tolist()
    )
    n = len(candidates)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 5.0 * rows))
    axes = axes.flatten()

    # Stable color assignment so the same industry has the same color
    # across all 10 pies.
    industries_global = (
        big.groupby("industry")["Amount"].sum().sort_values(ascending=False).index.tolist()
    )
    cmap = plt.get_cmap("tab20")
    color_for = {ind: cmap(i % 20) for i, ind in enumerate(industries_global) if ind != "Unclassified"}
    color_for["Other"] = (0.6, 0.6, 0.6)

    for i, cand in enumerate(candidates):
        ax = axes[i]
        sub = big[big["candidate"] == cand]
        total_big = float(sub["Amount"].sum())
        by_industry_full = sub.groupby("industry")["Amount"].sum()
        classified_total = float(by_industry_full.drop("Unclassified", errors="ignore").sum())
        coverage = classified_total / total_big * 100 if total_big else 0.0

        slices = _consolidate_for_pie(by_industry_full, total_big)
        if slices.sum() == 0:
            ax.text(0.5, 0.5, "No classified $", ha="center", va="center")
            ax.set_title(cand, fontsize=11)
            ax.axis("off")
            continue

        colors = [color_for.get(label, (0.7, 0.7, 0.7)) for label in slices.index]
        ax.pie(
            slices.values,
            labels=slices.index,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 8},
            colors=colors,
            pctdistance=0.78,
        )
        ax.set_title(
            f"{cand}\n>${AMOUNT_MIN:,} total: ${total_big:,.0f}   "
            f"({coverage:.0f}% classified)",
            fontsize=10,
        )

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"2026 CA Governor's race — Top-{TOP_N} candidates by total contributions\n"
        f"Industry breakdown of >${AMOUNT_MIN:,} contributions (classified $ only; Unclassified $ excluded)",
        fontsize=13,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def render_stacked_bar(big: pd.DataFrame, out_path: Path) -> None:
    """Horizontal stacked bar: each candidate is a row, x-axis is share
    of classified >$10K $, color stacks are industries. Unclassified is
    excluded so cross-candidate comparison reflects industry pattern."""
    cls_only = big[big["industry"] != "Unclassified"]
    pivot = (
        cls_only.pivot_table(index="candidate", columns="industry", values="Amount", aggfunc="sum")
        .fillna(0)
    )
    # Order rows by total classified $; columns by aggregate $ (largest first).
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]
    pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).index]
    pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(pivot.columns))]

    pivot.plot(kind="barh", stacked=True, ax=axes[0], color=colors, width=0.7)
    axes[0].set_title(f"Classified >${AMOUNT_MIN:,} contributions ($)")
    axes[0].set_xlabel("$")
    axes[0].set_ylabel("")
    axes[0].xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M"))
    axes[0].legend().remove()

    pct.plot(kind="barh", stacked=True, ax=axes[1], color=colors, width=0.7)
    axes[1].set_title(f"Same data as % share")
    axes[1].set_xlabel("% of classified $")
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, 100)
    axes[1].legend(
        title="Industry",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        fontsize=8,
        title_fontsize=9,
    )

    fig.suptitle(
        f"2026 CA Governor's race — Industry mix of classified >${AMOUNT_MIN:,} contributions, top-{TOP_N} candidates",
        fontsize=13,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------- main ----------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--donations", type=Path, default=DEFAULT_DONATIONS)
    p.add_argument("--classified", type=Path, default=DEFAULT_CLASSIFIED)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    args = p.parse_args(argv)

    print(f"Donations:  {args.donations}")
    print(f"Classified: {args.classified}")

    big, summary = build_industry_table(args.donations, args.classified)
    print(f"\n{len(big):,} >${AMOUNT_MIN:,} donations across top-{TOP_N} candidates")
    coverage = (
        big.loc[big["industry"] != "Unclassified", "Amount"].sum() / max(big["Amount"].sum(), 1) * 100
    )
    print(f"  classified share of $: {coverage:.1f}%")

    print("\nRendering figures:")
    render_pie_grid(big, args.out_dir / "candidate_industry_pies.png")
    render_stacked_bar(big, args.out_dir / "candidate_industry_stacked_bar.png")

    summary_path = args.out_dir.parent / "candidate_industry_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  wrote {summary_path}")

    # Per-candidate top-3 industry teaser
    print("\nTop-3 industries (classified $) per candidate:")
    for cand, sub in summary[summary["industry"] != "Unclassified"].groupby("candidate", observed=True):
        top3 = sub.nlargest(3, "total")
        parts = ", ".join(f"{r.industry} ${r.total:,.0f}" for r in top3.itertuples())
        print(f"  {cand:14s} {parts}")


if __name__ == "__main__":
    main()
