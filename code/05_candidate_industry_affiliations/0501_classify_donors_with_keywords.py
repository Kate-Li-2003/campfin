"""
0501_classify_donors_with_keywords.py

Two-part classification pipeline for high-value donors in a CA gov race.
Run end-to-end on a transactional donor file (e.g. governor_race_*.csv);
applies the same step-1 filter as 0302_classify_donors.py to isolate
qualifying high-value, company-affiliated donations and then runs:

  Part 1: masterfile match
          Join aggregated employers to running_list.csv (built by
          0301/0303 from OpenSecrets, H-1B, and prior EDD lookups).

  Part 2: keyword match (fallback)
          For employers still unmatched after Part 1, look for any
          whole-word keyword from Keywords_Manually_Collected.xlsx
          (employer_keywords + company_keywords sheets) inside the
          employer name; the longest matching keyword wins.

Output records carry two source-tracking columns:
  data_source_1: "masterfile" | "keyword match" | NaN (still unmatched)
  data_source_2: when data_source_1 == "masterfile", names the underlying
                 sub-source ("opensecrets" / "edd" / "h1b"); otherwise NaN.

A summary block at the end reports % assigned via masterfile vs keyword
match, both by employer count and by donation count.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = REPO_ROOT / "data/04_output_latest_data_pulls/governor_race_2026-04-27.csv"
DEFAULT_RUNNING_LIST = REPO_ROOT / "data/03_input/masterfile/running_list.csv"
DEFAULT_KEYWORDS = (
    REPO_ROOT
    / "data/03_input/training data (manual classifications)/Keywords_Manually_Collected.xlsx"
)
DEFAULT_OUT = REPO_ROOT / "output/05_output/donors_classified_masterfile_then_keywords.csv"

AMOUNT_MIN = 10000
NON_COMPANY_OCCUPATIONS = {
    "retired",
    "self-employed",
    "self employed",
    "not employed",
    "unemployed",
    "homemaker",
}

# Map running_list.source values -> data_source_2 token shown to the user.
SUB_SOURCE_MAP = {
    "opensecrets": "opensecrets",
    "edd": "edd",
    "h1b_employer": "h1b",
}

# Final column order for the output CSV.
OUT_COLS = [
    "employer",
    "n_donors",
    "employer_norm",
    "match_name",
    "matched_keyword",
    "level1_category",
    "level2_category",
    "level3_category",
    "naics_code",
    "naics_label",
    "data_source_1",
    "data_source_2",
]


# ---------- helpers ----------

def normalize_name(s: pd.Series) -> pd.Series:
    """Same normalization used to build running_list.csv (uppercase, drop
    non-alphanumerics, collapse whitespace)."""
    return (
        s.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9 ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _load_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


# ---------- step 0: filter + aggregate (same logic as 0302) ----------

def filter_and_aggregate(df: pd.DataFrame, amount_min: float) -> pd.DataFrame:
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

    print()
    print("=" * 70)
    print("STEP 0 — filter qualifying donations")
    print("=" * 70)
    print(f"  Total donations:                   {n_total:>9,}   ${total_amount:>15,.0f}")
    print(
        f"  Qualifying (>${int(amount_min):,}, no PAC ID, "
        f"company-affiliated occupation, non-blank employer):"
    )
    print(
        f"    rows:       {len(filtered):>9,}   "
        f"({len(filtered) / max(n_total, 1) * 100:>5.2f}% of input rows)"
    )
    print(
        f"    $ value:  ${filtered['Amount'].sum():>14,.0f}   "
        f"({filtered['Amount'].sum() / max(total_amount, 1) * 100:>5.2f}% of input $)"
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


# ---------- part 1: masterfile (running_list) match ----------

def match_masterfile(employers: pd.DataFrame, running_list_path: Path) -> pd.DataFrame:
    rl = pd.read_csv(running_list_path)
    rl = rl.drop_duplicates("name_norm", keep="first")
    keep = [
        "name_norm",
        "name",
        "level1_category",
        "level2_category",
        "level3_category",
        "naics_code",
        "naics_label",
        "source",
    ]
    rl_use = rl[[c for c in keep if c in rl.columns]].rename(columns={"name": "match_name"})

    out = employers.copy()
    out["employer_norm"] = normalize_name(out["employer"])
    merged = out.merge(rl_use, left_on="employer_norm", right_on="name_norm", how="left").drop(
        columns="name_norm"
    )

    matched = merged["match_name"].notna()
    merged["data_source_1"] = pd.NA
    merged["data_source_2"] = pd.NA
    merged["matched_keyword"] = pd.NA
    merged.loc[matched, "data_source_1"] = "masterfile"
    merged.loc[matched, "data_source_2"] = (
        merged.loc[matched, "source"].map(SUB_SOURCE_MAP).fillna(merged.loc[matched, "source"])
    )
    merged = merged.drop(columns="source")

    n_emp = len(merged)
    n_match = int(matched.sum())
    n_donations = int(merged["n_donors"].sum())
    n_match_donations = int(merged.loc[matched, "n_donors"].sum())

    print()
    print("=" * 70)
    print("PART 1 — masterfile match (running_list.csv)")
    print("=" * 70)
    print(f"  Reference rows in running_list:   {len(rl):>9,}")
    print(
        f"  Employers matched:                 {n_match:>9,}   "
        f"({n_match / max(n_emp, 1) * 100:>5.2f}% of {n_emp:,})"
    )
    print(
        f"  Donations matched:                 {n_match_donations:>9,}   "
        f"({n_match_donations / max(n_donations, 1) * 100:>5.2f}% of {n_donations:,})"
    )
    return merged


# ---------- part 2: keyword fallback ----------

def load_keywords(path: Path) -> pd.DataFrame:
    """Load `employer_keywords` + `company_keywords` sheets, expand
    comma/slash-separated rows into one keyword each, dedupe, and sort
    by descending length so longer (more specific) matches are tried
    first when scanning an employer name.

    Eliminated_* sheets are intentionally skipped — those are the
    keywords the curator deemed unsafe to substring-match.
    """
    cols = ["keywords", "level1_category", "level2_category", "level3_category"]

    emp = pd.read_excel(path, sheet_name="employer_keywords")
    com = pd.read_excel(path, sheet_name="company_keywords")
    if "level3_category" not in com.columns:
        com["level3_category"] = pd.NA

    df = pd.concat([emp[cols], com[cols]], ignore_index=True).dropna(subset=["keywords"])

    rows: list[dict] = []
    for _, r in df.iterrows():
        for part in re.split(r"\s*[,/]\s*", str(r["keywords"])):
            part = part.strip().lower()
            if part:
                rows.append(
                    {
                        "keyword": part,
                        "level1_category": r["level1_category"],
                        "level2_category": r["level2_category"],
                        "level3_category": r["level3_category"],
                    }
                )

    out = pd.DataFrame(rows).drop_duplicates(subset="keyword", keep="first")
    out = out.sort_values("keyword", key=lambda s: s.str.len(), ascending=False)
    return out.reset_index(drop=True)


def match_keywords(merged: pd.DataFrame, keywords_df: pd.DataFrame) -> pd.DataFrame:
    """Apply keyword fallback to rows where data_source_1 is still null."""
    unmatched_idx = merged.index[merged["data_source_1"].isna()].tolist()

    print()
    print("=" * 70)
    print("PART 2 — keyword match (fallback)")
    print("=" * 70)
    print(f"  Keywords loaded (after expansion + dedupe): {len(keywords_df):,}")
    print(f"  Employers unmatched after Part 1:           {len(unmatched_idx):,}")

    if not unmatched_idx:
        return merged

    # Pre-compile whole-word regex per keyword. Keywords are sorted by
    # length desc, so the first hit on an employer is the most specific.
    patterns = [
        (re.compile(rf"\b{re.escape(row.keyword)}\b"), row)
        for row in keywords_df.itertuples(index=False)
    ]

    n_kw_matched = 0
    for idx in unmatched_idx:
        emp_str = str(merged.at[idx, "employer"]).lower()
        for pat, row in patterns:
            if pat.search(emp_str):
                merged.at[idx, "level1_category"] = row.level1_category
                merged.at[idx, "level2_category"] = row.level2_category
                merged.at[idx, "level3_category"] = row.level3_category
                merged.at[idx, "matched_keyword"] = row.keyword
                merged.at[idx, "data_source_1"] = "keyword match"
                # data_source_2 stays NaN per spec
                n_kw_matched += 1
                break

    print(
        f"  Newly matched by keyword:                   {n_kw_matched:,}   "
        f"({n_kw_matched / max(len(unmatched_idx), 1) * 100:>5.2f}% of remaining)"
    )
    return merged


# ---------- summary ----------

def print_summary(merged: pd.DataFrame) -> None:
    n_emp = len(merged)
    n_donations = int(merged["n_donors"].sum())
    src = merged["data_source_1"].fillna("unmatched")
    by_emp = src.value_counts()
    by_don = merged.groupby(src)["n_donors"].sum()

    print()
    print("=" * 70)
    print("FINAL SUMMARY — % of dataset assigned via masterfile vs keyword")
    print("=" * 70)
    print(f"  Total employers: {n_emp:,}    Total donations: {n_donations:,}\n")
    for source in ("masterfile", "keyword match", "unmatched"):
        e = int(by_emp.get(source, 0))
        d = int(by_don.get(source, 0))
        print(
            f"  {source:14s}  employers: {e:>5,} ({e / max(n_emp, 1) * 100:>5.2f}%)   "
            f"donations: {d:>5,} ({d / max(n_donations, 1) * 100:>5.2f}%)"
        )

    # Masterfile sub-source breakdown.
    mf = merged[merged["data_source_1"] == "masterfile"]
    if len(mf):
        print(f"\n  Masterfile sub-sources (data_source_2):")
        for sub, count in mf["data_source_2"].value_counts().items():
            d = int(mf.loc[mf["data_source_2"] == sub, "n_donors"].sum())
            print(
                f"    {sub:14s} employers: {count:>5,} ({count / n_emp * 100:>5.2f}%)   "
                f"donations: {d:>5,} ({d / n_donations * 100:>5.2f}%)"
            )


# ---------- main ----------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--running-list", type=Path, default=DEFAULT_RUNNING_LIST)
    p.add_argument("--keywords", type=Path, default=DEFAULT_KEYWORDS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--amount-min", type=float, default=AMOUNT_MIN)
    args = p.parse_args(argv)

    print(f"Input: {args.input}")
    df = _load_input(args.input)
    employers = filter_and_aggregate(df, args.amount_min)
    merged = match_masterfile(employers, args.running_list)
    keywords = load_keywords(args.keywords)
    merged = match_keywords(merged, keywords)
    print_summary(merged)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in OUT_COLS if c in merged.columns] + [
        c for c in merged.columns if c not in OUT_COLS
    ]
    merged[cols].to_csv(args.out, index=False)
    print(f"\nWrote: {args.out}  ({len(merged):,} rows)")


if __name__ == "__main__":
    main()
