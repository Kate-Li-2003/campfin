"""
0504_classify_other_races.py

Apply the established classification pipeline to the *other* down-ballot
races (Lt. Governor and Insurance Commissioner) to see how much coverage
we can get without further manual labeling.

Differences vs 0502:
  - No manual NAICS sheet (we don't have one for these races) — instead,
    EDD live lookups serve as the third fallback so coverage doesn't
    drop off a cliff.
  - Step 1 filter is broadened to also accept *organizational* donors,
    not just individuals-with-employer. An entity qualifies if either
      (a) Contributor Employer is non-blank (individual w/ employer), or
      (b) Contributor Name looks like an organization (corporate suffix
          like LLC/Inc/Corp/etc., or no comma) — and we use that name
          itself as the entity to classify.
    This recovers the org donors (~$75M nationally on the Gov race) that
    the original step-1 filter was silently dropping.
  - Rows with a Contributor ID (typically committee-to-committee
    transfers) are KEPT and treated as organizations.
  - Rows whose Contributor Occupation is retired/self-employed/etc. are
    KEPT and classified as the new source "non-company individual"
    (skipping masterfile/keyword/EDD lookup).

Pipeline per race:
  STEP 0  filter + aggregate to (entity, n_donations)
  PART 1  masterfile match           (running_list.csv)
  PART 1.5 auto-classify non-company individuals
  PART 2  keyword match              (Keywords_Manually_Collected.xlsx)
  PART 3  EDD live lookup            (cached; no manual layer)
  + summary (% by entity / % by donation)
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "03_aggregating_data"))

from edd_naics_lookup import EDDClient  # noqa: E402

DEFAULT_INPUTS = [
    REPO_ROOT / "data/04_output_latest_data_pulls/lt_governor_race_2026.csv",
    REPO_ROOT / "data/04_output_latest_data_pulls/insurance_commissioner_race_2026.csv",
]
DEFAULT_RUNNING_LIST = REPO_ROOT / "data/03_input/masterfile/running_list.csv"
DEFAULT_KEYWORDS = (
    REPO_ROOT
    / "data/03_input/training data (manual classifications)/Keywords_Manually_Collected.xlsx"
)
DEFAULT_OUT_DIR = REPO_ROOT / "output/05_output"

AMOUNT_MIN = 10000
NON_COMPANY_OCCUPATIONS = {
    "retired",
    "self-employed",
    "self employed",
    "not employed",
    "unemployed",
    "homemaker",
}

# Corporate suffix tokens that flag a Contributor Name as an organization.
ORG_SUFFIX_RE = re.compile(
    r"\b("
    r"LLC|L\.?L\.?C\.?|LLP|"
    r"INC|INCORPORATED|CORP|CORPORATION|"
    r"LTD|LIMITED|LP|"
    r"CO|COMPANY|COMPANIES|"
    r"HOLDINGS|GROUP|PARTNERS|VENTURES|CAPITAL|FUND|TRUST|"
    r"FOUNDATION|ASSOCIATION|UNION|PAC|COMMITTEE|GUILD|SOCIETY|"
    r"INSTITUTE|COUNCIL|AGENCY|AUTHORITY|DISTRICT|BUREAU|"
    r"CHURCH|TEMPLE|MOSQUE|SYNAGOGUE|MINISTRIES|"
    r"HOSPITAL|UNIVERSITY|COLLEGE|SCHOOL|"
    r"FIRM|PARTNERSHIP|ENTERPRISES|RESORT|REALTY|PROPERTIES"
    r")\b",
    re.IGNORECASE,
)


# Dynamically load 0501 (filename starts with digits → not directly importable).
def _load_0501():
    path = Path(__file__).parent / "0501_classify_donors_with_keywords.py"
    spec = importlib.util.spec_from_file_location("kw_classifier", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kw = _load_0501()


# ---------- helpers ----------

def looks_like_org(name: str | float) -> bool:
    """True if the contributor name reads as an organization rather than
    an individual. Used when Contributor Employer is blank, to decide
    whether to keep the row (org self-donation) or drop it."""
    if pd.isna(name):
        return False
    s = str(name).strip()
    if not s:
        return False
    if ORG_SUFFIX_RE.search(s):
        return True
    # No corporate suffix — fall back to "no comma" heuristic. Individuals
    # are formatted "Last, First" so a no-comma name is usually an org.
    return "," not in s


def filter_and_aggregate_with_orgs(df: pd.DataFrame, amount_min: float) -> pd.DataFrame:
    """Step 0: filter to qualifying high-value rows then collapse to one
    row per entity. Three entity kinds are tracked:
      - "individual"             : person with an employer; entity = employer
      - "organization"           : org-like name OR has Contributor ID
                                   (committee-to-committee); entity = name
      - "non_company_individual" : Contributor Occupation is retired/
                                   self-employed/etc.; entity = name
    Drops only truly empty rows (no employer, no org-like name, no
    Contributor ID, no relevant occupation)."""
    n_total = len(df)
    total_amount = float(df["Amount"].sum())

    occ = df["Contributor Occupation"].fillna("").str.lower().str.strip()
    employer_str = df["Contributor Employer"].astype(str).str.strip()
    has_employer = df["Contributor Employer"].notna() & (employer_str != "")
    is_org = df["Contributor Name"].apply(looks_like_org)
    has_id = df["Contributor ID"].notna()
    is_nc = occ.isin(NON_COMPANY_OCCUPATIONS)

    qual = (
        (df["Amount"] > amount_min)
        & (has_employer | is_org | has_id | is_nc)
    )
    filtered = df.loc[qual].copy()

    f_has_employer = has_employer.loc[qual]
    f_is_org = is_org.loc[qual]
    f_has_id = has_id.loc[qual]
    f_is_nc = is_nc.loc[qual]

    # Priority: non_company_individual > organization > individual.
    # non-company individuals shouldn't be classified by employer (e.g. a
    # retiree's stale employer field); committee-to-committee rows (with
    # Contributor ID) should be treated as orgs not as individuals with
    # whatever the committee listed as "employer".
    filtered["entity_kind"] = "individual"
    filtered.loc[f_is_org | f_has_id, "entity_kind"] = "organization"
    filtered.loc[f_is_nc, "entity_kind"] = "non_company_individual"

    # Entity: employer for individuals, contributor name otherwise.
    is_individual = filtered["entity_kind"] == "individual"
    filtered["entity"] = filtered["Contributor Name"]
    filtered.loc[is_individual, "entity"] = filtered.loc[is_individual, "Contributor Employer"]

    # Preserve normalized occupation (used to label non-company-individual rows).
    filtered["occupation_norm"] = occ.loc[qual]

    n_qual = len(filtered)
    qual_amount = float(filtered["Amount"].sum())
    n_org_rows = int((filtered["entity_kind"] == "organization").sum())
    n_nc_rows = int((filtered["entity_kind"] == "non_company_individual").sum())
    n_ind_rows = int((filtered["entity_kind"] == "individual").sum())

    print()
    print("=" * 70)
    print("STEP 0 — filter qualifying donations (org + non-company aware)")
    print("=" * 70)
    print(f"  Total donations:                  {n_total:>9,}   ${total_amount:>15,.0f}")
    print(f"  Qualifying (>${int(amount_min):,}, any donor type):")
    print(
        f"    rows:    {n_qual:>9,}   "
        f"({n_qual / max(n_total, 1) * 100:>5.2f}% of input rows)"
    )
    print(
        f"    $ value: ${qual_amount:>14,.0f}   "
        f"({qual_amount / max(total_amount, 1) * 100:>5.2f}% of input $)"
    )
    print(f"    breakdown:  individual={n_ind_rows:,}  "
          f"organization={n_org_rows:,}  non-company={n_nc_rows:,}")

    aggregated = (
        filtered.groupby("entity", as_index=False)
        .agg(
            n_donors=("entity", "size"),
            amount_total=("Amount", "sum"),
            entity_kind=("entity_kind", "first"),
            occupation_norm=("occupation_norm", "first"),
        )
        .sort_values("amount_total", ascending=False)
        .reset_index(drop=True)
        .rename(columns={"entity": "employer"})  # keep schema compatible w/ 0501
    )
    print(f"  Unique entities (aggregated):     {len(aggregated):,}")
    return aggregated


# ---------- PART 1.5: auto-classify non-company individuals ----------

def classify_non_company_individuals(merged: pd.DataFrame) -> pd.DataFrame:
    """Mark rows whose entity_kind is 'non_company_individual' with
    data_source_1 == 'non-company individual' BEFORE keyword/EDD steps,
    so personal names aren't misclassified by industry keywords. The
    specific occupation (retired/self-employed/etc.) is carried in
    level2_category."""
    print()
    print("=" * 70)
    print("PART 1.5 — auto-classify non-company individuals")
    print("=" * 70)

    mask = (merged["entity_kind"] == "non_company_individual") & merged["data_source_1"].isna()
    n = int(mask.sum())
    print(f"  Rows tagged 'non-company individual': {n:,}")

    if n == 0:
        return merged

    occ_titled = (
        merged.loc[mask, "occupation_norm"].fillna("").str.title().replace("", pd.NA)
    )
    merged.loc[mask, "data_source_1"] = "non-company individual"
    merged.loc[mask, "level1_category"] = "Non-company individual"
    merged.loc[mask, "level2_category"] = occ_titled
    # naics_code / naics_label / match_name stay NaN — no NAICS for these.
    return merged


# ---------- EDD step (mirrors 0302's step 3 but with our entity column) ----------

def match_edd(merged: pd.DataFrame, delay: float = 1.0) -> pd.DataFrame:
    print()
    print("=" * 70)
    print("PART 3 — EDD lookup for unmatched entities")
    print("=" * 70)

    unmatched = merged[merged["data_source_1"].isna()]
    if unmatched.empty:
        print("  Nothing to look up.")
        return merged

    unique_norms = unmatched.drop_duplicates("employer_norm")
    print(f"  Unmatched rows:                   {len(unmatched):>5,}")
    print(f"  Distinct names to query:          {len(unique_norms):>5,}")

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
            f"  [{i:>3}/{len(unique_norms)}] {str(row.employer)[:50]:50s} -> "
            f"{r.status:11s} {r.naics_code or '':6s} {r.match_name or ''}",
            file=sys.stderr,
        )

    for col in (
        "edd_match_name",
        "edd_match_type",
        "edd_naics_code",
        "edd_naics_description",
        "edd_status",
    ):
        merged[col] = merged["employer_norm"].map(lambda k: edd_by_norm.get(k, {}).get(col))

    edd_hit = merged["edd_status"] == "ok"
    # NAICS 999990 is EDD's "Unknown" — treat as no useful classification.
    edd_hit = edd_hit & (merged["edd_naics_code"] != "999990")

    merged.loc[edd_hit, "match_name"] = merged.loc[edd_hit, "edd_match_name"]
    merged.loc[edd_hit, "naics_code"] = merged.loc[edd_hit, "edd_naics_code"]
    merged.loc[edd_hit, "naics_label"] = merged.loc[edd_hit, "edd_naics_description"]
    merged.loc[edd_hit, "data_source_1"] = "edd"

    n_edd = int(edd_hit.sum())
    print(
        f"  EDD matches with valid NAICS:     {n_edd:>5,}   "
        f"({n_edd / max(len(unique_norms), 1) * 100:>5.2f}% of unique queries)"
    )
    return merged


# ---------- summary ----------

def print_summary(merged: pd.DataFrame, race: str) -> None:
    n_emp = len(merged)
    n_donations = int(merged["n_donors"].sum())
    src = merged["data_source_1"].fillna("unmatched")
    by_emp = src.value_counts()
    by_don = merged.groupby(src)["n_donors"].sum()

    print()
    print("=" * 70)
    print(f"FINAL SUMMARY — {race}")
    print("=" * 70)
    print(f"  Total entities: {n_emp:,}    Total donations: {n_donations:,}\n")
    for source in (
        "masterfile",
        "non-company individual",
        "keyword match",
        "edd",
        "unmatched",
    ):
        e = int(by_emp.get(source, 0))
        d = int(by_don.get(source, 0))
        print(
            f"  {source:14s}  entities: {e:>4,} ({e / max(n_emp, 1) * 100:>5.2f}%)   "
            f"donations: {d:>4,} ({d / max(n_donations, 1) * 100:>5.2f}%)"
        )


# ---------- per-race driver ----------

def run_race(
    input_path: Path,
    running_list_path: Path,
    keywords_path: Path,
    out_dir: Path,
    amount_min: float = AMOUNT_MIN,
    delay: float = 1.0,
) -> dict:
    print("\n" + "#" * 70)
    print(f"# Race: {input_path.name}")
    print("#" * 70)

    df = kw._load_input(input_path)
    employers = filter_and_aggregate_with_orgs(df, amount_min)

    if employers.empty:
        print("  No qualifying entities for this race; skipping.")
        return {"race": input_path.stem, "n_entities": 0}

    # Reuse 0501's masterfile + keyword steps directly. Both operate on
    # the `employer` column we just renamed from `entity`.
    merged = kw.match_masterfile(employers, running_list_path)
    # Auto-tag non-company individuals so they bypass keyword + EDD.
    merged = classify_non_company_individuals(merged)
    keywords = kw.load_keywords(keywords_path)
    merged = kw.match_keywords(merged, keywords)
    # Replace 0501's "manual" with our EDD live-lookup fallback.
    merged = match_edd(merged, delay=delay)

    print_summary(merged, race=input_path.stem)

    out_path = out_dir / f"{input_path.stem}_classified.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"\n  Wrote: {out_path}  ({len(merged):,} rows)")

    src = merged["data_source_1"].fillna("unmatched")
    return {
        "race": input_path.stem,
        "n_entities": len(merged),
        "n_donations": int(merged["n_donors"].sum()),
        "by_source_entities": dict(src.value_counts()),
        "by_source_donations": dict(merged.groupby(src)["n_donors"].sum()),
    }


# ---------- main ----------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--inputs", type=Path, nargs="*", default=DEFAULT_INPUTS)
    p.add_argument("--running-list", type=Path, default=DEFAULT_RUNNING_LIST)
    p.add_argument("--keywords", type=Path, default=DEFAULT_KEYWORDS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--amount-min", type=float, default=AMOUNT_MIN)
    p.add_argument("--delay", type=float, default=1.0)
    args = p.parse_args(argv)

    results = []
    for inp in args.inputs:
        r = run_race(
            inp,
            args.running_list,
            args.keywords,
            args.out_dir,
            amount_min=args.amount_min,
            delay=args.delay,
        )
        results.append(r)

    # Cross-race recap.
    print("\n" + "=" * 70)
    print("CROSS-RACE COVERAGE RECAP")
    print("=" * 70)
    print(f"  {'race':40s} {'entities':>10s} {'matched':>10s} {'%':>7s}")
    for r in results:
        if not r.get("n_entities"):
            print(f"  {r['race']:40s} {0:>10}     —")
            continue
        ent = r["by_source_entities"]
        matched = sum(v for k, v in ent.items() if k != "unmatched")
        pct = matched / r["n_entities"] * 100
        print(
            f"  {r['race']:40s} {r['n_entities']:>10,} {matched:>10,} {pct:>6.2f}%"
        )


if __name__ == "__main__":
    main()
