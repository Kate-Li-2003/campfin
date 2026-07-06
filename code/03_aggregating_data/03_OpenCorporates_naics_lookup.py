"""
04_naics_lookup.py
==================
For individual donors in the 2026 CA Governor's race who donated >= $10,000,
look up each donor's employer in the OpenCorporates database and retrieve any
NAICS classification codes attached to that company record.

At the end of the run the script prints a coverage summary:
  • How many unique employers were found in OpenCorporates
  • How many of those have at least one NAICS code
  • How many qualifying donors work for a NAICS-classified employer
  • The NAICS codes themselves, sorted by total donation amount

────────────────────────────────────────────────────────────────────────────────
NAICS vs SIC in OpenCorporates
────────────────────────────────────────────────────────────────────────────────
OpenCorporates stores both SIC and NAICS codes in the `industry_codes` array.
Each entry looks like:

  { "industry_code": {
        "code":          "524113",
        "description":   "Direct Life Insurance Carriers",
        "code_scheme_id": "us_naics_2017",
        "code_scheme":    "NAICS 2017",
        "uid":            "us_naics_2017-524113"
    }
  }

This script identifies NAICS entries by checking for "naics" (case-insensitive)
in either `code_scheme` or `code_scheme_id`.  All matching NAICS codes for a
company are collected; typically there is one, but occasionally several.

────────────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────────────
  python 04_naics_lookup.py [options]

  --api-key    STR    OpenCorporates API key (prompted securely if omitted)
  --min-amount FLOAT  Minimum single-donation amount to include (default: 10000)
  --max-rows   INT    Limit to N employers — useful for a test run
  --resume            Skip employers already present in the output CSV
  --delay      FLOAT  Seconds between HTTP requests (default: 1.2)
  --output     PATH   Override output CSV path

────────────────────────────────────────────────────────────────────────────────
OUTPUT  (data/04_output/naics_lookup.csv)
────────────────────────────────────────────────────────────────────────────────
  employer              Raw employer string from the donation CSV
  n_donors              Number of qualifying donation rows for this employer
  total_donated         Sum of qualifying donations (USD)
  oc_found              "yes" / "no" — whether OpenCorporates returned a match
  oc_company_name       Matched company name in OpenCorporates
  oc_company_type       e.g. "Domestic LLC", "Foreign Corporation"
  oc_status             e.g. "Active", "Dissolved"
  oc_jurisdiction       Jurisdiction code (e.g. "us_ca", "us_de")
  oc_address            Registered address
  oc_url                OpenCorporates profile URL
  naics_found           "yes" / "no" — whether at least one NAICS code was found
  naics_codes           Pipe-separated NAICS codes (e.g. "524113|524114")
  naics_descriptions    Pipe-separated NAICS descriptions
  naics_schemes         Pipe-separated scheme names (e.g. "NAICS 2017")
  sic_codes             Pipe-separated SIC codes (for reference)
  sic_descriptions      Pipe-separated SIC descriptions (for reference)
  fetch_error           Any error message for this row
"""

from __future__ import annotations

import argparse
import csv
import getpass
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

GOV_RACE_PATH    = DATA_DIR / "01CalAccess_CampaignFinance_Data" / "governor_race_2026.csv"
OPENSECRETS_PATH = DATA_DIR / "ContributionOpenSecretsCategories (1).csv"
OUTPUT_DIR       = DATA_DIR / "04_output"
DEFAULT_OUTPUT   = OUTPUT_DIR / "naics_lookup.csv"

DEFAULT_MIN_AMOUNT = 10_000.0

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

OPENCORP_SEARCH_URL = "https://api.opencorporates.com/v0.4/companies/search"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


# ── HELPERS: DATA LOADING ─────────────────────────────────────────────────────

def _clean_amount(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[$,]", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _detect_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalise_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = name.upper()
    n = re.sub(r"[^A-Z0-9 ]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def load_unmatched_medium_plus_donors(
    gov_path: Path,
    os_path: Path,
    min_amount: float,
) -> pd.DataFrame:
    """
    Load individual donors with a single donation >= min_amount who have
    no OpenSecrets industry match, or whose match is only "Uncoded".
    """
    print(f"\n[1] Loading governor race data from {gov_path.name} …")
    df = pd.read_csv(gov_path, dtype=str, low_memory=False)
    print(f"    {len(df):,} rows loaded.")

    col_name = _detect_column(df, ["Contributor Name", "CTRIB_NAML"])
    col_occ  = _detect_column(df, ["Contributor Occupation", "CTRIB_OCC"])
    col_id   = _detect_column(df, ["Contributor ID", "CMTE_ID", "FILER_ID"])
    col_amt  = _detect_column(df, ["Amount", "AMOUNT", "TRAN_AMT1"])
    col_emp  = _detect_column(df, ["Contributor Employer", "CTRIB_EMP", "employer"])

    if col_amt is None:
        raise ValueError("Cannot find Amount column in the CSV.")
    if col_name is None:
        raise ValueError("Cannot find Contributor Name column in the CSV.")

    df["amount_clean"] = _clean_amount(df[col_amt])
    df["_name"]        = df[col_name].fillna("").str.strip()
    df["_employer"]    = df[col_emp].fillna("").str.strip() if col_emp else ""

    has_occ = (
        df[col_occ].notna() & df[col_occ].str.strip().ne("") & df[col_occ].str.lower().ne("nan")
        if col_occ else pd.Series(False, index=df.index)
    )
    has_id = (
        df[col_id].notna() & df[col_id].str.strip().ne("") & df[col_id].str.lower().ne("nan")
        if col_id else pd.Series(False, index=df.index)
    )
    df["donor_type"] = np.select(
        [has_id, has_occ], ["PAC/Committee", "Individual"], default="Organization"
    )

    indiv       = df[df["donor_type"] == "Individual"].copy()
    medium_plus = indiv[indiv["amount_clean"] >= min_amount].copy()
    print(
        f"    Individual donors with donation >= ${min_amount:,.0f}: "
        f"{len(medium_plus):,} rows ({medium_plus['_name'].nunique():,} unique names)"
    )

    if not os_path.exists():
        print(f"    WARNING: OpenSecrets file not found — treating all as unmatched.")
        medium_plus["os_matched"] = False
        medium_plus["os_level1"]  = None
    else:
        print(f"    Loading OpenSecrets from {os_path.name} …")
        os_df = pd.read_csv(os_path, dtype=str, low_memory=False)
        os_df.columns = [c.strip() for c in os_df.columns]
        os_df["name_norm"] = os_df["name"].apply(normalise_name)
        os_lookup = (
            os_df.drop_duplicates("name_norm")
            .set_index("name_norm")[["level1_category"]]
            .to_dict(orient="index")
        )
        medium_plus["name_norm"] = medium_plus["_name"].apply(normalise_name)
        medium_plus["os_matched"] = medium_plus["name_norm"].isin(os_lookup)
        medium_plus["os_level1"]  = medium_plus["name_norm"].map(
            {k: v["level1_category"] for k, v in os_lookup.items()}
        )

    unmatched = medium_plus[
        (~medium_plus["os_matched"]) |
        (medium_plus["os_level1"].fillna("Uncoded") == "Uncoded")
    ].copy()

    print(
        f"    Unmatched (no industry classification): "
        f"{len(unmatched):,} rows ({unmatched['_name'].nunique():,} unique donor names)"
    )
    print()
    return unmatched


def build_employer_table(donors: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by employer, drop blank/generic values, sort by total donated."""
    skip_vals = {
        "", "n/a", "na", "none", "self", "self-employed", "retired",
        "not employed", "homemaker", "student", "unemployed",
    }
    valid = donors[
        donors["_employer"].str.lower().str.strip().apply(
            lambda v: v not in skip_vals and len(v) > 0
        )
    ].copy()

    table = (
        valid.groupby("_employer")
        .agg(n_donors=("_name", "count"), total_donated=("amount_clean", "sum"))
        .reset_index()
        .rename(columns={"_employer": "employer"})
        .sort_values("total_donated", ascending=False)
        .reset_index(drop=True)
    )
    donors_with_employer = valid["_name"].nunique()
    donors_total         = donors["_name"].nunique()
    print(
        f"\n[2] {len(table):,} unique employers to look up "
        f"({donors_with_employer:,} / {donors_total:,} unique donors have a named employer).\n"
    )
    return table


# ── OPENCORPORATES NAICS LOOKUP ───────────────────────────────────────────────

def _is_naics(code_entry: dict) -> bool:
    """Return True if this industry_code entry is a NAICS classification."""
    ic = code_entry.get("industry_code", code_entry)  # handle both wrapped and flat
    scheme    = (ic.get("code_scheme", "")    or "").lower()
    scheme_id = (ic.get("code_scheme_id", "") or "").lower()
    return "naics" in scheme or "naics" in scheme_id


def _is_sic(code_entry: dict) -> bool:
    ic = code_entry.get("industry_code", code_entry)
    scheme    = (ic.get("code_scheme", "")    or "").lower()
    scheme_id = (ic.get("code_scheme_id", "") or "").lower()
    return "sic" in scheme or "sic" in scheme_id


def _parse_company(company: dict) -> dict:
    """Extract all relevant fields, separating NAICS from SIC codes."""
    ctype    = company.get("company_type", "")         or ""
    status   = company.get("current_status", "")       or ""
    juris    = company.get("jurisdiction_code", "")    or ""
    address  = company.get("registered_address_in_full", "") or ""
    oc_url   = company.get("opencorporates_url", "")   or ""
    co_name  = company.get("name", "")                 or ""

    naics_codes = []
    naics_descs = []
    naics_schms = []
    sic_codes   = []
    sic_descs   = []

    for entry in (company.get("industry_codes") or []):
        ic = entry.get("industry_code", entry)
        code  = (ic.get("code", "")        or "").strip()
        desc  = (ic.get("description", "") or "").strip()
        schm  = (ic.get("code_scheme", "") or "").strip()

        if _is_naics(entry):
            if code:
                naics_codes.append(code)
                naics_descs.append(desc)
                naics_schms.append(schm)
        elif _is_sic(entry):
            if code:
                sic_codes.append(code)
                sic_descs.append(desc)

    return {
        "oc_company_name": co_name,
        "oc_company_type": ctype,
        "oc_status":       status,
        "oc_jurisdiction": juris,
        "oc_address":      address,
        "oc_url":          oc_url,
        "naics_codes":     "|".join(naics_codes),
        "naics_descriptions": "|".join(naics_descs),
        "naics_schemes":   "|".join(naics_schms),
        "sic_codes":       "|".join(sic_codes),
        "sic_descriptions": "|".join(sic_descs),
    }


def opencorporates_naics_lookup(
    company_name: str,
    api_key: str,
    session: requests.Session,
) -> dict:
    """
    Search OpenCorporates for company_name.
    Strategy:
      1. Search California (jurisdiction_code=us_ca), take top result.
      2. If no CA result, search all US jurisdictions (no restriction).
      3. If still nothing, return empty result.

    Returns a dict with all fields for one CSV row.
    """
    empty = {
        "oc_found":           "no",
        "oc_company_name":    "",
        "oc_company_type":    "",
        "oc_status":          "",
        "oc_jurisdiction":    "",
        "oc_address":         "",
        "oc_url":             "",
        "naics_found":        "no",
        "naics_codes":        "",
        "naics_descriptions": "",
        "naics_schemes":      "",
        "sic_codes":          "",
        "sic_descriptions":   "",
        "fetch_error":        "",
    }

    def _search(params: dict) -> Optional[dict]:
        try:
            resp = session.get(OPENCORP_SEARCH_URL, params=params, timeout=20)
            resp.raise_for_status()
            companies = resp.json().get("results", {}).get("companies", [])
            if companies:
                return companies[0].get("company", {})
        except Exception as exc:
            empty["fetch_error"] = str(exc)
        return None

    base_params: dict = {"q": company_name, "order": "score", "per_page": 1}
    if api_key:
        base_params["api_token"] = api_key

    # Pass 1 — California only
    company = _search({**base_params, "jurisdiction_code": "us_ca"})

    # Pass 2 — US-wide fallback
    if company is None:
        company = _search(base_params)

    if company is None:
        return empty

    parsed = _parse_company(company)
    result = {**empty, "oc_found": "yes", **parsed}
    result["naics_found"] = "yes" if parsed["naics_codes"] else "no"
    return result


# ── CREDENTIAL PROMPT ─────────────────────────────────────────────────────────

def prompt_api_key() -> str:
    print(
        "\n── OpenCorporates API key ────────────────────────────────────────\n"
        "Required to query NAICS classifications via the OpenCorporates API.\n\n"
        "  • Account portal: https://opencorporates.com/api_accounts/new\n"
        "  • Your key appears in your account settings after sign-up.\n"
        "  • Without a key the API still works but is heavily rate-limited.\n\n"
        "  Leave blank to use the unauthenticated free tier (very slow).\n"
        "─────────────────────────────────────────────────────────────────\n"
    )
    try:
        return getpass.getpass("  Paste your OpenCorporates API key (hidden): ").strip()
    except (EOFError, OSError):
        return input("  Paste your OpenCorporates API key: ").strip()


# ── OUTPUT SCHEMA ─────────────────────────────────────────────────────────────

FIELDNAMES = [
    "employer", "n_donors", "total_donated",
    "oc_found", "oc_company_name", "oc_company_type", "oc_status",
    "oc_jurisdiction", "oc_address", "oc_url",
    "naics_found", "naics_codes", "naics_descriptions", "naics_schemes",
    "sic_codes", "sic_descriptions",
    "fetch_error",
]


# ── SUMMARY REPORT ────────────────────────────────────────────────────────────

def print_summary(results_path: Path, employer_table: pd.DataFrame) -> None:
    """Read the output CSV and print a coverage summary to stdout."""
    try:
        res = pd.read_csv(results_path, dtype=str)
    except Exception:
        print("\n[Summary] Could not read output file for summary.\n")
        return

    total_employers = len(res)
    oc_found        = (res["oc_found"] == "yes").sum()
    naics_found     = (res["naics_found"] == "yes").sum()

    # Merge back to count donors
    merged = employer_table.merge(
        res[["employer", "naics_found", "naics_codes", "naics_descriptions"]],
        on="employer",
        how="left",
    )
    donors_total        = int(merged["n_donors"].sum())
    donors_naics        = int(merged.loc[merged["naics_found"] == "yes", "n_donors"].sum())
    donations_total     = merged["total_donated"].sum()
    donations_naics     = merged.loc[merged["naics_found"] == "yes", "total_donated"].sum()

    pct_emp    = 100 * naics_found / total_employers if total_employers else 0
    pct_donors = 100 * donors_naics / donors_total   if donors_total   else 0
    pct_dollars= 100 * donations_naics / donations_total if donations_total else 0

    print("\n" + "=" * 65)
    print("NAICS COVERAGE SUMMARY")
    print("=" * 65)
    print(f"  Employers queried:                {total_employers:>6,}")
    print(f"  Found in OpenCorporates:          {oc_found:>6,}  ({100*oc_found/total_employers:.1f}%)")
    print(f"  With at least one NAICS code:     {naics_found:>6,}  ({pct_emp:.1f}% of employers queried)")
    print(f"  Donors at NAICS-classified firms: {donors_naics:>6,}  ({pct_donors:.1f}% of donors with employers)")
    print(f"  $ donated by NAICS-classified:  ${donations_naics:>10,.0f}  ({pct_dollars:.1f}% of total)")
    print("=" * 65)

    # Top NAICS codes by donation volume
    naics_rows = res[res["naics_found"] == "yes"].copy()
    if naics_rows.empty:
        print("\n  No NAICS codes found in this run.\n")
        return

    naics_rows = naics_rows.merge(
        employer_table[["employer", "n_donors", "total_donated"]],
        on="employer", how="left"
    )

    # Explode multi-code employers to one row per NAICS code
    rows = []
    for _, r in naics_rows.iterrows():
        codes = (r["naics_codes"] or "").split("|")
        descs = (r["naics_descriptions"] or "").split("|")
        for code, desc in zip(codes, descs):
            if code.strip():
                rows.append({
                    "naics_code":  code.strip(),
                    "description": desc.strip(),
                    "n_donors":    r.get("n_donors", 0),
                    "total_donated": float(r.get("total_donated", 0) or 0),
                })

    if not rows:
        return

    code_df = (
        pd.DataFrame(rows)
        .groupby(["naics_code", "description"])
        .agg(n_employers=("n_donors", "count"), n_donors=("n_donors", "sum"),
             total_donated=("total_donated", "sum"))
        .reset_index()
        .sort_values("total_donated", ascending=False)
    )

    print("\nTop NAICS codes by total donations:")
    print(f"  {'Code':<10} {'Employers':>10} {'Donors':>8} {'Total Donated':>15}  Description")
    print("  " + "-" * 75)
    for _, r in code_df.head(20).iterrows():
        print(
            f"  {r['naics_code']:<10} {int(r['n_employers']):>10} "
            f"{int(r['n_donors']):>8} ${r['total_donated']:>14,.0f}  {r['description']}"
        )
    print()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    out_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────
    donors        = load_unmatched_medium_plus_donors(GOV_RACE_PATH, OPENSECRETS_PATH, args.min_amount)
    employer_table = build_employer_table(donors)

    if args.max_rows:
        employer_table = employer_table.head(args.max_rows)
        print(f"    (--max-rows {args.max_rows}: processing {len(employer_table)} employers)\n")

    # ── resume ────────────────────────────────────────────────────────────
    done_employers: set = set()
    if args.resume and out_path.exists():
        ex = pd.read_csv(out_path, dtype=str)
        done_employers = set(ex["employer"].dropna().str.strip())
        print(f"    Resuming: {len(done_employers)} employers already done, skipping.\n")

    # ── API key ───────────────────────────────────────────────────────────
    api_key = args.api_key or ""
    if not api_key:
        api_key = prompt_api_key()
        if not api_key:
            print("  No key entered — using unauthenticated free tier (rate ~1 req/s).\n")

    # ── HTTP session ──────────────────────────────────────────────────────
    session = requests.Session()
    session.headers.update(HEADERS)

    total      = len(employer_table)
    to_process = total - len(done_employers)
    print(f"\n{'='*65}")
    print(f"Querying OpenCorporates for {to_process} employers …")
    print(f"{'='*65}\n")

    # ── open output CSV ───────────────────────────────────────────────────
    write_mode = "a" if (args.resume and out_path.exists()) else "w"
    with open(out_path, write_mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_mode == "w":
            writer.writeheader()

        done_count = len(done_employers)
        processed = 0
        for _, row in employer_table.iterrows():
            employer = row["employer"].strip()
            if employer in done_employers:
                continue

            processed += 1
            current = done_count + processed
            result = opencorporates_naics_lookup(employer, api_key, session)

            naics_label = f"NAICS: {result['naics_codes']}" if result["naics_found"] == "yes" \
                          else (f"SIC: {result['sic_codes']}" if result["sic_codes"] else "no codes")
            print(
                f"  [{current}/{total}] {employer[:55]:<55} "
                f"oc={result['oc_found']}  {naics_label}"
            )
            sys.stdout.flush()

            writer.writerow({
                "employer":     employer,
                "n_donors":     int(row["n_donors"]),
                "total_donated": round(float(row["total_donated"]), 2),
                **result,
            })
            fh.flush()

            time.sleep(args.delay)

    print(f"\nResults written to:\n  {out_path}\n")
    print_summary(out_path, employer_table)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Look up NAICS classifications for $10K+ donor employers "
            "via the OpenCorporates API."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--api-key", type=str, default="",
        help="OpenCorporates API key (prompted securely if omitted)"
    )
    p.add_argument(
        "--min-amount", type=float, default=DEFAULT_MIN_AMOUNT,
        help=f"Minimum single-donation amount (default: {DEFAULT_MIN_AMOUNT:,.0f})"
    )
    p.add_argument(
        "--max-rows", type=int, default=None,
        help="Limit to N employers — useful for a test run"
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip employers already present in the output CSV"
    )
    p.add_argument(
        "--delay", type=float, default=1.2,
        help="Seconds between HTTP requests (default: 1.2)"
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Override output CSV path"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
