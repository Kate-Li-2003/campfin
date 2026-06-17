"""
calaccess_clean.py
==================
Shared cleaning logic for CalAccess campaign-finance data.

This module is the SINGLE SOURCE OF TRUTH for how raw CalAccess RCPT_CD rows
are classified, normalised, and mapped into the Power Search (MapLight) column
format used by the master CSVs.  It is imported by:

  • pull_calaccess.py        — the weekly pull script (download → clean → append)
  • clean_walkthrough.py     — a marimo notebook that walks through each cleaning
                               step on real data and validates the output against
                               a Power Search export.

Keeping the logic here (rather than inline in the script) means the notebook
genuinely confirms the same code the script runs, instead of a separate copy
that can silently drift.

The functions fall into two groups:

  LOOKUP BUILDERS  (one pass over a whole table → a dict)
    build_committee_lookup_and_races  — FILERNAME_CD → committee names + filer→race
    build_filing_to_filer             — FILER_FILINGS_CD → filing→filer bridge

  ROW-LEVEL CLEANING  (one RCPT_CD row → one Power Search row)
    classify_committee_name           — committee name → GOV/LTG/INS/''
    fmt_date                          — raw CalAccess date → YYYY-MM-DD
    map_row                           — RCPT_CD row → Power Search dict
    dedup_key / dedup_key_series      — stable composite key for deduplication
"""

from __future__ import annotations

import hashlib
import io
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

# Either an in-memory buffer or a path on disk — both are accepted by
# pd.read_csv.  pull_calaccess.py now passes file paths (the tables are
# stream-extracted to disk to avoid holding multi-GB files in memory); the
# marimo walkthrough may pass a BytesIO.
TableSource = "io.BytesIO | str | Path"

# ── COLUMN MAPPING CONSTANTS: RCPT_CD → Power Search format ────────────────────
#
# The master CSV uses Power Search (MapLight) column names.  These maps convert
# the raw CalAccess codes into that format so existing downstream scripts work
# unchanged.

OFFICE_CODE_MAP = {
    "GOV":  "Governor",
    "LTG":  "Lieutenant Governor",
    "ATT":  "Attorney General",
    "SOS":  "Secretary of State",
    "CON":  "Controller",
    "TRE":  "Treasurer",
    "INS":  "Insurance Commissioner",
    "SPI":  "Superintendent of Public Instruction",
    "BOE":  "Board of Equalization",
    "SEN":  "State Senate",
    "ASM":  "State Assembly",
}

# FORM_TYPE in RCPT_CD distinguishes monetary (Schedule A) from
# non-monetary / in-kind (Schedule C) contributions.
FORM_TYPE_MAP = {
    "A":  "Monetary Contribution",
    "C":  "Non-Monetary Contribution",
}

# Columns written to the master CSV (must match the existing file exactly)
MASTER_COLUMNS = [
    "Transaction Type",
    "Cycle",
    "Election",
    "Start Date",
    "End Date",
    "Amount",
    "Recipient Name",
    "Recipient Committee",
    "Recipient Committee ID",
    "Office",
    "District",
    "Ballot Measure(s)",
    "Contributor Name",
    "Contributor ID",
    "Contributor City",
    "Contributor State",
    "Contributor Zip Code",
    "Contributor Employer",
    "Contributor Occupation",
    "Candidate Contribution",
    "Ballot Measure Contribution",
    "Allied Committee",
]

# Only include contributions from the 2025-2026 election cycle onward.
CYCLE_START_DATE = "2025-01-01"


# ── COMMITTEE NAME → RACE CLASSIFICATION ───────────────────────────────────────
#
# RCPT_CD rows almost never have OFFICE_CD populated — the office is associated
# with the *committee* (filer), not the receipt — and FILER_TO_FILER_TYPE_CD in
# the public CalAccess dump does not carry an office code either.  We therefore
# classify candidate-controlled committees from FILERNAME_CD by name pattern.
#
# Every observed candidate-controlled committee name in the master CSVs takes
# the form "<candidate> for [California] <Office> <year>" (case-insensitive).
# Requiring the "for" prefix excludes recall PACs and non-candidate committees
# whose names happen to contain the office word.  Order matters: LTG must be
# checked before GOV because "lieutenant governor" contains "governor".
RACE_NAME_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("LTG", re.compile(
        r"\bfor\s+(?:california\s+)?(?:lieutenant|lt\.?|lieu\.?)[ .]+governor\b",
        re.I,
    )),
    ("INS", re.compile(r"\bfor\s+(?:california\s+)?insurance\s+commissioner\b", re.I)),
    ("GOV", re.compile(r"\bfor\s+(?:california\s+)?governor\b", re.I)),
]


def classify_committee_name(name: str) -> str:
    """Return GOV/LTG/INS if the committee name matches that race, else ''."""
    if not name:
        return ""
    for code, pat in RACE_NAME_PATTERNS:
        if pat.search(name):
            return code
    return ""


def build_committee_lookup_and_races(
    filername_bytes: TableSource,
    seed_filers_per_race: dict[str, set[str]],
    target_races: set[str],
    verbose: bool = True,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Single pass over FILERNAME_CD.TSV.  Returns:
      committee_lookup: {filer_id: committee_name}, covering both FILER_ID and
                        XREF_FILER_ID so joins work either way.
      filer_to_race:    {filer_id: race_code} for every filer whose name
                        identifies it as a target candidate committee, unioned
                        with FILER_IDs already present in master CSVs.

    `target_races` is the set of race codes in scope (e.g. {"GOV", "LTG", "INS"}).
    `seed_filers_per_race` maps each race code to the set of FILER_IDs already in
    its master CSV so legacy / atypically-named committees stay in scope.
    """
    df = pd.read_csv(
        filername_bytes,
        sep="\t",
        dtype=str,
        encoding="latin-1",
        on_bad_lines="skip",
    )
    df.columns = [c.strip().upper() for c in df.columns]
    df["NAML"] = df["NAML"].fillna("").str.strip()
    df["NAMF"] = df.get("NAMF", pd.Series("", index=df.index)).fillna("").str.strip()
    df["full_name"] = df["NAML"]

    # Pre-classify each row's name once.  apply() preserves Python str returns
    # (map() coerces empty/None values to NaN, which fails membership checks).
    df["RACE"] = df["full_name"].apply(classify_committee_name).fillna("")

    committee_lookup: dict[str, str] = {}
    filer_to_race:    dict[str, str] = {}

    for col in ("FILER_ID", "XREF_FILER_ID"):
        if col not in df.columns:
            continue
        sub = df[df[col].notna()]
        for fid_raw, name, race in zip(sub[col], sub["full_name"], sub["RACE"]):
            fid = str(fid_raw).strip()
            if not fid:
                continue
            if fid not in committee_lookup:
                committee_lookup[fid] = name
            if race in target_races and fid not in filer_to_race:
                filer_to_race[fid] = race

    # Seed with FILER_IDs already in the master CSVs so legacy / atypically-
    # named committees stay in scope.
    seeded = 0
    for race, fids in seed_filers_per_race.items():
        for fid in fids:
            if fid and fid not in filer_to_race:
                filer_to_race[fid] = race
                seeded += 1

    if verbose:
        print(f"  Committee lookup built: {len(committee_lookup):,} entries.")
        by_race = Counter(filer_to_race.values())
        race_summary = ", ".join(
            f"{c}={by_race.get(c, 0):,}" for c in sorted(target_races)
        )
        print(
            f"  Filer→race lookup built: {len(filer_to_race):,} entries "
            f"({seeded:,} seeded from existing master CSVs)  [{race_summary}]"
        )
    return committee_lookup, filer_to_race


def build_filing_to_filer(
    filings_bytes: TableSource,
    target_filers: set[str],
    verbose: bool = True,
) -> dict[str, str]:
    """
    Build {FILING_ID: FILER_ID} for filings whose FILER_ID is in target_filers,
    excluding amendments superseded by a later sequence.

    A semi-annual filing (e.g. F460) gets a NEW FILING_ID each time it is
    amended, with FILING_SEQUENCE incrementing from 0.  Both the original and
    every amendment exist in RCPT_CD with overlapping receipts; if we keep them
    all we'd quadruple-count contributions.  Within each
    (FILER_ID, FORM_ID, PERIOD_ID) group we therefore keep only the rows sharing
    the maximum FILING_SEQUENCE — that drops superseded originals while still
    allowing event-driven forms like F496/F497 (which are filed many times per
    period at sequence 0) to keep all of their filings.

    Restricting to target filers also keeps the in-memory dict small
    (a few hundred thousand entries instead of ~2.6M for the full table).
    """
    df = pd.read_csv(
        filings_bytes, sep="\t", dtype=str, encoding="latin-1",
        on_bad_lines="skip",
        usecols=["FILER_ID", "FILING_ID", "FORM_ID", "PERIOD_ID", "FILING_SEQUENCE"],
    )
    df.columns = [c.strip().upper() for c in df.columns]
    df["FILER_ID"]  = df["FILER_ID"].fillna("").str.strip()
    df["FILING_ID"] = df["FILING_ID"].fillna("").str.strip()
    df["FORM_ID"]   = df["FORM_ID"].fillna("").str.strip().str.upper()
    df["PERIOD_ID"] = df["PERIOD_ID"].fillna("").str.strip()
    df["FILING_SEQUENCE"] = pd.to_numeric(
        df["FILING_SEQUENCE"], errors="coerce",
    ).fillna(0).astype(int)

    sub = df[df["FILER_ID"].isin(target_filers) & (df["FILING_ID"] != "")].copy()
    before = len(sub)

    # Keep only rows tied for the max sequence within each (filer, form, period).
    seq_max = sub.groupby(
        ["FILER_ID", "FORM_ID", "PERIOD_ID"],
    )["FILING_SEQUENCE"].transform("max")
    sub = sub[sub["FILING_SEQUENCE"] == seq_max]

    out = dict(zip(sub["FILING_ID"], sub["FILER_ID"]))
    if verbose:
        print(
            f"  Filing→filer bridge built: {len(out):,} filings spanning "
            f"{sub['FILER_ID'].nunique():,} target filers "
            f"({before - len(sub):,} superseded amendments dropped)."
        )
    return out


# ── ROW-LEVEL CLEANING ─────────────────────────────────────────────────────────

def fmt_date(val: str) -> str:
    """Normalise a CalAccess date to YYYY-MM-DD; return '' on failure.

    CalAccess raw dates come in as 'M/D/YYYY HH:MM:SS AM/PM'.  We split off the
    time component before parsing so we don't have to enumerate every
    permutation, and so the result matches the ISO-formatted dates in the
    existing master CSV (otherwise the dedup keys never align)."""
    if not isinstance(val, str) or not val.strip():
        return ""
    v = val.strip().split(" ")[0]  # drop the trailing 'HH:MM:SS AM/PM' if any
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(v, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return v


def map_row(row: pd.Series, committee_lookup: dict, filer_id: str, race: str) -> dict:
    """
    Convert one RCPT_CD row to the Power Search column format.

    `filer_id` is the recipient committee's FILER_ID, resolved upstream by
    bridging RCPT_CD.FILING_ID through FILER_FILINGS_CD (RCPT_CD itself does not
    carry FILER_ID).  `race` is the GOV/LTG/INS code that filer maps to.

    Cleaning steps applied
    ──────────────────────
    1.  Date normalisation  — RCPT_DATE / DATE_THRU converted from raw CalAccess
        formats (MM/DD/YYYY, YYYYMMDD, YYYY-MM-DD) to YYYY-MM-DD.  End Date falls
        back to Start Date when DATE_THRU is blank.

    2.  Cycle             — 4-digit year extracted from Start Date.

    3.  Transaction Type  — derived from FORM_TYPE:
          'A' → 'Monetary Contribution'
          'C' → 'Non-Monetary Contribution'   (in-kind / non-cash)
          other → 'Monetary Contribution' (safe default)

    4.  Recipient Name    — CAND_NAML + CAND_NAMF joined as "LAST, FIRST" and
        uppercased to match the existing file's convention.

    5.  Contributor Name  — CTRIB_NAML + CTRIB_NAMF joined as "Last, First"
        (title-case as it appears in the raw data).

    6.  Recipient Committee — looked up from FILERNAME_CD via FILER_ID (the
        recipient committee that filed the receipt), covering both FILER_ID and
        XREF_FILER_ID cross-references.

    7.  Amount            — converted to float, then formatted as a plain number
        string: trailing '.00' stripped (e.g. '25000' not '25000.00'),
        meaningful decimals preserved (e.g. '258.88').

    8.  Allied Committee  — Y/N flag: 'Y' if an intermediary (INTR_NAML) is
        present, 'N' otherwise.  Matches the boolean convention in the existing
        file (not the intermediary name string).

    9.  Candidate / Ballot Measure flags — Y/N derived from presence of
        CAND_NAML and BAL_NAME respectively.

    10. Whitespace stripping — all string fields stripped of leading/trailing
        whitespace inherited from the raw TSV.
    """
    rcpt_date = fmt_date(str(row.get("RCPT_DATE", "")))
    end_date  = fmt_date(str(row.get("DATE_THRU", ""))) or rcpt_date
    cycle     = rcpt_date[:4] if rcpt_date else ""

    committee = committee_lookup.get(filer_id, "")

    # Recipient name — uppercase to match Power Search convention
    cand_last  = str(row.get("CAND_NAML", "") or "").strip().upper()
    cand_first = str(row.get("CAND_NAMF", "") or "").strip().upper()
    recipient_name = f"{cand_last}, {cand_first}" if cand_first else cand_last

    # Contributor name — preserve original casing from raw data
    ctrib_last  = str(row.get("CTRIB_NAML", "") or "").strip()
    ctrib_first = str(row.get("CTRIB_NAMF", "") or "").strip()
    contrib_name = f"{ctrib_last}, {ctrib_first}" if ctrib_first else ctrib_last

    office_str = OFFICE_CODE_MAP.get(race, race)

    # Transaction type from FORM_TYPE (Schedule A = monetary, C = non-monetary)
    form_type = str(row.get("FORM_TYPE", "") or "").strip().upper()
    tran_type = FORM_TYPE_MAP.get(form_type, "Monetary Contribution")

    bal_name     = str(row.get("BAL_NAME", "") or "").strip()
    is_ballot    = "Y" if bal_name else "N"
    is_candidate = "Y" if cand_last else "N"

    # Allied Committee: Y/N flag — Y if an intermediary committee is present
    allied = "Y" if str(row.get("INTR_NAML", "") or "").strip() else "N"

    # Amount: strip trailing .00 but keep meaningful decimals
    try:
        amt_f  = float(row.get("AMOUNT", 0) or 0)
        amount = f"{amt_f:.2f}"
        if amount.endswith(".00"):
            amount = amount[:-3]
    except (ValueError, TypeError):
        amount = str(row.get("AMOUNT", ""))

    return {
        "Transaction Type":            tran_type,
        "Cycle":                       cycle,
        "Election":                    "0000-00-00",
        "Start Date":                  rcpt_date,
        "End Date":                    end_date,
        "Amount":                      amount,
        "Recipient Name":              recipient_name,
        "Recipient Committee":         committee,
        "Recipient Committee ID":      filer_id,
        "Office":                      office_str,
        "District":                    str(row.get("DIST_NO", "") or "").strip(),
        "Ballot Measure(s)":           bal_name,
        "Contributor Name":            contrib_name,
        "Contributor ID":              "",
        "Contributor City":            str(row.get("CTRIB_CITY", "") or "").strip(),
        "Contributor State":           str(row.get("CTRIB_ST", "") or "").strip(),
        "Contributor Zip Code":        str(row.get("CTRIB_ZIP4", "") or "").strip(),
        "Contributor Employer":        str(row.get("CTRIB_EMP", "") or "").strip(),
        "Contributor Occupation":      str(row.get("CTRIB_OCC", "") or "").strip(),
        "Candidate Contribution":      is_candidate,
        "Ballot Measure Contribution": is_ballot,
        "Allied Committee":            allied,
    }


# ── DEDUPLICATION KEY ───────────────────────────────────────────────────────────

def dedup_key(row: dict) -> str:
    """
    Stable composite key for a contribution row (Power Search format).
    Hashed to a short hex string to keep the in-memory set compact.
    """
    parts = "|".join([
        str(row.get("Start Date", "")),
        str(row.get("Amount", "")),
        str(row.get("Contributor Name", "")).upper().strip(),
        str(row.get("Recipient Committee ID", "")),
        str(row.get("Transaction Type", "")),
    ])
    return hashlib.md5(parts.encode()).hexdigest()


def dedup_key_series(df: pd.DataFrame) -> pd.Series:
    """Vectorised dedup key for a whole DataFrame."""
    return (
        df.get("Start Date", pd.Series("", index=df.index)).fillna("").astype(str)
        + "|"
        + df.get("Amount", pd.Series("", index=df.index)).fillna("").astype(str)
        + "|"
        + df.get("Contributor Name", pd.Series("", index=df.index)).fillna("").str.upper().str.strip()
        + "|"
        + df.get("Recipient Committee ID", pd.Series("", index=df.index)).fillna("").astype(str)
        + "|"
        + df.get("Transaction Type", pd.Series("", index=df.index)).fillna("").astype(str)
    ).apply(lambda s: hashlib.md5(s.encode()).hexdigest())
