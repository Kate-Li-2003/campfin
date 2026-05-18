"""
pull_calaccess.py
=================
Weekly pull of CA Governor's race campaign contributions from the CalAccess
raw data dump (updated daily by the CA Secretary of State).

HOW IT WORKS
────────────
1. Downloads dbwebexport.zip from campaignfinance.cdn.sos.ca.gov.
2. Stream-extracts three tables — no others are written to disk:
     • RCPT_CD.TSV          (receipts; only carries FILING_ID, not FILER_ID)
     • FILERNAME_CD.TSV     (committee names per filer)
     • FILER_FILINGS_CD.TSV (FILING_ID → FILER_ID bridge)
3. Builds a {FILER_ID: race_code} lookup by classifying each FILERNAME_CD
   row by committee name pattern (GOV / LTG / INS), seeded with FILER_IDs
   already in the master CSVs so legacy / atypically-named committees
   stay in scope.  (RCPT_CD.OFFICE_CD is essentially never populated, and
   the public CalAccess dump has no single table that maps a recipient
   committee to its office, so the committee name is the most reliable
   signal.)
4. Builds a {FILING_ID: FILER_ID} bridge from FILER_FILINGS_CD restricted
   to filings whose filer is in the lookup above.
5. Filters RCPT_CD for FILING_ID ∈ bridge AND RCPT_DATE >= 2025-01-01,
   resolving each receipt's recipient committee via the bridge.
6. Applies data cleaning to map raw CalAccess columns → the same column
   format used by the existing master CSVs so downstream scripts work
   unchanged.  See map_row() docstring for the full list of cleaning
   steps.
7. Deduplicates against the master CSV using a composite key
   (RCPT_DATE + AMOUNT + contributor name + FILER_ID + transaction type).
8. Appends only new rows to the master CSV.
9. Saves run metadata in .pull_state.json (last run date, row counts).

SCHEDULING (run every Monday at 6 AM)
──────────────────────────────────────
Add to crontab via `crontab -e`:

  0 6 * * 1  cd "/Users/kateli/Desktop/CalMatters/campaign finance categorization" && \
             /Users/kateli/etf_venv/bin/python code/04_donations_data_pull/pull_calaccess.py \
             >> code/04_donations_data_pull/pull.log 2>&1

USAGE
─────
  python code/04_donations_data_pull/pull_calaccess.py           # normal run
  python code/04_donations_data_pull/pull_calaccess.py --dry-run # show counts, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tempfile
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ── PATHS ─────────────────────────────────────────────────────────────────────

# Project root is two levels above this file:
#   code/04_donations_data_pull/pull_calaccess.py → code/ → project root.
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = BASE_DIR / "data"
STATE_FILE  = Path(__file__).resolve().parent / ".pull_state.json"
LOG_SEP     = "─" * 65

CALACCESS_ZIP_URL = "https://campaignfinance.cdn.sos.ca.gov/dbwebexport.zip"

# Races to extract — office code → output CSV path.
# All use the same format, date filter, and dedup logic.
RACES: dict[str, Path] = {
    "GOV": DATA_DIR / "01CalAccess_CampaignFinance_Data" / "governor_race_2026-04-27.csv",
    "LTG": DATA_DIR / "01CalAccess_CampaignFinance_Data" / "lt_governor_race_2026.csv",
    "INS": DATA_DIR / "01CalAccess_CampaignFinance_Data" / "insurance_commissioner_race_2026.csv",
}

# Chunk size for reading the large RCPT_CD.TSV (rows per batch)
CHUNK_SIZE = 100_000

# Only include contributions from the 2025-2026 election cycle onward
CYCLE_START_DATE = "2025-01-01"

# ── COLUMN MAPPING: RCPT_CD → Power Search format ─────────────────────────────
#
# The master CSV uses Power Search (MapLight) column names.  This mapping
# converts the raw CalAccess columns into that format so existing downstream
# scripts work unchanged.

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


# ── STATE ─────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"last_run": None, "master_rows": 0, "runs": []}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── DOWNLOAD & EXTRACT ────────────────────────────────────────────────────────

def download_zip(url: str) -> Path:
    """
    Download the CalAccess ZIP to a temp file with a progress indicator.
    Returns the path to the temp file.
    """
    print(f"Downloading CalAccess database from:\n  {url}")
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp_path = Path(tmp.name)

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    start = time.time()

    with open(tmp_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            if chunk:
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct  = 100 * downloaded / total
                    mb   = downloaded / 1e6
                    elapsed = time.time() - start
                    speed   = mb / elapsed if elapsed > 0 else 0
                    print(
                        f"\r  {pct:5.1f}%  {mb:6.1f} MB  ({speed:.1f} MB/s)",
                        end="", flush=True,
                    )

    elapsed = time.time() - start
    print(f"\r  Done.  {downloaded/1e6:.1f} MB downloaded in {elapsed:.0f}s.       ")
    return tmp_path


def extract_tables(zip_path: Path) -> tuple[io.BytesIO, io.BytesIO, io.BytesIO]:
    """
    Open the ZIP and return in-memory BytesIO objects for RCPT_CD.TSV,
    FILERNAME_CD.TSV, and FILER_FILINGS_CD.TSV without extracting other
    files.  FILER_FILINGS_CD is needed to map RCPT_CD.FILING_ID →
    recipient-committee FILER_ID (RCPT_CD itself does not carry FILER_ID).
    """
    print("Extracting RCPT_CD.TSV, FILERNAME_CD.TSV, FILER_FILINGS_CD.TSV …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        def _find(target: str) -> Optional[str]:
            # Handle possible subdirectory prefix inside the ZIP
            for n in names:
                if n.upper().endswith(target.upper()):
                    return n
            return None

        rcpt_name      = _find("RCPT_CD.TSV")
        filername_name = _find("FILERNAME_CD.TSV")
        filings_name   = _find("FILER_FILINGS_CD.TSV")

        if rcpt_name is None:
            raise FileNotFoundError("RCPT_CD.TSV not found in ZIP.")
        if filername_name is None:
            raise FileNotFoundError("FILERNAME_CD.TSV not found in ZIP.")
        if filings_name is None:
            raise FileNotFoundError("FILER_FILINGS_CD.TSV not found in ZIP.")

        rcpt_bytes     = io.BytesIO(zf.read(rcpt_name))
        filername_bytes = io.BytesIO(zf.read(filername_name))
        filings_bytes   = io.BytesIO(zf.read(filings_name))

    print("  Extraction complete.")
    return rcpt_bytes, filername_bytes, filings_bytes


def build_filing_to_filer(
    filings_bytes: io.BytesIO,
    target_filers: set[str],
) -> dict[str, str]:
    """
    Build {FILING_ID: FILER_ID} for filings whose FILER_ID is in target_filers,
    excluding amendments superseded by a later sequence.

    A semi-annual filing (e.g. F460) gets a NEW FILING_ID each time it is
    amended, with FILING_SEQUENCE incrementing from 0.  Both the original and
    every amendment exist in RCPT_CD with overlapping receipts; if we keep
    them all we'd quadruple-count contributions.  Within each
    (FILER_ID, FORM_ID, PERIOD_ID) group we therefore keep only the rows
    sharing the maximum FILING_SEQUENCE — that drops superseded originals
    while still allowing event-driven forms like F496/F497 (which are filed
    many times per period at sequence 0) to keep all of their filings.

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
    print(
        f"  Filing→filer bridge built: {len(out):,} filings spanning "
        f"{sub['FILER_ID'].nunique():,} target filers "
        f"({before - len(sub):,} superseded amendments dropped)."
    )
    return out


# ── COMMITTEE NAME LOOKUP + FILER → RACE CLASSIFICATION ─────────────────────
#
# RCPT_CD rows almost never have OFFICE_CD populated — the office is associated
# with the *committee* (filer), not the receipt — and FILER_TO_FILER_TYPE_CD
# in the public CalAccess dump does not carry an office code either.  Power
# Search builds its filer→office mapping from form-410 / form-501 records that
# aren't all reliably present in this dump, so we classify candidate-controlled
# committees from FILERNAME_CD by name pattern and seed with FILER_IDs already
# in the master CSVs.  Receipt-date filtering downstream excludes anything that
# slips through from prior cycles.

# Match the candidate-committee naming convention used in CalAccess: every
# observed candidate-controlled committee name in the master CSVs takes the
# form "<candidate> for [California] <Office> <year>" (case-insensitive).
# Requiring the "for" prefix excludes recall PACs and non-candidate committees
# whose names happen to contain the office word.  Order matters: LTG must be
# checked before GOV because "lieutenant governor" contains "governor".
_RACE_NAME_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("LTG", re.compile(
        r"\bfor\s+(?:california\s+)?(?:lieutenant|lt\.?|lieu\.?)[ .]+governor\b",
        re.I,
    )),
    ("INS", re.compile(r"\bfor\s+(?:california\s+)?insurance\s+commissioner\b", re.I)),
    ("GOV", re.compile(r"\bfor\s+(?:california\s+)?governor\b", re.I)),
]


def _classify_committee_name(name: str) -> str:
    """Return GOV/LTG/INS if the committee name matches that race, else ''."""
    if not name:
        return ""
    for code, pat in _RACE_NAME_PATTERNS:
        if pat.search(name):
            return code
    return ""


def build_committee_lookup_and_races(
    filername_bytes: io.BytesIO,
    seed_filers_per_race: dict[str, set[str]],
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Single pass over FILERNAME_CD.TSV.  Returns:
      committee_lookup: {filer_id: committee_name}, covering both FILER_ID and
                        XREF_FILER_ID so joins work either way.
      filer_to_race:    {filer_id: race_code} for every filer whose name
                        identifies it as a GOV / LTG / INS candidate committee,
                        unioned with FILER_IDs already present in master CSVs.
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
    df["RACE"] = df["full_name"].apply(_classify_committee_name).fillna("")

    target_races = set(RACES.keys())
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

    print(f"  Committee lookup built: {len(committee_lookup):,} entries.")
    by_race = Counter(filer_to_race.values())
    race_summary = ", ".join(
        f"{c}={by_race.get(c, 0):,}" for c in sorted(RACES)
    )
    print(
        f"  Filer→race lookup built: {len(filer_to_race):,} entries "
        f"({seeded:,} seeded from existing master CSVs)  [{race_summary}]"
    )
    return committee_lookup, filer_to_race


# ── ROW MAPPING ──────────────────────────────────────────────────────────────

def _fmt_date(val: str) -> str:
    """Normalise RCPT_DATE to YYYY-MM-DD; return '' on failure.

    CalAccess raw dates come in as 'M/D/YYYY HH:MM:SS AM/PM'.  We split off
    the time component before parsing so we don't have to enumerate every
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
    bridging RCPT_CD.FILING_ID through FILER_FILINGS_CD (RCPT_CD itself does
    not carry FILER_ID).  `race` is the GOV/LTG/INS code that filer maps to.

    Cleaning steps applied
    ──────────────────────
    1.  Date normalisation  — RCPT_DATE / DATE_THRU converted from raw
        CalAccess formats (MM/DD/YYYY, YYYYMMDD, YYYY-MM-DD) to YYYY-MM-DD.
        End Date falls back to Start Date when DATE_THRU is blank.

    2.  Cycle             — 4-digit year extracted from Start Date.

    3.  Transaction Type  — derived from FORM_TYPE:
          'A' → 'Monetary Contribution'
          'C' → 'Non-Monetary Contribution'   (in-kind / non-cash)
          other → 'Monetary Contribution' (safe default)

    4.  Recipient Name    — CAND_NAML + CAND_NAMF joined as "LAST, FIRST"
        and uppercased to match the existing file's convention.

    5.  Contributor Name  — CTRIB_NAML + CTRIB_NAMF joined as "Last, First"
        (title-case as it appears in the raw data).

    6.  Recipient Committee — looked up from FILERNAME_CD via FILER_ID
        (the recipient committee that filed the receipt), covering both
        FILER_ID and XREF_FILER_ID cross-references.

    7.  Amount            — converted to float, then formatted as a plain
        number string: trailing '.00' stripped (e.g. '25000' not '25000.00'),
        meaningful decimals preserved (e.g. '258.88').

    8.  Allied Committee  — Y/N flag: 'Y' if an intermediary (INTR_NAML) is
        present, 'N' otherwise.  Matches the boolean convention in the
        existing file (not the intermediary name string).

    9.  Candidate / Ballot Measure flags — Y/N derived from presence of
        CAND_NAML and BAL_NAME respectively.

    10. Whitespace stripping — all string fields stripped of leading/trailing
        whitespace inherited from the raw TSV.
    """
    rcpt_date = _fmt_date(str(row.get("RCPT_DATE", "")))
    end_date  = _fmt_date(str(row.get("DATE_THRU", ""))) or rcpt_date
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


# ── DEDUPLICATION KEY ─────────────────────────────────────────────────────────

def _dedup_key(row: dict) -> str:
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


def _dedup_key_series(df: pd.DataFrame) -> pd.Series:
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


# ── PROCESS RCPT_CD ───────────────────────────────────────────────────────────

def process_rcpt(
    rcpt_bytes: io.BytesIO,
    committee_lookup: dict,
    filer_to_race: dict[str, str],
    filing_to_filer: dict[str, str],
    existing_keys: dict[str, set],
) -> dict[str, list[dict]]:
    """
    Read RCPT_CD.TSV in chunks.  For each row, resolve FILING_ID →
    recipient FILER_ID via filing_to_filer; a row is kept iff that
    FILER_ID is in filer_to_race AND RCPT_DATE >= CYCLE_START_DATE.
    """
    target_races = set(RACES.keys())
    print(
        f"Scanning RCPT_CD.TSV for races: {', '.join(sorted(target_races))} "
        f"({len(filer_to_race):,} target filers, "
        f"{len(filing_to_filer):,} target filings) …"
    )

    new_rows: dict[str, list[dict]] = {code: [] for code in target_races}
    total_scanned = 0
    total_matched = 0

    reader = pd.read_csv(
        rcpt_bytes,
        sep="\t",
        dtype=str,
        encoding="latin-1",
        on_bad_lines="skip",
        chunksize=CHUNK_SIZE,
    )

    for chunk_num, chunk in enumerate(reader, 1):
        chunk.columns = [c.strip().upper() for c in chunk.columns]
        total_scanned += len(chunk)

        if "FILING_ID" not in chunk.columns:
            continue

        filing_ids = chunk["FILING_ID"].fillna("").astype(str).str.strip()

        # Resolve recipient FILER_ID via the filings bridge.  Rows whose
        # filing isn't on a target candidate committee map to NaN.
        chunk["_FILER_ID"] = filing_ids.map(filing_to_filer)
        race_mask = chunk["_FILER_ID"].notna()

        # 2025-2026 cycle only
        dates = pd.to_datetime(chunk["RCPT_DATE"], errors="coerce", format="mixed")
        cycle_mask = dates >= CYCLE_START_DATE

        matched = chunk[race_mask & cycle_mask]
        total_matched += len(matched)

        for _, row in matched.iterrows():
            fid    = str(row["_FILER_ID"]).strip()
            race   = filer_to_race.get(fid, "")
            if race not in target_races:
                continue
            mapped = map_row(row, committee_lookup, filer_id=fid, race=race)
            key    = _dedup_key(mapped)
            if key not in existing_keys[race]:
                new_rows[race].append(mapped)
                existing_keys[race].add(key)

        print(
            f"\r  Chunk {chunk_num}: scanned {total_scanned:>9,} rows | "
            f"matched: {total_matched:>6,} | "
            + "  ".join(f"{c}={len(new_rows[c])}" for c in sorted(target_races)),
            end="", flush=True,
        )

    print(f"\n  Scan complete. {total_scanned:,} total rows scanned.")
    return new_rows


# ── EXISTING MASTER CSV ───────────────────────────────────────────────────────

def load_all_existing_keys() -> tuple[
    dict[str, set],
    dict[str, int],
    dict[str, set[str]],
]:
    """
    For every race in RACES, load dedup keys, row counts, and the set of
    Recipient Committee IDs already present in its CSV.  The third return
    value is used to seed the filer→race lookup so legacy committees remain
    in scope even if FILER_TO_FILER_TYPE_CD doesn't currently classify them.
    """
    all_keys:    dict[str, set]      = {}
    all_counts:  dict[str, int]      = {}
    all_filers:  dict[str, set[str]] = {}
    for code, path in RACES.items():
        if not path.exists():
            print(f"  [{code}] No existing file — will create: {path.name}")
            all_keys[code]   = set()
            all_counts[code] = 0
            all_filers[code] = set()
        else:
            df = pd.read_csv(path, dtype=str, low_memory=False)
            keys = set(_dedup_key_series(df).tolist())
            filers = set(
                df.get("Recipient Committee ID", pd.Series(dtype=str))
                  .fillna("").astype(str).str.strip()
            )
            filers.discard("")
            print(
                f"  [{code}] {len(df):,} existing rows | {len(keys):,} dedup keys "
                f"| {len(filers):,} committee IDs  ({path.name})"
            )
            all_keys[code]   = keys
            all_counts[code] = len(df)
            all_filers[code] = filers
    return all_keys, all_counts, all_filers


# ── APPEND NEW ROWS ───────────────────────────────────────────────────────────

def append_all_new_rows(
    new_rows: dict[str, list[dict]],
    dry_run: bool,
) -> None:
    """Append new rows to each race's CSV; write header if the file is new."""
    for code, rows in new_rows.items():
        path = RACES[code]
        if not rows:
            print(f"  [{code}] No new rows to append.")
            continue
        new_df = pd.DataFrame(rows, columns=MASTER_COLUMNS)
        if dry_run:
            print(f"  [{code}] [DRY RUN] Would append {len(new_df):,} rows to {path.name}.")
            print(new_df.head(3).to_string(index=False))
            continue
        write_header = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_csv(path, mode="a", header=write_header, index=False)
        print(f"  [{code}] Appended {len(new_df):,} new rows → {path.name}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    state = load_state()

    print(LOG_SEP)
    print(f"CalAccess Multi-Race Pull  —  {datetime.now():%Y-%m-%d %H:%M:%S}")
    for code, path in RACES.items():
        print(f"  {code}: {path.name}")
    print(LOG_SEP)

    # 1. Load existing dedup keys, row counts, and committee IDs for all races
    print("Loading existing CSV files …")
    existing_keys, existing_counts, existing_filers = load_all_existing_keys()

    # 2. Download ZIP
    zip_path = download_zip(CALACCESS_ZIP_URL)

    try:
        # 3. Stream-extract the tables we need
        rcpt_bytes, filername_bytes, filings_bytes = extract_tables(zip_path)

        # 4. One pass over FILERNAME_CD: build both the filer-id → committee
        #    name lookup and the filer-id → race lookup (latter classified by
        #    name pattern, seeded with existing master CSV filer IDs).
        committee_lookup, filer_to_race = build_committee_lookup_and_races(
            filername_bytes, existing_filers,
        )

        # 5. Bridge FILING_ID → FILER_ID via FILER_FILINGS_CD.  RCPT_CD only
        #    carries FILING_ID, so this is how we resolve each receipt to its
        #    recipient committee.  Restricted to target filers to keep memory
        #    small.
        filing_to_filer = build_filing_to_filer(
            filings_bytes, set(filer_to_race),
        )

        # 6. Process RCPT_CD — filter, map, deduplicate across all races
        new_rows = process_rcpt(
            rcpt_bytes, committee_lookup, filer_to_race,
            filing_to_filer, existing_keys,
        )

    finally:
        zip_path.unlink(missing_ok=True)
        print("  Temp ZIP deleted.")

    # 6. Append new rows to each race's CSV
    print(LOG_SEP)
    append_all_new_rows(new_rows, dry_run=args.dry_run)

    # 7. Update state file
    if not args.dry_run:
        now_str = datetime.now(timezone.utc).isoformat()
        state["last_run"] = now_str
        state.setdefault("runs", []).append({
            "timestamp": now_str,
            "new_rows":  {c: len(rows) for c, rows in new_rows.items()},
        })
        save_state(state)

    # 8. Summary
    print(LOG_SEP)
    print(f"  {'Race':<6}  {'Previous':>10}  {'New':>8}  {'Total':>10}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*10}")
    for code in sorted(RACES):
        prev  = existing_counts[code]
        added = len(new_rows[code])
        print(f"  {code:<6}  {prev:>10,}  {added:>8,}  {prev+added:>10,}")
    print(LOG_SEP)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Pull 2026 CA Governor, Lt. Governor, and Insurance Commissioner "
            "contributions from CalAccess raw data and append new rows to each "
            "race's CSV."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Download and process data but do not write anything"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
