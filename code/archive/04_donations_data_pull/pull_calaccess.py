"""
pull_calaccess.py
=================
Weekly pull of CA Governor's race (plus Lt. Governor and Insurance
Commissioner) campaign contributions from the CalAccess raw data dump
(updated daily by the CA Secretary of State).

This is a plain Python script — run it directly from the terminal:

    python code/04_donations_data_pull/pull_calaccess.py             # prompts you to pick race(s)
    python code/04_donations_data_pull/pull_calaccess.py --races GOV # pull just one race (no prompt)
    python code/04_donations_data_pull/pull_calaccess.py --races all # pull every race (no prompt)
    python code/04_donations_data_pull/pull_calaccess.py --dry-run   # show counts, write nothing

Run with no --races flag and a terminal attached and the script prints a
numbered menu (GOV / LTG / INS) and pulls only the race(s) you choose.

The data-cleaning logic (committee→race classification, date normalisation,
row mapping, dedup keys) lives in calaccess_clean.py so the same code can be
imported and confirmed in clean_governor_export.py. This script owns only the
download / extract / filter / append orchestration.

HOW IT WORKS
────────────
1. Downloads dbwebexport.zip from campaignfinance.cdn.sos.ca.gov (streamed to a
   temp file on disk, not held in memory).
2. Stream-extracts just three tables to temp files on disk — copied in bounded
   chunks so the multi-GB RCPT_CD is never loaded whole into RAM, then read
   back in chunks.  No other ZIP members are written.
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
6. Applies data cleaning (calaccess_clean.map_row) to map raw CalAccess
   columns → the same column format used by the existing master CSVs so
   downstream scripts work unchanged.
7. Deduplicates against the master CSV using a composite key
   (RCPT_DATE + AMOUNT + contributor name + FILER_ID + transaction type).
8. Appends only new rows to the master CSV.
9. Saves run metadata in .pull_state.json (last run date, row counts).

SCHEDULING (run every Monday at 6 AM)
──────────────────────────────────────
Add to crontab via `crontab -e` (point at a Python with pandas + requests):

A cron job has no terminal, so pass --races explicitly (the script otherwise
defaults to all races when no terminal is detected):

  0 6 * * 1  cd "/Users/kateli/Desktop/CalMatters/campaign finance categorization" && \
             .venv/bin/python code/04_donations_data_pull/pull_calaccess.py --races all \
             >> code/04_donations_data_pull/pull.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Cleaning logic lives in calaccess_clean.py (same directory) so the
# clean_governor_export.py notebook can import and confirm the exact same code.
# The script's directory is on sys.path[0] when run directly, so a plain
# `import` of the sibling module works from any working directory.
from calaccess_clean import (
    CYCLE_START_DATE,
    MASTER_COLUMNS,
    build_committee_lookup_and_races,
    build_filing_to_filer,
    dedup_key,
    dedup_key_series,
    map_row,
)

# ── PATHS ─────────────────────────────────────────────────────────────────────

# Project root is two levels above this file:
#   code/04_donations_data_pull/pull_calaccess.py → code/ → project root.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
STATE_FILE = Path(__file__).resolve().parent / ".pull_state.json"
LOG_SEP = "─" * 65

CALACCESS_ZIP_URL = "https://campaignfinance.cdn.sos.ca.gov/dbwebexport.zip"

# Full catalog of races this script knows how to pull — office code → output
# CSV path.  All use the same format, date filter, and dedup logic.
ALL_RACES: dict[str, Path] = {
    "GOV": DATA_DIR / "01CalAccess_CampaignFinance_Data" / "governor_race_2026.csv",
    "LTG": DATA_DIR / "01CalAccess_CampaignFinance_Data" / "lt_governor_race_2026.csv",
    "INS": DATA_DIR / "01CalAccess_CampaignFinance_Data" / "insurance_commissioner_race_2026.csv",
}

# Human-readable label for each office code, shown in the selection menu.
RACE_LABELS: dict[str, str] = {
    "GOV": "Governor",
    "LTG": "Lieutenant Governor",
    "INS": "Insurance Commissioner",
}

# Friendly aliases accepted on input (in addition to the canonical codes
# above) so users can type the abbreviations they expect, e.g. "IC".
RACE_ALIASES: dict[str, str] = {
    "GOVERNOR": "GOV",
    "LT": "LTG",
    "LTGOV": "LTG",
    "IC": "INS",
    "INSURANCE": "INS",
}

# The races actually processed this run.  Populated at startup from the user's
# selection (interactive menu, --races flag, or the non-interactive default).
# All downstream functions read this dict, so narrowing it here scopes the
# entire pull to just the chosen races.
RACES: dict[str, Path] = {}

# Chunk size for reading the large RCPT_CD.TSV (rows per batch).  Kept modest so
# peak memory stays low — RCPT_CD has millions of rows and reading it all at once
# can exhaust RAM.
CHUNK_SIZE = 50_000

# Only these RCPT_CD columns are read.  Everything map_row() and the date filter
# need is here; skipping the other ~25 columns cuts per-chunk memory and parse
# time substantially.  Matched against the normalised (stripped/upper-cased)
# header so it works regardless of the raw header's casing or whitespace.
RCPT_NEEDED_COLS = {
    "FILING_ID", "RCPT_DATE", "DATE_THRU", "FORM_TYPE",
    "CAND_NAML", "CAND_NAMF", "CTRIB_NAML", "CTRIB_NAMF",
    "AMOUNT", "BAL_NAME", "INTR_NAML", "DIST_NO",
    "CTRIB_CITY", "CTRIB_ST", "CTRIB_ZIP4", "CTRIB_EMP", "CTRIB_OCC",
}


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


# ── DOWNLOAD & EXTRACT ──────────────────────────────────────────────────────────

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
                    pct = 100 * downloaded / total
                    mb = downloaded / 1e6
                    elapsed = time.time() - start
                    speed = mb / elapsed if elapsed > 0 else 0
                    print(
                        f"\r  {pct:5.1f}%  {mb:6.1f} MB  ({speed:.1f} MB/s)",
                        end="", flush=True,
                    )

    elapsed = time.time() - start
    print(f"\r  Done.  {downloaded/1e6:.1f} MB downloaded in {elapsed:.0f}s.       ")
    return tmp_path


def extract_tables(zip_path: Path, dest_dir: Path) -> tuple[Path, Path, Path]:
    """
    Stream-extract RCPT_CD.TSV, FILERNAME_CD.TSV, and FILER_FILINGS_CD.TSV from
    the ZIP to temp files under dest_dir, and return their paths.  No other ZIP
    members are written.  FILER_FILINGS_CD is needed to map RCPT_CD.FILING_ID →
    recipient-committee FILER_ID (RCPT_CD itself does not carry FILER_ID).

    Each member is copied to disk in bounded 1 MB chunks (rather than read whole
    into memory) so we never hold the multi-GB RCPT_CD table in RAM — pandas
    then reads it back from disk in chunks.  This is the key safeguard against
    the out-of-memory crash that a full in-memory read can cause.
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

        out_paths: list[Path] = []
        for target in ("RCPT_CD.TSV", "FILERNAME_CD.TSV", "FILER_FILINGS_CD.TSV"):
            member = _find(target)
            if member is None:
                raise FileNotFoundError(f"{target} not found in ZIP.")
            out_path = dest_dir / target
            # Stream the member straight to disk in 1 MB chunks.
            with zf.open(member) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            out_paths.append(out_path)

    print("  Extraction complete (streamed to temp files on disk).")
    return out_paths[0], out_paths[1], out_paths[2]


# ── PROCESS RCPT_CD ───────────────────────────────────────────────────────────

def process_rcpt(
    rcpt_path: Path,
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
        rcpt_path,
        sep="\t",
        dtype=str,
        encoding="latin-1",
        on_bad_lines="skip",
        chunksize=CHUNK_SIZE,
        # Read only the columns we use — matched on the normalised header so
        # casing/whitespace don't matter.  Keeps each chunk small in memory.
        usecols=lambda c: c.strip().upper() in RCPT_NEEDED_COLS,
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
            fid = str(row["_FILER_ID"]).strip()
            race = filer_to_race.get(fid, "")
            if race not in target_races:
                continue
            mapped = map_row(row, committee_lookup, filer_id=fid, race=race)
            key = dedup_key(mapped)
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
    all_keys: dict[str, set] = {}
    all_counts: dict[str, int] = {}
    all_filers: dict[str, set[str]] = {}
    for code, path in RACES.items():
        if not path.exists():
            print(f"  [{code}] No existing file — will create: {path.name}")
            all_keys[code] = set()
            all_counts[code] = 0
            all_filers[code] = set()
        else:
            df = pd.read_csv(path, dtype=str, low_memory=False)
            keys = set(dedup_key_series(df).tolist())
            filers = set(
                df.get("Recipient Committee ID", pd.Series(dtype=str))
                  .fillna("").astype(str).str.strip()
            )
            filers.discard("")
            print(
                f"  [{code}] {len(df):,} existing rows | {len(keys):,} dedup keys "
                f"| {len(filers):,} committee IDs  ({path.name})"
            )
            all_keys[code] = keys
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
    tmp_dir = Path(tempfile.mkdtemp(prefix="calaccess_"))

    try:
        # 3. Stream-extract the tables we need to temp files on disk (never the
        #    whole multi-GB RCPT_CD table into RAM).
        rcpt_path, filername_path, filings_path = extract_tables(zip_path, tmp_dir)

        # ZIP is no longer needed once the members are extracted — delete it now
        # to free ~1.5 GB of disk before the memory-heavier processing begins.
        zip_path.unlink(missing_ok=True)

        # 4. One pass over FILERNAME_CD: build both the filer-id → committee
        #    name lookup and the filer-id → race lookup (latter classified by
        #    name pattern, seeded with existing master CSV filer IDs).
        committee_lookup, filer_to_race = build_committee_lookup_and_races(
            filername_path, existing_filers, target_races=set(RACES),
        )

        # 5. Bridge FILING_ID → FILER_ID via FILER_FILINGS_CD.  RCPT_CD only
        #    carries FILING_ID, so this is how we resolve each receipt to its
        #    recipient committee.  Restricted to target filers to keep memory
        #    small.
        filing_to_filer = build_filing_to_filer(
            filings_path, set(filer_to_race),
        )

        # 6. Process RCPT_CD — filter, map, deduplicate across all races
        new_rows = process_rcpt(
            rcpt_path, committee_lookup, filer_to_race,
            filing_to_filer, existing_keys,
        )

    finally:
        zip_path.unlink(missing_ok=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("  Temp ZIP and extracted tables deleted.")

    # 7. Append new rows to each race's CSV
    print(LOG_SEP)
    append_all_new_rows(new_rows, dry_run=args.dry_run)

    # 8. Update state file
    if not args.dry_run:
        now_str = datetime.now(timezone.utc).isoformat()
        state["last_run"] = now_str
        state.setdefault("runs", []).append({
            "timestamp": now_str,
            "new_rows": {c: len(rows) for c, rows in new_rows.items()},
        })
        save_state(state)

    # 9. Summary
    print(LOG_SEP)
    print(f"  {'Race':<6}  {'Previous':>10}  {'New':>8}  {'Total':>10}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*10}")
    for code in sorted(RACES):
        prev = existing_counts[code]
        added = len(new_rows[code])
        print(f"  {code:<6}  {prev:>10,}  {added:>8,}  {prev+added:>10,}")
    print(LOG_SEP)


# ── RACE SELECTION ────────────────────────────────────────────────────────────

def parse_race_tokens(raw: str) -> list[str]:
    """
    Turn a free-form string like "1 3", "GOV,INS", or "all" into an ordered,
    de-duplicated list of canonical race codes.  Accepts menu numbers (1-based,
    in ALL_RACES order), canonical codes, and the aliases in RACE_ALIASES.
    Raises ValueError on any unrecognised token.
    """
    codes = list(ALL_RACES.keys())
    selected: list[str] = []
    for tok in raw.replace(",", " ").split():
        t = tok.strip().upper()
        if not t:
            continue
        if t in ("ALL", "*"):
            for code in codes:
                if code not in selected:
                    selected.append(code)
            continue
        if t.isdigit():
            idx = int(t) - 1
            if not 0 <= idx < len(codes):
                raise ValueError(f"'{tok}' is out of range (pick 1-{len(codes)}).")
            code = codes[idx]
        else:
            code = RACE_ALIASES.get(t, t)
            if code not in ALL_RACES:
                raise ValueError(f"'{tok}' is not a known race.")
        if code not in selected:
            selected.append(code)
    if not selected:
        raise ValueError("No race selected.")
    return selected


def select_races_interactive() -> list[str]:
    """Prompt the user to choose one or more races from a numbered menu."""
    codes = list(ALL_RACES.keys())
    print("\nWhich race(s) do you want to pull from CalAccess?")
    for i, code in enumerate(codes, 1):
        print(f"  {i}) {code:<4} — {RACE_LABELS.get(code, code)}")
    print(
        "\nEnter numbers or codes separated by spaces/commas "
        '(e.g. "1 3" or "GOV INS"),\n'
        'or "all" for every race.  Press Enter for all.'
    )
    while True:
        try:
            raw = input("> ").strip()
        except EOFError:
            raw = ""
        if not raw:
            return codes  # bare Enter → all races
        try:
            return parse_race_tokens(raw)
        except ValueError as exc:
            print(f"  {exc}  Try again.")


def resolve_races(args: argparse.Namespace) -> dict[str, Path]:
    """
    Decide which races to pull, in priority order:
      1. --races flag, if given (non-interactive, scriptable).
      2. Interactive menu, if stdin is a terminal.
      3. All races, otherwise (e.g. a cron job with no terminal attached).
    Returns the {code: output_path} subset of ALL_RACES to process.
    """
    if args.races:
        codes = parse_race_tokens(args.races)
    elif sys.stdin.isatty():
        codes = select_races_interactive()
    else:
        codes = list(ALL_RACES.keys())
        print("No terminal detected and no --races given; pulling all races.")
    return {code: ALL_RACES[code] for code in codes}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Pull 2026 CA Governor, Lt. Governor, and Insurance Commissioner "
            "contributions from CalAccess raw data and append new rows to each "
            "race's CSV.  Run with no --races flag to choose races interactively."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--races", metavar="CODES", default=None,
        help=(
            "Comma/space-separated races to pull (e.g. 'GOV,INS' or 'all'). "
            "Codes: " + ", ".join(f"{c}={RACE_LABELS[c]}" for c in ALL_RACES)
            + ".  If omitted, you'll be prompted to choose interactively."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Download and process data but do not write anything",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        selected = resolve_races(args)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(2)
    # Narrow the module-level RACES dict in place so every downstream function
    # (which reads this same dict) is scoped to just the chosen races.
    RACES.clear()
    RACES.update(selected)
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
