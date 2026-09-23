"""
diagnose_office_codes.py
========================
Downloads the CalAccess ZIP and prints every OFFICE_CD value that appears
in RCPT_CD for RCPT_DATE >= 2025-01-01, with row counts.

Usage:
  python donations_data_pull/diagnose_office_codes.py
"""

import io
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

CALACCESS_ZIP_URL = "https://campaignfinance.cdn.sos.ca.gov/dbwebexport.zip"
CYCLE_START_DATE  = "2025-01-01"
CHUNK_SIZE        = 100_000


def download_zip(url: str) -> Path:
    print(f"Downloading {url} …")
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp_path = Path(tmp.name)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    start = time.time()
    with open(tmp_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {100*downloaded/total:5.1f}%  {downloaded/1e6:.1f} MB", end="", flush=True)
    print(f"\r  Done. {downloaded/1e6:.1f} MB in {time.time()-start:.0f}s.      ")
    return tmp_path


def main() -> None:
    zip_path = download_zip(CALACCESS_ZIP_URL)
    try:
        print("Extracting RCPT_CD.TSV …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            rcpt_name = next(n for n in zf.namelist() if n.upper().endswith("RCPT_CD.TSV"))
            rcpt_bytes = io.BytesIO(zf.read(rcpt_name))

        print(f"Scanning for RCPT_DATE >= {CYCLE_START_DATE} …")
        counts: Counter = Counter()
        total = 0
        reader = pd.read_csv(
            rcpt_bytes, sep="\t", dtype=str, encoding="latin-1",
            on_bad_lines="skip", chunksize=CHUNK_SIZE,
        )
        for i, chunk in enumerate(reader, 1):
            chunk.columns = [c.strip().upper() for c in chunk.columns]
            total += len(chunk)
            if "OFFICE_CD" not in chunk.columns:
                continue
            dates = pd.to_datetime(chunk["RCPT_DATE"], errors="coerce", format="mixed")
            cycle_mask = dates >= CYCLE_START_DATE
            sub = chunk[cycle_mask]
            counts.update(sub["OFFICE_CD"].str.strip().str.upper().value_counts().to_dict())
            print(f"\r  chunk {i}: {total:,} rows scanned", end="", flush=True)

        print(f"\n\nOFFICE_CD values in 2025+ contributions ({sum(counts.values()):,} rows total):\n")
        for code, n in counts.most_common():
            print(f"  {code:<10}  {n:>8,}")
    finally:
        zip_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
