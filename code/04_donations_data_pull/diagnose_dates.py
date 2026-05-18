"""
diagnose_dates.py
=================
Reads the first 500K rows of RCPT_CD.TSV and reports:
  - Sample of raw RCPT_DATE values
  - How many dates parsed successfully vs. failed
  - Distribution of successfully-parsed years

Run from the project root:
  python donations_data_pull/diagnose_dates.py
"""

import io
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

CALACCESS_ZIP_URL = "https://campaignfinance.cdn.sos.ca.gov/dbwebexport.zip"
SAMPLE_ROWS = 500_000


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

        print(f"Reading first {SAMPLE_ROWS:,} rows …")
        df = pd.read_csv(
            rcpt_bytes, sep="\t", dtype=str, encoding="latin-1",
            on_bad_lines="skip", nrows=SAMPLE_ROWS,
        )
        df.columns = [c.strip().upper() for c in df.columns]

        print(f"\n--- Raw RCPT_DATE samples (first 20 non-null values) ---")
        samples = df["RCPT_DATE"].dropna().head(20).tolist()
        for s in samples:
            print(f"  {repr(s)}")

        print(f"\n--- Null/blank RCPT_DATE ---")
        null_count = df["RCPT_DATE"].isna().sum() + (df["RCPT_DATE"].fillna("").str.strip() == "").sum()
        print(f"  {null_count:,} of {len(df):,} rows ({100*null_count/len(df):.1f}%) have blank/null RCPT_DATE")

        print(f"\n--- Date parse success rate ---")
        parsed = pd.to_datetime(df["RCPT_DATE"], errors="coerce", format="mixed")
        failed = parsed.isna().sum()
        print(f"  Parsed OK : {len(df)-failed:,}")
        print(f"  Failed    : {failed:,}  ({100*failed/len(df):.1f}%)")

        print(f"\n--- Year distribution of successfully-parsed dates ---")
        years = parsed.dropna().dt.year.value_counts().sort_index()
        for yr, cnt in years.items():
            print(f"  {yr}:  {cnt:,}")

    finally:
        zip_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
