"""Deep dive on data errors, top donors, and trends."""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/Users/kateli/Desktop/CalMatters/campaign finance categorization/data")
gov = pd.read_csv(DATA_DIR / "governor_race_2026-04-27.csv", low_memory=False)

print("="*80)
print("DATA QUALITY: GOVERNOR FILE")
print("="*80)

# Bad cycles
bad_cycles = gov[gov['Cycle'].isin([2048, 2124, 2147])]
print(f"Bad cycle rows (2048/2124/2147): {len(bad_cycles)}")
print(bad_cycles[['Cycle', 'End Date', 'Amount', 'Recipient Committee', 'Contributor Name']].head(15).to_string())

# Date sanity
gov['End Date Parsed'] = pd.to_datetime(gov['End Date'], errors='coerce')
bad_dates = gov[gov['End Date Parsed'].isna() | (gov['End Date Parsed'].dt.year > 2030)]
print(f"\nRows with invalid/future End Date: {len(bad_dates)}")
print(bad_dates[['End Date', 'Cycle', 'Amount', 'Recipient Committee', 'Contributor Name']].head(10).to_string())

# Filter to 2026 cycle only
print("\n" + "="*80)
print("2026 CYCLE FOCUS - GOVERNOR")
print("="*80)
g26 = gov[gov['Cycle'] == 2026].copy()
g26['End Date Parsed'] = pd.to_datetime(g26['End Date'], errors='coerce')
print(f"Rows in 2026 cycle: {len(g26):,}")
print(f"Total $ in 2026 cycle: ${g26['Amount'].sum():,.2f}")
print(f"Date range: {g26['End Date Parsed'].min()} → {g26['End Date Parsed'].max()}")

# Per candidate totals
print("\n--- 2026 cycle: Top recipients by $ raised ---")
totals = g26.groupby('Recipient Committee').agg(
    total=('Amount', 'sum'),
    txns=('Amount', 'count'),
    avg=('Amount', 'mean'),
    max_donation=('Amount', 'max'),
    min_donation=('Amount', 'min'),
    unique_donors=('Contributor Name', 'nunique')
).sort_values('total', ascending=False)
print(totals.head(20).to_string())

# Negative amounts (refunds, corrections)
print("\n--- Negative amounts (refunds/corrections) ---")
negs = g26[g26['Amount'] < 0].sort_values('Amount')
print(f"Count: {len(negs)}, Total: ${negs['Amount'].sum():,.2f}")
print(negs[['End Date Parsed', 'Amount', 'Recipient Committee', 'Contributor Name', 'Contributor Employer']].head(15).to_string())

# Huge amounts
print("\n--- Largest positive contributions (top 25) ---")
big = g26[g26['Amount'] > 100000].sort_values('Amount', ascending=False)
print(f"Donations >$100k: {len(big)}, Total: ${big['Amount'].sum():,.2f}")
print(big[['End Date Parsed', 'Amount', 'Recipient Committee', 'Contributor Name', 'Contributor Employer', 'Contributor Occupation']].head(25).to_string())

# Out of state donors
print("\n--- Out-of-state donor totals to 2026 governor candidates ---")
oos = g26[g26['Contributor State'].notna() & (g26['Contributor State'].str.upper() != 'CA')]
print(f"Out of state contributions: {len(oos):,} txns, ${oos['Amount'].sum():,.2f}")
print("Top states (excl CA):")
print(oos.groupby(oos['Contributor State'].str.upper())['Amount'].agg(['sum', 'count']).sort_values('sum', ascending=False).head(15).to_string())

# Self-funding indicators
print("\n--- Candidate self-contributions ---")
self_funds = g26[g26['Candidate Contribution'] == 'Y']
print(f"Self-funded entries: {len(self_funds)}, Total: ${self_funds['Amount'].sum():,.2f}")
print("Top by candidate:")
print(self_funds.groupby('Recipient Committee')['Amount'].sum().sort_values(ascending=False).head(15).to_string())
