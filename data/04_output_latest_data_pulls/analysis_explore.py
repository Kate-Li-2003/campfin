"""Quick exploratory analysis of the three CSVs."""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/Users/kateli/Desktop/CalMatters/campaign finance categorization/data")
gov = pd.read_csv(DATA_DIR / "governor_race_2026-04-27.csv", low_memory=False)
ins = pd.read_csv(DATA_DIR / "insurance_commissioner_race_2026.csv", low_memory=False)
lt  = pd.read_csv(DATA_DIR / "lt_governor_race_2026.csv", low_memory=False)

print("="*80)
print("GOVERNOR RACE")
print("="*80)
print(f"Rows: {len(gov):,}")
print(f"Columns: {list(gov.columns)}")
print(f"Transaction Types: {gov['Transaction Type'].value_counts().to_dict()}")
print(f"Recipient Committees (top 30):")
print(gov['Recipient Committee'].value_counts().head(30))
print()
print(f"Unique Recipient Committees: {gov['Recipient Committee'].nunique()}")
print()

# Date sanity
gov['End Date Parsed'] = pd.to_datetime(gov['End Date'], errors='coerce')
print(f"Date range (parsed): {gov['End Date Parsed'].min()} → {gov['End Date Parsed'].max()}")
print("End Date raw - top weirdest values:")
print(gov['End Date'].astype(str).str[:4].value_counts().head(15))

print()
print("Amounts:")
print(gov['Amount'].describe())
print()
print(f"Sum Amount: ${gov['Amount'].sum():,.2f}")
print()
print(f"Cycles: {sorted(gov['Cycle'].dropna().unique().tolist())[:50]}")

print()
print("="*80)
print("INSURANCE COMMISSIONER RACE")
print("="*80)
print(f"Rows: {len(ins):,}")
print(f"Recipient Committees: {ins['Recipient Committee'].value_counts().to_dict()}")
print()
print(f"Sum Amount: ${ins['Amount'].sum():,.2f}")
ins['End Date Parsed'] = pd.to_datetime(ins['End Date'], errors='coerce')
print(f"Date range: {ins['End Date Parsed'].min()} → {ins['End Date Parsed'].max()}")

print()
print("="*80)
print("LT GOVERNOR RACE")
print("="*80)
print(f"Rows: {len(lt):,}")
print(f"Recipient Committees: {lt['Recipient Committee'].value_counts().to_dict()}")
print(f"Sum Amount: ${lt['Amount'].sum():,.2f}")
lt['End Date Parsed'] = pd.to_datetime(lt['End Date'], errors='coerce')
print(f"Date range: {lt['End Date Parsed'].min()} → {lt['End Date Parsed'].max()}")
