"""Investigate data issues and donor patterns more deeply."""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/Users/kateli/Desktop/CalMatters/campaign finance categorization/data")
gov = pd.read_csv(DATA_DIR / "governor_race_2026-04-27.csv", low_memory=False)
ins = pd.read_csv(DATA_DIR / "insurance_commissioner_race_2026.csv", low_memory=False)
lt  = pd.read_csv(DATA_DIR / "lt_governor_race_2026.csv", low_memory=False)

# ---- Data integrity: Candidate Contribution field ----
print("="*80)
print("DATA QUALITY: 'Candidate Contribution' field (should mark candidate self-funds)")
print("="*80)
print("Governor file values:")
print(gov['Candidate Contribution'].value_counts(dropna=False))
print("Allied Committee:")
print(gov['Allied Committee'].value_counts(dropna=False))
print("Ballot Measure Contribution:")
print(gov['Ballot Measure Contribution'].value_counts(dropna=False))

# ---- End Date issue ----
print("\n--- End Date sample (non-null vs null) ---")
print(f"Null End Date: {gov['End Date'].isna().sum():,}")
print(f"Non-null End Date: {gov['End Date'].notna().sum():,}")
print(f"Cycle vs End Date null status:")
print(pd.crosstab(gov['Cycle'], gov['End Date'].isna()).head(20))

print("\n--- Sample rows with End Date present ---")
print(gov[gov['End Date'].notna()].head(5)[['Cycle', 'Start Date', 'End Date', 'Recipient Committee']].to_string())

print("\n--- Sample rows with End Date NaN, 2026 cycle ---")
print(gov[(gov['End Date'].isna()) & (gov['Cycle'] == 2026)].head(5)[['Cycle', 'Start Date', 'End Date', 'Recipient Committee', 'Amount']].to_string())

# ---- 2026 cycle: Real self-funding via name matching ----
print("\n" + "="*80)
print("REAL CANDIDATE SELF-FUNDING (matching contributor name to committee name)")
print("="*80)
g26 = gov[gov['Cycle'] == 2026].copy()
# Real self-funding: candidate giving to own committee
def detect_self(row):
    cname = str(row['Contributor Name']).lower()
    rcomm = str(row['Recipient Committee']).lower()
    # last names of known candidates
    candidates = ['steyer', 'porter', 'hilton', 'bianco', 'becerra', 'swalwell', 'yee', 'mahan',
                  'villaraigosa', 'atkins', 'ware', 'ahn', 'thurmond', 'kounalakis']
    for c in candidates:
        if c in cname and c in rcomm:
            return c
    return None

g26['self_candidate'] = g26.apply(detect_self, axis=1)
selffund = g26[g26['self_candidate'].notna()]
print("Self-funding by candidate (donor surname matches committee):")
print(selffund.groupby('self_candidate')['Amount'].agg(['sum', 'count']).sort_values('sum', ascending=False).to_string())

# ---- Top non-self contributors to 2026 governor race ----
print("\n" + "="*80)
print("TOP NON-SELF CONTRIBUTORS to 2026 governor race")
print("="*80)
nonself = g26[g26['self_candidate'].isna()]
top_donors = nonself.groupby('Contributor Name').agg(
    total=('Amount', 'sum'),
    count=('Amount', 'count'),
    n_recipients=('Recipient Committee', 'nunique')
).sort_values('total', ascending=False).head(40)
print(top_donors.to_string())

# ---- IE committees identification ----
print("\n" + "="*80)
print("IE / ALLIED COMMITTEES (committees not directly controlled by candidate)")
print("="*80)
# Detect IE/sponsored committees by name pattern
ie_pattern = g26['Recipient Committee'].str.contains('SPONSORED BY|SUPPORTING|OPPOSING|COALITION|PROJECT|PAC$', case=False, na=False, regex=True)
ie_rows = g26[ie_pattern]
print(f"IE/sponsored committee rows: {len(ie_rows):,}, total ${ie_rows['Amount'].sum():,.2f}")
print("Top IE committees:")
print(ie_rows.groupby('Recipient Committee')['Amount'].sum().sort_values(ascending=False).head(15).to_string())

# ---- Sector look: oil/gas, tech, labor ----
print("\n" + "="*80)
print("SECTOR FLAGS in 2026 governor race")
print("="*80)
emp = nonself['Contributor Employer'].fillna('').str.lower() + ' ' + nonself['Contributor Name'].fillna('').str.lower()
occ = nonself['Contributor Occupation'].fillna('').str.lower()

def sector_match(text, keywords):
    return text.str.contains('|'.join(keywords), na=False)

sectors = {
    'Oil/Gas': ['chevron','exxon','occidental','pg&e','pge','sempra','socal gas','western states petroleum','marathon','phillips 66','valero'],
    'Tech': ['google','meta','facebook','apple inc','microsoft','amazon','salesforce','oracle','nvidia','openai','anthropic','sequoia','andreessen','a16z','iconiq','bot company','zynga','sean n. parker','webflow'],
    'Crypto': ['coinbase','ripple','fairshake','crypto','andreessen','a16z'],
    'Real Estate': ['realtor','real estate','property','development'],
    'Labor': ['union','seiu','afscme','teamster','carpenter','laborer','firefighter','nurse association','teacher','cta','ufcw'],
    'Health': ['davita','kaiser','dignity','sutter','blue shield','anthem','pharma','medical','hospital','nurses'],
    'Tribal/Gaming': ['indian','tribe','gaming','casino','rincon','agua caliente','pechanga'],
    'Entertainment': ['mgm','disney','netflix','warner','universal','sony pictures','hbo','paramount'],
}
for sector, kws in sectors.items():
    mask = sector_match(emp, kws) | sector_match(nonself['Contributor Name'].fillna('').str.lower(), kws)
    s = nonself[mask]
    print(f"  {sector:18s} txns={len(s):>5}  sum=${s['Amount'].sum():>14,.2f}  top recipient={s.groupby('Recipient Committee')['Amount'].sum().idxmax() if len(s)>0 else '—'}")

# ---- Insurance Commissioner ----
print("\n" + "="*80)
print("INSURANCE COMMISSIONER RACE 2026 - candidates and top donors")
print("="*80)
i26 = ins[ins['Cycle'] == 2025].copy()  # active 2025-26 filings
print(f"Rows: {len(i26)}")
print("Per-candidate totals:")
print(ins.groupby('Recipient Committee')['Amount'].agg(['sum', 'count']).sort_values('sum', ascending=False).to_string())

print("\nTop insurance commissioner donors:")
print(ins.groupby('Contributor Name')['Amount'].sum().sort_values(ascending=False).head(20).to_string())

# ---- Lt Governor ----
print("\n" + "="*80)
print("LT GOVERNOR RACE - candidates and top donors")
print("="*80)
print("Per-candidate totals:")
print(lt.groupby('Recipient Committee')['Amount'].agg(['sum', 'count']).sort_values('sum', ascending=False).to_string())
print("\nTop lt gov donors:")
print(lt.groupby('Contributor Name')['Amount'].sum().sort_values(ascending=False).head(20).to_string())
