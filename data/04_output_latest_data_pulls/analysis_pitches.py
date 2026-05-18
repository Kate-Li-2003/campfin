"""Final dive — looking at specific patterns for story pitches."""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/Users/kateli/Desktop/CalMatters/campaign finance categorization/data")
gov = pd.read_csv(DATA_DIR / "governor_race_2026-04-27.csv", low_memory=False)
ins = pd.read_csv(DATA_DIR / "insurance_commissioner_race_2026.csv", low_memory=False)
lt  = pd.read_csv(DATA_DIR / "lt_governor_race_2026.csv", low_memory=False)

g26 = gov[gov['Cycle'].isin([2025, 2026])].copy()

# ---- 1. Mahan refund pattern detail ----
print("="*80)
print("MAHAN 'CALIFORNIA BACK TO BASICS' — refunds and donors detail")
print("="*80)
mahan_back = g26[g26['Recipient Committee'].str.contains('MAHAN.*BACK TO BASICS', na=False, regex=True)]
print(f"Total rows: {len(mahan_back)}, sum: ${mahan_back['Amount'].sum():,.2f}")
print("All transactions for this committee:")
m_sort = mahan_back.sort_values('Amount')
print(m_sort[['Start Date','Amount','Contributor Name','Contributor Employer','Contributor Occupation']].to_string())

print()
print("="*80)
print("MAHAN main committee — top donors")
print("="*80)
mahan_main = g26[g26['Recipient Committee'] == 'MAHAN FOR GOVERNOR 2026']
print(f"Total: ${mahan_main['Amount'].sum():,.2f}, txns: {len(mahan_main)}")
print(mahan_main.groupby('Contributor Name')['Amount'].sum().sort_values(ascending=False).head(20).to_string())

# ---- 2. Tubbs IE pattern ----
print()
print("="*80)
print("TUBBS LIEUTENANT GOVERNOR — who's funding the IE?")
print("="*80)
tubbs_ie = lt[lt['Recipient Committee'].str.contains('TUBBS.*FRIENDS', na=False, regex=True)]
print(tubbs_ie[['Start Date','Amount','Contributor Name','Contributor Employer','Contributor Occupation']].sort_values('Amount', ascending=False).to_string())

# ---- 3. Wells Fargo / US Treasury entries — investigate ----
print()
print("="*80)
print("STRANGE DONORS - Wells Fargo, US Treasury as 'donors'?")
print("="*80)
strange = lt[lt['Contributor Name'].str.contains('Wells Fargo|United States Treasury|Stifel', na=False, regex=True)]
print(strange[['Start Date','Transaction Type','Amount','Recipient Committee','Contributor Name','Contributor Occupation']].to_string())

# ---- 4. Hilton small-dollar v Bianco small-dollar ----
print()
print("="*80)
print("REPUBLICAN GRASSROOTS: Hilton vs Bianco vs Ware (small donors)")
print("="*80)
for committee in ['HILTON FOR GOVERNOR 2026; STEVE','BIANCO FOR GOVERNOR 2026','WARE FOR GOVERNOR 2026; BUTCH']:
    sub = g26[g26['Recipient Committee']==committee]
    pos = sub[sub['Amount']>0]
    print(f"\n{committee}:")
    print(f"  Total raised: ${pos['Amount'].sum():,.2f}")
    print(f"  Txns: {len(pos)}, unique donors: {pos['Contributor Name'].nunique()}")
    print(f"  Median: ${pos['Amount'].median():.2f}, Mean: ${pos['Amount'].mean():.2f}")
    # Distribution
    tiers = pd.cut(pos['Amount'], bins=[-1, 50, 200, 1000, 10000, 40000, 1e9],
                   labels=['≤$50','$51-200','$201-1k','$1k-10k','$10k-40k','>$40k'])
    print(f"  Donor amount tiers:")
    print((tiers.value_counts().reindex(['≤$50','$51-200','$201-1k','$1k-10k','$10k-40k','>$40k'])).to_string())
    # Out-of-state pct
    oos = pos[pos['Contributor State'].notna() & (pos['Contributor State'].str.upper()!='CA')]
    print(f"  Out of state: {len(oos)} txns / ${oos['Amount'].sum():,.2f} ({100*oos['Amount'].sum()/pos['Amount'].sum():.1f}% of total)")

# ---- 5. Porter dominant fundraising — vs IE? ----
print()
print("="*80)
print("PORTER vs HILTON — small-dollar machines compared")
print("="*80)
for committee in ['PORTER FOR GOVERNOR 2026; KATIE','HILTON FOR GOVERNOR 2026; STEVE','BIANCO FOR GOVERNOR 2026','BECERRA FOR GOVERNOR 2026','SWALWELL FOR GOVERNOR 2026; ERIC']:
    sub = g26[g26['Recipient Committee']==committee]
    pos = sub[sub['Amount']>0]
    print(f"{committee}: ${pos['Amount'].sum():,.0f} from {pos['Contributor Name'].nunique():,} donors, median ${pos['Amount'].median():.0f}")

# ---- 6. PG&E / Realtors anti-Steyer detail ----
print()
print("="*80)
print("ANTI-STEYER COMMITTEE — who's funding it?")
print("="*80)
anti = g26[g26['Recipient Committee'].str.contains('NO ON STEYER', na=False, regex=True) |
           g26['Recipient Committee'].str.contains('CALIFORNIANS FOR RESILIENT AND AFFORDABLE ENERGY', na=False, regex=True)]
print(anti[['Start Date','Amount','Recipient Committee','Contributor Name']].sort_values('Amount', ascending=False).to_string())

# ---- 7. Pro-Steyer IE? ----
print()
print("="*80)
print("STEYER'S OWN IE (Coalition of Housing/Labor) — who's funding it?")
print("="*80)
steyer_ie = g26[g26['Recipient Committee'].str.contains('STEYER.*HOUSING.*LABOR', na=False, regex=True)]
print(steyer_ie[['Start Date','Amount','Contributor Name']].sort_values('Amount', ascending=False).to_string())

# ---- 8. Out of state Republican donors ----
print()
print("="*80)
print("OUT OF STATE big-money donors to 2026 governor (>$5k single contrib)")
print("="*80)
oos_big = g26[(g26['Amount']>=5000) &
              g26['Contributor State'].notna() &
              (g26['Contributor State'].str.upper()!='CA')]
print(oos_big[['Amount','Recipient Committee','Contributor Name','Contributor City','Contributor State','Contributor Employer']].sort_values('Amount', ascending=False).head(30).to_string())

# ---- 9. Crypto / fintech ----
print()
print("="*80)
print("CRYPTO / FINTECH donors (governor + lt gov + ins comm)")
print("="*80)
combined = pd.concat([g26.assign(race='Gov'),
                       ins.assign(race='InsComm'),
                       lt.assign(race='LtGov')])
crypto_kw = 'ripple|coinbase|crypto|kraken|paxos|circle|grayscale|gemini|fairshake|larsen, christian|armstrong, brian'
mask = (combined['Contributor Employer'].fillna('').str.lower().str.contains(crypto_kw, regex=True) |
        combined['Contributor Name'].fillna('').str.lower().str.contains(crypto_kw, regex=True))
print(combined[mask][['race','Amount','Recipient Committee','Contributor Name','Contributor Employer']].sort_values('Amount', ascending=False).to_string())

# ---- 10. Tech billionaires ----
print()
print("="*80)
print("BIG NAME individual donors (>= $100k single contribution) — governor race")
print("="*80)
big_ind = g26[(g26['Amount']>=100000) & ~g26['Contributor Name'].fillna('').str.contains(',\\s*nan', regex=True, na=False)]
print(big_ind[['Start Date','Amount','Recipient Committee','Contributor Name','Contributor Employer','Contributor Occupation','Contributor City']].sort_values('Amount', ascending=False).to_string())
