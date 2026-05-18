"""
0502_classify_donors_with_manual.py

Three-source classification pipeline for high-value donors. Re-uses
0501's masterfile + keyword logic and layers on a manually-labelled NAICS
sheet as a third fallback source.

Precedence (highest -> lowest):
  1. Masterfile match  (running_list.csv)        -> data_source_1="masterfile"
  2. Keyword match     (Keywords_Manually...xlsx) -> data_source_1="keyword match"
  3. Manual NAICS      (Manual NAICS Class...xlsx) -> data_source_1="manual"

Per spec, the script-derived match (1+2) wins on conflict; manual labels
only fill in employers that 0501 left unmatched. Cases where 0501 found a
match but the manual sheet *also* had a label (i.e., 0501 won a conflict)
are logged separately to a `conflicts.csv` so you can spot-check them.

Output adds two columns beyond what 0501 produces:
  manual_source_note: text from the manual sheet's classification_source
                      column (e.g. "manual: snippet (podcast)") — only
                      populated for rows where data_source_1 == "manual".
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = REPO_ROOT / "data/04_output_latest_data_pulls/governor_race_2026-04-27.csv"
DEFAULT_RUNNING_LIST = REPO_ROOT / "data/03_input/masterfile/running_list.csv"
DEFAULT_KEYWORDS = (
    REPO_ROOT
    / "data/03_input/training data (manual classifications)/Keywords_Manually_Collected.xlsx"
)
DEFAULT_MANUAL = (
    REPO_ROOT
    / "data/03_input/training data (manual classifications)/Manual NAICS Classifications (+Employer Descriptions) - populated.xlsx"
)
DEFAULT_OUT = REPO_ROOT / "output/05_output/donors_classified_with_manual.csv"
DEFAULT_CONFLICTS = REPO_ROOT / "output/05_output/donors_classified_conflicts.csv"

AMOUNT_MIN = 10000

# Standard NAICS 2-digit sector names — used to fall back on a readable
# label when the manual sheet has only a 2-digit code (no naics_desc).
NAICS_SECTORS = {
    "11": "Agriculture, Forestry, Fishing and Hunting",
    "21": "Mining, Quarrying, and Oil and Gas Extraction",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "32": "Manufacturing",
    "33": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail Trade",
    "45": "Retail Trade",
    "48": "Transportation and Warehousing",
    "49": "Transportation and Warehousing",
    "51": "Information",
    "52": "Finance and Insurance",
    "53": "Real Estate and Rental and Leasing",
    "54": "Professional, Scientific, and Technical Services",
    "55": "Management of Companies and Enterprises",
    "56": "Administrative and Support and Waste Management",
    "61": "Educational Services",
    "62": "Health Care and Social Assistance",
    "71": "Arts, Entertainment, and Recreation",
    "72": "Accommodation and Food Services",
    "81": "Other Services (except Public Administration)",
    "92": "Public Administration",
}


# Dynamically load 0501 (filename starts with digits, not importable via
# normal `import`). All 0501 top-level code runs here except main().
def _load_0501():
    path = Path(__file__).parent / "0501_classify_donors_with_keywords.py"
    spec = importlib.util.spec_from_file_location("kw_classifier", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kw = _load_0501()


# ---------- manual sheet loader ----------

def load_manual(path: Path) -> pd.DataFrame:
    """Read the 'Manual Classifications' sheet, picking the most specific
    NAICS code available per row (4-digit first, then 2-digit), and a
    label that prefers the curator's free-text desc over the standard
    sector name."""
    df = pd.read_excel(path, sheet_name="Manual Classifications")
    df = df[df["employer"].notna()].copy()
    df["employer_norm"] = kw.normalize_name(df["employer"])
    df = df.drop_duplicates("employer_norm", keep="first")

    def _resolve(row: pd.Series) -> tuple[str, str]:
        for col in ("4-digit NAICS classification", "2-Digit NAICS Classification"):
            v = row.get(col)
            if pd.isna(v):
                continue
            code = str(int(v)) if isinstance(v, float) else str(v).strip()
            desc = row.get("naics_desc")
            if pd.notna(desc) and str(desc).strip():
                return code, str(desc).strip()
            return code, NAICS_SECTORS.get(code[:2], "")
        return "", ""

    resolved = df.apply(lambda r: pd.Series(_resolve(r), index=["naics_code", "naics_label"]), axis=1)
    df["manual_naics_code"] = resolved["naics_code"]
    df["manual_naics_label"] = resolved["naics_label"]
    df["manual_source_note"] = df["classification_source"].fillna("")
    df["manual_has_label"] = df["manual_naics_code"].astype(bool) | df["manual_naics_label"].astype(bool)

    return df[
        [
            "employer_norm",
            "manual_naics_code",
            "manual_naics_label",
            "manual_source_note",
            "manual_has_label",
        ]
    ]


# ---------- part 3: layer manual onto 0501 output ----------

def layer_manual(
    merged: pd.DataFrame, manual: pd.DataFrame, conflicts_path: Path | None = None
) -> pd.DataFrame:
    out = merged.copy()
    if "manual_source_note" not in out.columns:
        out["manual_source_note"] = pd.NA

    lookup = manual.set_index("employer_norm")

    n_unmatched_before = int(out["data_source_1"].isna().sum())
    promoted = 0
    conflicts: list[dict] = []

    for idx in out.index:
        norm = out.at[idx, "employer_norm"]
        if norm not in lookup.index:
            continue
        m = lookup.loc[norm]
        if not bool(m["manual_has_label"]):
            continue

        if pd.isna(out.at[idx, "data_source_1"]):
            # Promote into the manual classification.
            out.at[idx, "naics_code"] = m["manual_naics_code"]
            out.at[idx, "naics_label"] = m["manual_naics_label"]
            out.at[idx, "manual_source_note"] = m["manual_source_note"]
            out.at[idx, "data_source_1"] = "manual"
            promoted += 1
        else:
            # 0501 matched it AND manual has a label too — log the
            # disagreement for the user. 0501 wins per spec.
            conflicts.append(
                {
                    "employer": out.at[idx, "employer"],
                    "n_donors": out.at[idx, "n_donors"],
                    "script_data_source_1": out.at[idx, "data_source_1"],
                    "script_data_source_2": out.at[idx, "data_source_2"],
                    "script_match_name": out.at[idx, "match_name"],
                    "script_level1_category": out.at[idx, "level1_category"],
                    "script_naics_code": out.at[idx, "naics_code"],
                    "script_naics_label": out.at[idx, "naics_label"],
                    "manual_naics_code": m["manual_naics_code"],
                    "manual_naics_label": m["manual_naics_label"],
                    "manual_source_note": m["manual_source_note"],
                }
            )

    print()
    print("=" * 70)
    print("PART 3 — manual NAICS classifications (fallback)")
    print("=" * 70)
    print(f"  Manual rows w/ usable label:        {int(manual['manual_has_label'].sum()):>9,}")
    print(f"  Employers unmatched after Part 2:   {n_unmatched_before:>9,}")
    print(
        f"  Newly matched by manual labels:     {promoted:>9,}   "
        f"({promoted / max(n_unmatched_before, 1) * 100:>5.2f}% of remaining)"
    )
    print(f"  Conflicts (script won, manual differs): {len(conflicts):>5}")

    if conflicts and conflicts_path is not None:
        conflicts_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(conflicts).to_csv(conflicts_path, index=False)
        print(f"  Conflicts logged to:  {conflicts_path}")

    return out


# ---------- summary ----------

def print_three_part_summary(merged: pd.DataFrame) -> None:
    n_emp = len(merged)
    n_donations = int(merged["n_donors"].sum())
    src = merged["data_source_1"].fillna("unmatched")
    by_emp = src.value_counts()
    by_don = merged.groupby(src)["n_donors"].sum()

    print()
    print("=" * 70)
    print("FINAL SUMMARY — three-source classification pipeline")
    print("=" * 70)
    print(f"  Total employers: {n_emp:,}    Total donations: {n_donations:,}\n")
    for source in ("masterfile", "keyword match", "manual", "unmatched"):
        e = int(by_emp.get(source, 0))
        d = int(by_don.get(source, 0))
        print(
            f"  {source:14s}  employers: {e:>5,} ({e / max(n_emp, 1) * 100:>5.2f}%)   "
            f"donations: {d:>5,} ({d / max(n_donations, 1) * 100:>5.2f}%)"
        )

    mf = merged[merged["data_source_1"] == "masterfile"]
    if len(mf):
        print("\n  Masterfile sub-sources (data_source_2):")
        for sub, count in mf["data_source_2"].value_counts().items():
            d = int(mf.loc[mf["data_source_2"] == sub, "n_donors"].sum())
            print(
                f"    {sub:14s} employers: {count:>5,} ({count / n_emp * 100:>5.2f}%)   "
                f"donations: {d:>5,} ({d / n_donations * 100:>5.2f}%)"
            )


# ---------- main ----------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--running-list", type=Path, default=DEFAULT_RUNNING_LIST)
    p.add_argument("--keywords", type=Path, default=DEFAULT_KEYWORDS)
    p.add_argument("--manual", type=Path, default=DEFAULT_MANUAL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--conflicts-out", type=Path, default=DEFAULT_CONFLICTS)
    p.add_argument("--amount-min", type=float, default=AMOUNT_MIN)
    args = p.parse_args(argv)

    print(f"Input: {args.input}")
    df = kw._load_input(args.input)

    # Part 0/1/2 via 0501.
    employers = kw.filter_and_aggregate(df, args.amount_min)
    merged = kw.match_masterfile(employers, args.running_list)
    keywords = kw.load_keywords(args.keywords)
    merged = kw.match_keywords(merged, keywords)

    # Part 3.
    manual = load_manual(args.manual)
    merged = layer_manual(merged, manual, conflicts_path=args.conflicts_out)

    print_three_part_summary(merged)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"\nWrote: {args.out}  ({len(merged):,} rows)")


if __name__ == "__main__":
    main()
