"""
0305_extend_running_list_from_manual.py

Fold manually-relabeled NAICS classifications into running_list.csv as the
TOP-priority, ground-authority source.

Manual labels are treated as deterministic truth:
  - they OVERRIDE any existing running_list row on normalized-name collision
    (manual beats opensecrets / h1b / edd / crosswalk), so subsequent
    masterfile matches in the 05 stage return the manual NAICS code
    deterministically; and
  - because 0701 trains directly off running_list.csv, they automatically
    become training rows for the NAICS model (no change needed in 0701).

Run order: 0301 build -> 0303 edd -> 0304 crosswalk -> 0305 manual (LAST,
so manual wins) -> 0701 retrain.

Input
-----
A PowerSearch-format file (donation-level rows with `Contributor *`
columns) that also carries a `naics_code_manual` column holding the
hand-checked NAICS code. The entity key is derived the same way the 05
pipeline derives it (employer for individuals; contributor name for
organizations / committees / non-company individuals) so the keys line up
when 05 matches against the masterfile.

Usage
-----
    python 0305_extend_running_list_from_manual.py \
        --manual "data/03_input/training data (manual classifications)/manual_relabeled_data_062826.xlsx"
    python 0305_extend_running_list_from_manual.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANUAL = (
    REPO_ROOT
    / "data/03_input/training data (manual classifications)/manual_relabeled_data_062826.xlsx"
)
DEFAULT_RUNNING_LIST = REPO_ROOT / "data/03_input/masterfile/running_list.csv"
# Optional code -> label sources, for readability of the masterfile output.
CROSSWALK_MAPPED = REPO_ROOT / "data/03_input/masterfile/NAICS_OpenSecrets_Crosswalk_Mapped.xlsx"
CUSTOM_LABELS = REPO_ROOT / "data/03_input/masterfile/custom_naics_labels.csv"

MANUAL_SOURCE = "manual"

RUNNING_LIST_COLS = [
    "name",
    "name_norm",
    "level1_category",
    "level2_category",
    "level3_category",
    "naics_code",
    "naics_label",
    "source",
]

# Mirror the entity-kind logic used in 0504_classify_other_races.py so the
# derived entity (and thus name_norm) matches what the 05 stage produces.
NON_COMPANY_OCCUPATIONS = {
    "retired", "self-employed", "self employed", "not employed",
    "unemployed", "homemaker",
}
ORG_SUFFIX_RE = re.compile(
    r"\b("
    r"LLC|L\.?L\.?C\.?|LLP|INC|INCORPORATED|CORP|CORPORATION|LTD|LIMITED|LP|"
    r"CO|COMPANY|COMPANIES|HOLDINGS|GROUP|PARTNERS|VENTURES|CAPITAL|FUND|TRUST|"
    r"FOUNDATION|ASSOCIATION|UNION|PAC|COMMITTEE|GUILD|SOCIETY|INSTITUTE|COUNCIL|"
    r"AGENCY|AUTHORITY|DISTRICT|BUREAU|CHURCH|TEMPLE|MOSQUE|SYNAGOGUE|MINISTRIES|"
    r"HOSPITAL|UNIVERSITY|COLLEGE|SCHOOL|FIRM|PARTNERSHIP|ENTERPRISES|RESORT|"
    r"REALTY|PROPERTIES)\b",
    re.IGNORECASE,
)


def normalize_name(s: pd.Series) -> pd.Series:
    """Same normalization used to build running_list.csv (0301/0501)."""
    return (
        s.fillna("").astype(str).str.upper()
        .str.replace(r"[^A-Z0-9 ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Column resolver that is case-insensitive and treats '.', '_', and
    space as equivalent separators, so PowerSearch ('Contributor Name') and
    R-style dotted ('Contributor.Name') headers both resolve."""
    def key(c: str) -> str:
        return re.sub(r"[\s._]+", " ", str(c)).strip().lower()
    norm = {key(c): c for c in df.columns}
    for cand in candidates:
        if key(cand) in norm:
            return norm[key(cand)]
    return None


def _clean_naics(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return "" if s.lower() in ("", "nan") else s


def derive_entity(name: str, employer: str, occupation: str, has_id: bool) -> str:
    """Employer for individuals; contributor name for orgs / committees /
    non-company individuals. Mirrors 0504.filter_and_aggregate_with_orgs."""
    name = "" if pd.isna(name) else str(name).strip()
    employer = "" if pd.isna(employer) else str(employer).strip()
    occ = ("" if pd.isna(occupation) else str(occupation)).lower().strip()
    if occ in NON_COMPANY_OCCUPATIONS:
        return name
    if has_id or (name and ORG_SUFFIX_RE.search(name)) or ("," not in name and name):
        return name
    return employer or name


def load_label_lookup() -> dict[str, str]:
    """Best-effort code -> label map so manual rows carry a readable
    naics_label in the masterfile. naics_code is the authority; label is
    cosmetic."""
    lut: dict[str, str] = {}
    if CROSSWALK_MAPPED.exists():
        cw = pd.read_excel(CROSSWALK_MAPPED)
        cw.columns = [c.strip() for c in cw.columns]
        code_c = _col(cw, "2-Digit NAICS Codes", "naics_code")
        lab_c = _col(cw, "NAICS Sector Description", "naics_label")
        if code_c and lab_c:
            for c, l in zip(cw[code_c], cw[lab_c]):
                k = _clean_naics(c).replace("*", "").replace("?", "")
                if k and k not in lut and pd.notna(l):
                    lut[k] = str(l).replace(" (not in source list)", "").strip()
    if CUSTOM_LABELS.exists():
        cl = pd.read_csv(CUSTOM_LABELS, dtype=str)
        for r in cl.itertuples(index=False):
            k = _clean_naics(getattr(r, "naics_code", None))
            if k and pd.notna(getattr(r, "naics_label", None)):
                lut[k] = getattr(r, "naics_label")
    return lut


def build_manual_rows(manual_path: Path) -> pd.DataFrame:
    df = pd.read_excel(manual_path) if manual_path.suffix.lower() in (".xlsx", ".xls") \
        else pd.read_csv(manual_path)

    name_c = _col(df, "Contributor Name")
    emp_c = _col(df, "Contributor Employer")
    occ_c = _col(df, "Contributor Occupation")
    id_c = _col(df, "Contributor ID")
    naics_c = _col(df, "naics_code_manual")
    if naics_c is None:
        raise SystemExit("Could not find a 'naics_code_manual' column in the manual file.")
    if name_c is None and emp_c is None:
        raise SystemExit("Need a 'Contributor Name' or 'Contributor Employer' column.")

    names = df[name_c] if name_c else pd.Series([""] * len(df))
    emps = df[emp_c] if emp_c else pd.Series([""] * len(df))
    occs = df[occ_c] if occ_c else pd.Series([""] * len(df))
    has_id = df[id_c].notna() if id_c else pd.Series([False] * len(df))

    entity = [
        derive_entity(n, e, o, bool(i))
        for n, e, o, i in zip(names, emps, occs, has_id)
    ]
    out = pd.DataFrame({
        "name": entity,
        "naics_code": df[naics_c].map(_clean_naics),
    })
    # Optional manual sub-labels if the file happens to carry them.
    for tgt, *cands in [
        ("level1_category", "level1_category_manual", "level1_category"),
        ("level2_category", "level2_category_manual", "level2_category"),
        ("level3_category", "level3_category_manual", "level3_category"),
    ]:
        c = _col(df, *cands)
        out[tgt] = df[c].astype(object) if c else pd.NA

    out["name"] = out["name"].astype(str).str.strip()
    out = out[(out["name"] != "") & (out["naics_code"] != "")]
    out["name_norm"] = normalize_name(out["name"])
    out = out[out["name_norm"] != ""]

    # Resolve duplicate entities: keep the modal manual code, log conflicts.
    conflicts = []
    rows = []
    for key, g in out.groupby("name_norm"):
        codes = g["naics_code"].value_counts()
        if len(codes) > 1:
            conflicts.append((g["name"].iloc[0], dict(codes)))
        best = codes.index[0]
        r = g[g["naics_code"] == best].iloc[0]
        rows.append(r)
    out = pd.DataFrame(rows).reset_index(drop=True)

    lut = load_label_lookup()
    out["naics_label"] = out["naics_code"].map(lambda c: lut.get(c, pd.NA))
    out["source"] = MANUAL_SOURCE

    if conflicts:
        print(f"\n  WARNING: {len(conflicts)} entities had conflicting manual codes "
              f"(kept the modal value):")
        for nm, d in conflicts[:15]:
            print(f"    {nm!r}: {d}")

    return out[RUNNING_LIST_COLS]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manual", type=Path, default=DEFAULT_MANUAL)
    p.add_argument("--running-list", type=Path, default=DEFAULT_RUNNING_LIST)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    manual = build_manual_rows(args.manual)
    rl = pd.read_csv(args.running_list, dtype=str)

    print(f"Manual file:        {args.manual.name}")
    print(f"  authoritative manual entities (deduped): {len(manual):,}")
    print(f"  manual NAICS distribution (top): "
          f"{dict(manual['naics_code'].value_counts().head(8))}")
    print(f"\nrunning_list before: {len(rl):,} rows  "
          f"{dict(rl['source'].value_counts())}")

    overlap = rl["name_norm"].isin(set(manual["name_norm"]))
    print(f"  existing rows overridden by manual:      {int(overlap.sum()):,}")
    print(f"  brand-new manual rows added:             "
          f"{len(manual) - int(overlap.sum()):,}")

    # Manual wins: drop colliding rows, then place manual FIRST so any
    # downstream drop_duplicates(keep='first') also resolves to manual.
    kept = rl[~overlap]
    out = pd.concat([manual[RUNNING_LIST_COLS], kept[RUNNING_LIST_COLS]],
                    ignore_index=True)

    print(f"\nrunning_list after:  {len(out):,} rows  "
          f"{dict(out['source'].value_counts())}")

    if args.dry_run:
        print("\n--dry-run: not writing. Sample manual rows:")
        print(manual[["name", "naics_code", "naics_label"]].head(12).to_string(index=False))
        return

    backup = args.running_list.with_suffix(args.running_list.suffix + ".pre_manual_bak")
    if not backup.exists():
        shutil.copy2(args.running_list, backup)
        print(f"\nBacked up original to: {backup.name}")
    out.to_csv(args.running_list, index=False)
    print(f"Wrote: {args.running_list}")


if __name__ == "__main__":
    main()
