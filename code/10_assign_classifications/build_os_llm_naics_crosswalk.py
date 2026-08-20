"""build_os_llm_naics_crosswalk.py

OpenSecrets (0806_match_opensecrets.Rmd) and the LLM (09_llm_classifier) each
produce their own naics_code_os / naics_code_llm values independently of
0805_classify_contributors.py's rule-based path -- the only path that runs
codes through build_naics_crosswalk() to translate raw/legacy codes to the
current custom sector scheme. So a code like "33" (raw NAICS manufacturing)
can reach assign_final_classification.Rmd's code_final un-translated (should
be "31-33"), instead of a properly resolved custom code.

This script collects every distinct raw naics_code_os / naics_code_llm value
across the OS-match and LLM-classification output files, runs them through
the SAME build_naics_crosswalk() used by 0805 (not a re-implementation, to
avoid the two ever drifting apart), and writes a small lookup table:
raw_code -> (custom_code, custom_label). assign_final_classification.Rmd and
assign_pac_final_classification.Rmd join os_code/llm_code against this
lookup right after loading, before resolution -- the same treatment
rule_code already gets upstream in 0805.

Usage
-----
    python build_os_llm_naics_crosswalk.py
    python build_os_llm_naics_crosswalk.py --out 10_outputs/os_llm_naics_crosswalk.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "classify_contributors_0805",
    REPO_ROOT / "08_alternative_pipeline" / "0805_classify_contributors.py",
)
_m0805 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _m0805
_spec.loader.exec_module(_m0805)
build_naics_crosswalk = _m0805.build_naics_crosswalk
load_old_to_new_map = _m0805.load_old_to_new_map

DEFAULT_SOURCES = [
    (REPO_ROOT / "08_alternative_pipeline/08_outputs/contributions_os_matches.csv", "naics_code_os"),
    (REPO_ROOT / "08_alternative_pipeline/08_outputs/pac_contributions_os_matches.csv", "naics_code_os"),
    (REPO_ROOT / "09_llm_classifier/09_outputs/classification_full_expanded_2026-08-13.csv", "naics_code_llm"),
    (REPO_ROOT / "09_llm_classifier/09_outputs/pac_classification_batch1_full_expanded_pac_2026-08-13.csv", "naics_code_llm"),
]

DEFAULT_OUT = Path(__file__).parent / "10_outputs" / "os_llm_naics_crosswalk.csv"


def collect_raw_codes(sources: list[tuple[Path, str]]) -> set[str]:
    codes: set[str] = set()
    for path, col in sources:
        if not path.exists():
            print(f"  skip (not found): {path}")
            continue
        df = pd.read_csv(path, dtype=str, usecols=lambda c: c == col)
        vals = df[col].dropna().unique()
        codes |= set(vals)
        print(f"  {path.name}: {len(vals)} unique {col} values")
    return codes


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    print("Collecting raw codes from OS/LLM output files:")
    raw_codes = collect_raw_codes(DEFAULT_SOURCES)
    print(f"\n{len(raw_codes):,} distinct raw codes found")

    old_to_new_map = load_old_to_new_map()
    crosswalk = build_naics_crosswalk(old_to_new_map=old_to_new_map)

    rows = []
    n_changed = 0
    for raw in sorted(raw_codes):
        custom_code, custom_label = crosswalk(raw)
        if pd.isna(custom_code):
            custom_code, custom_label = raw, pd.NA  # leave unresolved codes as-is
        else:
            n_changed += int(str(custom_code) != str(raw))
        rows.append({"raw_code": raw, "custom_code": custom_code, "custom_label": custom_label})

    out_df = pd.DataFrame(rows)
    print(f"\n{n_changed} of {len(rows)} raw codes translated to a different custom code:")
    print(out_df[out_df["raw_code"] != out_df["custom_code"]].to_string(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
