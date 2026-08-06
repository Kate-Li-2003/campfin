"""
0704_benchmark_reviewed.py

Score the classification stack against a manually reviewed benchmark file
("Governor contribution classifications" export: naics_final_classification
= reviewer ground truth, ml_naics_code = the model's prediction at the time
of the run).

Because the ML predictions are precomputed in the file, this script can
measure the effect of the post-ML layers (keyword priors, custom-code regex
overrides, 99 fallback — see CUSTOM_CODES.md) WITHOUT the sentence encoder:

  baseline  = ml_naics_code alone
  stack     = custom rules  >  keyword prior (when ml conf < threshold)
              >  ml_naics_code  >  99 fallback

Scoring is at the normalized 2-digit sector level (31/32/33 ~ 31-33 etc.).
Only rows where BOTH ml_naics_code and naics_final_classification exist are
scored, so baseline and stack are compared on the same set.

Run:
  python code/07_ml_unmatched_classifier/0704_benchmark_reviewed.py
  python .../0704_benchmark_reviewed.py --prior-threshold 0.6 --per-code
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyword_priors as kp  # noqa: E402
import text_features as tf  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = (
    REPO_ROOT / "data/03_input/benchmarks/governor_contribution_classifications.xlsx"
)
DEFAULT_SHEET = "Governor contribution classific"
DEFAULT_RULES = REPO_ROOT / "data/03_input/masterfile/custom_naics_labels.csv"

_RANGE_MAP = {"31": "31-33", "32": "31-33", "33": "31-33",
              "44": "44-45", "45": "44-45", "48": "48-49", "49": "48-49"}


def norm2(v):
    """Normalize to OLD-scheme parent codes for scoring: the benchmark
    truth predates the custom sub-codes, so 52a/52b collapse to 52,
    77a/77b to 77, etc., and 100 (Retired/Homemaker/Student) scores as
    the old 99 bucket. Re-benchmark at sub-code level once a reviewed
    file exists in the new scheme."""
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if not s:
        return None
    if re.fullmatch(r"\d{2}[a-z]", s):
        s = s[:2]
    if s == "100":
        return "99"
    if s in ("31-33", "44-45", "48-49"):
        return s
    p = s[:2]
    return _RANGE_MAP.get(p, p)


def load_benchmark(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, header=1)
    df.columns = [str(c).strip() for c in df.columns]
    need = ["Contributor.Name", "naics_final_classification", "ml_naics_code"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"benchmark missing columns: {missing}")
    return df


def custom_rule_code(name: str, rules: pd.DataFrame) -> str | None:
    """Apply the custom_naics_labels.csv regexes in file order (later rows
    overwrite earlier matches — same semantics as 0501's override pass)."""
    hit = None
    for r in rules.itertuples(index=False):
        if pd.isna(r.name_regex):
            continue
        if re.search(r.name_regex, name, re.IGNORECASE):
            hit = str(r.naics_code)
    return hit


def is_individual(row) -> bool:
    """Benchmark-side stand-in for 0504's entity kinds: 'Last, First' name
    with no org tokens and a usable employer field -> individual."""
    name = tf.clean_field(row.get("Contributor.Name"))
    if not name or tf._ORG_TOKEN_RE.search(name):
        return False
    return bool(re.match(r"^[^,]+,\s*\S+", name))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    p.add_argument("--sheet", type=str, default=DEFAULT_SHEET)
    p.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    p.add_argument("--prior-threshold", type=float, default=kp.DEFAULT_PRIOR_THRESHOLD)
    p.add_argument("--per-code", action="store_true", help="per-code breakdown")
    p.add_argument("--entity-level", action="store_true",
                   help="dedupe donations to unique (name, employer) entities")
    args = p.parse_args(argv)

    df = load_benchmark(args.benchmark, args.sheet)
    rules = pd.read_csv(args.rules, dtype=str)
    # New-schema CSVs mark occupation-based rules (e.g. 100); only
    # employer-name regexes apply to contributor names here.
    if "applies_to" in rules.columns:
        rules = rules[rules["applies_to"] != "occupation"]

    df["truth"] = df["naics_final_classification"].map(norm2)
    df["ml"] = df["ml_naics_code"].map(norm2)
    conf = pd.to_numeric(df.get("ml_naics_code_conf"), errors="coerce")

    scored = df[df["truth"].notna() & df["ml"].notna()].copy()
    scored_conf = conf.loc[scored.index]
    if args.entity_level:
        keep = ~scored.duplicated(subset=["Contributor.Name", "Contributor.Employer"])
        scored, scored_conf = scored[keep], scored_conf[keep]
    print(f"benchmark: {args.benchmark.name}  scored rows: {len(scored):,}"
          f"{' (entity level)' if args.entity_level else ''}")

    # ---- build the stacked prediction per row ----
    stack, layer = [], []
    for i, row in scored.iterrows():
        indiv = is_individual(row)
        name = tf.clean_field(row.get("Contributor.Name"))
        employer = tf.clean_field(row.get("Contributor.Employer"))
        occupation = tf.clean_field(row.get("Contributor.Occupation"))
        entity_name = employer if (indiv and employer) else name

        # 1. custom-code regexes on the ENTITY name — authoritative,
        #    mirrors apply_custom_label_overrides: in 0504/0702 the entity
        #    for an individual donor IS their employer, so the rules see
        #    employer strings there too (e.g. "VAL Property AI" -> 79).
        rule = custom_rule_code(entity_name, rules)
        if rule is not None:
            stack.append(norm2(rule)), layer.append("custom rule")
            continue

        # 2. keyword prior (employer tokens then occupation tokens),
        #    overriding ML only when naics conf < threshold. The prior sees
        #    the same string 0702's entity-level `employer` column holds:
        #    the employer for individuals, the org/committee name for orgs.
        cand = employer if (indiv and employer) else entity_name
        emp_for_prior = (
            "" if tf.is_junk_employer(cand, name if indiv else None) else cand
        )
        prior = kp.prior_for(emp_for_prior, occupation)
        ml = row["ml"]
        c = scored_conf.loc[i]
        if prior is not None and (pd.isna(c) or c < args.prior_threshold) \
                and norm2(prior) != ml:
            stack.append(norm2(prior)), layer.append("keyword prior")
            continue

        # 3. the ML prediction; 4. 99 fallback (can't occur here since we
        #    restrict to rows with an ML prediction).
        stack.append(ml if ml is not None else "99")
        layer.append("ml" if ml is not None else "99 fallback")

    scored["stack"] = stack
    scored["layer"] = layer

    base_acc = (scored["ml"] == scored["truth"]).mean()
    stack_acc = (scored["stack"] == scored["truth"]).mean()
    print(f"\n  baseline (ml only):     top1 = {base_acc:.3f}")
    print(f"  stack (rules+priors):   top1 = {stack_acc:.3f}")

    print("\n  by layer:")
    for lay, grp in scored.groupby("layer"):
        acc = (grp["stack"] == grp["truth"]).mean()
        print(f"    {lay:<14s} n={len(grp):>4,}  acc={acc:.3f}")

    fixed = ((scored["ml"] != scored["truth"]) & (scored["stack"] == scored["truth"])).sum()
    broke = ((scored["ml"] == scored["truth"]) & (scored["stack"] != scored["truth"])).sum()
    print(f"\n  errors fixed by stack: {fixed:,}   correct rows broken: {broke:,}")

    if broke:
        print("\n  --- broken rows (were right, now wrong) ---")
        b = scored[(scored["ml"] == scored["truth"]) & (scored["stack"] != scored["truth"])]
        print(b[["Contributor.Name", "Contributor.Employer", "Contributor.Occupation",
                 "ml", "stack", "truth", "layer"]].to_string(index=False, max_colwidth=36))

    if args.per_code:
        print("\n  per-code (truth) accuracy, baseline -> stack:")
        for code, grp in scored.groupby("truth"):
            b = (grp["ml"] == grp["truth"]).mean()
            s = (grp["stack"] == grp["truth"]).mean()
            print(f"    {code:<6s} n={len(grp):>4,}  {b:.2f} -> {s:.2f}")

    print("\n  --- remaining stack errors (top confusions) ---")
    rem = scored[scored["stack"] != scored["truth"]]
    print(rem.groupby(["stack", "truth"]).size().sort_values(ascending=False).head(15))


if __name__ == "__main__":
    main()
