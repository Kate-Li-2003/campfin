"""NAICS + OpenSecrets classification from web_search.py's industry summaries.

Reads one or more web_search.py output CSVs, classifies each unique
classification unit (entity x employer x occupation alias) by NAICS code and
OpenSecrets category, and writes a slim + full CSV of results. Run this after
web_search.py.

Examples:
    python classify_web_results.py --input 09_outputs/web_search_output_test20_2026-08-01.csv --test 20
    python classify_web_results.py --input 09_outputs/web_search_output_full_2026-08-01.csv --full
    python classify_web_results.py --input out1.csv out2.csv --full --provider anthropic
"""

import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from config import COLUMNS, require_columns
from naics_data import ALL_NAICS as NAICS_DESCRIPTIONS

_DIR = Path(__file__).resolve().parent

PROVIDER_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4.1",
    "gemini": "gemini-2.5-flash",
}

_NO_SUMMARY = {"not employed", "unknown", "", "did not find", "none"}
_NO_OCCUPATION = {"none", "unknown", "not employed", "retired", "homemaker", "student"}
_UNINFORMATIVE = {"NONE", "UNKNOWN", "N/A", "NA", "NOT EMPLOYED", "UNEMPLOYED", "RETIRED", "HOMEMAKER", "STUDENT"}


def _load_os_categories(filename="open_secrets_level2_categories.csv") -> list[str]:
    path = _DIR / "09_inputs" / filename
    with open(path, newline="", encoding="utf-8") as f:
        return [row["x"] for row in csv.DictReader(f) if row["x"].strip()]


def _load_naics_reference(filename: str, key_col: str) -> dict[str, str]:
    """Load the richer NAICS reference used as the classifier's actual code list.

    naics_data.py's NAICS_SECTORS/NAICS_INDUSTRY_GROUPS are NOT used here --
    per project convention those are only for looking up a description for an
    already-assigned code (see NAICS_DESCRIPTIONS below), not for telling the
    LLM what codes are valid.
    """
    path = _DIR / "09_inputs" / filename
    with open(path, newline="", encoding="utf-8") as f:
        return {row[key_col]: row["description"] for row in csv.DictReader(f)}


OS_CATEGORIES = _load_os_categories()
NAICS_SECTOR_REFERENCE = _load_naics_reference("naics_sector_title_expanded.csv", "naics_sector")
NAICS_INDUSTRY_REFERENCE = _load_naics_reference("naics_industry_title_expanded.csv", "naics_industry")

# '100' isn't a real NAICS code, so it isn't in the CSV -- add it manually so it
# shows up as a valid choice (kept distinct from '99', which means "unclear").
NAICS_SECTOR_REFERENCE["100"] = "Retired, Homemaker, Student, or Unemployed"


def _format_naics_industry_reference() -> str:
    lines = ["=== 2-DIGIT SECTORS ==="]
    for code, desc in sorted(NAICS_SECTOR_REFERENCE.items()):
        lines.append(f"  {code}: {desc}")
    lines.append("")
    lines.append("=== 4-DIGIT INDUSTRY GROUPS ===")
    current_sector = None
    for code in sorted(NAICS_INDUSTRY_REFERENCE):
        sector = code[:2]
        if sector != current_sector:
            current_sector = sector
            lines.append(f"  -- {sector}: {NAICS_SECTOR_REFERENCE.get(sector, '')} --")
        lines.append(f"  {code}: {NAICS_INDUSTRY_REFERENCE[code]}")
    return "\n".join(lines)


def _valid_codes_for(digits: int) -> dict[str, str]:
    return NAICS_INDUSTRY_REFERENCE if digits == 4 else NAICS_SECTOR_REFERENCE


def validate_naics(code: str, digits: int = 2) -> str:
    """Validate against NAICS_SECTOR/INDUSTRY_REFERENCE (the CSVs), falling back
    to the 2-digit sector or '00' if unrecognized."""
    code = str(code).strip()
    if code in _valid_codes_for(digits):
        return code
    # pseudo-sector codes (88/90/91/99/100, ...) have no 4-digit children but
    # are still valid at any digit level
    if code in NAICS_SECTOR_REFERENCE:
        return code
    prefix = code[:2]
    if prefix in NAICS_SECTOR_REFERENCE:
        return prefix
    return "00"


def build_system_prompt(digits: int = 2) -> str:
    if digits == 4:
        code_label = "4-digit NAICS industry group code"
        code_list = _format_naics_industry_reference()
    else:
        code_label = "2-digit NAICS sector code"
        code_list = "\n".join(f"  {c}: {d}" for c, d in sorted(NAICS_SECTOR_REFERENCE.items()))

    os_list = "\n".join(f"  - {cat}" for cat in OS_CATEGORIES)

    return (
        "You are an expert at classifying businesses by NAICS industry code.\n"
        "Each entry below is a pre-researched industry summary for a California campaign finance contributor.\n"
        f"Assign each the single most appropriate {code_label} based solely on the information provided.\n\n"
        "Rules:\n"
        "- Always use a code from the valid list below.\n"
        "- For individuals: classify by the employer's industry\n"
        "- For self-employed / sole proprietors: classify by the occupation's industry\n"
        "- Any PAC or political committee, including those sponsored by unions, associations and political parties, should be classified as an '88' NAICS code.\n"
        "- Candidate committees are entities like 'John Duarte for Congress.'\n"
        "- When only an occupation is provided (no industry summary): classify based on that occupation's industry\n"
        "- Native American tribes and tribal governments should be classified as a '92' NAICS code.\n"
        "- For retired, unemployed, homemaker, or student contributors: if an industry summary is provided, classify by their career or former industry. If no useful information is available, use code '100'.\n"
        "- Use code '99' when no useful information is available to determine an industry.\n"
        '- naics_confidence: "high" if the summary is robust and clearly maps to one code; '
        '"medium" if classifying on occupation alone or summary is not specific about the organization\'s products or industry; '
        '"low" if the information is ambiguous, lacks detail, multi-sector, could match multiple NAICS codes or an exact search result was not found.\n'
        "- open_secrets_confidence: same logic as naics_confidence.\n"
        "- reasoning: 1 sentence citing the specific detail in the summary or occupation that drove the decision\n\n"
        f"VALID NAICS CODES:\n{code_list}"
        "- open_secrets_category: assign the single most appropriate OpenSecrets industry category from the list below\n"
        f"\nVALID OPENSECRETS CATEGORIES:\n{os_list}"
    )


class NAICSResult(BaseModel):
    naics_code: str
    open_secrets_category: str
    naics_confidence: Literal["high", "medium", "low"]
    open_secrets_confidence: Literal["high", "medium", "low"]
    reasoning: str


class NAICSBatch(BaseModel):
    results: list[NAICSResult]


NOT_EMPLOYED_RESULT = NAICSResult(
    naics_code="100",
    open_secrets_category="Retired/Homemaker/Student/Unemployed",
    naics_confidence="high",
    open_secrets_confidence="high",
    reasoning="Contributor listed as not employed.",
)

NAICS_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "naics_code": {"type": "string"},
                    "open_secrets_category": {"type": "string"},
                    "naics_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "open_secrets_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reasoning": {"type": "string"},
                },
                "required": ["naics_code", "open_secrets_category", "naics_confidence", "open_secrets_confidence", "reasoning"],
            }
        }
    },
    "required": ["results"],
}


def make_client(provider: str):
    if provider == "gemini":
        from google import genai
        return genai.Client()
    if provider == "openai":
        import openai
        return openai.OpenAI()
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()
    raise ValueError(f"Unknown provider: {provider}")


def _call_gemini(client, user_prompt: str, system_prompt: str, model: str) -> list[NAICSResult]:
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=NAICSBatch,
        ),
    )
    try:
        return NAICSBatch.model_validate_json(response.text).results
    except (ValidationError, Exception) as e:
        print(f"  Parse error: {e}")
        return []


def _call_openai(client, user_prompt: str, system_prompt: str, model: str) -> list[NAICSResult]:
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=NAICSBatch,
        temperature=0,
    )
    parsed = response.choices[0].message.parsed
    return parsed.results if parsed else []


def _call_anthropic(client, user_prompt: str, system_prompt: str, model: str) -> list[NAICSResult]:
    tool = {
        "name": "submit_classifications",
        "description": "Submit NAICS classifications for all provided contributors",
        "input_schema": NAICS_TOOL_SCHEMA,
    }
    response = client.messages.create(
        model=model,
        max_tokens=8096,
        system=system_prompt,
        tools=[tool],
        tool_choice={"type": "tool", "name": "submit_classifications"},
        messages=[{"role": "user", "content": user_prompt}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_classifications":
            try:
                return NAICSBatch.model_validate(block.input).results
            except ValidationError as e:
                print(f"  Validation error: {e}")
    return []


_PROVIDER_FNS = {
    "gemini": _call_gemini,
    "openai": _call_openai,
    "anthropic": _call_anthropic,
}


def classify_from_summaries(
    client,
    summaries: list[str],
    occupations: list[str] | None = None,
    digits: int = 2,
    provider: str = "gemini",
    batch_size: int = 50,
) -> list[NAICSResult | None]:
    model = PROVIDER_MODELS[provider]
    system_prompt = build_system_prompt(digits)
    call_fn = _PROVIDER_FNS[provider]
    effective_batch = batch_size if digits == 2 else min(batch_size, 15)

    final: list[NAICSResult | None] = [None] * len(summaries)
    to_classify: list[tuple[int, str]] = []

    for i, summary in enumerate(summaries):
        raw_occ = occupations[i] if occupations else None
        occupation = ("" if raw_occ is None or not isinstance(raw_occ, str) else raw_occ).strip()
        summary_clean = (summary or "").strip()

        if summary_clean.lower() in _NO_SUMMARY:
            if occupation and occupation.lower() not in _NO_OCCUPATION:
                to_classify.append((i, f"Occupation (no industry summary available): {occupation}"))
            else:
                final[i] = NOT_EMPLOYED_RESULT
        else:
            to_classify.append((i, f"Industry summary: {summary_clean}"))

    sep = "\n\n" + "=" * 60 + "\n\n"
    for start in tqdm(range(0, len(to_classify), effective_batch), desc=provider):
        chunk = to_classify[start: start + effective_batch]
        user_prompt = (
            f"Classify these {len(chunk)} contributors.\n\n"
            + ("=" * 60 + "\n\n")
            + sep.join(
                f"Contributor {i + 1}:\n{entry_text}"
                for i, (_, entry_text) in enumerate(chunk)
            )
            + "\n\n" + "=" * 60 + "\n\n"
            + f"Return exactly {len(chunk)} classifications in the 'results' array, in the same order."
        )
        try:
            results = call_fn(client, user_prompt, system_prompt, model)
        except Exception as e:
            print(f"  Chunk error at {start}: {e}")
            continue

        if len(results) != len(chunk):
            print(f"  Warning: expected {len(chunk)}, got {len(results)}")

        for (original_idx, _), result in zip(chunk, results):
            code = validate_naics(result.naics_code, digits)
            if code != result.naics_code:
                result = NAICSResult(
                    naics_code=code,
                    naics_confidence=result.naics_confidence,
                    open_secrets_category=result.open_secrets_category,
                    open_secrets_confidence=result.open_secrets_confidence,
                    reasoning=result.reasoning + f" (Code corrected from {result.naics_code}.)",
                )
            final[original_idx] = result

    return final


def _is_uninformative(value) -> bool:
    return str(value).strip().upper() in _UNINFORMATIVE


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test", type=int, metavar="N", help="Classify a random sample of N unique classification units.")
    mode.add_argument("--full", action="store_true", help="Classify every unique unit in the input file(s).")
    p.add_argument("--input", nargs="+", required=True, metavar="CSV",
                    help="One or more web_search.py output CSVs (concatenated).")
    p.add_argument("--digits", type=int, choices=[2, 4], default=2, help="NAICS code granularity (default: 2).")
    p.add_argument("--provider", choices=list(PROVIDER_MODELS), default="gemini")
    p.add_argument("--batch-size", type=int, default=50, help="Max classification units per LLM call (default: 50).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for --test sampling (default: 42).")
    p.add_argument("--output-dir", default=str(_DIR / "09_outputs"))
    p.add_argument("--tag", help="Label appended to output filenames (default: today's date).")
    return p.parse_args()


def main():
    args = parse_args()
    load_dotenv()
    client = make_client(args.provider)
    print(f"{args.provider} client ready")

    frames = []
    for path in args.input:
        frame = pd.read_csv(path)
        require_columns(frame, path)
        for col in ("industry_summary", "confidence"):
            if col not in frame.columns:
                raise KeyError(
                    f"{path} is missing column {col!r} -- is --input a web_search.py output file?"
                )
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    print(f"{len(data):,} rows from {len(args.input)} file(s)")

    # backfill prominence columns for backward compat with older web_search outputs
    if "is_prominent" not in data.columns:
        data["is_prominent"] = False
    if "prominence_reason" not in data.columns:
        data["prominence_reason"] = ""

    unit_col = COLUMNS["unit_id"]
    name_col = COLUMNS["name"]
    employer_col = COLUMNS["employer"]
    occ_col = COLUMNS["occupation"]

    for col in (employer_col, occ_col):
        data[col] = data[col].fillna("")

    unique = data.drop_duplicates(subset=[unit_col]).reset_index(drop=True)
    print(f"{len(unique):,} unique classification units")

    if args.test:
        n = min(args.test, len(unique))
        unique_to_run = unique.sample(n=n, random_state=args.seed).reset_index(drop=True)
        data = data[data[unit_col].isin(unique_to_run[unit_col])].reset_index(drop=True)
        print(f"TEST run: classifying {len(unique_to_run):,} of {len(unique):,} unique units")
    else:
        unique_to_run = unique
        print(f"FULL run: classifying {len(unique_to_run):,} unique units")

    not_employed_mask = unique_to_run.apply(
        lambda r: _is_uninformative(r[employer_col]) and _is_uninformative(r[occ_col]), axis=1
    )
    has_real_summary = ~unique_to_run["industry_summary"].str.strip().str.lower().isin(_NO_SUMMARY)

    # not-employed with a real web summary → send to LLM (classify career/former industry)
    to_classify = unique_to_run[~not_employed_mask | (not_employed_mask & has_real_summary)].reset_index(drop=True)
    auto_retired = unique_to_run[not_employed_mask & ~has_real_summary].reset_index(drop=True)
    print(f"  {len(to_classify):,} to classify via LLM, {len(auto_retired):,} auto-assigned (not employed, no summary)")

    results = classify_from_summaries(
        client,
        to_classify["industry_summary"].fillna("").tolist(),
        occupations=to_classify[occ_col].fillna("").tolist(),
        digits=args.digits,
        provider=args.provider,
        batch_size=args.batch_size,
    )
    print(f"Classified {sum(r is not None for r in results)}/{len(results)} unique units ({len(data)} total rows)")

    result_map = {}
    for unit_id, result in zip(to_classify[unit_col], results):
        result_map[unit_id] = result
    for unit_id in auto_retired[unit_col]:
        result_map[unit_id] = NOT_EMPLOYED_RESULT

    def _lookup(unit_id, field):
        r = result_map.get(unit_id)
        return getattr(r, field, "") if r else ""

    data["naics_code_llm"] = data[unit_col].map(lambda u: _lookup(u, "naics_code"))
    data["open_secrets_category"] = data[unit_col].map(lambda u: _lookup(u, "open_secrets_category"))
    data["naics_confidence"] = data[unit_col].map(lambda u: _lookup(u, "naics_confidence"))
    data["open_secrets_confidence"] = data[unit_col].map(lambda u: _lookup(u, "open_secrets_confidence"))
    data["naics_reasoning"] = data[unit_col].map(lambda u: _lookup(u, "reasoning"))
    # naics_data.py is used only to attach a description to the assigned code,
    # never as the reference the LLM classifies against -- its code space
    # doesn't fully overlap with NAICS_SECTOR_REFERENCE, so this can be blank.
    data["naics_description"] = data["naics_code_llm"].map(lambda c: NAICS_DESCRIPTIONS.get(c, ""))

    tag = args.tag or date.today().isoformat()
    mode = f"test{args.test}" if args.test else "full"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slim_cols = [
        unit_col, COLUMNS["entity_id"], name_col, employer_col, occ_col,
        "industry_summary", "confidence", "is_prominent", "prominence_reason",
        "naics_code_llm", "naics_description", "open_secrets_category",
        "naics_confidence", "open_secrets_confidence", "naics_reasoning",
    ]
    slim_cols = [c for c in slim_cols if c in data.columns]

    slim_path = out_dir / f"classification_slim_{mode}_{tag}.csv"
    full_path = out_dir / f"classification_full_{mode}_{tag}.csv"
    data[slim_cols].to_csv(slim_path, index=False)
    data.to_csv(full_path, index=False)
    print(f"Saved slim output to {slim_path}")
    print(f"Saved full output to {full_path}")


if __name__ == "__main__":
    main()
