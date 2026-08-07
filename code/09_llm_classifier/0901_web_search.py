"""Web-search industry lookups for the LLM classifier.

Runs one Gemini + Google Search lookup per classification unit (entity x
employer x occupation alias) in the pipeline's classification input, and
saves the input rows back out with industry_summary / urls / confidence
columns attached. Run this first -- classify_web_results.py consumes its
output.

Examples:
    python web_search.py --test 20
    python web_search.py --full
    python web_search.py --full --skip-existing 09_outputs/web_search_output_test20_2026-08-01.csv
"""

import argparse
import re
import time
from datetime import date
from pathlib import Path 
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from config import COLUMNS, DEFAULT_INPUT_PATH, require_columns

MODEL = "gemini-2.5-flash"

_NOT_EMPLOYED = {
    "not employed", "unknown", "retired", "homemaker", "student", "n/a", "none",
    "unemployed", "housewife", "househusband", "stay at home", "stay-at-home",
    "na", "-", "nan",
}
_BAD_SUMMARIES = {"did not find", "", "unknown"}

search_cache: dict = {}
raw_response_cache: dict = {}
parse_error_names: set = set()


class IndustrySearchResult(BaseModel):
    industry_summary: str
    urls: list[str]
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"


def _is_not_employed(employer: str, occupation: str) -> bool:
    return (employer or "").strip().lower() in _NOT_EMPLOYED \
       and (occupation or "").strip().lower() in _NOT_EMPLOYED


def search_industry_gemini(client, name: str, employer: str, occupation: str = "") -> tuple[IndustrySearchResult, str]:
    from google.genai import types

    user_msg = (
        f"Contributor name: {name}\n"
        f"Employer: {employer or 'Non-individual'}\n"
        f"Occupation: {occupation or 'Not provided'}"
    )

    last_exc = None
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a research assistant identifying California campaign finance contributors. "
                        "Use web search to look up the contributor and return a single JSON object with these keys:\n"
                        '- "industry_summary": 1-2 sentences describing ONLY the organization\'s or employer\'s business. Do NOT include URLs, confidence ratings, or any metadata in this field.\n'
                        '  - For individuals: describe what the employer does.\n'
                        '  - For non-individual contributors (employer is "Non-individual"): describe the entity and its purpose.\n'
                        '  - If the contributor is a PAC or political committee, note that, and also describe what the PAC supports.\n'
                        '  - If you cannot find any information, set to "Unknown".\n'
                        '- "urls": list of URLs most useful for this contributor. Empty list if none found.\n'
                        '- "confidence": "high" if clearly identified and detailed information on the contributor is available, \n'
                        '"medium" if clear match, but the description of the contributor is short or vague. \n'
                        '"low" if multiple possible matches, partial match, or limited info, "unknown" if nothing found.\n'
                        "Return ONLY a raw JSON object. No markdown, no prose, no explanation -- just the JSON."
                    ),
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            break  # success -- exit retry loop
        except Exception as e:
            last_exc = e
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < 3:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"  503 on attempt {attempt + 1} for {name!r}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    else:
        raise last_exc  # all 4 attempts failed

    import json

    text = re.sub(r"^```(?:json)?\s*\n?", "", (response.text or "").strip())
    text = re.sub(r"\n?```$", "", text.strip())

    # extract JSON even if the model wraps it in prose
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start: end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        parse_error_names.add(name)
        print(f"  Parse error for {name!r}: model returned prose | Raw: {text[:200]}")
        data = None

    if data is not None:
        try:
            if "confidence" in data:
                data["confidence"] = data["confidence"].lower()
            return IndustrySearchResult(**data), response.text
        except ValidationError as e:
            parse_error_names.add(name)
            print(f"  Validation error for {name!r}: {e} | Raw: {text[:200]}")

    urls = []
    try:
        for chunk in response.candidates[0].grounding_metadata.grounding_chunks or []:
            if chunk.web and chunk.web.uri:
                urls.append(chunk.web.uri)
    except AttributeError:
        pass

    summary_text = text if data is None else data.get("industry_summary", text)
    return IndustrySearchResult(industry_summary=summary_text, urls=urls, confidence="low"), response.text


def run_searches(client, rows) -> None:
    """rows: iterable of (unit_id, name, employer, occupation)."""
    for unit_id, name, employer, occ in tqdm(rows, total=len(rows)):
        if _is_not_employed(employer, occ):
            search_cache[unit_id] = IndustrySearchResult(industry_summary="Not employed", urls=[])
            continue
        try:
            result, raw = search_industry_gemini(client, name, employer, occ)
            search_cache[unit_id] = result
            raw_response_cache[unit_id] = raw
        except Exception as e:
            print(f"  Error for {name!r} / {employer!r}: {e}")
            search_cache[unit_id] = IndustrySearchResult(industry_summary="Did not find", urls=[], confidence="unknown")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test", type=int, metavar="N", help="Run on a random sample of N classification units.")
    mode.add_argument("--full", action="store_true", help="Run on every row in the input file.")
    p.add_argument("--input", default=DEFAULT_INPUT_PATH,
                    help=f"Classification input CSV (default: {DEFAULT_INPUT_PATH})")
    p.add_argument("--output", help="Output CSV path (default: 09_outputs/web_search_output_<mode>_<date>.csv)")
    p.add_argument("--skip-existing", nargs="*", default=[], metavar="CSV",
                    help="Prior web_search.py output file(s); units already present there are skipped.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for --test sampling (default: 42).")
    return p.parse_args()


def default_output_path(args) -> str:
    mode = f"test{args.test}" if args.test else "full"
    out_dir = Path(__file__).parent / "09_outputs"
    out_dir.mkdir(exist_ok=True)
    return str(out_dir / f"web_search_output_{mode}_{date.today().isoformat()}.csv")


def main():
    args = parse_args()
    output_path = args.output or default_output_path(args)

    load_dotenv()
    from google import genai
    client = genai.Client()
    print("Gemini client ready")

    df = pd.read_csv(args.input)
    str_cols = df.select_dtypes(include=["object", "str"]).columns
    df[str_cols] = df[str_cols].fillna("")
    require_columns(df, args.input)

    unit_col = COLUMNS["unit_id"]
    name_col = COLUMNS["name"]
    employer_col = COLUMNS["employer"]
    occ_col = COLUMNS["occupation"]

    before = len(df)
    df = df.drop_duplicates(subset=[unit_col]).reset_index(drop=True)
    if len(df) != before:
        print(f"Dropped {before - len(df):,} duplicate rows sharing a {unit_col}")

    for prior_path in args.skip_existing:
        prior = pd.read_csv(prior_path)
        require_columns(prior, prior_path, needed=("unit_id",))
        seen = set(prior[unit_col].astype(str))
        n_before = len(df)
        df = df[~df[unit_col].astype(str).isin(seen)].reset_index(drop=True)
        print(f"Skipped {n_before - len(df):,} rows already in {prior_path}")

    if args.test:
        n = min(args.test, len(df))
        sample = df.sample(n=n, random_state=args.seed).reset_index(drop=True)
        print(f"TEST run: {len(sample):,} of {len(df):,} rows")
    else:
        sample = df
        print(f"FULL run: {len(sample):,} rows")

    if sample.empty:
        print("Nothing to do.")
        return

    rows = list(zip(sample[unit_col], sample[name_col], sample[employer_col], sample[occ_col]))

    print(f"Searching {len(rows):,} classification units...")
    run_searches(client, rows)

    def _get(unit_id, field):
        result = search_cache.get(unit_id)
        return getattr(result, field, [] if field == "urls" else "")

    sample["industry_summary"] = sample[unit_col].map(lambda u: _get(u, "industry_summary"))
    sample["urls"] = sample[unit_col].map(lambda u: _get(u, "urls"))
    sample["confidence"] = sample[unit_col].map(lambda u: _get(u, "confidence"))

    # retry pass: re-run any entry with a bad or empty summary
    retry_rows = [
        (uid, name, employer, occ) for uid, name, employer, occ in rows
        if search_cache.get(uid, IndustrySearchResult(industry_summary="", urls=[])).industry_summary.strip().lower() in _BAD_SUMMARIES
    ]
    if retry_rows:
        print(f"Retrying {len(retry_rows):,} entries with no/bad summary...")
        run_searches(client, retry_rows)
        sample["industry_summary"] = sample[unit_col].map(lambda u: _get(u, "industry_summary"))
        sample["urls"] = sample[unit_col].map(lambda u: _get(u, "urls"))
        sample["confidence"] = sample[unit_col].map(lambda u: _get(u, "confidence"))

    if parse_error_names:
        print(f"Parse errors for {len(parse_error_names)} entries (first 10): {sorted(parse_error_names)[:10]}")

    sample.to_csv(output_path, index=False)
    print(f"Saved {len(sample):,} rows to {output_path}")


if __name__ == "__main__":
    main()
