"""Web-search industry lookups for the LLM classifier.

Runs one Gemini + Google Search lookup per classification unit (entity x
employer x occupation alias) in the pipeline's classification input, and
saves the input rows back out with industry_summary / urls / confidence
columns attached. Run this first -- classify_web_results.py consumes its
output.

The script automatically loads 09_outputs/web_search_cache.csv (if it exists)
to skip already-searched contributors, and appends new results to it after
each run. Build the cache from prior outputs first with build_web_search_cache.py.

Examples:
    python web_search.py --test 20
    python web_search.py --full
    python web_search.py --full --skip-existing some_other_output.csv
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
from standardization_helpers import (
    normalize_for_key, normalize_employer_for_key, NOT_EMPLOYED,
    lightly_process_name, lightly_process_employer, standardize_occupation_employer,
    standardize_city, standardize_state,
)

MODEL = "gemini-2.5-flash"

CACHE_PATH = Path(__file__).resolve().parent / "09_outputs" / "web_search_cache.csv"

_CACHE_COLS = [
    "search_key",
    "Contributor.Name", "Contributor.Employer",   # raw source — most stable reference
    "standardized_name", "standardized_employer_name",
    "standardized_city", "Contributor.State",     # search context for not-employed
    "industry_summary", "urls", "confidence", "is_prominent", "prominence_reason",
]

_NOT_EMPLOYED = NOT_EMPLOYED  # imported from standardization_helpers
_BAD_SUMMARIES = {"did not find", ""}  # "unknown" is a valid result for unidentifiable people

search_cache: dict = {}
raw_response_cache: dict = {}
parse_error_names: set = set()


class IndustrySearchResult(BaseModel):
    industry_summary: str
    urls: list[str]
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"
    is_prominent: bool = False
    prominence_reason: str = ""


def _is_not_employed(employer: str, occupation: str) -> bool:
    e = (employer or "").strip().lower()
    o = (occupation or "").strip().lower()
    return (not e or e in _NOT_EMPLOYED) and (not o or o in _NOT_EMPLOYED)


def _make_key(name_raw: str, employer_raw: str, employer_processed: str, occ: str,
              city: str = "", state: str = "", entity_type: str = "individual") -> tuple:
    """
    Composite dedup/cache key. Components are normalized via normalize_for_key()
    (uppercase + alphanumeric only) so the key is stable across processing changes.
    City+state included only for not-employed individuals (never for organizations,
    whose standardized_employer_name is legitimately blank).
    employer_processed is used only for the _is_not_employed() check.
    """
    key_name     = normalize_for_key(name_raw)
    key_employer = normalize_employer_for_key(employer_raw)
    if entity_type == "individual" and _is_not_employed(employer_processed, occ):
        return (key_name, key_employer, normalize_for_key(city), normalize_for_key(state))
    return (key_name, key_employer, "", "")


def search_industry_gemini(
    client, name: str, employer: str, occupation: str = "",
    city: str = "", state: str = "",
    not_employed: bool = False,
) -> tuple[IndustrySearchResult, str]:
    from google.genai import types

    if not_employed:
        user_msg = (
            f"Contributor name: {name}\n"
            f"City: {city or 'Not provided'}\n"
            f"State: {state or 'Not provided'}"
        )
        system_instruction = (
            "You are a research assistant searching for information about California campaign finance donors. "
            "This person lists no employer or occupation. Identifying the specific individual by name alone is difficult — be wary multiple people may share this name. "
            "Use web search to try to find this person. If you can identify them, provide a 1-2 sentence summary of who they are. "
            "If you cannot confidently identify the right person, set industry_summary to 'Unknown'. "
            "Set 'is_prominent' to true if the person appears to be a billionaire, major business executive, "
            "influential political figure, major donor, or someone else notable enough to warrant further review. "
            "Set 'prominence_reason' to a brief explanation if is_prominent is true, otherwise leave it empty. "
            "Return ONLY a raw JSON object with keys: industry_summary, urls, confidence, is_prominent (boolean), prominence_reason (string). "
            "No markdown, no prose -- just the JSON."
        )
    else:
        user_msg = (
            f"Contributor name: {name}\n"
            f"Employer: {employer or 'Non-individual'}\n"
            f"Occupation: {occupation or 'Not provided'}"
        )
        system_instruction = (
            "You are a research assistant identifying California campaign finance contributors. "
            "Use web search to look up the contributor and return a single JSON object with these keys:\n"
            '- "industry_summary": 1-2 sentences describing ONLY the organization\'s or employer\'s business. Do NOT include URLs, confidence ratings, or any metadata in this field.\n'
            '  - For individuals: describe what the employer does.\n'
            '  - For non-individual contributors (employer is "Non-individual"): describe the entity and its purpose.\n'
            '  - If the contributor is a PAC or political committee, note that, and also describe what the PAC supports.\n'
            '  - If you cannot find any information, set to "Unknown".\n'
            '- "urls": list of URLs most useful for this contributor. Empty list if none found.\n'
            '- "confidence": "high" if clearly identified and detailed information on the contributor is available, '
            '"medium" if clear match, but the description of the contributor is short or vague. '
            '"low" if multiple possible matches, partial match, or limited info, "unknown" if nothing found.\n'
            '- "is_prominent": true if the person appears to be a billionaire, major business executive, influential political figure, or major donor.\n'
            '- "prominence_reason": brief explanation if is_prominent is true, otherwise empty string.\n'
            "Return ONLY a raw JSON object. No markdown, no prose, no explanation -- just the JSON."
        )

    last_exc = None
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
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
                data["confidence"] = str(data["confidence"]).lower()
            if "prominence_reason" not in data:
                data["prominence_reason"] = ""
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
    is_prominent = data.get("is_prominent", False) if data else False
    prominence_reason = data.get("prominence_reason", "") if data else ""
    return IndustrySearchResult(
        industry_summary=summary_text, urls=urls, confidence="low",
        is_prominent=is_prominent, prominence_reason=prominence_reason,
    ), response.text


def run_searches(client, rows) -> None:
    """rows: iterable of (search_key, name, employer, occupation, city, state)."""
    for unit_id, name, employer, occ, city, state in tqdm(rows, total=len(rows)):
        not_employed = _is_not_employed(employer, occ)
        try:
            result, raw = search_industry_gemini(
                client, name, employer, occ,
                city=city, state=state, not_employed=not_employed,
            )
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

    name_raw_col  = COLUMNS["name_raw"]
    emp_raw_col   = COLUMNS["employer_raw"]
    name_col      = COLUMNS["name"]
    employer_col  = COLUMNS["employer"]
    occ_col       = COLUMNS["occupation"]
    city_col      = COLUMNS.get("city")
    state_col     = COLUMNS.get("state")

    states = df[state_col].fillna("") if state_col and state_col in df.columns else pd.Series("", index=df.index)

    entity_type_col = COLUMNS["entity_type"]
    # Lightly-processed names/occ for the Gemini search prompt.
    df["_search_name"]     = df[name_col].map(lightly_process_name)
    # For individuals: lightly_process_employer (includes standardize_occupation_employer).
    # If standardized_employer_name is blank (R dropped "N/A" etc.), fall back to
    # standardize_occupation_employer on the raw value so Gemini sees "NONE" not blank.
    # Organizations legitimately have a blank standardized_employer_name — no fallback.
    df["_search_employer"] = [
        lightly_process_employer(std) if (std.strip() or etype == "organization")
        else standardize_occupation_employer((raw or "").strip().upper())
        for std, raw, etype in zip(df[employer_col], df[emp_raw_col], df[entity_type_col])
    ]
    df["_search_occ"]      = df[occ_col].map(lambda o: standardize_occupation_employer(o or ""))
    # Re-apply standardize_city() to catch imperfect upstream R output (e.g. "SAN JOSE, CA 95126").
    # Fall back to raw Contributor.City when standardized_city is blank.
    city_raw_col = "Contributor.City"
    std_cities  = df[city_col].fillna("")  if city_col     and city_col     in df.columns else pd.Series("", index=df.index)
    raw_cities  = df[city_raw_col].fillna("") if city_raw_col in df.columns               else pd.Series("", index=df.index)
    df["_city"] = [standardize_city(s) if s.strip() else standardize_city(r)
                   for s, r in zip(std_cities, raw_cities)]

    # Cache key uses normalize_for_key() on raw columns — stable across processing changes.
    df["_search_key"] = [
        _make_key(nr, er, ep, o, c, s, et)
        for nr, er, ep, o, c, s, et in zip(
            df[name_raw_col], df[emp_raw_col], df["_search_employer"], df["_search_occ"],
            df["_city"], states, df[entity_type_col]
        )
    ]

    before = len(df)
    df = df.drop_duplicates(subset=["_search_key"]).reset_index(drop=True)
    if len(df) != before:
        print(f"Dropped {before - len(df):,} duplicate rows sharing the same search key")

    # Always check the persistent cache first, then any --skip-existing files
    key_strs = df["_search_key"].map(lambda k: "|".join(k))
    skip_paths = ([str(CACHE_PATH)] if CACHE_PATH.exists() else []) + list(args.skip_existing)
    for prior_path in skip_paths:
        prior = pd.read_csv(prior_path)
        prior_str = prior.select_dtypes(include=["object", "str"]).columns
        prior[prior_str] = prior[prior_str].fillna("")

        if "search_key" in prior.columns:
            seen = set(prior["search_key"].dropna().str.strip())
        elif "standardized_name" in prior.columns and "processed_employer_name" in prior.columns:
            # Old notebook format: no city/state, so not-employed city distinctions are lost
            seen = set(
                (prior["standardized_name"] + "|" + prior["processed_employer_name"] + "||").values
            )
            print(f"Note: {Path(prior_path).name} is old format — not-employed city/state distinctions not preserved.")
        else:
            print(f"Warning: {Path(prior_path).name} has no recognized key columns — skipping.")
            continue

        n_before = len(df)
        df = df[~key_strs.isin(seen)].reset_index(drop=True)
        key_strs = df["_search_key"].map(lambda k: "|".join(k))
        print(f"Skipped {n_before - len(df):,} rows already in {Path(prior_path).name}")

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

    cities_s = sample["_city"]
    states_s = sample[state_col].fillna("") if state_col and state_col in sample.columns else pd.Series("", index=sample.index)

    rows = list(zip(
        sample["_search_key"], sample["_search_name"], sample["_search_employer"], sample["_search_occ"],
        cities_s, states_s,
    ))

    print(f"Searching {len(rows):,} classification units...")
    run_searches(client, rows)

    def _get(key, field):
        result = search_cache.get(key)
        return getattr(result, field, [] if field == "urls" else "")

    sample["industry_summary"]  = sample["_search_key"].map(lambda k: _get(k, "industry_summary"))
    sample["urls"]              = sample["_search_key"].map(lambda k: _get(k, "urls"))
    sample["confidence"]        = sample["_search_key"].map(lambda k: _get(k, "confidence"))
    sample["is_prominent"]      = sample["_search_key"].map(lambda k: _get(k, "is_prominent"))
    sample["prominence_reason"] = sample["_search_key"].map(lambda k: _get(k, "prominence_reason"))

    # retry pass: re-run any entry with a bad or empty summary
    # "unknown" is intentionally excluded — it's a valid result when the person can't be identified
    retry_rows = [
        row for row in rows
        if search_cache.get(row[0], IndustrySearchResult(industry_summary="", urls=[])).industry_summary.strip().lower() in _BAD_SUMMARIES
    ]
    if retry_rows:
        print(f"Retrying {len(retry_rows):,} entries with no/bad summary...")
        run_searches(client, retry_rows)
        sample["industry_summary"]  = sample["_search_key"].map(lambda k: _get(k, "industry_summary"))
        sample["urls"]              = sample["_search_key"].map(lambda k: _get(k, "urls"))
        sample["confidence"]        = sample["_search_key"].map(lambda k: _get(k, "confidence"))
        sample["is_prominent"]      = sample["_search_key"].map(lambda k: _get(k, "is_prominent"))
        sample["prominence_reason"] = sample["_search_key"].map(lambda k: _get(k, "prominence_reason"))

    if parse_error_names:
        print(f"Parse errors for {len(parse_error_names)} entries (first 10): {sorted(parse_error_names)[:10]}")

    # Serialize key for cache and future --skip-existing use
    sample["search_key"] = sample["_search_key"].map(lambda k: "|".join(k))
    sample = sample.drop(columns=["_search_key"])

    sample.to_csv(output_path, index=False)
    print(f"Saved {len(sample):,} rows to {output_path}")

    # Append new results to the persistent cache
    new_cache_rows = sample[[c for c in _CACHE_COLS if c in sample.columns]].copy()
    if CACHE_PATH.exists():
        existing = pd.read_csv(CACHE_PATH)
        updated = pd.concat([existing, new_cache_rows], ignore_index=True)
        updated = updated.drop_duplicates(subset=["search_key"], keep="last")
    else:
        CACHE_PATH.parent.mkdir(exist_ok=True)
        updated = new_cache_rows
    updated.to_csv(CACHE_PATH, index=False)
    print(f"Cache updated: {len(updated):,} total entries at {CACHE_PATH}")


if __name__ == "__main__":
    main()
