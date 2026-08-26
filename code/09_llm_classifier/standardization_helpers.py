"""Python ports of the R standardization functions in 08_alternative_pipeline/standardization_helpers.R.

Used by 0901_web_search.py, its notebook, and build_web_search_cache.py to replicate
the name processing applied upstream in the R pipeline.
"""

import re


# ---------------------------------------------------------------------------
# Minimal key normalization — used for cache keys ONLY
# Deliberately has no decisions: uppercase + alphanumeric only + collapse spaces.
# This will never change across pipeline versions, making cache keys stable.
# ---------------------------------------------------------------------------

def normalize_for_key(s: str) -> str:
    """Uppercase, strip all non-alphanumeric, collapse whitespace. Nothing more."""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]", " ", s)
    return " ".join(s.split())


# Canonical "not employed" markers — used by _is_not_employed() and normalize_employer_for_key()
NOT_EMPLOYED = {
    "not employed", "unknown", "retired", "homemaker", "student", "n/a", "none",
    "unemployed", "housewife", "househusband", "stay at home", "stay-at-home",
    "na", "-", "nan",
}


def normalize_employer_for_key(s: str) -> str:
    """Normalize a raw employer string for use as a cache key component.

    Applies standardize_occupation_employer() first (maps N/A → NONE,
    NOT EMPLOYED → NONE, etc.), then maps any not-employed result to ''
    so that RETIRED / UNKNOWN / N/A / blank all produce the same key.
    """
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    if not s.strip():
        return ""
    normalized = standardize_occupation_employer(s.strip().upper())
    if normalized.lower() in NOT_EMPLOYED:
        return ""
    return normalize_for_key(s)


# ---------------------------------------------------------------------------
# Basic normalization (mirrors standardize_names())
# ---------------------------------------------------------------------------

def standardize_names(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.upper()
    s = s.replace(" & ", " AND ")
    s = re.sub(r"#", "NUMBER", s)
    s = re.sub(r" NO ", " NUMBER ", s)
    s = re.sub(r"[.']", "", s)
    s = re.sub(r",,", ",", s)
    s = re.sub(r"\?=s", "'", s)
    s = re.sub(r"\?", "", s)
    s = re.sub(r" ,", ",", s)
    return " ".join(s.split())


def fix_typos(s: str) -> str:
    s = s.upper()
    s = s.replace("ENTITITES", "ENTITIES")
    s = s.replace("ENTITES", "ENTITIES")
    s = re.sub(r"COMMITTE ", "COMMITTEE ", s)
    s = s.replace("CAMMITTEE", "COMMITTEE")
    s = s.replace("VACINITY", "VICINITY")
    s = s.replace("AND AFFILIATES ENTITIES", "AND AFFILIATED ENTITIES")
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# City / state normalization
# ---------------------------------------------------------------------------

def standardize_city(s: str) -> str:
    """
    Mirrors R standardize_city(). Note: trailing state abbreviation is stripped
    BEFORE uppercasing to avoid matching mid-word lowercase letter pairs.
    """
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = re.sub(r"\s+[A-Z]{2}$", "", s)   # strip trailing state abbrev (before upper)
    s = s.upper()
    s = re.sub(r",.*$", "", s)            # remove everything after a comma
    s = re.sub(r"\s+\d{5}(-\d{4})?$", "", s)  # remove trailing zip
    s = re.sub(r"[^A-Z0-9 ]", "", s)     # remove non-alphanumeric (equiv to [^[:alnum:] ])
    return " ".join(s.split())


def standardize_state(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.upper()
    s = re.sub(r"\s+\d{5}(-\d{4})?$", "", s)
    s = re.sub(r"[^A-Z0-9 ]", "", s)
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# PAC / union info removal
# ---------------------------------------------------------------------------

def remove_pac_info(s: str) -> str:
    s = re.sub(r"FED\s*ID\s*NUMBER\s*[C]\d+", "", s)
    s = re.sub(r"\s*\([^)]*(ID|FPPC|FEC)[^)]*\)", "", s)
    s = re.sub(r"\s*(ID(?:\s*NUMBER)?|FPPC|FEC)\s*[A-Z0-9]+$", "", s)
    return " ".join(s.split())


def remove_unit_info(s: str) -> str:
    s = re.sub(r"\s+LOCAL(\s+UNION)?(\s+NO|\s+NUMBER)?\s*\d+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+NUMBER\s*\d+", "", s, flags=re.IGNORECASE)
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Occupation / employer normalization
# ---------------------------------------------------------------------------

def standardize_occupation_employer(s: str) -> str:
    _exact = [
        (r"^N/A$", "NONE"), (r"^NA$", "NONE"), (r"^N A$", "NONE"),
        (r"^BLANK$", "UNKNOWN"),
        (r"^NONE OF YOUR BUSINESS$", "UNKNOWN"),
        (r"^PREFER NOT TO DISCLOSE$", "UNKNOWN"),
        (r"^NOT EMPLOYED \(RETIRED\)$", "RETIRED"),
        (r"^NONE \(RETIRED\)$", "RETIRED"),
        (r"^NOT EMPLOYED-RETIRED$", "RETIRED"),
        (r"^NONE-RETIRED$", "RETIRED"),
        (r"^RETIRED NONE$", "RETIRED"),
        (r"^RETIRED NOT EMPLOYED", "RETIRED"),
        (r"^NOT EMPLOYED$", "NONE"),
        (r"^UNEMPLOYED$", "NONE"),
        (r"^NO$", "NONE"),
        (r"^NOT EMPOYED$", "NONE"),
        (r"^NOT EMLOYED$", "NONE"),
        (r"^NOT-EMPLOYED$", "NONE"),
        (r"^NOT RMPLOYED$", "NONE"),
        (r"^NOT EMPLOYYED$", "NONE"),
        (r"^A, N \/$", "NONE"),
    ]
    _partial = [
        ("SELF-EMPLOYED", "SELF EMPLOYED"),
        ("CHIEF EXECUTIVE OFFICER", "CEO"),
        ("CHIEF TECHNOLOGY OFFICER", "CTO"),
        ("CHIEF OPERATING OFFICER", "COO"),
        ("CHIEF FINANCIAL OFFICER", "CFO"),
        (r"EXEC ", "EXECUTIVE "),
        (r"EXEC$", "EXECUTIVE"),
        (r"^INFORMATION REQUESTED$", "UNKNOWN"),
        (r"INFORMATION REQUESTED-?\s*", ""),
    ]
    for pattern, replacement in _exact + _partial:
        s = re.sub(pattern, replacement, s)
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Lighter processing for Gemini search prompts (per the R code in 0901)
# Preserves business text and committee language — useful context for the LLM.
# ---------------------------------------------------------------------------

def lightly_process_name(standardized: str) -> str:
    """
    Lighter processed_name for the search prompt:
      remove_pac_info -> remove_unit_info -> remove parentheticals
      -> expand CA abbreviation -> str_squish
    """
    s = remove_pac_info(standardized)
    s = remove_unit_info(s)
    s = re.sub(r"\s*\(.*?\)", "", s)           # remove parenthetical content
    s = re.sub(r" CA ", " CALIFORNIA ", s)
    return " ".join(s.split())


def lightly_process_employer(standardized: str) -> str:
    """
    Lighter processed_employer_name for the search prompt:
      remove_unit_info -> standardize_occupation_employer -> str_squish
    Retains business type info (LLC, Inc, etc.) — useful context for the LLM.
    """
    s = remove_unit_info(standardized)
    s = standardize_occupation_employer(s)
    return " ".join(s.split())
