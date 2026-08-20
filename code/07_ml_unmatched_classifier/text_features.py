"""
text_features.py

Shared embed-text construction for the 07 ML classifier, used by BOTH
0701 (training augmentation) and 0702 (prediction) so the text format the
model was trained on always matches the text format it predicts on.

Format versions (recorded as text_format.txt in the models dir):
  v1 (legacy): "{employer}; occupation: {occ}"  — employer-led, no junk
      handling. Matches models trained on bare company names only.
  v2:          junk employers dropped; occupation-led for individuals:
      individual: "occupation: {occ}; employer: {emp}"
      org:        "{emp}"
      Rationale (benchmark evidence): personal/junk employer strings carry
      no industry signal and drag predictions toward 54; occupations like
      "farmer"/"investor"/"broadcaster" carry strong signal but were being
      drowned out by the leading employer name.
"""

from __future__ import annotations

import re

import pandas as pd

FORMAT_VERSION = "v2"

# Occupation values that carry no industry signal.
JUNK_OCCUPATIONS = {
    "", "none", "n/a", "na", "unknown", "not employed", "self",
    "information requested", "requested", "refused", "declined",
    "refunded","self-employed","self employed"
}

# Employer strings that carry no industry signal. Covers "Self Employed",
# "Self Employed - Kelly Day", "Self-Employed-Phil Mickelson", "Refunded",
# "Retired", etc. (Employers that are just the donor's own name are caught
# by looks_like_person_name when the caller has no contributor name.)
_JUNK_EMPLOYER_RE = re.compile(
    r"^\s*("
    r"self[\s\-]?employ(ed|ment)?([\s\-].*)?"
    r"|refunded|anonymous|unknown|none|n/?a"
    r"|not employed|unemployed|retired|homemaker|housewife|student"
    r"|same|best efforts?|information requested|requested|declined|refused"
    r")\s*$",
    re.IGNORECASE,
)

# "Surname, First [Middle]" or "First [M.] Last" with 2-4 title-cased
# tokens and no org markers -> likely a person, not a company.
_ORG_TOKEN_RE = re.compile(
    r"\b(INC|LLC|LLP|LP|LTD|CORP|CO|COMPANY|GROUP|PARTNERS|ASSOCIATES|"
    r"HOLDINGS|ENTERPRISES|FARMS|PAC|COMMITTEE|FUND|TRUST|FOUNDATION)\b\.?",
    re.IGNORECASE,
)


def clean_field(v) -> str:
    """NaN/blank/'nan' -> ''."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("", "nan") else s


def is_junk_employer(employer, contributor_name: str | None = None) -> bool:
    """True when the employer string carries no industry signal."""
    emp = clean_field(employer)
    if not emp:
        return True
    if _JUNK_EMPLOYER_RE.match(emp):
        return True
    # Employer that is just the donor's own name ("James D. Jameson" for
    # contributor "Jameson, James") — compare last/first tokens.
    if contributor_name:
        name = clean_field(contributor_name)
        if "," in name:
            last, _, first = (t.strip() for t in name.partition(","))
            first = first.split()[0] if first.split() else ""
            emp_l = emp.lower()
            if (
                last
                and first
                and last.lower() in emp_l
                and first.lower() in emp_l
                and not _ORG_TOKEN_RE.search(emp)
            ):
                return True
    return False


def clean_occupation(occupation) -> str:
    occ = clean_field(occupation)
    return "" if occ.lower() in JUNK_OCCUPATIONS else occ


def build_embed_text(
    employer,
    occupation,
    entity_kind: str | None = None,
    contributor_name: str | None = None,
) -> str:
    """Compose the string fed to the sentence encoder (format v2).

    Junk employers are dropped entirely. For individuals the occupation
    leads (strong plain-language industry signal); for organizations the
    entity name is the signal and occupation is normally absent.
    """
    emp = clean_field(employer)
    if is_junk_employer(emp, contributor_name):
        emp = ""
    occ = clean_occupation(occupation)

    is_individual = (entity_kind or "").strip().lower() == "individual"
    if emp and occ:
        if is_individual:
            return f"occupation: {occ}; employer: {emp}"
        return f"{emp}; occupation: {occ}"
    if occ:
        return f"occupation: {occ}"
    if emp:
        return emp
    return ""


def occupation_train_text(occupation: str) -> str:
    """Training-time text for a bare occupation seed example. Must match
    the serving-side occupation-only branch of build_embed_text."""
    return f"occupation: {clean_occupation(occupation)}"
