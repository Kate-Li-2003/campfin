"""
keyword_priors.py

High-signal keyword -> NAICS(-style) code priors, applied on top of the ML
prediction in 0702. Motivated by the reviewed governor-race benchmark:
the embedding classifier over-predicts 54/81 and ignores strong industry
tokens ("Farms" -> 11, "Energy" -> 22, "Media" -> 51, "Dept of Commerce"
-> 92) and strong occupations ("Farmer" -> 11, "Quantitative Trader" ->
52, "Broadcaster" -> 51).

Policy (approved): a fired prior OVERRIDES the ML prediction only when the
ML naics confidence is below a threshold (default 0.55); above it, the ML
prediction stands. The fired prior is always recorded in
`prior_naics_code` for analyst transparency.

Precedence within this module: employer-name tokens are checked first
(most specific rules first in the list), then occupation tokens. The
general area of business outweighs the business type: "Healthcare Central
LLC" fires 62 (HEALTH) rather than anything suggested by "LLC".
"""

from __future__ import annotations

import re

import pandas as pd

DEFAULT_PRIOR_THRESHOLD = 0.55

# ---- employer / entity-name token priors (ordered: first match wins) ----
# More specific / higher-signal rules first. All matched case-insensitively.
EMPLOYER_PRIORS: list[tuple[str, str]] = [
    # Government (federal/state/local agencies as employer).
    (r"\b(DEPARTMENT OF|DEPT\.? OF|U\.?S\.? (GOVERNMENT|ARMY|NAVY|AIR FORCE)|"
     r"SMALL BUSINESS ADMINISTRATION|SOCIAL SECURITY ADMINISTRATION|"
     r"CITY OF|COUNTY OF|STATE OF|FEDERAL (GOVERNMENT|RESERVE|BUREAU))\b", "92"),
    # Oil & gas beats generic energy.
    (r"\b(OIL|PETROLEUM|DRILLING|GAS (EXPLORATION|PRODUCTION)|FRACKING|WELL SERVICES)\b", "21"),
    # Construction beats "electrical"/"power" utility tokens.
    (r"\b(CONSTRUCTION|BUILDERS|CONTRACTING|CONTRACTORS|ELECTRICAL? SERVICES|"
     r"PLUMBING|ROOFING|PAVING|DRYWALL|EXCAVAT\w*|HOMEBUILD\w*|MASONRY|HVAC)\b", "23"),
    (r"\b(ENERGY|UTILIT\w*|SOLAR|WIND POWER|HYDRO\w*|POWER (CO|COMPANY|CORP|AUTHORITY))\b", "22"),
    # Agriculture.
    (r"\b(FARMS?|RANCH(ES)?|AGRICULTUR\w*|AG|ORCHARDS?|VINEYARDS?|GROWERS?|"
     r"DAIRY|HARVEST\w*|NURSERY|CATTLE|CROP\w*)\b", "11"),
    # Health care.
    (r"\b(HEALTH\w*|MEDICAL|HOSPITALS?|CLINICS?|DENTAL|PHARMACY|HOSPICE|"
     r"WELLNESS|SURGERY|PEDIATRIC\w*|CARDIOLOGY|ONCOLOGY|PSYCHIATR\w*)\b", "62"),
    # Tech / software / AI (code 79 is broad per the Codes sheet). Checked
    # before education so "Academia.edu" -> 79, not 61.
    (r"\b(SOFTWARE|AI|ARTIFICIAL INTELLIGENCE|TECHNOLOG\w*|TECH|ROBOTICS|"
     r"CYBERSECURITY|SEMICONDUCTORS?|LABS)\b|\.(COM|IO|AI|EDU)\b", "79"),
    # Education.
    (r"\b(SCHOOLS?|UNIVERSIT\w*|COLLEGES?|ACADEM(Y|IES)|EDUCATIONAL?)\b", "61"),
    # Media / information.
    # NETWORKS plural only: "LATV Networks" is media, "Freedom Financial
    # Network" is not.
    (r"\b(MEDIA|FILMS?|STUDIOS?|ENTERTAINMENT|PRODUCTIONS?|BROADCAST\w*|"
     r"TELEVISION|TV|NETWORKS|PUBLISHING|PICTURES|RECORDS|TELECOM\w*|"
     r"COMMUNICATIONS?|PODCAST\w*|STREAMING|SHOW)\b", "51"),
    # Transportation & warehousing.
    (r"\b(AVIATION|AIRLINES?|TRUCKING|LOGISTICS|FREIGHT|SHIPPING|"
     r"TRANSPORT\w*|WAREHOUS\w*|MOVING & STORAGE)\b", "48-49"),
    # Wholesale.
    (r"\b(WHOLESALE|DISTRIBUT\w*|SUPPLY|SUPPLIES)\b", "42"),
    # Retail.
    (r"\b(RETAIL\w*|STORES?|SUPERMARKETS?|DEALERSHIPS?|AUTO (SALES|DEALERS?))\b", "44-45"),
    # Manufacturing.
    (r"\b(MANUFACTUR\w*|INDUSTRIES|STEEL|ALUMINUM|PLASTICS?|PHARMA\w*|"
     r"BIOTECH\w*|BIOLOGY|BIOSCIENCES?|LABORATORIES|WINER(Y|IES)|BREWER\w*|"
     r"BEVERAGE|FOODS|PACKAGING|AEROSPACE)\b", "31-33"),
    # Accommodation & food service.
    (r"\b(RESTAURANTS?|HOTELS?|RESORTS?|CATERING|CAFES?|HOSPITALITY|BAKER(Y|IES))\b", "72"),
    # Admin & support services.
    (r"\b(STAFFING|PERSONNEL|RECRUIT\w*|LANDSCAP\w*|JANITORIAL|"
     r"PEST CONTROL|SECURITY (SERVICES|GUARD)|CLEANING)\b", "56"),
    # Sports / arts / recreation.
    (r"\b(SPORTS?|LEAGUE|ATHLETICS?|STADIUM|GOLF (CLUB|COURSE)|FITNESS|"
     r"MUSEUMS?|GALLER(Y|IES))\b", "71"),
    # Real estate (before finance so "Realty Capital" -> 53).
    (r"\b(REAL ESTATE|REALTY|REALTORS?|PROPERTIES|PROPERTY (MANAGEMENT|GROUP|CO)|"
     r"LAND (CO|COMPANY|MANAGEMENT)|APARTMENTS?|COMMUNITIES)\b", "53"),
    # Finance & insurance. (Reviewers code bare "Holdings" companies 52
    # more often than 55, so HOLDINGS deliberately has no prior.)
    (r"\b(BANKS?|BANCORP|CAPITAL|INVESTMENTS?|FINANCIAL|FINANCE|INSURANCE|"
     r"VENTURES?|EQUITY|ASSET MANAGEMENT|CREDIT UNION|TRADING|HEDGE|WEALTH|"
     r"MORTGAGE|SECURITIES)\b", "52"),
    # Law / professional services.
    (r"\b(LAW (FIRM|GROUP|OFFICES?|CORP)|ATTORNEYS?( AT LAW)?|LEGAL|"
     r"ACCOUNTING|ARCHITECT\w*|ENGINEERING|CONSULT\w*)\b", "54"),
]

# ---- occupation token priors (checked when no employer prior fired) ----
OCCUPATION_PRIORS: list[tuple[str, str]] = [
    (r"\b(PETROLEUM|OIL|DRILLING) (ENGINEER|EXECUTIVE)\b", "21"),
    (r"\b(SOFTWARE (ENGINEER|DEVELOPER)|DATA SCIENTIST|PROGRAMMER|CTO)\b", "79"),
    (r"\b(FARMER|RANCHER|GROWER|FARM(ING)?|AGRICULTURE|WINEGRAPE)\b", "11"),
    (r"\b(INVESTOR|TRADER|BANKER|VENTURE CAPITAL\w*|PRIVATE EQUITY|"
     r"FINANCE|FINANCIAL (ADVISOR|PLANNER)|FUND MANAGER|HEDGE FUND)\b", "52"),
    (r"\b(REAL ESTATE|REALTOR|RE (DEV\w*|MANAGER|INVEST\w*)|PROPERTY MANAGER)\b", "53"),
    (r"\b(PHYSICIAN|DOCTOR|DENTIST|NURSE|SURGEON|CHIROPRACT\w*|VETERINAR\w*|"
     r"PSYCHOLOG\w*|THERAPIST|PHARMACIST|OPTOMETRIST|HEALTHCARE)\b", "62"),
    (r"\b(BROADCASTER|JOURNALIST|FILMMAKER|(FILM|TV|MOVIE|MUSIC) PRODUCER|"
     r"PUBLISHER|EDITOR)\b", "51"),
    (r"\b(ARTIST|ACTOR|ACTRESS|MUSICIAN|GOLFER|ATHLETE|SCREENWRITER|"
     r"ENTERTAINER|SINGER)\b", "71"),
    (r"\b(ATTORNEY|LAWYER|ACCOUNTANT|CPA|ARCHITECT|ENGINEER|CONSULTANT|LOBBYIST)\b", "54"),
    (r"\b(TEACHER|PROFESSOR|EDUCATOR|PRINCIPAL|SUPERINTENDENT)\b", "61"),
    (r"\b(CONTRACTOR|BUILDER|CONSTRUCTION)\b", "23"),
    (r"\b(REFUNDED|RETIRED|NOT EMPLOYED|UNEMPLOYED|HOMEMAKER)\b", "99"),
]

CODE_LABELS = {
    "11": "Agriculture, Forestry, Fishing and Hunting",
    "21": "Mining, Quarrying, and Oil and Gas Extraction",
    "22": "Utilities",
    "23": "Construction",
    "31-33": "Manufacturing",
    "42": "Wholesale Trade",
    "44-45": "Retail Trade",
    "48-49": "Transportation and Warehousing",
    "51": "Information",
    "52": "Finance and Insurance",
    "53": "Real Estate and Rental and Leasing",
    "54": "Professional, Scientific, and Technical Services",
    "55": "Management of Companies and Enterprises",
    "56": "Administrative and Support and Waste Management and Remediation Services",
    "61": "Educational Services",
    "62": "Health Care and Social Assistance",
    "71": "Arts, Entertainment, and Recreation",
    "72": "Accommodation and Food Services",
    "76": "Associations (business & professional)",
    "77": "Unions (labor & trade)",
    "78": "Defense contractors",
    "79": "Technology, software and AI companies",
    "81": "Other Services (except Public Administration)",
    "88": "PAC / Political Committee",
    "90": "Candidate committee",
    "91": "Political party",
    "92": "Public Administration",
    "99": "Unknown / Uncategorized",
}

_EMPLOYER_COMPILED = [(re.compile(p, re.IGNORECASE), c) for p, c in EMPLOYER_PRIORS]
_OCCUPATION_COMPILED = [(re.compile(p, re.IGNORECASE), c) for p, c in OCCUPATION_PRIORS]


def prior_for(employer, occupation) -> str | None:
    """Return the prior code for one entity, or None if nothing fires.
    Employer-name tokens win over occupation tokens."""
    emp = "" if pd.isna(employer) else str(employer)
    occ = "" if pd.isna(occupation) else str(occupation)
    if emp:
        for rx, code in _EMPLOYER_COMPILED:
            if rx.search(emp):
                return code
    if occ:
        for rx, code in _OCCUPATION_COMPILED:
            if rx.search(occ):
                return code
    return None


def apply_keyword_priors(
    df: pd.DataFrame,
    mask: pd.Series,
    conf_col: str = "ml_naics_code_conf",
    threshold: float = DEFAULT_PRIOR_THRESHOLD,
    employer_col: str = "employer",
    occupation_col: str = "occupation_norm",
) -> pd.DataFrame:
    """Record fired priors in `prior_naics_code` for all `mask` rows and
    override naics_code/naics_label (data_source_1 = 'keyword prior') where
    the ML confidence is below `threshold` (or missing).

    Only rows in `mask` (the ML-handled, previously-unmatched rows) are
    touched; masterfile/keyword/EDD classifications are never overridden.
    """
    if "prior_naics_code" not in df.columns:
        df["prior_naics_code"] = pd.Series([pd.NA] * len(df), dtype="object")

    occ = (
        df[occupation_col]
        if occupation_col in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )
    fired = {
        i: p
        for i in df.index[mask]
        if (p := prior_for(df.at[i, employer_col], occ.at[i])) is not None
    }
    if not fired:
        print("  keyword priors: none fired")
        return df

    idx = pd.Index(fired.keys())
    df.loc[idx, "prior_naics_code"] = pd.Series(fired)

    conf = df[conf_col] if conf_col in df.columns else pd.Series(index=df.index, dtype=float)
    low_conf = conf.reindex(idx).isna() | (conf.reindex(idx) < threshold)
    ml_val = (
        df["ml_naics_code"].reindex(idx).astype("string")
        if "ml_naics_code" in df.columns
        else pd.Series(pd.NA, index=idx, dtype="string")
    )
    disagree = ml_val.isna() | (ml_val != pd.Series(fired).astype("string"))
    override_idx = idx[(low_conf & disagree).to_numpy()]

    if "naics_code" in df.columns:
        df["naics_code"] = df["naics_code"].astype(object)
    for i in override_idx:
        code = fired[i]
        df.at[i, "naics_code"] = code
        if "naics_label" in df.columns:
            df.at[i, "naics_label"] = CODE_LABELS.get(code, pd.NA)
        df.at[i, "data_source_1"] = "keyword prior"

    print(
        f"  keyword priors: fired on {len(fired):,} rows; overrode ML "
        f"(conf < {threshold:.2f} & disagree) on {len(override_idx):,}"
    )
    return df
