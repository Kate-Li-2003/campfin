"""
Classify IE Spenders - 2026 CA Governor Race
=============================================
Applies the NLP pipeline from Label_Organization-Categories-NLP
(TF-IDF vectorization + KNN search) to classify the industry/sector
affiliation of independent expenditure (IE) committees spending in
the 2026 California gubernatorial race.

Data source: /data/040726_CalAccess_BulkData (CalAccess bulk download, 2026-04-07)

IE form types in CalAccess:
  F496  — 24-hour Late Independent Expenditure Report (used in 2026 race).
            Cover record in CVR_CAMPAIGN_DISCLOSURE_CD; expenditure amounts
            are not included in the bulk EXPN_CD file for this form type.
  F465  — General Independent Expenditure Report (historical races).
            Cover record in CVR_CAMPAIGN_DISCLOSURE_CD; line-item amounts
            in EXPN_CD (FORM_TYPE='F465P3').

Classification target: FILER_NAML — the IE committee/organization name.
"""

import re
import subprocess
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
BULK_DIR   = DATA_DIR / "040726_CalAccess_BulkData" / "DATA"
TRAIN_PATH = BASE_DIR / "Label-Organization-Categories-NLP" / "code" / "data" / "processed" / "OrganizationsFull.tsv"
OUTPUT_PATH = DATA_DIR / "ie_spenders_classified.csv"

CVR_PATH   = BULK_DIR / "CVR_CAMPAIGN_DISCLOSURE_CD.TSV"
EXPN_PATH  = BULK_DIR / "EXPN_CD.TSV"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  TAXONOMY  (mirrors classify_contributors.py)
# ─────────────────────────────────────────────────────────────────────────────

INDUSTRY_SEEDS: dict[str, list[str]] = {
    "Political": [
        "for governor 2026", "for senate 2024", "for assembly 2026",
        "steyer for governor", "villaraigosa for governor", "newsom for governor",
        "bass for mayor", "campaign committee", "exploratory committee",
        "ballot committee", "victory fund", "political action committee",
        "democratic party", "republican party", "progressive campaign",
    ],
    "Ridesharing": [
        "Uber", "Lyft", "DoorDash", "Instacart", "Postmates", "Grubhub",
        "gig economy platform", "ride share", "rideshare", "food delivery app",
        "TaskRabbit", "delivery network", "on-demand transport",
    ],
    "Crypto/AI": [
        "blockchain", "cryptocurrency", "bitcoin", "ethereum", "crypto exchange",
        "web3", "NFT", "decentralized finance", "defi", "coinbase", "binance",
        "artificial intelligence", "machine learning", "deep learning",
        "OpenAI", "Anthropic", "generative AI", "large language model",
        "AI startup", "neural network", "data science platform",
    ],
    "Environment": [
        "environmental defense", "sierra club", "nature conservancy",
        "climate action", "conservation", "clean air", "clean water",
        "sustainability", "environmental advocacy", "green",
        "carbon", "emissions", "wildlife", "land trust", "ocean",
        "350 org", "Earthjustice", "NRDC", "League of Conservation Voters",
    ],
    "Oil": [
        "oil", "petroleum", "crude", "drilling", "refinery", "fracking",
        "natural gas extraction", "pipeline", "offshore drilling",
        "Chevron", "ExxonMobil", "Shell", "BP", "ConocoPhillips",
        "California Resources Corporation", "Ensign United States Drilling",
        "Aera Energy", "Freeport-McMoRan",
    ],
    "Utilities": [
        "electric utility", "water utility", "gas utility",
        "public utility", "PG&E", "Southern California Edison",
        "Southern California Gas", "San Diego Gas", "LADWP",
        "power company", "electric cooperative", "municipal utility",
        "nuclear power", "renewable energy developer", "solar utility",
        "wind farm operator",
    ],
    "Labor": [
        "union", "workers", "labor council", "AFL-CIO", "teamsters",
        "carpenters union", "electricians union", "plumbers union",
        "trade council", "SEIU", "teachers union", "nurses association",
        "United Food Workers", "Operating Engineers",
        "Western States Regional Council of Carpenters",
        "International Brotherhood", "United Auto Workers",
        "firefighters", "police officers association", "peace officers",
        "public employees", "service employees",
    ],
    "Lawyers/Lobbying": [
        "law firm", "attorneys", "legal services", "counsel", "litigation",
        "solicitors", "lawyers", "legal group", "lobbyist", "lobbying firm",
        "government relations", "public affairs",
        "Jones Day", "Latham Watkins", "Skadden Arps", "King Spalding",
        "Manatt Phelps Phillips", "Gibson Dunn", "Covington Burling",
        "Sidley Austin", "Paul Hastings", "O'Melveny Myers",
    ],
    "Health Care": [
        "hospital", "medical center", "health system", "clinic", "pharmacy",
        "pharmaceutical", "biotech", "healthcare", "wellness", "dental",
        "surgery", "urgent care", "health plan", "medical group",
        "Kaiser Permanente", "Cedars Sinai", "UCLA Health", "community health",
        "health foundation", "children's hospital", "HMO", "health insurance",
    ],
    "Ag": [
        "farm", "agriculture", "farming", "ranching", "ranch", "vineyard",
        "winery", "dairy", "produce", "crop", "livestock", "fisheries",
        "aquaculture", "organic farm", "agricultural cooperative",
        "Malibu Valley Farms", "grower", "harvester", "agribusiness",
    ],
    "Real Estate": [
        "real estate", "realty", "properties", "construction",
        "builders", "housing developer", "commercial real estate",
        "property management", "land development", "homebuilder",
        "CBRE", "JLL", "Brookfield", "Lennar", "KB Home",
        "apartment complex", "residential development",
    ],
    "Tech": [
        "technology", "software", "tech company", "systems", "digital",
        "cyber", "cloud computing", "semiconductor", "hardware",
        "internet platform", "IT services", "SaaS", "IoT",
        "Google", "Apple", "Microsoft", "Salesforce", "Oracle", "Meta",
        "Cisco", "IBM", "Intel", "Qualcomm", "Amazon Web Services",
    ],
}

# Keyword map for fast first-pass classification.
# Ordered most-specific-first; first match wins.
KEYWORD_MAP: dict[str, list[str]] = {
    "Ridesharing": [
        "uber", "lyft", "doordash", "instacart", "postmates", "grubhub",
        "rideshare", "ride share", "ride-share", "gig economy",
        "on-demand delivery",
    ],
    "Crypto/AI": [
        "blockchain", "cryptocurrency", "bitcoin", "ethereum", "crypto",
        "web3", "nft", "defi", "decentralized", "coinbase", "binance",
        "artificial intelligence", "machine learning", "deep learning",
        "openai", "anthropic", "generative ai", "large language",
        "ai startup", "neural network",
    ],
    "Environment": [
        "environmental defense", "sierra club", "nature conservancy",
        "climate action", "conservation", "clean air", "clean water",
        "sustainability", "environmental advocacy", "green energy",
        "carbon", "emissions", "wildlife", "land trust", "earthjustice",
        "nrdc", "league of conservation", "350",
    ],
    "Oil": [
        "oil", "petroleum", "crude", "drilling", "refinery", "fracking",
        "natural gas", "pipeline", "offshore", "oilfield",
        "chevron", "exxon", "shell", "conoco", "california resources",
        "aera energy", "freeport",
    ],
    "Utilities": [
        "electric utility", "water utility", "gas utility",
        "public utility", "pge", "pg&e", "southern california edison",
        "socal gas", "southern california gas", "san diego gas",
        "ladwp", "power company", "electric cooperative",
        "municipal utility", "nuclear power", "solar utility",
        "wind farm",
    ],
    "Labor": [
        "union", "labor council", "afl-cio", "afl cio", "teamster",
        "carpenters", "electricians", "plumbers", "trade council",
        "seiu", "brotherhood", "guild", "federation of workers",
        "nurses", "teachers", "operating engineers",
        "united food workers", "firefighters", "fire fighters",
        "peace officers", "police officers", "public employees",
        "school employees", "faculty members", "labor organizations",
        "service employees international",
    ],
    "Lawyers/Lobbying": [
        "law firm", "law office", "legal services", "attorney",
        "attorneys", "counsel", "litigation", "solicitor", "lawyer",
        "esquire", "llp", "apc", "lobbyist", "lobbying",
        "government relations", "public affairs consulting",
    ],
    "Health Care": [
        "health", "medical", "hospital", "clinic", "pharmacy", "pharma",
        "biotech", "dental", "dentist", "physician", "surgery",
        "wellness", "care center", "health plan", "med center",
        "healthcare", "therapeutics", "diagnostics", "oncology", "hmo",
    ],
    "Ag": [
        "farm", "agriculture", "agricultural", "ranch", "ranching",
        "vineyard", "winery", "dairy", "produce", "livestock",
        "crop", "fisheries", "aquaculture", "grower", "harvester",
        "agribusiness",
    ],
    "Real Estate": [
        "real estate", "realty", "properties", "construction",
        "builder", "housing", "development", "developer", "land company",
        "commercial real", "homebuilder", "property management",
        "apartment", "residential development",
    ],
    "Tech": [
        "technology", "software", "tech", "systems", "digital",
        "cyber", "cloud", "semiconductor", "hardware", "internet",
        "platform", "saas", "iot", "robotics", "data analytics",
        "it services", "network solutions",
    ],
    # Political is last — many IE committee names mention the candidate/office
    # but the spender's identity (labor, environment, etc.) often takes priority.
    # We only fall through to Political if no other keyword matched.
    "Political": [
        "for governor", "for senate", "for assembly", "for mayor",
        "for congress", "for supervisor", "for controller",
        "for treasurer", "for attorney general", "for president",
        "campaign committee", "campaign fund", "exploratory committee",
        "ballot committee", "victory fund", "democratic party",
        "republican party", "political action",
    ],
}

# Substrings that disqualify a text from keyword classification
KEYWORD_BLOCKLIST = [
    "state farm",
    "farm credit",
]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PREPROCESSING  (mirrors classify_contributors.py)
# ─────────────────────────────────────────────────────────────────────────────

ABBREV_MAP = {
    r"\bintl\b": "international",
    r"\bsys\b": "systems",
    r"\btech\b": "technology",
    r"\bsvcs\b": "services",
    r"\bsvc\b": "service",
    r"\bmfg\b": "manufacturing",
    r"\bdist\b": "distribution",
    r"\bassoc\b": "associates",
    r"\bdev\b": "development",
    r"\bmgmt\b": "management",
    r"\bgrp\b": "group",
    r"\bcorp\b": "corporation",
    r"\bllc\b": "limited liability company",
    r"\bllp\b": "limited liability partnership",
    r"\binc\b\.?": "incorporated",
    r"\bltd\b\.?": "limited",
    r"\bco\b\.?": "company",
    # NOTE: do NOT expand "pac" here — it causes false Political matches on
    # every committee name containing "PAC" (e.g. "Opportunity PAC").
}


ENTITY_STRIP = re.compile(
    r"\b(incorporated|inc|llc|llp|l\.l\.c\.|corp|corporation|"
    r"limited|ltd|company|co|plc|lp|l\.p\.)\b\.?",
    re.IGNORECASE,
)

# Strip only the CANDIDATE-specific suffix from IE committee names so that
# the underlying organization identity (e.g. "California Professional Firefighters")
# is isolated.  Coalition descriptions (" -- A coalition of teachers, nurses...")
# are preserved because they contain useful classification signals.
IE_CANDIDATE_SUFFIX = re.compile(
    r"[,\s;-]+"
    r"(?:supporting|in support of|opposing|no on|yes on)"
    r"\s+.+?\s+(?:for governor|for senate|for assembly|for mayor|for congress)"
    r"[^,;]*$",
    re.IGNORECASE,
)


def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower().strip()
    for pattern, replacement in ABBREV_MAP.items():
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
    t = ENTITY_STRIP.sub("", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def strip_ie_suffixes(name: str) -> str:
    """
    Strip candidate-specific suffixes from IE committee names.
    Keeps coalition/org descriptions that reveal the filer's industry identity.

    Examples:
      "California Prof. Firefighters IE PAC"         → unchanged (no candidate suffix)
      "CA Back to Basics Supporting Matt Mahan..."   → "CA Back to Basics"
      "Opportunity PAC -- A coalition of teachers..."→ unchanged (coalition text kept)
    """
    cleaned = IE_CANDIDATE_SUFFIX.sub("", name).strip(" ,;-")
    return cleaned if cleaned else name


# ─────────────────────────────────────────────────────────────────────────────
# 3.  KEYWORD CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def keyword_classify(text: str):
    """Return industry label on keyword match, else None."""
    t = text.lower().strip()
    if any(blocked in t for blocked in KEYWORD_BLOCKLIST):
        return None
    for industry, kws in KEYWORD_MAP.items():
        for kw in kws:
            if re.search(r"\b" + re.escape(kw) + r"\b", t):
                return industry
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TF-IDF + KNN  (mirrors classify_contributors.py)
# ─────────────────────────────────────────────────────────────────────────────

REPO_TO_TARGET = {
    "Computer Equipment & Services":               "Tech",
    "Electronics Manufacturing & Services":        "Tech",
    "Telecom Services & Equipment":                "Tech",
    "Telephone Utilities":                         "Tech",
    "Cable TV":                                    "Tech",
    "Miscellaneous Communications & Electronics":  "Tech",
    "General Trade Unions":                        "Labor",
    "Public Sector Unions":                        "Labor",
    "Transportation Unions":                       "Labor",
    "Oil & Gas":                                   "Oil",
    "Miscellaneous Energy":                        "Oil",
    "Electric Utilities":                          "Utilities",
    "Water Utilities":                             "Utilities",
    "Waste Management":                            "Utilities",
    "Crop Production & Basic Processing":          "Ag",
    "Dairy":                                       "Ag",
    "Livestock":                                   "Ag",
    "Poultry & Eggs":                              "Ag",
    "Forestry & Forest Products":                  "Ag",
    "Agricultural Services & Products":            "Ag",
    "Miscellaneous Agriculture":                   "Ag",
    "Farm Bureau":                                 "Ag",
    "Candidate Committees":                        "Political",
    "Candidate Contributions":                     "Political",
    "Party Committees":                            "Political",
    "Leadership PACs":                             "Political",
    "Joint Candidate Committee":                   "Political",
    "Liberal Policy Organization":                 "Political",
    "Conservative Policy Organization":            "Political",
    "Christian Conservative":                      "Political",
    "Health Professionals":                        "Health Care",
    "Hospitals & Nursing Homes":                   "Health Care",
    "Pharmaceuticals & Health Products":           "Health Care",
    "Health Services":                             "Health Care",
    "Miscellaneous Health":                        "Health Care",
    "Health & Welfare Policy":                     "Health Care",
    "Pro-Environmental Policy":                    "Environment",
    "Environmental Services & Equipment":          "Environment",
    "Animal Rights":                               "Environment",
    "Real Estate":                                 "Real Estate",
    "Home Builders":                               "Real Estate",
    "Construction Services":                       "Real Estate",
    "General Contractors":                         "Real Estate",
    "Special Trade Contractors":                   "Real Estate",
    "Building Materials & Equipment":              "Real Estate",
    "Lawyers & Lobbyists":                         "Lawyers/Lobbying",
    "Accountants":                                 "Lawyers/Lobbying",
}


def build_knn_model(k: int = 3):
    """Build TF-IDF vectorizers and training corpus from OrganizationsFull.tsv."""
    import scipy.sparse as sp

    tsv = pd.read_csv(TRAIN_PATH, sep="\t", dtype=str, low_memory=False)
    tsv["target"] = tsv["Industry"].map(REPO_TO_TARGET)
    tsv = tsv.dropna(subset=["target", "name_org"])
    print(f"  Training orgs after crosswalk: {len(tsv):,} ({tsv['target'].nunique()} categories)")

    labels = tsv["target"].tolist()
    texts  = [preprocess(n) for n in tsv["name_org"].tolist()]

    # Supplement with hand-crafted seeds for categories absent from training data
    for cat in ("Crypto/AI", "Ridesharing"):
        for seed in INDUSTRY_SEEDS.get(cat, []):
            labels.append(cat)
            texts.append(preprocess(seed))

    # Additional Labor seeds for firefighters/police/public safety
    extra_labor = [
        "California Professional Firefighters", "firefighters association",
        "police officers association", "peace officers research association",
        "public safety employees", "correctional officers",
        "sheriff deputies", "fire protection district",
    ]
    for seed in extra_labor:
        labels.append("Labor")
        texts.append(preprocess(seed))

    print(f"  Total training examples: {len(labels):,}")

    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=5000, min_df=2, max_df=0.9, sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5),
        max_features=3000, min_df=2, max_df=0.9, sublinear_tf=True,
    )

    X_word  = word_vec.fit_transform(texts)
    X_char  = char_vec.fit_transform(texts)
    X_train = sp.hstack([X_word, X_char])

    return word_vec, char_vec, X_train, labels


def knn_predict(query: str, word_vec, char_vec, X_train, train_labels,
                k: int = 3):
    """Weighted cosine-similarity KNN."""
    import scipy.sparse as sp
    q = sp.hstack([word_vec.transform([query]), char_vec.transform([query])])
    sims = cosine_similarity(q, X_train)[0]
    top_k = np.argsort(sims)[::-1][:k]
    votes: dict[str, float] = defaultdict(float)
    for idx in top_k:
        votes[train_labels[idx]] += sims[idx]
    best  = max(votes, key=lambda x: votes[x])
    total = sum(votes.values())
    return best, (votes[best] / total if total > 0 else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN CLASSIFICATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def classify_filer(
    filer_name: str,
    word_vec, char_vec, X_train, train_labels: list[str],
    k: int = 3,
    min_confidence: float = 0.40,
) -> dict:
    """
    Classify an IE filer name into an industry category.
    Uses the same pipeline as classify_contributors.py:
      1. Keyword match on raw name
      2. Keyword match on stripped name (removes IE-committee boilerplate)
      3. KNN on preprocessed stripped name
    """
    raw      = filer_name.strip()
    stripped = strip_ie_suffixes(raw)           # remove "... for Governor 2026" etc.
    proc     = preprocess(stripped)
    proc_raw = preprocess(raw)

    # Step 1: keyword on raw full name first — catches coalition descriptions
    # (e.g. "Opportunity PAC -- A coalition of teachers, nurses...")
    kw = keyword_classify(raw.lower()) or keyword_classify(proc_raw)

    # Step 2: if no sector keyword yet, try the stripped name
    if not kw:
        kw = keyword_classify(stripped.lower()) or keyword_classify(proc)

    if kw:
        return {
            "ie_spender_text": stripped or raw,
            "industry_category": kw,
            "confidence": 0.90,
            "method": "keyword",
        }

    # Step 3: KNN on preprocessed stripped name
    if proc:
        knn_cat, knn_conf = knn_predict(proc, word_vec, char_vec, X_train, train_labels, k)
    else:
        knn_cat, knn_conf = "Unknown", 0.0

    if knn_conf >= min_confidence:
        return {
            "ie_spender_text": stripped or raw,
            "industry_category": knn_cat,
            "confidence": round(knn_conf, 3),
            "method": "knn",
        }

    return {
        "ie_spender_text": stripped or raw,
        "industry_category": knn_cat if knn_conf > 0.0 else "Unknown",
        "confidence": round(knn_conf, 3),
        "method": "knn_low_conf",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_governor_ie_cvr(start_year: int = 2024) -> pd.DataFrame:
    """
    Load IE cover records (F496 + F465) for the CA Governor race.

    start_year: filter to filings with RPT_DATE >= this year.
                Default 2024 targets the 2026 gubernatorial primary.
                Set to None to include all historical governor IEs.

    Returns one row per unique filing (latest amendment kept).
    """
    print(f"  Loading CVR_CAMPAIGN_DISCLOSURE_CD ({CVR_PATH.stat().st_size / 1e6:.0f} MB)…")
    cvr = pd.read_csv(CVR_PATH, sep="\t", on_bad_lines="skip",
                      encoding="latin1", low_memory=False)
    print(f"  CVR rows: {len(cvr):,}")

    # Filter to governor IE form types
    ie_mask = cvr["FORM_TYPE"].isin(["F496", "F465"])
    gov_mask = cvr["OFFICE_CD"] == "GOV"
    # Also catch any rows where OFFIC_DSCR mentions Governor
    gov_dscr_mask = cvr["OFFIC_DSCR"].str.contains("Governor", case=False, na=False)
    df = cvr[ie_mask & (gov_mask | gov_dscr_mask)].copy()
    print(f"  Governor IE filings (all years): {len(df):,}")

    # Date filter
    df["RPT_DATE_parsed"] = pd.to_datetime(df["RPT_DATE"], errors="coerce",
                                           infer_datetime_format=True)
    if start_year is not None:
        df = df[df["RPT_DATE_parsed"].dt.year >= start_year].copy()
        print(f"  Governor IE filings ({start_year}+): {len(df):,}")

    # Keep only the latest amendment for each FILING_ID
    df = (df.sort_values(["FILING_ID", "AMEND_ID"])
            .groupby("FILING_ID", as_index=False)
            .last())
    print(f"  Unique filings after dedup: {len(df):,}")

    # Build clean candidate name
    df["Candidate_Name"] = (
        df["CAND_NAML"].fillna("") + " " + df["CAND_NAMF"].fillna("")
    ).str.strip()

    return df


def load_f465_expenditures(filing_ids) -> pd.DataFrame:
    """
    Extract F465P3 expenditure line items from EXPN_CD for the given FILING_IDs.
    Uses awk for efficient extraction from the large (2.8 GB) file.

    Strategy: write the target FILING_IDs to a temp file, then awk reads that
    file into a lookup set and filters EXPN_CD rows where FORM_TYPE=F465P3 and
    FILING_ID is in the set.

    Returns a DataFrame with FILING_ID, PAYEE_NAML, AMOUNT, EXPN_DATE, EXPN_DSCR.
    If extraction fails, returns an empty DataFrame (amounts won't be available).
    """
    if not filing_ids:
        return pd.DataFrame()

    import tempfile
    print(f"  Extracting F465P3 line items for {len(filing_ids)} filings from EXPN_CD…")
    print(f"  (This may take several minutes — EXPN_CD is {EXPN_PATH.stat().st_size / 1e9:.1f} GB)")

    # Write filing IDs to a temp file so awk can load them into a set
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fid_file:
        fid_path = fid_file.name
        for fid in filing_ids:
            fid_file.write(f"{int(fid)}\n")

    # awk: load IDs file into id_set, then filter EXPN_CD for F465P3 rows in the set
    awk_cmd = (
        f"awk -F'\\t' '"
        f"NR==FNR{{id_set[$1]=1; next}} "
        f"FNR==1{{print; next}} "
        f"($5==\"F465P3\" && ($1 in id_set)){{print}}"
        f"' '{fid_path}' '{EXPN_PATH}'"
    )

    try:
        result = subprocess.run(
            awk_cmd, shell=True, capture_output=True, timeout=600
        )
        # Clean up temp file
        Path(fid_path).unlink(missing_ok=True)

        raw_output = result.stdout.decode("latin1", errors="replace")
        lines = raw_output.strip()
        if not lines or lines.count("\n") == 0:
            print("  No F465P3 rows found for these FILING_IDs.")
            print("  Note: the EXPN_CD bulk dump only covers F465P3 line items for")
            print("  filings up to ~ID 1.1M (approx. pre-2009). More recent F465")
            print("  line-item amounts are not in this bulk download.")
            return pd.DataFrame()

        from io import StringIO
        expn = pd.read_csv(StringIO(lines), sep="\t",
                           on_bad_lines="skip", low_memory=False)
        expn["FILING_ID"] = pd.to_numeric(expn["FILING_ID"], errors="coerce")
        expn = expn[expn["FILING_ID"].isin(filing_ids)]
        expn["AMOUNT"] = pd.to_numeric(expn["AMOUNT"], errors="coerce")
        print(f"  F465P3 line items found: {len(expn):,}")
        return expn[["FILING_ID", "PAYEE_NAML", "AMOUNT", "EXPN_DATE", "EXPN_DSCR"]].copy()

    except subprocess.TimeoutExpired:
        Path(fid_path).unlink(missing_ok=True)
        print("  Warning: EXPN_CD extraction timed out. Proceeding without line-item amounts.")
        return pd.DataFrame()
    except Exception as e:
        Path(fid_path).unlink(missing_ok=True)
        print(f"  Warning: EXPN_CD extraction failed ({e}). Proceeding without line-item amounts.")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  RUN
# ─────────────────────────────────────────────────────────────────────────────

def main(start_year: int = 2024):
    """
    Classify IE spenders in the CA Governor race.

    Args:
        start_year: Only include filings with RPT_DATE >= this year.
                    Default 2024 captures the 2026 gubernatorial primary.
                    Pass None to classify all historical governor IEs.
    """
    print("=" * 65)
    print("IE SPENDER CLASSIFICATION — CA GOVERNOR RACE")
    print("=" * 65)

    # ── 1. Load CVR governor IE records ──────────────────────────────────────
    print("\n[1] Loading governor IE filings from CVR…")
    cvr_df = load_governor_ie_cvr(start_year=start_year)

    if cvr_df.empty:
        print("No governor IE filings found. Check date range or data path.")
        return

    # ── 2. Load F465 line-item amounts from EXPN_CD ───────────────────────────
    f465_ids = set(
        cvr_df.loc[cvr_df["FORM_TYPE"] == "F465", "FILING_ID"]
        .dropna().astype(int).tolist()
    )
    print(f"\n[2] Loading F465 expenditure line items ({len(f465_ids)} F465 filings)…")
    if f465_ids:
        expn_df = load_f465_expenditures(f465_ids)
    else:
        expn_df = pd.DataFrame()
        print("  No F465 filings in this date range — skipping EXPN_CD extraction.")

    # Note about F496 amounts
    f496_count = (cvr_df["FORM_TYPE"] == "F496").sum()
    if f496_count > 0:
        print(f"\n  Note: {f496_count} F496 (24-hour IE) filings found. "
              "Expenditure amounts for F496 are not stored in the CalAccess "
              "bulk EXPN_CD file. Filer names and candidate info are available.")

    # ── 3. Build NLP model ────────────────────────────────────────────────────
    print(f"\n[3] Building TF-IDF + KNN model from {TRAIN_PATH.name}…")
    word_vec, char_vec, X_train, train_labels = build_knn_model()

    # ── 4. Classify each unique filer ─────────────────────────────────────────
    print("\n[4] Classifying IE spenders…")
    unique_filers = cvr_df["FILER_NAML"].dropna().unique()
    filer_cache: dict[str, dict] = {}
    for name in unique_filers:
        filer_cache[name] = classify_filer(
            name, word_vec, char_vec, X_train, train_labels
        )
    print(f"  Classified {len(filer_cache)} unique IE committees.")

    # ── 5. Build output DataFrame ─────────────────────────────────────────────
    print("\n[5] Assembling output…")

    # Attach classification results to CVR rows
    class_rows = cvr_df["FILER_NAML"].map(filer_cache)
    class_df   = pd.DataFrame(list(class_rows))

    out = pd.DataFrame({
        "Filing_ID":              cvr_df["FILING_ID"].values,
        "Amend_ID":               cvr_df["AMEND_ID"].values,
        "Form_Type":              cvr_df["FORM_TYPE"].values,
        "RPT_Date":               cvr_df["RPT_DATE"].values,
        "Filer_ID":               cvr_df["FILER_ID"].values,
        "Filer_Name":             cvr_df["FILER_NAML"].values,
        "Filer_City":             cvr_df["FILER_CITY"].values,
        "Filer_State":            cvr_df["FILER_ST"].values,
        "Candidate_Name":         cvr_df["Candidate_Name"].values,
        "Sup_Opp":                cvr_df["SUP_OPP_CD"].values,
        "From_Date":              cvr_df["FROM_DATE"].values,
        "Thru_Date":              cvr_df["THRU_DATE"].values,
        "IE_Spender_Text":        class_df["ie_spender_text"].values,
        "Industry_Category":      class_df["industry_category"].values,
        "Classification_Confidence": class_df["confidence"].values,
        "Classification_Method":  class_df["method"].values,
    })

    # Join F465 line-item amounts (total per filing)
    if not expn_df.empty:
        amt_by_filing = (
            expn_df.groupby("FILING_ID")["AMOUNT"]
            .sum()
            .reset_index()
            .rename(columns={"AMOUNT": "IE_Amount_F465"})
        )
        out = out.merge(amt_by_filing, on="Filing_ID", how="left")
    else:
        out["IE_Amount_F465"] = pd.NA

    # ── 6. Save ───────────────────────────────────────────────────────────────
    print(f"\n[6] Saving to {OUTPUT_PATH}…")
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"  {len(out):,} rows written.")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("INDUSTRY CATEGORY DISTRIBUTION (by unique filer)")
    print("=" * 65)
    by_cat = out.groupby("Industry_Category")["Filer_Name"].nunique().sort_values(ascending=False)
    for cat, n in by_cat.items():
        print(f"  {cat:<42}  {n:>4} filer(s)")

    print("\n" + "=" * 65)
    print("IE SPENDERS BY CATEGORY")
    print("=" * 65)
    for _, row in out.sort_values(["Industry_Category", "Filer_Name"]).iterrows():
        sup_opp = "SUPPORT" if row["Sup_Opp"] == "S" else "OPPOSE" if row["Sup_Opp"] == "O" else row["Sup_Opp"]
        amt     = f"${row['IE_Amount_F465']:,.0f}" if pd.notna(row.get("IE_Amount_F465")) else "amt N/A"
        print(f"  [{row['Industry_Category']:<20}]  {sup_opp:<8}  "
              f"{row['Candidate_Name']:<22}  {row['RPT_Date'].split()[0]:<12}  {amt:<12}  "
              f"{row['Filer_Name']}")

    print("\n" + "=" * 65)
    print("CLASSIFICATION METHOD BREAKDOWN")
    print("=" * 65)
    for method, n in out["Classification_Method"].value_counts().items():
        pct = n / len(out) * 100
        print(f"  {method:<25}  {n:>4}  ({pct:.1f}%)")

    print(f"\nDone. Output saved to:\n  {OUTPUT_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify IE spenders in the CA Governor race using TF-IDF + KNN."
    )
    parser.add_argument(
        "--start-year", type=int, default=2024,
        help="Only include filings with RPT_DATE >= this year (default: 2024 for 2026 race). "
             "Use 2000 for all historical governor IEs.",
    )
    args = parser.parse_args()
    main(start_year=args.start_year)
