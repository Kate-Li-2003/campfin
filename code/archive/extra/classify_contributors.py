"""
Contributor Industry Classifier
================================
Applies NLP techniques from github.com/bernicelau430/Label-Organization-Categories-NLP
to classify industry categories for donors in the CA Governor Race 2026 dataset.

Approach (mirrors the repo):
  1. Text preprocessing  – abbreviation expansion, entity-type stripping,
     special-char removal, stopword filtering
  2. Keyword feature detection  – fast first-pass classification
  3. TF-IDF + cosine-similarity KNN  – fallback for ambiguous names
  4. Occupation signal  – used as a tiebreaker / booster
"""

import re
import pickle
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  TAXONOMY  (Industry → representative seed organizations for KNN seeding)
# ─────────────────────────────────────────────────────────────────────────────

INDUSTRY_SEEDS: dict[str, list[str]] = {
    # ── checked before broad Tech bucket ──────────────────────────────────────
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
    # ── energy split into Oil vs Utilities ────────────────────────────────────
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
    # ── labor ─────────────────────────────────────────────────────────────────
    "Labor": [
        "union", "workers", "labor council", "AFL-CIO", "teamsters",
        "carpenters union", "electricians union", "plumbers union",
        "trade council", "SEIU", "teachers union", "nurses association",
        "United Food Workers", "Operating Engineers",
        "Western States Regional Council of Carpenters",
        "International Brotherhood", "United Auto Workers",
    ],
    # ── professional services split ───────────────────────────────────────────
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
    # ── broadest tech bucket – last so specific ones match first ──────────────
    "Tech": [
        "technology", "software", "tech company", "systems", "digital",
        "cyber", "cloud computing", "semiconductor", "hardware",
        "internet platform", "IT services", "SaaS", "IoT",
        "Google", "Apple", "Microsoft", "Salesforce", "Oracle", "Meta",
        "Cisco", "IBM", "Intel", "Qualcomm", "Amazon Web Services",
    ],
    # ── catch-all for non-employed ────────────────────────────────────────────
    "Not Employed / Retired": [
        "not employed", "retired", "homemaker", "self-employed",
        "n/a", "none", "unemployed",
    ],
}

# Keyword lookup — ordered most-specific-first so first match wins
# (Crypto/AI and Ridesharing fire before the broad Tech bucket)
KEYWORD_MAP: dict[str, list[str]] = {
    "Political": [
        "for governor", "for senate", "for assembly", "for mayor",
        "for congress", "for supervisor", "for controller",
        "for treasurer", "for attorney general", "for president",
        "campaign committee", "campaign fund", "exploratory committee",
        "ballot committee", "victory fund", "democratic party",
        "republican party", "political action",
    ],
    "Ridesharing": [
        "uber", "lyft", "doordash", "instacart", "postmates", "grubhub",
        "rideshare", "ride share", "ride-share", "gig economy",
        "tasrabbit", "on-demand delivery",
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
        "nurses association", "teachers union", "operating engineers",
        "united food workers",
    ],
    "Lawyers/Lobbying": [
        "law firm", "law office", "legal services", "attorney",
        "attorneys", "counsel", "litigation", "solicitor", "lawyer",
        "esquire", "llp", "apc", "lawfirm", "lobbyist", "lobbying",
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
        "agribusiness", "malibu valley farms",
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
    "Not Employed / Retired": [
        "not employed", "retired", "homemaker", "self-employed",
        "n/a", "none", "unemployed",
    ],
}

# Occupation → industry hints (used as a tiebreaker)
OCC_HINTS: dict[str, str] = {
    "attorney": "Lawyers/Lobbying",
    "lawyer": "Lawyers/Lobbying",
    "counsel": "Lawyers/Lobbying",
    "solicitor": "Lawyers/Lobbying",
    "lobbyist": "Lawyers/Lobbying",
    "physician": "Health Care",
    "doctor": "Health Care",
    "dentist": "Health Care",
    "nurse": "Health Care",
    "pharmacist": "Health Care",
    "engineer": "Tech",
    "developer": "Tech",
    "software": "Tech",
    "farmer": "Ag",
    "rancher": "Ag",
    "candidate": "Political",
    "politician": "Political",
    "retired": "Not Employed / Retired",
    "homemaker": "Not Employed / Retired",
    "not employed": "Not Employed / Retired",
    # deliberately generic — not assigned
    "executive": None,
    "president": None,
    "ceo": None,
    "owner": None,
    "manager": None,
    "director": None,
    "officer": None,
    "consultant": None,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PREPROCESSING  (mirrors repo's processing.py and model_training.py)
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
}

ENTITY_STRIP = re.compile(
    r"\b(incorporated|inc|llc|llp|l\.l\.c\.|corp|corporation|"
    r"limited|ltd|company|co|plc|lp|l\.p\.)\b\.?",
    re.IGNORECASE,
)


def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower().strip()
    # Expand abbreviations
    for pattern, replacement in ABBREV_MAP.items():
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
    # Remove entity suffixes
    t = ENTITY_STRIP.sub("", t)
    # Strip punctuation except spaces
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ─────────────────────────────────────────────────────────────────────────────
# 3.  KEYWORD-FIRST CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

# Substrings that disqualify a text from keyword classification.
# Checked before keyword matching to block false positives like
# "State Farm" (insurance) or "Farm Credit" (financial institution).
KEYWORD_BLOCKLIST = [
    "state farm",    # insurance brand — appears standalone or inside agent names
    "farm credit",   # agricultural lender, not a farm
]

def keyword_classify(text: str):
    """Return industry label if any keyword matches, else None.
    Uses word-boundary matching and a blocklist to avoid false positives."""
    t = text.lower().strip()
    if any(blocked in t for blocked in KEYWORD_BLOCKLIST):
        return None
    for industry, kws in KEYWORD_MAP.items():
        for kw in kws:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, t):
                return industry
    return None


def occupation_hint(occ: str):
    occ_low = occ.lower().strip()
    for key, cat in OCC_HINTS.items():
        if key in occ_low:
            return cat
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TF-IDF + KNN  — trained on repo's OrganizationsFull.tsv
# ─────────────────────────────────────────────────────────────────────────────

# Crosswalk: repo's 109 Industry labels → our 12 target categories.
# Categories absent from the training data (Crypto/AI, Ridesharing) are
# handled entirely by the keyword pre-pass above.
REPO_TO_TARGET = {
    # Tech
    "Computer Equipment & Services":          "Tech",
    "Electronics Manufacturing & Services":   "Tech",
    "Telecom Services & Equipment":           "Tech",
    "Telephone Utilities":                    "Tech",
    "Cable TV":                               "Tech",
    "Miscellaneous Communications & Electronics": "Tech",
    # Labor
    "General Trade Unions":                   "Labor",
    "Public Sector Unions":                   "Labor",
    "Transportation Unions":                  "Labor",
    # Oil
    "Oil & Gas":                              "Oil",
    "Miscellaneous Energy":                   "Oil",
    # Utilities
    "Electric Utilities":                     "Utilities",
    "Water Utilities":                        "Utilities",
    "Waste Management":                       "Utilities",
    # Ag
    "Crop Production & Basic Processing":     "Ag",
    "Dairy":                                  "Ag",
    "Livestock":                              "Ag",
    "Poultry & Eggs":                         "Ag",
    "Forestry & Forest Products":             "Ag",
    "Agricultural Services & Products":       "Ag",
    "Miscellaneous Agriculture":              "Ag",
    "Farm Bureau":                            "Ag",
    # Beer/Wine/Liquor, Food & Beverage, Food Processing excluded —
    # these span restaurants, distributors and candy companies that
    # aren't meaningfully Agriculture and were pulling in false positives.
    # Political
    "Candidate Committees":                   "Political",
    "Candidate Contributions":                "Political",
    "Party Committees":                       "Political",
    "Leadership PACs":                        "Political",
    "Joint Candidate Committee":              "Political",
    "Liberal Policy Organization":            "Political",
    "Conservative Policy Organization":       "Political",
    "Christian Conservative":                 "Political",
    # Health Care
    "Health Professionals":                   "Health Care",
    "Hospitals & Nursing Homes":              "Health Care",
    "Pharmaceuticals & Health Products":      "Health Care",
    "Health Services":                        "Health Care",
    "Miscellaneous Health":                   "Health Care",
    "Health & Welfare Policy":                "Health Care",
    # Environment
    "Pro-Environmental Policy":               "Environment",
    "Environmental Services & Equipment":     "Environment",
    "Animal Rights":                          "Environment",
    # Real Estate
    "Real Estate":                            "Real Estate",
    "Home Builders":                          "Real Estate",
    "Construction Services":                  "Real Estate",
    "General Contractors":                    "Real Estate",
    "Special Trade Contractors":              "Real Estate",
    "Building Materials & Equipment":         "Real Estate",
    # Lawyers/Lobbying
    "Lawyers & Lobbyists":                    "Lawyers/Lobbying",
    "Accountants":                            "Lawyers/Lobbying",
}

TRAINING_DATA_PATH = Path("/Users/kateli/Desktop/CalMatters/campaign finance categorization/data/OrganizationsFull.tsv")


def build_knn_model(k: int = 3):
    """
    Build TF-IDF vectorizer and labeled training corpus from the repo's
    OrganizationsFull.tsv (18,626 CA lobbying orgs with Industry labels).
    Repo Industry labels are crosswalked to our 12 target categories.
    The hand-crafted seeds in INDUSTRY_SEEDS supplement for categories
    absent from the training data (Crypto/AI, Ridesharing).
    """
    import scipy.sparse as sp

    # ── Load repo training data ───────────────────────────────────────────────
    tsv = pd.read_csv(TRAINING_DATA_PATH, sep="\t", dtype=str, low_memory=False)
    # Map repo Industry label → target category; drop rows with no mapping
    tsv["target"] = tsv["Industry"].map(REPO_TO_TARGET)
    tsv = tsv.dropna(subset=["target", "name_org"])
    print(f"  Repo training orgs after crosswalk: {len(tsv):,} "
          f"(from {tsv['target'].nunique()} categories)")

    labels = tsv["target"].tolist()
    texts  = [preprocess(n) for n in tsv["name_org"].tolist()]

    # ── Supplement with hand-crafted seeds for missing categories ─────────────
    missing_cats = {"Crypto/AI", "Ridesharing"}
    for cat in missing_cats:
        if cat in INDUSTRY_SEEDS:
            for seed in INDUSTRY_SEEDS[cat]:
                labels.append(cat)
                texts.append(preprocess(seed))
    print(f"  Supplemental seeds added for: {missing_cats}")
    print(f"  Total training examples: {len(labels):,}")

    # ── Fit dual TF-IDF vectorizers (repo's optimal params) ───────────────────
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=3000,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )

    X_word  = word_vec.fit_transform(texts)
    X_char  = char_vec.fit_transform(texts)
    X_train = sp.hstack([X_word, X_char])

    return word_vec, char_vec, X_train, labels


def knn_predict(query: str, word_vec, char_vec, X_train, train_labels: list[str], k: int = 3) -> tuple[str, float]:
    """Weighted cosine-similarity KNN, matching repo's voting mechanism."""
    import scipy.sparse as sp
    q_word = word_vec.transform([query])
    q_char = char_vec.transform([query])
    q = sp.hstack([q_word, q_char])

    sims = cosine_similarity(q, X_train)[0]
    top_k_idx = np.argsort(sims)[::-1][:k]

    # Weighted vote by industry
    votes: dict[str, float] = defaultdict(float)
    for idx in top_k_idx:
        votes[train_labels[idx]] += sims[idx]

    best = max(votes, key=lambda x: votes[x])
    total = sum(votes.values())
    confidence = votes[best] / total if total > 0 else 0.0
    return best, confidence


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN CLASSIFICATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def get_text_to_classify(row: pd.Series) -> tuple[str, str]:
    """
    Decide what text represents the 'organization' for this row.
    - Individual donors  → Contributor Employer (+ Occupation as hint)
    - Organizational donors (no occupation) → Contributor Name itself
    """
    employer = str(row.get("Contributor Employer", "") or "").strip()
    occupation = str(row.get("Contributor Occupation", "") or "").strip()
    name = str(row.get("Contributor Name", "") or "").strip()

    # Treat blank / n/a employer as missing
    if employer.lower() in ("", "n/a", "none", "nan"):
        employer = ""

    # If no employer and no occupation → organizational donor → use name
    if not employer and not occupation:
        return name, ""

    org_text = employer if employer else name
    return org_text, occupation


def classify_row(
    row: pd.Series,
    word_vec,
    char_vec,
    X_train,
    train_labels: list[str],
    k: int = 3,
    min_confidence: float = 0.40,
) -> dict:
    org_text, occupation = get_text_to_classify(row)
    processed = preprocess(org_text)
    occ_processed = preprocess(occupation)

    # --- Step 1: special-case "not employed / retired" ---
    for marker in ("not employed", "retired", "homemaker", "n/a", "none", "unemployed"):
        if marker in processed or marker in occ_processed:
            return {
                "org_text": org_text,
                "industry_category": "Not Employed / Retired",
                "confidence": 1.0,
                "method": "special_case",
            }

    # --- Step 2: keyword match on org text ---
    kw_result = keyword_classify(processed) or keyword_classify(org_text.lower())

    # --- Step 3: keyword match on occupation ---
    occ_result = occupation_hint(occupation)

    # --- Step 4: KNN fallback ---
    if processed:
        knn_result, knn_conf = knn_predict(processed, word_vec, char_vec, X_train, train_labels, k)
    else:
        knn_result, knn_conf = "Unknown", 0.0

    # --- Combine signals (mirrors repo's weighted voting) ---
    if kw_result:
        # Keyword match is high-confidence
        return {
            "org_text": org_text,
            "industry_category": kw_result,
            "confidence": 0.90,
            "method": "keyword",
        }
    elif knn_conf >= min_confidence:
        # Use KNN result, but boost with occupation hint if they agree
        final = knn_result
        if occ_result and occ_result != knn_result:
            # Slight tiebreak: keep KNN but note disagreement
            pass
        return {
            "org_text": org_text,
            "industry_category": final,
            "confidence": round(knn_conf, 3),
            "method": "knn",
        }
    elif occ_result:
        # Fall back to occupation signal
        return {
            "org_text": org_text,
            "industry_category": occ_result,
            "confidence": 0.60,
            "method": "occupation_hint",
        }
    else:
        return {
            "org_text": org_text,
            "industry_category": knn_result if knn_conf > 0.0 else "Unknown",
            "confidence": round(knn_conf, 3),
            "method": "knn_low_conf",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  RUN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    INPUT = Path("/Users/kateli/Desktop/CalMatters/campaign finance categorization/data/governor_race_2026-03-20.csv")
    OUTPUT = Path("/Users/kateli/Desktop/CalMatters/campaign finance categorization/data/governor_race_classified.csv")

    print("Loading data …")
    df = pd.read_csv(INPUT, dtype=str)
    print(f"  {len(df):,} rows loaded.")

    print("Building KNN model from repo's OrganizationsFull.tsv …")
    word_vec, char_vec, X_train, train_labels = build_knn_model(k=3)

    print("Classifying contributors …")
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        res = classify_row(row, word_vec, char_vec, X_train, train_labels, k=3)
        results.append(res)
        if (i + 1) % 5000 == 0:
            print(f"  … {i+1:,} / {len(df):,} processed")

    results_df = pd.DataFrame(results)

    # Attach results back to original dataframe
    df["Org_Text_Used"] = results_df["org_text"].values
    df["Industry_Category"] = results_df["industry_category"].values
    df["Classification_Confidence"] = results_df["confidence"].values
    df["Classification_Method"] = results_df["method"].values

    print(f"\nSaving to {OUTPUT} …")
    df.to_csv(OUTPUT, index=False)

    # ── Summary stats ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INDUSTRY CATEGORY DISTRIBUTION")
    print("=" * 60)
    counts = df["Industry_Category"].value_counts()
    total = len(df)
    for cat, n in counts.items():
        bar = "█" * int(n / total * 40)
        print(f"  {cat:<42}  {n:>6,}  ({n/total*100:5.1f}%)  {bar}")

    print("\n" + "=" * 60)
    print("CLASSIFICATION METHOD BREAKDOWN")
    print("=" * 60)
    for method, n in df["Classification_Method"].value_counts().items():
        print(f"  {method:<20}  {n:>6,}  ({n/total*100:5.1f}%)")

    print("\n" + "=" * 60)
    print("TOP CONTRIBUTORS BY AMOUNT (with category)")
    print("=" * 60)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    top = (
        df.groupby(["Contributor Name", "Industry_Category"])["Amount"]
        .sum()
        .reset_index()
        .sort_values("Amount", ascending=False)
        .head(20)
    )
    for _, r in top.iterrows():
        print(f"  ${r['Amount']:>12,.2f}  {r['Contributor Name']:<45}  {r['Industry_Category']}")

    print("\n" + "=" * 60)
    print("TOTAL CONTRIBUTIONS BY INDUSTRY CATEGORY")
    print("=" * 60)
    by_industry = (
        df.groupby("Industry_Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )
    for cat, amt in by_industry.items():
        print(f"  {cat:<42}  ${amt:>14,.2f}")

    print(f"\nDone. Output: {OUTPUT}")


if __name__ == "__main__":
    main()
