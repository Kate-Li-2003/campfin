"""
03_webscraping_addtl_data.py
============================
For individual donors in the 2026 CA Governor's race who:
  (a) donated >= $10,000 in a single transaction ("medium spender" or above), AND
  (b) are NOT matched to an industry classification in the OpenSecrets database
      (i.e., no match, OR matched as "Uncoded"),

fetch a short description of the donor's employer from four sources:

  1. SerpAPI              — returns the Google search snippet for the top result
                            (e.g., "personal injury attorney in Los Angeles, CA"
                            for a query like "Arash Khorsandi attorney").
                            REQUIRES API KEY (free tier: 100 searches/month).

  2. DuckDuckGo Instant   — DuckDuckGo Instant Answer API; returns an AbstractText
                            for entities with Wikipedia/Wikidata entries; silent on
                            small private firms.
                            NO API KEY REQUIRED (completely free).

  3. OpenCorporates       — searches the official California Secretary of State
                            company registry via the OpenCorporates API; returns
                            company type, SIC industry code description, and
                            registered address — best for registered LLCs/corps.
                            NO KEY REQUIRED for free tier (rate-limited ~1 req/s).
                            OPTIONAL API KEY lifts the rate limit.

  4. Bloomberg Profiles   — attempts to load the company's public Bloomberg profile
                            page; falls back to a Bloomberg site-search if the
                            direct URL fails.
                            NO API KEY / LOGIN REQUIRED (public pages only).

A fifth column ("about_description") is filled by following the top SerpAPI result
URL and extracting the page's <meta description> or opening paragraph — useful when
the search snippet is short or cut off.

────────────────────────────────────────────────────────────────────────────────
WHEN YOU WILL BE PROMPTED FOR CREDENTIALS
────────────────────────────────────────────────────────────────────────────────
All prompts happen BEFORE the fetch loop starts, so you won't be interrupted
mid-run.  The order is:

  Step A — SerpAPI key (if --no-serp is NOT set)
    • Required.  Get one at https://serpapi.com/ (100 free searches/month).
    • If you leave it blank the script automatically skips SerpAPI lookups.

  Step B — OpenCorporates API token (if --no-opencorp is NOT set)
    • Optional.  Get one at https://opencorporates.com/api_accounts/new
    • If you leave it blank, the script still runs OpenCorporates queries but
      at the free unauthenticated rate limit (~1 request/second).

DuckDuckGo and Bloomberg require no credentials at any stage.

────────────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────────────
  python 03_webscraping_addtl_data.py [options]

  --min-amount  FLOAT  Minimum single-donation amount (default: 10000)
  --max-rows    INT    Limit to N employers — useful for a test run
  --resume             Skip employers already present in the output CSV
  --no-serp            Skip SerpAPI calls
  --no-ddg             Skip DuckDuckGo Instant Answer calls
  --no-opencorp        Skip OpenCorporates calls
  --no-bloomberg       Skip Bloomberg profile/search calls
  --no-about           Skip About-page scraping (follows SerpAPI top-result URL)
  --delay       FLOAT  Seconds between HTTP requests (default: 1.5)
  --output      PATH   Override output CSV path

────────────────────────────────────────────────────────────────────────────────
OUTPUT  (data/03_output/employer_descriptions.csv)
────────────────────────────────────────────────────────────────────────────────
  employer              raw employer string from the donation CSV
  n_donors              number of qualifying donation rows for this employer
  total_donated         sum of qualifying donations (USD)
  serp_snippet          Google search snippet from SerpAPI
  serp_url              URL of the top SerpAPI result
  ddg_abstract          DuckDuckGo AbstractText (Wikipedia-sourced description)
  ddg_answer            DuckDuckGo direct answer (if any)
  oc_company_type       OpenCorporates: company type (e.g. "Domestic Corporation")
  oc_status             OpenCorporates: current status (e.g. "Active")
  oc_sic_description    OpenCorporates: SIC industry code description
  oc_address            OpenCorporates: registered address
  oc_url                OpenCorporates: profile URL
  bloomberg_description Bloomberg profile description
  bloomberg_url         Bloomberg URL used
  about_description     Meta-description / opening text from top SerpAPI result page
  about_url             URL scraped for about_description
  fetch_errors          Pipe-separated list of any error messages
"""

from __future__ import annotations

import argparse
import csv
import getpass
import re
import time
import urllib.parse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

GOV_RACE_PATH    = DATA_DIR / "01CalAccess_CampaignFinance_Data" / "governor_race_2026-03-20.csv"
OPENSECRETS_PATH = DATA_DIR / "ContributionOpenSecretsCategories (1).csv"
OUTPUT_DIR       = DATA_DIR / "03_output"
DEFAULT_OUTPUT   = OUTPUT_DIR / "employer_descriptions.csv"

DEFAULT_MIN_AMOUNT = 10_000.0

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

SERPAPI_URL         = "https://serpapi.com/search.json"
DDG_API_URL         = "https://api.duckduckgo.com/"
OPENCORP_SEARCH_URL = "https://api.opencorporates.com/v0.4/companies/search"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Domains skipped by the About-page scraper (SERP or blocked domains)
SKIP_DOMAINS_ABOUT = {
    "google.com", "google.co", "bing.com", "yahoo.com", "duckduckgo.com",
    "bloomberg.com", "linkedin.com", "facebook.com", "twitter.com",
    "youtube.com", "wikipedia.org", "wikimedia.org",
}

# ── HELPERS: DATA LOADING ─────────────────────────────────────────────────────

def _clean_amount(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[$,]", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _detect_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalise_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = name.upper()
    n = re.sub(r"[^A-Z0-9 ]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def load_unmatched_medium_plus(
    gov_path: Path,
    os_path: Path,
    min_amount: float,
) -> pd.DataFrame:
    """
    Reproduce the Section 3-4 logic from 02_donation_analysis to identify
    individual donors with:
      - single-donation amount >= min_amount
      - no OpenSecrets match, OR matched only as "Uncoded"
    """
    print(f"\n[1] Loading governor race data from {gov_path.name} …")
    df = pd.read_csv(gov_path, dtype=str, low_memory=False)
    print(f"    {len(df):,} rows loaded.")

    col_name = _detect_column(df, ["Contributor Name", "CTRIB_NAML"])
    col_occ  = _detect_column(df, ["Contributor Occupation", "CTRIB_OCC"])
    col_id   = _detect_column(df, ["Contributor ID", "CMTE_ID", "FILER_ID"])
    col_amt  = _detect_column(df, ["Amount", "AMOUNT", "TRAN_AMT1"])
    col_emp  = _detect_column(df, ["Contributor Employer", "CTRIB_EMP", "employer"])

    if col_amt is None:
        raise ValueError("Cannot find Amount column in the CSV.")
    if col_name is None:
        raise ValueError("Cannot find Contributor Name column in the CSV.")

    df["amount_clean"] = _clean_amount(df[col_amt])
    df["_name"]        = df[col_name].fillna("").str.strip()
    df["_employer"]    = df[col_emp].fillna("").str.strip() if col_emp else ""

    has_occ = (
        df[col_occ].notna() & df[col_occ].str.strip().ne("") & df[col_occ].str.lower().ne("nan")
        if col_occ else pd.Series(False, index=df.index)
    )
    has_id = (
        df[col_id].notna() & df[col_id].str.strip().ne("") & df[col_id].str.lower().ne("nan")
        if col_id else pd.Series(False, index=df.index)
    )
    df["donor_type"] = np.select(
        [has_id, has_occ], ["PAC/Committee", "Individual"], default="Organization"
    )

    indiv       = df[df["donor_type"] == "Individual"].copy()
    medium_plus = indiv[indiv["amount_clean"] >= min_amount].copy()
    print(
        f"    Individual donors with donation >= ${min_amount:,.0f}: "
        f"{len(medium_plus):,} rows ({medium_plus['_name'].nunique():,} unique names)"
    )

    if not os_path.exists():
        print(f"    WARNING: OpenSecrets file not found — treating all as unmatched.")
        medium_plus["os_matched"] = False
        medium_plus["os_level1"]  = None
    else:
        print(f"    Loading OpenSecrets from {os_path.name} …")
        os_df = pd.read_csv(os_path, dtype=str, low_memory=False)
        os_df.columns = [c.strip() for c in os_df.columns]
        os_df["name_norm"] = os_df["name"].apply(normalise_name)
        os_lookup = (
            os_df.drop_duplicates("name_norm")
            .set_index("name_norm")[["level1_category"]]
            .to_dict(orient="index")
        )
        medium_plus["name_norm"] = medium_plus["_name"].apply(normalise_name)
        medium_plus["os_matched"] = medium_plus["name_norm"].isin(os_lookup)
        medium_plus["os_level1"]  = medium_plus["name_norm"].map(
            {k: v["level1_category"] for k, v in os_lookup.items()}
        )

    unmatched = medium_plus[
        (~medium_plus["os_matched"]) |
        (medium_plus["os_level1"].fillna("Uncoded") == "Uncoded")
    ].copy()

    print(
        f"    Unmatched (no industry classification): "
        f"{len(unmatched):,} rows ({unmatched['_name'].nunique():,} unique donor names)"
    )
    print()
    return unmatched


def build_employer_table(unmatched: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by employer, skip blank/generic values, sort by total donated."""
    skip_vals = {
        "", "n/a", "na", "none", "self", "self-employed", "retired",
        "not employed", "homemaker", "student", "unemployed",
    }
    valid = unmatched[
        unmatched["_employer"].str.lower().str.strip().apply(
            lambda v: v not in skip_vals and len(v) > 0
        )
    ].copy()

    table = (
        valid.groupby("_employer")
        .agg(n_donors=("_name", "count"), total_donated=("amount_clean", "sum"))
        .reset_index()
        .rename(columns={"_employer": "employer"})
        .sort_values("total_donated", ascending=False)
        .reset_index(drop=True)
    )
    print(f"[2] {len(table):,} unique employers to look up "
          f"(blank/self/retired employers excluded).\n")
    return table


# ── HELPERS: HTTP ─────────────────────────────────────────────────────────────

def _get(url: str, session: requests.Session, timeout: int = 12) -> Optional[requests.Response]:
    try:
        resp = session.get(url, headers=BROWSER_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp
    except Exception:
        return None


def _soup(resp: requests.Response) -> BeautifulSoup:
    return BeautifulSoup(resp.text, "html.parser")


def _domain(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return ""


# ── SOURCE 1: SERPAPI ─────────────────────────────────────────────────────────

def serpapi_search(
    query: str,
    api_key: str,
    session: requests.Session,
) -> tuple:
    """
    Call the SerpAPI Google Search endpoint.
    Returns (snippet, top_result_url).

    SerpAPI docs: https://serpapi.com/search-api
    Free tier:    100 searches/month.
    Paid:         $50/month for 5,000 searches.
    """
    params = {
        "q":       query,
        "api_key": api_key,
        "engine":  "google",
        "num":     3,       # fetch top 3 so we have fallbacks for About-page scraper
        "hl":      "en",
        "gl":      "us",
    }
    try:
        resp = session.get(SERPAPI_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        # SerpAPI returns an "error" key if the key is invalid/quota exceeded
        if "error" in data:
            return "", f"ERROR:{data['error']}"

        organic = data.get("organic_results", [])
        if organic:
            top = organic[0]
            snippet = top.get("snippet", "").replace("\n", " ").strip()
            url     = top.get("link", "")
            return snippet, url
        return "", ""
    except Exception as exc:
        return "", f"ERROR:{exc}"


# ── SOURCE 2: DUCKDUCKGO INSTANT ANSWER API ───────────────────────────────────

def duckduckgo_search(query: str, session: requests.Session) -> tuple:
    """
    Call the DuckDuckGo Instant Answer API (no key required, free).
    Returns (abstract_text, direct_answer).

    AbstractText is sourced from Wikipedia/Wikidata — reliable for well-known
    companies and public figures; returns empty string for small private firms.
    The 'answer' field covers DuckDuckGo's own direct answers (e.g. calculations,
    definitions).

    DDG API docs: https://api.duckduckgo.com/
    """
    params = {
        "q":              query,
        "format":         "json",
        "no_redirect":    "1",
        "no_html":        "1",
        "skip_disambig":  "1",
    }
    try:
        resp = session.get(DDG_API_URL, params=params, timeout=12)
        resp.raise_for_status()
        data = resp.json()

        abstract = data.get("AbstractText", "").strip()
        answer   = data.get("Answer", "").strip()

        # If both are empty, try the first RelatedTopic text as a fallback
        if not abstract and not answer:
            topics = data.get("RelatedTopics", [])
            for t in topics:
                if isinstance(t, dict) and t.get("Text", "").strip():
                    abstract = t["Text"].strip()[:300]
                    break

        return abstract[:400] if abstract else "", answer[:200] if answer else ""
    except Exception as exc:
        return "", f"ERROR:{exc}"


# ── SOURCE 3: OPENCORPORATES API ─────────────────────────────────────────────

def opencorporates_search(
    company_name: str,
    api_token: str,   # empty string = unauthenticated (free, rate-limited)
    session: requests.Session,
) -> tuple:
    """
    Search the OpenCorporates API for the company in the California registry
    (jurisdiction_code=us_ca).  Falls back to a US-wide search if no CA result.

    Returns (company_type, status, sic_description, address, oc_url).

    OpenCorporates API docs: https://api.opencorporates.com/
    Free (no key):  ~1 request/second rate limit.
    With API token: higher limits; get one at opencorporates.com/api_accounts/new
    """
    params = {
        "q":                 company_name,
        "jurisdiction_code": "us_ca",
        "order":             "score",
        "per_page":          1,
    }
    if api_token:
        params["api_token"] = api_token

    def _parse(company: dict) -> tuple:
        ctype   = company.get("company_type", "") or ""
        status  = company.get("current_status", "") or ""
        address = company.get("registered_address_in_full", "") or ""
        oc_url  = company.get("opencorporates_url", "") or ""

        # SIC / industry code description — the closest thing to a one-liner category
        sic_desc = ""
        for code_entry in (company.get("industry_codes") or []):
            if isinstance(code_entry, dict):
                sic_desc = code_entry.get("description", "")
                if sic_desc:
                    break

        return ctype, status, sic_desc, address, oc_url

    try:
        resp = session.get(OPENCORP_SEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        companies = (
            data.get("results", {}).get("companies", [])
        )
        if companies:
            co = companies[0].get("company", {})
            return _parse(co)

        # Retry without jurisdiction restriction
        params.pop("jurisdiction_code", None)
        resp2 = session.get(OPENCORP_SEARCH_URL, params=params, timeout=15)
        resp2.raise_for_status()
        data2 = resp2.json()
        companies2 = data2.get("results", {}).get("companies", [])
        if companies2:
            co2 = companies2[0].get("company", {})
            return _parse(co2)

        return "", "", "", "", ""
    except Exception as exc:
        return "", "", f"ERROR:{exc}", "", ""


# ── SOURCE 4: BLOOMBERG PROFILES ─────────────────────────────────────────────

BLOOMBERG_PROFILE_URL = "https://www.bloomberg.com/profile/company/{slug}"
_BLOOMBERG_TICKER_RE  = re.compile(r"/profile/company/([^/?#\"]+)", re.IGNORECASE)


def _bloomberg_direct(slug: str, session: requests.Session) -> tuple:
    url  = BLOOMBERG_PROFILE_URL.format(slug=urllib.parse.quote(slug))
    resp = _get(url, session)
    if resp is None:
        return "", url
    soup = _soup(resp)
    for selector in (
        "p.description",
        'div[data-type="companyDescription"] p',
        'section[data-id="description"] p',
        'div.description__content p',
    ):
        tag = soup.select_one(selector)
        if tag and tag.get_text(strip=True):
            return tag.get_text(" ", strip=True)[:400], url
    for h in soup.find_all(["h2", "h3"], string=re.compile(r"description", re.I)):
        p = h.find_next("p")
        if p and p.get_text(strip=True):
            return p.get_text(" ", strip=True)[:400], url
    return "", url


def _bloomberg_search(company_name: str, session: requests.Session) -> tuple:
    search_url = (
        "https://www.bloomberg.com/search?query="
        + urllib.parse.quote(company_name)
        + "&type=Company"
    )
    resp = _get(search_url, session)
    if resp is None:
        return "", search_url
    soup = _soup(resp)
    for a in soup.find_all("a", href=True):
        m = _BLOOMBERG_TICKER_RE.search(a["href"])
        if m:
            desc, profile_url = _bloomberg_direct(m.group(1), session)
            if desc:
                return desc, profile_url
    for card in soup.select("article, div.search-result, li.result"):
        txt = card.get_text(" ", strip=True)
        if company_name.split()[0].lower() in txt.lower():
            return txt[:300], search_url
    return "", search_url


def bloomberg_lookup(company_name: str, session: requests.Session) -> tuple:
    slug_guess = re.sub(r"[^A-Z0-9:.]", "", company_name.upper().replace(" ", ""))
    if slug_guess:
        desc, url = _bloomberg_direct(slug_guess, session)
        if desc:
            return desc, url
    return _bloomberg_search(company_name, session)


# ── SOURCE 5: ABOUT-PAGE SCRAPER ─────────────────────────────────────────────

def about_page_description(
    serp_url: str,
    session: requests.Session,
) -> tuple:
    """
    Follow the top SerpAPI result URL and extract the page's meta description
    or first substantial paragraph.  Skips SERP pages and blocked domains.
    Returns (description, url_used).
    """
    if not serp_url or serp_url.startswith("ERROR:") or _domain(serp_url) in SKIP_DOMAINS_ABOUT:
        return "", serp_url

    resp = _get(serp_url, session)
    if resp is None:
        return "", serp_url
    soup = _soup(resp)

    meta = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if meta and meta.get("content", "").strip():
        return meta["content"].strip()[:400], serp_url

    og = soup.find("meta", property="og:description")
    if og and og.get("content", "").strip():
        return og["content"].strip()[:400], serp_url

    for container_sel in ("main", "article", "#content", ".content", "body"):
        container = soup.select_one(container_sel)
        if container:
            for p in container.find_all("p"):
                text = p.get_text(" ", strip=True)
                if len(text) > 60:
                    return text[:400], serp_url

    return "", serp_url


# ── CREDENTIAL PROMPTS ────────────────────────────────────────────────────────

def _prompt_hidden(prompt: str) -> str:
    """
    Prompt for a secret value, hiding input if a TTY is available.
    Falls back to plain input() if getpass cannot attach to a terminal.
    """
    try:
        return getpass.getpass(prompt).strip()
    except (EOFError, OSError):
        # Non-interactive context or no TTY — fall back to visible input
        return input(prompt).strip()


def prompt_serpapi_key() -> str:
    print(
        "\n── SerpAPI key ───────────────────────────────────────────────────\n"
        "SerpAPI is a third-party Google SERP scraper. No CSE configuration\n"
        "needed — just sign up for an account and copy your API key.\n\n"
        "  • Sign up / log in: https://serpapi.com/\n"
        "  • After login, your key is on the Dashboard page.\n"
        "  Free tier: 100 searches/month.\n"
        "  Paid tier: $50/month for 5,000 searches.\n\n"
        "  Leave blank to skip SerpAPI lookups for this run.\n"
        "─────────────────────────────────────────────────────────────────\n"
    )
    return _prompt_hidden("  Paste your SerpAPI key (hidden, or press Enter to skip): ")


def prompt_opencorporates_token() -> str:
    print(
        "\n── OpenCorporates API token (optional) ───────────────────────────\n"
        "The OpenCorporates API works WITHOUT a token (free, ~1 req/sec).\n"
        "An API token raises the rate limit and unlocks additional data.\n\n"
        "  • Sign up: https://opencorporates.com/api_accounts/new\n"
        "  • Your token appears in your account settings after sign-up.\n\n"
        "  Leave blank to use the unauthenticated free tier.\n"
        "─────────────────────────────────────────────────────────────────\n"
    )
    return _prompt_hidden(
        "  Paste your OpenCorporates token (hidden, or press Enter to skip): "
    )


# ── MAIN PROCESSING LOOP ──────────────────────────────────────────────────────

FIELDNAMES = [
    "employer", "n_donors", "total_donated",
    # SerpAPI
    "serp_snippet", "serp_url",
    # DuckDuckGo
    "ddg_abstract", "ddg_answer",
    # OpenCorporates
    "oc_company_type", "oc_status", "oc_sic_description", "oc_address", "oc_url",
    # Bloomberg
    "bloomberg_description", "bloomberg_url",
    # About-page scraper
    "about_description", "about_url",
    # Errors
    "fetch_errors",
]


def run(args: argparse.Namespace) -> None:
    out_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── load + filter data ────────────────────────────────────────────────
    unmatched = load_unmatched_medium_plus(GOV_RACE_PATH, OPENSECRETS_PATH, args.min_amount)
    employers  = build_employer_table(unmatched)

    if args.max_rows:
        employers = employers.head(args.max_rows)
        print(f"    (--max-rows {args.max_rows}: capped at {len(employers)} employers)\n")

    # ── resume: skip already-processed employers ──────────────────────────
    done_employers: set = set()
    if args.resume and out_path.exists():
        ex = pd.read_csv(out_path, dtype=str)
        done_employers = set(ex["employer"].dropna().str.strip())
        print(f"    Resuming: {len(done_employers)} employers already done, skipping.\n")

    # ── collect ALL credentials BEFORE the loop starts ────────────────────
    print("=" * 65)
    print("CREDENTIAL COLLECTION  (all prompts now, before fetching begins)")
    print("=" * 65)

    serp_api_key = ""
    if not args.no_serp:
        serp_api_key = prompt_serpapi_key()
        if not serp_api_key:
            print("  No SerpAPI key entered — SerpAPI lookups will be skipped.\n")
            args.no_serp = True

    oc_token = ""
    if not args.no_opencorp:
        oc_token = prompt_opencorporates_token()
        if not oc_token:
            print("  No token entered — using OpenCorporates free (unauthenticated) tier.\n")

    if not args.no_bloomberg:
        print("\n  Bloomberg: no credentials needed — using public profile pages only.\n")
    if not args.no_ddg:
        print("  DuckDuckGo: no credentials needed — free Instant Answer API.\n")

    print("=" * 65)
    print(f"Starting fetch for {len(employers) - len(done_employers)} employers …")
    print("=" * 65 + "\n")

    # ── HTTP session ──────────────────────────────────────────────────────
    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)

    # ── open output CSV ───────────────────────────────────────────────────
    write_mode = "a" if (args.resume and out_path.exists()) else "w"
    with open(out_path, write_mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_mode == "w":
            writer.writeheader()

        total = len(employers)
        for idx, row in employers.iterrows():
            employer = row["employer"].strip()
            if employer in done_employers:
                continue

            print(f"  [{idx + 1}/{total}] {employer}")
            errors: list = []

            # 1. SerpAPI ───────────────────────────────────────────────────
            serp_snippet = serp_url = ""
            if not args.no_serp:
                serp_snippet, serp_url = serpapi_search(employer, serp_api_key, session)
                if serp_url.startswith("ERROR:"):
                    errors.append(f"serp:{serp_url}")
                    serp_snippet = serp_url = ""
                elif serp_snippet:
                    print(f"       SerpAPI: {serp_snippet[:90]}{'…' if len(serp_snippet) > 90 else ''}")
                time.sleep(args.delay)

            # 2. DuckDuckGo ────────────────────────────────────────────────
            ddg_abstract = ddg_answer = ""
            if not args.no_ddg:
                ddg_abstract, ddg_answer = duckduckgo_search(employer, session)
                if ddg_answer.startswith("ERROR:"):
                    errors.append(f"ddg:{ddg_answer}")
                    ddg_abstract = ddg_answer = ""
                elif ddg_abstract:
                    print(f"           DDG: {ddg_abstract[:90]}{'…' if len(ddg_abstract) > 90 else ''}")
                time.sleep(args.delay)

            # 3. OpenCorporates ────────────────────────────────────────────
            oc_type = oc_status = oc_sic = oc_addr = oc_url_val = ""
            if not args.no_opencorp:
                oc_type, oc_status, oc_sic, oc_addr, oc_url_val = opencorporates_search(
                    employer, oc_token, session
                )
                if oc_sic.startswith("ERROR:"):
                    errors.append(f"oc:{oc_sic}")
                    oc_type = oc_status = oc_sic = oc_addr = oc_url_val = ""
                elif oc_sic or oc_type:
                    print(f"   OpenCorp SIC: {oc_sic or '(none)'} | type: {oc_type or '(none)'}")
                time.sleep(args.delay)

            # 4. Bloomberg ─────────────────────────────────────────────────
            bloomberg_description = bloomberg_url = ""
            if not args.no_bloomberg:
                bloomberg_description, bloomberg_url = bloomberg_lookup(employer, session)
                if bloomberg_description:
                    print(f"     Bloomberg: {bloomberg_description[:90]}{'…' if len(bloomberg_description) > 90 else ''}")
                time.sleep(args.delay)

            # 5. About page (follows SerpAPI top result) ───────────────────
            about_description = about_url = ""
            if not args.no_about and serp_url:
                about_description, about_url = about_page_description(serp_url, session)
                if about_description:
                    print(f"      About pg: {about_description[:90]}{'…' if len(about_description) > 90 else ''}")
                time.sleep(args.delay)

            writer.writerow({
                "employer":              employer,
                "n_donors":              int(row["n_donors"]),
                "total_donated":         round(float(row["total_donated"]), 2),
                "serp_snippet":          serp_snippet,
                "serp_url":              serp_url,
                "ddg_abstract":          ddg_abstract,
                "ddg_answer":            ddg_answer,
                "oc_company_type":       oc_type,
                "oc_status":             oc_status,
                "oc_sic_description":    oc_sic,
                "oc_address":            oc_addr,
                "oc_url":                oc_url_val,
                "bloomberg_description": bloomberg_description,
                "bloomberg_url":         bloomberg_url,
                "about_description":     about_description,
                "about_url":             about_url,
                "fetch_errors":          " | ".join(errors),
            })
            fh.flush()   # flush after each row so partial runs are recoverable

    print(f"\nDone.  Results written to:\n  {out_path}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch one-liner descriptions for unmatched medium+ donors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--min-amount",    type=float, default=DEFAULT_MIN_AMOUNT,
                   help=f"Minimum single-donation amount (default: {DEFAULT_MIN_AMOUNT:,.0f})")
    p.add_argument("--max-rows",      type=int,   default=None,
                   help="Limit to N employers — useful for a test run")
    p.add_argument("--resume",        action="store_true",
                   help="Skip employers already in the output CSV")
    p.add_argument("--no-serp",       action="store_true",
                   help="Skip SerpAPI calls")
    p.add_argument("--no-ddg",        action="store_true",
                   help="Skip DuckDuckGo Instant Answer calls")
    p.add_argument("--no-opencorp",   action="store_true",
                   help="Skip OpenCorporates calls")
    p.add_argument("--no-bloomberg",  action="store_true",
                   help="Skip Bloomberg profile/search calls")
    p.add_argument("--no-about",      action="store_true",
                   help="Skip About-page scraping")
    p.add_argument("--delay",         type=float, default=1.5,
                   help="Seconds between HTTP requests (default: 1.5)")
    p.add_argument("--output",        type=str,   default=None,
                   help="Override output CSV path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
