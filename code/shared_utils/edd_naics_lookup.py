"""
edd_naics_lookup.py

Look up the NAICS industry code for a given employer name by scraping the
California EDD Labor Market Info "Employers by Name" search:
    https://labormarketinfo.edd.ca.gov/aspdotnet/databrowsing/empMain.aspx

There is no public EDD API for this dataset — only the ASP.NET WebForms UI.
The site is, however, GET-friendly: once you know the URL params, both the
results list and the detail page can be fetched with plain GET requests
(no VIEWSTATE postback needed). Flow:

  1. Results page (alphabetical by employer name, scoped to CA):
       empResults.aspx?menuChoice=emp&searchType=Keyword
                      &keyword=<NAME>&geogArea=0601000000
  2. Detail page for the first result:
       empDetails.aspx?menuChoice=emp&empid=<EMPID>&geogArea=0601000000
  3. The detail page contains a literal string:
       "Industry Description: <description> (NAICS code: <code>)"

Usage:
    # single
    python edd_naics_lookup.py "Google Inc"

    # batch — enrich an existing CSV. Reads each row, looks up the value in
    # --name-col, appends NAICS / NAICS_description / NAICS_match_name /
    # NAICS_status columns, and writes a new CSV preserving the originals.
    # Unique employer names are deduped before querying, so a 50K-row donor
    # file with 4K distinct employers makes only ~4K requests.
    python edd_naics_lookup.py --csv donors.csv --name-col employer \\
        --out donors_with_naics.csv

The script is polite by default: ~1 req/sec, on-disk cache so repeat runs
don't re-hit the site. Override with --delay / --no-cache.

Match strategies (--strategy / strategy= argument to lookup):

  "best"        (default) Prefer rows whose employer name exactly matches
                the query (case- and punctuation-insensitive). If none,
                prefer rows whose name *starts with* the query. Within each
                tier, pick the row with the largest employer size class. If
                no prefix match, fall back to the largest overall.
  "first"       Take the first row of the results table (alphabetic).
  "largest"     Take the row with the largest employer size class.
  "exact"       Take the largest exact match; if none, no_results.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup

BASE = "https://labormarketinfo.edd.ca.gov/aspdotnet/databrowsing"
RESULTS_URL = BASE + "/empResults.aspx"
DETAIL_URL = BASE + "/empDetails.aspx"

# geogArea code for "State of California" — dropdown value on EmpGeog.aspx.
CALIFORNIA_GEOG = "0601000000"

DEFAULT_CACHE = Path(__file__).resolve().parent / ".edd_naics_cache.json"
DEFAULT_UA = "CalMatters-CampaignFinance-Research/1.0 (kateli@stanford.edu)"

VALID_STRATEGIES = ("best", "first", "largest", "exact")


def _name_key(s: str | None) -> str:
    """Normalize a name for case/punctuation-insensitive comparison.
    Mirrors opensecrets `name_norm`: uppercase, drop non-alphanumerics
    (keeping spaces), collapse whitespace.
    """
    if not s:
        return ""
    s = re.sub(r"[^A-Z0-9 ]+", " ", s.upper())
    return re.sub(r"\s+", " ", s).strip()


def _size_lower_bound(size_class: str | None) -> int:
    """Extract the lower bound of an EDD size class (e.g. '100-249
    employees' -> 100, '1,000-4,999 Employees' -> 1000). Returns -1 for
    unknown/unparseable so it sorts last under max()."""
    if not size_class:
        return -1
    m = re.match(r"\s*([\d,]+)", size_class)
    return int(m.group(1).replace(",", "")) if m else -1


def _select_match(
    rows: list[dict], query: str, strategy: str
) -> tuple[dict | None, str | None]:
    """Pick one row from the EDD results table according to `strategy`.

    Returns (chosen_row, match_type) where match_type is one of
    "exact" | "starts_with" | "largest" | "first" | None — useful for
    downstream auditing of match confidence. The results table is sorted
    alphabetically, so the literal first row is rarely the most relevant
    for common-word queries; the default "best" strategy layers in
    name-match preference and size-class preference.
    """
    if not rows:
        return None, None
    if strategy == "first":
        return rows[0], "first"

    qn = _name_key(query)
    by_size = lambda r: _size_lower_bound(r.get("size_class"))  # noqa: E731

    if qn:
        exact = [r for r in rows if _name_key(r.get("name")) == qn]
        if exact:
            return max(exact, key=by_size), "exact"

        starts = [r for r in rows if _name_key(r.get("name")).startswith(qn + " ")]
        if starts:
            return max(starts, key=by_size), "starts_with"

    if strategy == "exact":
        return None, None
    # "best" and "largest" fall back to the largest overall row.
    return max(rows, key=by_size), "largest"


@dataclass
class LookupResult:
    query: str
    status: str = "ok"  # "ok" | "no_results" | "error"
    match_name: str | None = None
    naics_code: str | None = None
    naics_description: str | None = None  # long form, from detail page
    industry_short: str | None = None  # short form, from results table
    business_description: str | None = None
    address: str | None = None
    city: str | None = None
    size_class: str | None = None
    num_results: int = 0
    empid: str | None = None
    strategy_used: str | None = None
    match_type: str | None = None  # exact | starts_with | largest | first
    error: str | None = None
    candidates: list[dict] = field(default_factory=list)  # populated only when requested


class EDDClient:
    def __init__(
        self,
        delay: float = 1.0,
        timeout: float = 30.0,
        user_agent: str = DEFAULT_UA,
        cache_path: Path | None = DEFAULT_CACHE,
        verify_tls: bool = False,
    ):
        self.delay = delay
        self.timeout = timeout
        self._last_request = 0.0

        ctx = ssl.create_default_context()
        if not verify_tls:
            # EDD's cert chain occasionally fails verification on macOS; the
            # data is public so we don't need TLS auth, only encryption.
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ctx)
        )
        self._opener.addheaders = [("User-Agent", user_agent)]

        self.cache_path = cache_path
        self._cache: dict[str, dict] = {}
        if cache_path and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                self._cache = {}

    # ---------- HTTP ----------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.monotonic()

    def _get(self, url: str) -> str:
        self._throttle()
        req = urllib.request.Request(url)
        with self._opener.open(req, timeout=self.timeout) as r:
            return r.read().decode("utf-8", errors="ignore")

    # ---------- cache ----------

    def _save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.write_text(json.dumps(self._cache, indent=2))

    # ---------- parsing ----------

    @staticmethod
    def _parse_results(html: str) -> list[dict]:
        """Extract the rows of empResults.aspx into dicts.

        Returns an empty list if the search produced no matches.
        """
        soup = BeautifulSoup(html, "html.parser")
        # Locate the table that holds detail-page links.
        target = next(
            (
                t
                for t in soup.find_all("table")
                if t.find("a", href=lambda h: h and "empDetails" in h)
            ),
            None,
        )
        if target is None:
            return []

        rows: list[dict] = []
        for tr in target.find_all("tr", class_="tableData"):
            tds = tr.find_all("td")
            if len(tds) < 6:
                continue
            link = tds[1].find("a")
            href = link["href"] if link else ""
            empid_match = re.search(r"empid=(\d+)", href)
            rows.append(
                {
                    "name": " ".join(tds[1].get_text(" ", strip=True).split()),
                    "address": " ".join(tds[2].get_text(" ", strip=True).split()),
                    "city": " ".join(tds[3].get_text(" ", strip=True).split()),
                    "industry_short": " ".join(tds[4].get_text(" ", strip=True).split()),
                    "size_class": " ".join(tds[5].get_text(" ", strip=True).split()),
                    "empid": empid_match.group(1) if empid_match else None,
                    "detail_href": href,
                }
            )
        return rows

    @staticmethod
    def _parse_detail(html: str) -> dict:
        """Pull the labelled fields off an empDetails.aspx page."""
        text = re.sub(r"\s+", " ", BeautifulSoup(html, "html.parser").get_text(" "))

        out: dict = {}
        # "Industry Description: <desc> (NAICS code: <code>)"
        m = re.search(
            r"Industry Description:\s*(.+?)\s*\(NAICS code:\s*([0-9]+)\)",
            text,
            re.IGNORECASE,
        )
        if m:
            out["naics_description"] = m.group(1).strip()
            out["naics_code"] = m.group(2).strip()

        for label, key in [
            ("Business Description", "business_description"),
            ("Employer Size Class", "size_class"),
        ]:
            m = re.search(
                rf"{re.escape(label)}:\s*(.+?)\s*(?:&nbsp;|Mailing Address|Contact|Website|Industry Description|Employer|$)",
                text,
            )
            if m:
                out[key] = m.group(1).strip().rstrip(":").strip()
        return out

    # ---------- public ----------

    def lookup(
        self,
        name: str,
        *,
        strategy: str = "best",
        geog_area: str = CALIFORNIA_GEOG,
        keep_candidates: bool = False,
    ) -> LookupResult:
        """Look up NAICS for one employer name.

        See module docstring for the available `strategy` values. Default
        "best" prefers exact name matches, then starts-with matches, then
        largest size class. Set keep_candidates=True to retain the full
        list of result-page rows on the LookupResult.
        """
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {VALID_STRATEGIES}, got {strategy!r}"
            )

        query = name.strip()
        cache_key = f"{geog_area}|{strategy}|{query.upper()}"

        if cache_key in self._cache:
            return LookupResult(**self._cache[cache_key])

        result = LookupResult(query=query, strategy_used=strategy)
        try:
            params = {
                "menuChoice": "emp",
                "searchType": "Keyword",
                "keyword": query,
                "geogArea": geog_area,
            }
            results_html = self._get(f"{RESULTS_URL}?{urllib.parse.urlencode(params)}")
            rows = self._parse_results(results_html)
            result.num_results = len(rows)

            chosen, match_type = _select_match(rows, query, strategy)
            if chosen is None:
                result.status = "no_results"
            else:
                result.match_type = match_type
                result.match_name = chosen["name"]
                result.address = chosen["address"]
                result.city = chosen["city"]
                result.industry_short = chosen["industry_short"]
                result.size_class = chosen["size_class"]
                result.empid = chosen["empid"]

                if chosen["empid"]:
                    detail_html = self._get(
                        f"{DETAIL_URL}?menuChoice=emp"
                        f"&empid={chosen['empid']}&geogArea={geog_area}"
                    )
                    detail = self._parse_detail(detail_html)
                    result.naics_code = detail.get("naics_code")
                    result.naics_description = detail.get("naics_description")
                    result.business_description = detail.get("business_description")
                    # Detail page's size_class is canonical; overwrite if present.
                    if detail.get("size_class"):
                        result.size_class = detail["size_class"]

                if keep_candidates:
                    result.candidates = rows

        except Exception as e:
            result.status = "error"
            result.error = f"{type(e).__name__}: {e}"

        self._cache[cache_key] = asdict(result)
        self._save_cache()
        return result

    def lookup_batch(self, names: Iterable[str], **kwargs) -> list[LookupResult]:
        return [self.lookup(n, **kwargs) for n in names]


# ---------- CLI ----------

# Columns appended to the input CSV in batch mode. Prefixed so they don't
# collide with caller column names.
ENRICH_COLS = [
    "NAICS",
    "NAICS_description",
    "NAICS_match_name",
    "NAICS_match_type",  # exact | starts_with | largest | first — match confidence
    "NAICS_status",
]


def _print_single(result: LookupResult) -> None:
    print(f"Query:       {result.query}")
    print(f"Strategy:    {result.strategy_used}")
    print(f"Status:      {result.status} ({result.num_results} candidates)")
    if result.status == "ok":
        print(f"Match:       {result.match_name} — {result.city}  ({result.match_type})")
        print(f"NAICS:       {result.naics_code}  {result.naics_description}")
        print(f"Industry:    {result.industry_short}")
        print(f"Business:    {result.business_description}")
        print(f"Size:        {result.size_class}")
    elif result.status == "error":
        print(f"Error:       {result.error}")


def _run_csv(
    client: EDDClient,
    in_path: Path,
    out_path: Path,
    name_col: str,
    strategy: str,
) -> None:
    """Enrich an input CSV with NAICS columns from EDD lookups.

    All original columns are preserved verbatim; four columns are appended:
        NAICS, NAICS_description, NAICS_match_name, NAICS_status

    Unique employer names are deduped before querying so a 50K-row donor
    file with 4K distinct employers makes ~4K requests, not 50K.
    """
    with in_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if name_col not in fieldnames:
        sys.exit(f"--name-col {name_col!r} not in {fieldnames}")

    unique_names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        n = (row.get(name_col) or "").strip()
        if n and n not in seen:
            seen.add(n)
            unique_names.append(n)

    print(
        f"{len(rows):,} input rows; {len(unique_names):,} unique employer names",
        file=sys.stderr,
    )

    lookups: dict[str, LookupResult] = {}
    for i, name in enumerate(unique_names, 1):
        r = client.lookup(name, strategy=strategy)
        lookups[name] = r
        print(
            f"[{i}/{len(unique_names)}] {name[:50]:50s} -> "
            f"{r.status:11s} {r.naics_code or '':6s} {r.match_name or ''}",
            file=sys.stderr,
        )

    out_fields = list(fieldnames)
    for c in ENRICH_COLS:
        if c not in out_fields:
            out_fields.append(c)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        # extrasaction="ignore" tolerates malformed input rows where
        # DictReader stashed overflow values under key None.
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            n = (row.get(name_col) or "").strip()
            r = lookups.get(n)
            if r is None:
                row["NAICS"] = ""
                row["NAICS_description"] = ""
                row["NAICS_match_name"] = ""
                row["NAICS_match_type"] = ""
                row["NAICS_status"] = "empty_query"
            else:
                row["NAICS"] = r.naics_code or ""
                row["NAICS_description"] = r.naics_description or ""
                row["NAICS_match_name"] = r.match_name or ""
                row["NAICS_match_type"] = r.match_type or ""
                row["NAICS_status"] = r.status
            writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("name", nargs="?", help="Single employer name to look up")
    p.add_argument("--csv", type=Path, help="Batch mode: input CSV to enrich")
    p.add_argument(
        "--name-col",
        default="employer",
        help="Column in --csv that holds the employer name (default: employer)",
    )
    p.add_argument("--out", type=Path, help="Batch mode: output CSV path")
    p.add_argument(
        "--strategy",
        default="best",
        choices=VALID_STRATEGIES,
        help="Match-selection strategy (default: best)",
    )
    p.add_argument("--delay", type=float, default=1.0, help="Seconds between requests")
    p.add_argument("--no-cache", action="store_true", help="Disable on-disk cache")
    args = p.parse_args(argv)

    client = EDDClient(
        delay=args.delay,
        cache_path=None if args.no_cache else DEFAULT_CACHE,
    )

    if args.csv:
        if not args.out:
            sys.exit("--out is required with --csv")
        _run_csv(client, args.csv, args.out, args.name_col, args.strategy)
    elif args.name:
        _print_single(client.lookup(args.name, strategy=args.strategy))
    else:
        p.error("provide either NAME or --csv")


if __name__ == "__main__":
    main()
