"""Builds the batch prompt for Step 2 of the Claude web-search classification plan.

`build_batch_prompt()` reads the NAICS
code list live from `09_inputs/naics_sector_title_expanded_with_custom_codes.csv`
every time it's called, so the prompt always reflects the current file. Call this
from the orchestrating step right before dispatching each Agent() batch call --
the returned string is the complete, ready-to-send prompt, codes and units included.

Current policy: SEARCH-OPTIONAL. This first pass targets the units Gemini already
classified, specifically to compare Claude's own-knowledge/lightly-verified answers
against Gemini's search-grounded ones -- so a web search is only required when Claude
isn't already confident, not for every unit (that's the "lighter" in "lighter first
pass"). `used_web_search` is recorded on every unit so the comparison can separate
confident-from-training-knowledge answers from search-verified ones. This is
deliberately looser than a "never guess, always search" policy -- if a stricter mode
is wanted later (e.g. once we move past the Gemini-comparison units), change
INSTRUCTIONS below rather than the calling code.
"""

import pandas as pd

CODES_PATH = "09_inputs/naics_sector_title_expanded_with_custom_codes.csv"

INSTRUCTIONS = """You are classifying California campaign finance contributors. For each unit listed
below, you must:

1. First consider whether you are already confident about this contributor's industry
   from your own training knowledge -- e.g. a well-known public company, a widely
   known individual's employer, an unambiguous entity name like "Los Angeles Unified
   School District." If you are genuinely confident, you may answer directly without
   searching; set used_web_search=false.
   If you are NOT confident -- an unfamiliar name, a generic or ambiguous employer
   name, an entity that could be confused with a similarly-named one, or anything
   where a wrong guess could mislead -- call the WebSearch tool before answering; set
   used_web_search=true. When in doubt, search: only skip the search when you'd bet on
   the answer. If a search is ambiguous or returns nothing useful, run one more, more
   targeted search (add employer, city, or state) -- you do not need more than two
   searches per unit. Only write "Unknown" after searching and finding nothing
   relevant -- never guess to fill in a plausible-sounding answer once you've decided
   to search.

2. Determine entity_type: "individual" (a person) or "organization" (company, PAC,
   union, association, committee, tribe, government body, party committee, etc).

3. Write industry_summary (1-2 sentences, no URLs or confidence ratings inside it):
   - Individuals: describe what their employer does.
   - Self-employed / sole proprietors: describe the occupation's field.
   - Organizations: describe the entity and its business/purpose. If it is a PAC or
     political committee, say so and describe what it supports.
   - If you cannot find anything after searching, set to "Unknown".

4. Assign exactly one naics_code from the VALID CODES list below.
   Rules:
   - Always use a code from the valid list below.
   - For individuals: classify by the employer's industry.
   - For self-employed/sole proprietors: classify by the occupation's industry.
   - When only an occupation is available (no employer/industry info): classify by
     that occupation's industry.
   - Native American tribes and tribal governments -> "92a".
   - Use "99" only when no useful information exists to determine an industry.
   - "88" is ONLY for entities explicitly identified as a registered PAC, political
     action committee, contributor committee, or political action fund -- e.g. the
     name literally contains "PAC," "Political Action Committee," or "COPE," or your
     research confirms it's a distinct registered political committee. Do NOT use "88"
     for a union, trade/professional association, corporation, or any other
     organization just because it's politically active, lobbies, or has a
     separately-registered affiliated PAC -- classify the parent organization by its
     own actual industry instead (a union is "77," not "88"; an industry association
     is "76," not "88," even if it has PAC-like activity).
   - Unitemized dues or intermediary contributions passed through a PAC should be
     classified by the underlying organization's actual industry when it can be
     determined (e.g. a union's PAC dues -> "77"), not as "88."
   - Technology/software/AI companies -> "79" regardless of which industry vertical
     the product serves (health-tech, fintech, ad-tech, etc. are all "79," not the
     industry they serve).
   - Family trusts -> "100" ONLY when the contributor is literally the trust itself
     (name reads as a trust, e.g. "The Smith Family Trust," "Jane Doe 1992 Revocable
     Trust") -- narrow; do not extend to LLCs/holding companies merely owned by or
     affiliated with a trust.
   - Defense contractors / companies whose primary business is military/defense
     products or services -> "92," even if their formal NAICS code would otherwise be
     manufacturing (31-33).
   - Companies primarily in oil/gas/petroleum extraction, drilling, refining, or
     distribution -> "21," even if their formal NAICS code would otherwise be
     manufacturing (31/32) or wholesale trade ("80").
   - Vineyards -> "11" (agriculture/grape-growing), even though winemaking might
     otherwise suggest Beverage Manufacturing.
   - Integrated health systems that are both an insurer and a hospital/care operator
     (e.g. Kaiser Permanente, Sutter Health) -> "52" (Finance and Insurance), not "60".
   - Tribal casinos and other tribal gaming operations -> "92a" (Native American
     tribes), not "70" (Entertainment/Gambling).
   - Ambulance / ambulance services -> "92" (Government/Public Safety), not "60"
     (Healthcare).

5. Assign naics_confidence: "high" (clearly identified, detailed info found),
   "medium" (clear match but summary is short/vague), "low" (partial match/limited
   info), or "unknown" (nothing found despite searching).

6. Record used_web_search (true/false) and the URLs you actually used, if any (empty
   list if you didn't search or found none).

7. Give a 1-sentence reasoning citing the specific detail that drove the naics_code
   choice.
"""

RETURN_FORMAT = """Return one JSON object per unit, same order as given, with keys:
search_key, entity_type, industry_summary, naics_code, naics_confidence,
used_web_search (boolean), urls (list), reasoning."""


def load_naics_code_list() -> str:
    """Live-loads the current code list -- never hardcoded/copy-pasted."""
    codes = pd.read_csv(CODES_PATH)
    return "\n".join(
        f"{row.naics_sector}\t{row.description}" for row in codes.itertuples()
    )


def format_units(units: list[dict]) -> str:
    """units: list of dicts with the queue's per-unit fields."""
    blocks = []
    for i, u in enumerate(units, 1):
        blocks.append(
            f"Unit {i} (search_key={u['search_key']}):\n"
            f"  Name: {u.get('standardized_name', '')}\n"
            f"  Employer: {u.get('standardized_employer_name', '') or 'None'}\n"
            f"  Occupation: {u.get('standardized_occupation', '') or 'None'}\n"
            f"  City/State: {u.get('standardized_city', '')}, {u.get('Contributor.State', '')}\n"
            f"  Existing entity_type label (context only, don't defer to it): "
            f"{u.get('entity_type', '')}"
        )
    return "\n\n".join(blocks)


def build_batch_prompt(units: list[dict]) -> str:
    """Assembles the complete prompt for one Agent() batch call.

    `units` should be the `status == "pending"` rows of
    `09_outputs/claude_classification_queue.csv` for this batch, as a list of dicts
    (e.g. `df[df.status == "pending"].head(35).to_dict("records")`).
    """
    return (
        f"{INSTRUCTIONS}\n"
        f"VALID NAICS CODES (code<TAB>description):\n{load_naics_code_list()}\n\n"
        f"{'=' * 60}\n\n"
        f"{format_units(units)}\n\n"
        f"{'=' * 60}\n\n"
        f"{RETURN_FORMAT}"
    )
