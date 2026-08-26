"""Shared config for the LLM classifier pipeline (web_search.py + classify_web_results.py).

COLUMNS maps this pipeline's logical fields to the actual column names produced
upstream by 08_alternative_pipeline (0802_entity_resolution.Rmd ->
08_outputs/classification_input.csv).

If column names change, update the values below -- web_search.py and
classify_web_results.py reference fields only through this dict.
"""

from pathlib import Path

DEFAULT_INPUT_PATH = str(
    Path(__file__).resolve().parent.parent
    / "08_alternative_pipeline" / "08_outputs" / "classification_input.csv"
)

COLUMNS = {
    "entity_id":   "entity_id",
    "entity_type": "entity_type",   # "individual" | "organization"
    # Raw source columns — used to build the cache key via normalize_for_key()
    # (uppercase + alphanumeric only). Keeping the key tied to raw columns means
    # it never changes regardless of how downstream processing evolves.
    "name_raw":    "Contributor.Name",
    "employer_raw": "Contributor.Employer",
    # Standardized columns — used to compute lightly-processed names for Gemini
    "name":        "standardized_name",
    "employer":    "standardized_employer_name",
    "occupation":  "processed_occupation",   # for _is_not_employed() check
    "city":        "standardized_city",
    "state":       "Contributor.State",
    "amount":      "Amount",
}

# Fields the LLM classifier can't run without.
REQUIRED_COLUMNS = ("name", "employer", "occupation")


def require_columns(df, path, needed=REQUIRED_COLUMNS):
    """Raise a clear error if the upstream pipeline's column names have moved on us."""
    missing = [COLUMNS[k] for k in needed if k in COLUMNS and COLUMNS[k] not in df.columns]
    if missing:
        raise KeyError(
            f"{path} is missing expected column(s): {missing}.\n"
            f"Available columns: {list(df.columns)}\n"
            "If entity_resolution renamed these columns, "
            "update COLUMNS in config.py to match -- nothing else needs to change."
        )
