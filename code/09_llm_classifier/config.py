"""Shared config for the LLM classifier pipeline (web_search.py + classify_web_results.py).

COLUMNS maps this pipeline's logical fields to the actual column names produced
upstream by 08_alternative_pipeline (0801_entity_resolution.Rmd /
find_pac_contributors.Rmd -> 08_inputs/classification_input_combined.csv, one
row per classification unit = entity x employer x occupation alias).

Those column names are not finalized yet. If they change, update the values
below -- web_search.py and classify_web_results.py reference fields only
through this dict, so nothing else needs to change.
"""

from pathlib import Path

DEFAULT_INPUT_PATH = str(
    Path(__file__).resolve().parent.parent
    / "08_alternative_pipeline" / "08_inputs" / "classification_input_combined.csv"
)

COLUMNS = {
    "unit_id":     "classification_unit_id",
    "entity_id":   "entity_id",
    "entity_type": "entity_type",   # "individual" | "organization"
    "name":        "canonical_name",
    "employer":    "employer",
    "occupation":  "occupation",
    "amount":      "total_amount",
}

# Fields the LLM classifier can't run without.
REQUIRED_COLUMNS = ("unit_id", "name", "employer", "occupation")


def require_columns(df, path, needed=REQUIRED_COLUMNS):
    """Raise a clear error if the upstream pipeline's column names have moved on us."""
    missing = [COLUMNS[k] for k in needed if COLUMNS[k] not in df.columns]
    if missing:
        raise KeyError(
            f"{path} is missing expected column(s): {missing}.\n"
            f"Available columns: {list(df.columns)}\n"
            "If entity_resolution/find_pac_contributors renamed these columns, "
            "update COLUMNS in config.py to match -- nothing else needs to change."
        )
