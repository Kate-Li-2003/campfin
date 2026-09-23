# 03 LLM Classifier

This pipeline classifies ALL contribution records (except those with classifications from a previous pipeline run) using LLMs.

The process has two classification paths, both of which feed into the 04 pipeline:

1. **Gemini path** — Gemini + Google Search grounding produces an industry summary for each contributor, then classifies based on that summary
2. **Claude path** — Claude classifies contributors directly, using its own knowledge first and searching only when uncertain. Claude path added to use our Claude credits - Claude code does the classifications (we don't have API credits)

Both outputs are used in `04_assign_classifications/assign_final_classification.Rmd` to cross-check and reconcile classifications.

The pipeline deduplicates contributors to unique (name × employer × occupation) units before making any API calls, so each unique entity is only classified once regardless of how many contribution records they appear in. Results are then expanded back to one row per raw contribution record.

---

## Script Order

### One-time setup

- `0300_naics_descriptions_processing.Rmd` : Builds the NAICS code descriptions referenced by both LLMs — for each code, includes descriptions of each subsector to provide more context. Also adds descriptions for custom codes (PACs, unions, retired, etc.). Must be run before the Python notebooks.
    - outputs: `03_inputs/naics_sector_title_expanded_with_custom_codes.csv`, `03_inputs/open_secrets_level2_categories.csv`
- `build_web_search_cache.py` : One-time utility that seeds the Gemini web search cache from any prior pipeline outputs. 

### Each pipeline run

#### Gemini path

1. `0301_web_search.ipynb` : For each unique contributor unit, calls Gemini with Google Search grounding to produce an `industry_summary`. Results are cached to `web_search_cache.csv` so re-runs only query new entities.
    - outputs: `03_outputs/web_search_*.csv` (industry summaries) and updated `03_outputs/web_search_cache.csv`

2. `0302_classify_web_results.ipynb` : Reads the web search output and sends each contributor (in batches) to Gemini with a list of valid NAICS codes and OpenSecrets categories. Returns `naics_code`, `open_secrets_category`, confidence scores, and reasoning. Automatically runs the expansion step at the end.
    - outputs: `03_outputs/classification_full_expanded_*.csv`

#### Claude path

3. `0303_build_claude_classification_queue.ipynb` : Builds the classification input for Claude and runs batches using Claude Code agents (`claude_batch_prompt.py`). Claude is "search-optional" — it draws on training knowledge for well-known entities and performs a web search when uncertain. Has a `used_web_search` flag to indicate when it used web search.
    - outputs: `03_outputs/claude_classification_queue.csv` (input queue), `03_outputs/claude_classification_results.csv` (final output)

---

## Downstream use

Both `classification_full_expanded_*.csv` (Gemini) and `claude_classification_results.csv` (Claude) are read by `04_assign_classifications/assign_final_classification.Rmd`. The pipeline defaults to Gemini classifications since that is the more established pipeline, but Claude classifications are used to assess agreement (disagreements flagged for review).

---

## Helper files

- `config.py` : Maps logical field names to actual column names in the upstream CSV; shared by all Python scripts
- `naics_data.py` : NAICS 2022 reference data as Python dicts; used for code description lookups
- `standardization_helpers.py` : Python ports of the R name standardization functions from the 01 pipeline; ensures cache keys are stable across runs
- `expand_classifications.py` : Joins classification results back onto every raw contribution row by `search_key`; called automatically by the `0302` notebook
- `claude_batch_prompt.py` : Builds the full classification prompt sent to Claude, including the NAICS code list (read from `03_inputs/naics_sector_title_expanded_with_custom_codes.csv`) and per-unit contributor data. Called from `0303`.

---

## Key inputs

- `../01_contributor_pipeline/01_outputs/classification_input_combined_*.csv` : Produced by `0102c` in the 01 pipeline
- `03_inputs/digit_2022_Codes.xlsx` : Official NAICS 2022 code spreadsheet
- `03_inputs/naics_sector_title_expanded_with_custom_codes.csv` : Produced by `0300_naics_descriptions_processing.Rmd`
- `03_inputs/open_secrets_level2_categories.csv` : Produced by `0300_naics_descriptions_processing.Rmd`
- `03_outputs/web_search_cache.csv` : Cache of all prior Gemini web searches; checked before each API call

---

## Key outputs

- `03_outputs/web_search_cache.csv` : Cache of Gemini web search results; updated after each run
- `03_outputs/classification_full_expanded_*.csv` : Gemini path output — one row per raw contribution UUID with `naics_code_llm`, `naics_description`, `naics_confidence`, `open_secrets_category`, `naics_reasoning`, and `industry_summary` attached
- `03_outputs/claude_classification_results.csv` : Claude path output — one row per contributor unit with `naics_code`, `industry_summary`, `used_web_search`, and reasoning

---

## API keys required

- Google API key with Search Grounding enabled (for `0301_web_search.ipynb`)

