# 09 LLM Classifier

This pipeline classified ALL contribution records (except those with classifications from a previous pipeline run) using the LLM. 

Two step process:
1. **Web search** — Gemini + Google Search grounding produces an industry summary for each contributor (can reuse these even when classification schema changes)
2. **Classification** — an LLM reads those summaries and assigns the codes (re-run when classification schema changes)

The pipeline deduplicates contributors to unique (name × employer × occupation) units before making any API calls, so each unique entity is only searched and classified once regardless of how many contributions they appear in. Results are then expanded back to one row per raw contribution record.

---

## Notebooks vs. scripts

Each main step has both a **Jupyter notebook** (`.ipynb`) and a **Python script** (`.py`). The notebooks are the active development versions and are what I've been running The `.py` scripts are not fully up to date yet — they are intended to eventually replace the notebooks for production runs, but for now I've been using the notebooks.

---

## Script Order

### One-time setup


- `0900_naics_descriptions_processing.Rmd` : Builds the description for each NAICS code that is referenced by the LLM to assign classifications — for each NAICS code, it includes the description of each subsector in that parent sector with the aim of providing more detailed descriptions. Also adds custom project codes (PACs, unions, retired, etc.). Must be run before the Python scripts. 
- `build_web_search_cache.py` : One-time utility that seeds the web search cache from any prior pipeline outputs, so already-searched entities don't need to be re-queried. Skip this if starting fresh with no prior outputs. 

### Each pipeline run

1. `0901_web_search.ipynb` : For each unique contributor unit, calls Gemini with Google Search grounding to produce an `industry_summary`. Results are cached to `web_search_cache.csv` so re-runs only query new entities. 
    - outputs: `09_outputs/web_search_*.csv` (industry summaries for each contributor) and updated `web_search_cache.csv` 
2. `0902_classify_web_results.ipynb` : Reads the web search output and sends each contributor (in batches of 50) to an LLM with a list of valid NAICS codes and OpenSecrets categories. Returns `naics_code`, `open_secrets_category`, confidence scores, and reasoning. Then expands results back to full UUID coverage. 
    - outputs: `09_outputs/classification_full_expanded_*.csv` 

The notebook for step 2 automatically runs the expansion step at the end. For the `.py` script instead, run `expand_classifications.py` separately afterward.

---

## Helper files

- `config.py` : Maps logical field names to the actual column names in the upstream CSV; shared by all Python scripts (originally implemented this to be robust to changes in names of columns as we finalized the entity resolution process)
- `naics_data.py` : NAICS 2022 reference data as Python dicts; used for looking up descriptions of assigned codes (data produced by `0900_naics_descriptions_processing.Rmd` is much more detailed, I should combined into one thing)
- `standardization_helpers.py` : Python ports of the R name standardization functions from the 08 pipeline; ensures cache keys are stable across runs 
- `expand_classifications.py` : Joins deduplicated classification results back onto every raw contribution row by `search_key`; called automatically by the 0902 notebook 


## Key inputs

- `../08_alternative_pipeline/08_outputs/classification_input_combined.csv` : Produced by `0802_entity_resolution_simplified.Rmd` in the 08 pipeline 
- `09_inputs/digit_2022_Codes.xlsx` : Official NAICS 2022 code spreadsheet (manually downloaded) from NAICS
- `09_inputs/naics_sector_title_expanded_with_custom_codes.csv` : Produced by `0900_naics_descriptions_processing.Rmd` 
- `09_inputs/open_secrets_level2_categories.csv` : Produced by `0900_naics_descriptions_processing.Rmd` 
- `09_outputs/web_search_cache.csv` : Persistent cache of all prior web searches; checked before each API call 

---

## Key outputs

- `09_outputs/web_search_cache.csv` : Persistent cache of all web search results; updated after each run 
- `09_outputs/web_search_*.csv` : Per-batch web search results with `industry_summary` for each contributor unit 
- `09_outputs/classification_full_expanded_*.csv` : Final output: one row per raw contribution UUID with `naics_code_llm`, `naics_description`, `naics_confidence`, `open_secrets_category`, `open_secrets_confidence`, `naics_reasoning`, `industry_summary`, and `is_prominent` attached 

The expanded classification file feeds into the 10 pipeline (`assign_final_classification.Rmd`), which does a UUID-based join to produce final classifications.

---

## LLM configuration

The default model for both steps is `gemini-2.5-flash`. Step 2 also supports `claude-sonnet-4-6` and `gpt-4.1` — configurable at the top of the notebook.

We are looking to transition to have Claude Code do the classifications directly instead of using Gemini. 
