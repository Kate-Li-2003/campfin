# 04 Assign Classifications

This pipeline takes the outputs from the 01 (rule-based), 02 (ML), and 03 (LLM: Gemini + Claude) classifiers, and chooses a single "best" NAICS code per contributor. It then computes an industry funding breakdown for each candidate and ballot measure.

The pipeline traces each PAC's funding back to its underlying donors' industries and attributes PAC contributions proportionally.

---

## Script Order

1. `0401_build_os_llm_naics_crosswalk.py` : Translates raw NAICS codes from the OpenSecrets matches and LLM classifications into the project's custom sector scheme. Must run before anything else.
    - output: `04_outputs/os_llm_naics_crosswalk.csv`

2. `0402_assign_final_classification.Rmd` : Reconciles rule-based (01), LLM (03 Gemini and Claude), ML (02), and OpenSecrets codes for direct $5k+ contributors into a single `code_final` per UUID. Flags uncertain cases for human review.
    - outputs: `04_outputs/final_classifications_before_review.csv`, `04_outputs/manual_review_queue.csv`

3. `0403_match_pac_classifications.Rmd` : Computes an industry distribution for each PAC using its classified contributors. PAC-to-PAC contributions are resolved recursively.
    - output: `04_outputs/pac_industry_breakdown.csv`

4. `0404_build_race_prop_breakdown_input.R` : Assembles all contributions with industry codes attached. For PAC contributions, replaces each PAC row with one row per industry using the distributions from step 3.
    - output: `04_outputs/race_prop_breakdown_input.csv`

5. `0405_compute_breakdown.R` : Aggregates by candidate and industry to produce final dollar amounts and percentages. Verifies that percentages sum to 100% within each candidate.
    - output: `04_outputs/industry_breakdown_by_race.csv`

---

## Between steps 2–3: human review

Step 2 produces `04_outputs/manual_review_queue.csv` — a combined queue of flagged rows, not-employed individuals, and a confidence QA sample. Review decisions should be saved to `04_inputs/manual_review_queue_reviewed.csv` before running steps 3–5. PAC-specific review decisions go into `04_inputs/pac_manual_review.csv`

---

## Helpers

- `resolve_classifications.R` : Sourced by `0402_assign_final_classification.Rmd`. Defines the resolution logic including source priority rules, confidence thresholds, overrides (pre-classified rows, retired/not-employed codes, PAC/candidate codes), and the functions `resolve_code_one()` and `resolve_category_one()` that pick a winner when multiple classifiers disagree.
- `build_entity_classification_registry.R` : Builds a persistent registry of classification decisions across runs.

---

## Logic for choosing the "best" code

Each contributor has up to four classification sources: rule-based (01), LLM Gemini (03), LLM Claude (03), ML (02), and OpenSecrets match (01). The priority order is:

1. **Hard overrides** — pre-classified (manually reviewed) rows are locked; "100" code from pattern matching
2. **Trusted "rule"" tiers** — employer lookup, running list, and keyword match from the rule-based classifier
3. **Agreement between sources** — OS + LLM agree; rule + OS agree; rule + LLM agree (disagreement flagged for review)
4. **Single high-confidence source** - fallback is the code assigned by Gemini, but any classification based solely on the LLM is flagged for human review. 
5. **Fallback to "99"** (unknown/uncategorized)

Rows that fall through to step 4 or 5 are flagged for human review. Rows where the LLM disagrees with the final code are also flagged.

---

## Key inputs

- `../01_contributor_pipeline/01_outputs/classified_contributors_*.csv` : Rule-based classifier output
- `../01_contributor_pipeline/01_outputs/combined_contributions_os_matches_*.csv` : OpenSecrets matches 
- `../01_contributor_pipeline/01_outputs/classification_input_combined_*.csv` : $5k+ contributors with amounts (for PAC traversal)
- `../03_llm_classifier/03_outputs/classification_input_combined_expanded_full_*.csv` : Gemini LLM classifications for direct contributors
- `../03_llm_classifier/03_outputs/claude_classification_results.csv` : Claude LLM classifications
- `../../data/07_output_ml_classification/08_entities_with_ml_*.csv` : ML classifier predictions (02 pipeline)
- `../01_contributor_pipeline/01_inputs/already_classified_contributions.csv` : Pre-classified contributions (locked from override)
- `04_inputs/pac_manual_review.csv` : Human review decisions for PACs (federal, unresolvable, etc.)
- `04_inputs/manual_review_queue_reviewed.csv` : Human review decisions for flagged direct contributors

---

## Key outputs

- `04_outputs/final_classifications_before_review.csv` : One row per direct-contributor UUID with `code_final`, `code_final_source`, review flags
- `04_outputs/manual_review_queue.csv` : Combined human review queue: flagged rows, code-100 individuals, and QA confidence sample
- `04_outputs/pac_industry_breakdown.csv` : Industry distribution per PAC — dollar amount and percentage by sector
- `04_outputs/race_prop_breakdown_input.csv` : All contributions to tracked races with codes; PAC contributions exploded by industry
- `04_outputs/industry_breakdown_by_race.csv` : **Final editorial output** — industry $ and % breakdown per candidate and ballot measure
