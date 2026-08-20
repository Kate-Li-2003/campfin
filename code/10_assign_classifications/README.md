# 10 Assign Classifications

This pipeline takes the outputs from the 08 (rule-based), 09 (LLM), and 07 (ML) classifiers, reconciles them into a single final NAICS code per contributor, and computes an industry funding breakdown for each candidate and ballot measure.

The pipeline traces each PAC's funding back to its underlying donor industries and attributes PAC contributions proportionally.

---

## Script Order

1. `build_os_llm_naics_crosswalk.py` : Translates raw NAICS codes from the OpenSecrets matcher and LLM classifier into the project's custom sector scheme. Must run before anything else. 
    - outputs: `10_outputs/os_llm_naics_crosswalk.csv` 
2. `assign_final_classification.Rmd` : Reconciles rule-based, LLM, ML, and OpenSecrets codes for direct $5k+ contributors into a single `code_final` per UUID. Flags uncertain cases for human review. 
    - outputs: `10_outputs/final_classifications_before_review.csv`, `10_outputs/manual_review_queue.csv` 
3. `match_pac_classifications.Rmd` : Computes an industry distribution ($ and % by sector) for each PAC using its classified contributors. PAC-within-PAC contributions are resolved recursively in topological order. 
    - output: `10_outputs/pac_industry_breakdown.csv` 
4. `build_race_prop_breakdown_input.R` : Assembles all contributions with industry codes attached. For PAC contributions, replaces each PAC row with one row per industry using the distributions from step 4. 
    - output: `10_outputs/race_prop_breakdown_input.csv` 
5.  `compute_breakdown.R` : Aggregates by candidate and industry to produce final dollar amounts and percentages. Verifies that percentages sum to 100% within each candidate. 
    - output: `10_outputs/industry_breakdown_by_race.csv` 


### Between steps 2–3 and step 4: human review

Step 2 produces `10_outputs/manual_review_queue.csv` — a combined queue of flagged rows, not-employed individuals, and a confidence QA sample. Decisions from that review should be incorporated before running steps 4–6. PAC-specific review decisions go into `10_inputs/pac_manual_review.csv`, which step 4 reads to handle federal PACs and PACs with insufficient data.

---

## Helpers

`resolve_classifications.R` is sourced by `assign_final_classification.Rmd` ; it defines the resolution logic including source priority rules, confidence thresholds, overrides (pre-classified rows, retired/not-employed codes, PAC/candidate codes), and the functions `resolve_code_one()` and `resolve_category_one()` that pick a winner when multiple classifiers disagree.

---

## Code reconciliation logic

Each contributor has up to four classification sources: rule-based (08), LLM (09), ML (07), and OpenSecrets match (08). The priority order is:

1. **Hard overrides** — pre-classified (manually reviewed) rows are locked; "100" (retired/not-employed) from any reliable source wins; PAC/candidate codes from reliable sources are locked
2. **High-trust rule tiers** — employer lookup, running list, and keyword match from the rule-based classifier
3. **Agreement between sources** — OS + LLM agree, or rule + OS agree, or rule + LLM agree
4. **Single high-confidence source**
5. **Fallback to "99"** (unknown/uncategorized)

Rows that fall through to step 4 or 5 are flagged for human review. Rows where the LLM disagrees with the final code are also flagged for review.

---

## Key inputs

- `../08_alternative_pipeline/08_outputs/classified_contributors_combined.csv` : Rule-based classifier output (08 pipeline) 
- `../08_alternative_pipeline/08_outputs/combined_contributions_os_matches.csv` : OpenSecrets matches for direct contributors (08 pipeline) 
- `../08_alternative_pipeline/08_outputs/over_5k_tagged.csv` : $5k+ contributors to PACs with amounts (08 pipeline) 
- `../08_alternative_pipeline/08_outputs/pac_traversal_order.csv` : Topological sort of PAC dependency graph (08 pipeline) 
- `../09_llm_classifier/09_outputs/classification_input_combined_expanded_full_*.csv` : LLM classifications for direct contributors (09 pipeline) |
- `../../data/07_output_ml_classification/08_entities_with_ml_*.csv` : ML classifier predictions (07 pipeline) 
- `../08_alternative_pipeline/08_inputs/already_classified_contributions.csv` : Manually pre-classified contributions (locked from override) 
- `10_inputs/pac_manual_review.csv` : Human review decisions for PACs (federal, unresolvable, etc.) 

---

## Key outputs

- `10_outputs/final_classifications_before_review.csv` : One row per direct-contributor UUID with `code_final`, `code_final_source`, review flags 
- `10_outputs/manual_review_queue.csv` : Combined human review queue: flagged rows, code-100 individuals, and QA confidence sample 
- `10_outputs/pac_industry_breakdown.csv` : Industry distribution per PAC — dollar amount and percentage by sector 
- `10_outputs/race_prop_breakdown_input.csv` : All contributions to tracked races with codes, PAC contributions exploded by industry 
- `10_outputs/industry_breakdown_by_race.csv` : **Final editorial output**: industry $ and % breakdown per candidate and ballot measure 

`industry_breakdown_by_race.csv` is the end product with every dollar, including PAC money, attributed back to its underlying donor industries.
