# 08 Alternative Pipeline

This pipeline processed contributions from specific 2026 races — the Governor's race (Becerra vs. Hilton), the Insurance Commissioner race, and Propositions 40, 41, and 42 — and assigns each contributor an industry classification code.

The pipeline has two phases:

1. **Entity resolution** — grouping contributor records that refer to the same underlying person or organization, assigning stable UUIDs, and determining which entities cross the $5,000 reporting threshold
2. **Classification** — assigning each qualifying entity a NAICS or custom industry code through a waterfall of matching methods (prior classifications, employer databases, keyword lists, OpenSecrets reference data)

---

## Script Order

### One-time setup

Run these once before the first pipeline run, or when the underlying source data changes. They build reference files used by the classification step.

- `0800_opensecrets_data_processing.Rmd` : Cleans and deduplicates two OpenSecrets/Follow the Money datasets into a single classification reference file 
    - output: `running_list_opensecrets_alt.csv`) 
- `0804_build_running_list.py` : Combines H1B employer data and EDD employer data into `running_list_alt.csv`, resolving conflicts where the same employer name maps to different NAICS codes 

### Each pipeline run

Run these in order each time there is new contribution data to process.

1. `0801_data_pull.R` : Pulls contributions from the Datasette database set up by Jeremia, normalizes all fields, and flags entity type (individual vs. org) 
    - output:  `08_inputs/power_search_contributions_normalized.csv` 
2. `0802_entity_resolution_simplified.Rmd` : Two purposes (1) aggregates entities to determine which cross the $5k threshold and identifies PACs for separate processing (2) applies fuzzy matching and custom record linkage algorithm to surface name variants to review. 
    - outputs: `08_outputs/pac_input.csv`, `08_outputs/classification_input.csv` 
    - I need to clean up this notebook and separate the name variant surfacing from the aggregation because right now it's too confusing for someone else to run. 
    - NOTE: `0802_entity_resolution.Rmd` contains the OLD entity resolution process and is not being used
3.  `0803_find_pac_contributors.Rmd` : Traverses PAC contribution graphs up to 5 levels deep to trace money back to original funders; assesses how much of each PAC's money is traceable. Produces the PAC review queue. 
    - outputs : `08_outputs/pac_classifiability.csv` (stats for each PAC), `08_outputs/pac_classification_input.csv` (direct contributors to PACs that will be fed through the classification pipeline)
4. Re-run `0802_entity_resolution_simplified.Rmd` : Surface name variants from the PAC contributors -> name variants manually added to name variant map shared with Jeremia, which will then be used to aggregate entities/
5. `0805_classify_contributors.py` : Replaces prior masterfile approach. Applies classification waterfall for each contributor: - prior classifications -> running list (EDD + H1B employers) -> optional EDD lookup -> keyword matching -> identity overrides (PACs, unions, government, retired, etc.) 
    - output:  `08_outputs/classified_contributors.csv` |
6. `0806_match_opensecrets.Rmd` : Matches contributors against the OpenSecrets reference data using exact, fuzzy, and token-similarity matching; maps OpenSecrets categories to the custom NAICS schema 
    - output: `08_outputs/combined_contributions_os_matches.csv` |
7. `0807_update_index.Rmd` : Appends newly classified entities to the persistent classification index; existing entries are never overwritten 
    - output: `08_outputs/entity_classification_index.csv` 
    - NOTE: I have NOT been using this
8. `0808_extend_running_list_from_edd.py` : Promotes EDD-derived classifications from this run into `running_list_alt.csv` so future runs can match them directly 
    - output: `../../data/03_input/masterfile/running_list_alt.csv` |

> **Note on steps 2–4:** `0802` and `0803` are interdependent. `0802` produces `pac_input.csv`, which `0803` needs. `0803` then produces `pac_classifiability.csv`, which `0802` needs to finish. In practice: run `0802` through the entity resolution sections until it outputs `pac_input.csv`, then run `0803` in full, then re-run or continue `0802` to produce the final `classification_input.csv`.

### Between runs: human review

After step 2, two review queues are written to `08_outputs/`:

These two output files are manually reviewed to confirm whether name variants belong to the same entity. If they do,
the 2026 name map (https://docs.google.com/spreadsheets/d/1THwQtvw5s9ZO9n7QHncekl4kRMc1Nf3MJHViSYALDps/edit?gid=897118445#gid=897118445) is updated to include the name variants. Normalized name will then be used to aggregate contributors.
- `org_name_variant_review_queue_[date].csv` 
- `indiv_name_variant_review_queue.csv` 

Review decisions for orgs go back into `08_inputs/org_name_variant_review_queue_w_review.csv`. Individual decisions go into the 2026 Contributor Name Mapping Google Sheet (exported to `08_inputs/2026 Contributor name mapping and descriptions - Names.csv`).

After 0803_find_pac_contributors.Rmd is run, a PAC review queue is produced. This is manually reviewed by John who determines how we are able to classify the <$5k contributors to these PACs. 


---

## Helper files


- `standardization_helpers.R` : Shared R functions used across all R scripts: name standardization, typo fixing, business suffix removal, individual vs. org detection, name parsing 
-  `parse_names.R` : Earlier version of the name parsing functions; `standardization_helpers.R` is the canonical source (NOT used)
- `0802_entity_resolution.Rmd` : **Deprecated.** Older version of entity resolution — do not use. Use `0802_entity_resolution_simplified.Rmd` instead

---

## Key inputs

- `08_inputs/power_search_contributions_normalized.csv` : Fetched from Power Search Datasette `0801` 
- `08_inputs/already_classified_contributions.csv` : Prior classified contributions from earlier pipeline runs 
- `08_inputs/entity_registry.csv` : Persistent entity UUID store; maintained across runs 
- `08_inputs/name_entity_mapping.csv` : Persistent raw name → UUID mapping; updated by `0802` 
- `08_inputs/org_name_variant_review_queue_w_review.csv` : Human-reviewed org name variant decisions 
- `08_inputs/FILERNAME_CD.TSV` : CalAccess filer name/ID reference (download from calaccess.californiacivicdata.org) This is needed to normalize Contributor.IDs since different ids are sometimes used for the same org
- `../../data/03_input/masterfile/custom_naics_labels_updated.csv` : Regex rules for identity overrides (PACs, unions, government, retired, etc.) 

---

## Key outputs


- `08_outputs/classified_contributors_combined.csv` : Primary output: each contributor entity with NAICS code, label, and classification source |
- `08_outputs/combined_contributions_os_matches.csv` : Contributors matched to OpenSecrets level-1/2/3 categories and mapped to custom NAICS codes
- `08_outputs/entity_classification_index.csv` : Cumulative index of all classification decisions across runs (not currently using this, but should do a better job of incorporating)
- `../../data/03_input/masterfile/running_list_alt.csv` : Alternative running list (alternative to 05 pipeline / previous masterfile), updated in place after each run 
