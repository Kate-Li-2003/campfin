# 01 Contributor Pipeline

This pipeline processes contributions from specific 2026 races — the Governor's race, the Insurance Commissioner race and all propositions — and assigns each contributor an industry classification code.

The pipeline has two phases:

1. **Entity resolution** — grouping contributor records that refer to the same underlying person or organization, assigning stable UUIDs, and determining which entities cross the $5,000 reporting threshold
2. **Classification** — assigning each qualifying entity a NAICS or custom industry code through a waterfall of matching methods (prior classifications, employer databases, keyword lists, OpenSecrets reference data)

---

## Script Order

### One-time setup

Run these once before the first pipeline run.

- `0100_opensecrets_data_processing.Rmd` : Cleans and deduplicates two OpenSecrets/Follow the Money datasets into a single classification reference file
    - output: `../../data/03_input/masterfile/running_list_opensecrets_alt.csv`
- `0104_build_running_list.py` : Combines H1B employer data and EDD employer data into `running_list_alt.csv`, resolving conflicts where the same employer name maps to different NAICS codes

### Each pipeline run

Run these in order each time there is new contribution data to process.

1. `0101_data_pull.R` : Pulls contributions from the Datasette database, normalizes all fields, and flags entity type (individual vs. org)
    - output: `01_inputs/power_search_contributions_normalized.csv`

2. `0102_entity_resolution.Rmd` : Applies fuzzy matching and custom record linkage to surface name variants of contributors for manual review. **See the run order note below — this notebook has a dependency on `0102b`.**
    - outputs (first pass): `01_outputs/org_name_variant_review_queue_[date].csv`, `01_outputs/indiv_name_variant_review_queue.csv`
    - outputs (second pass): `01_outputs/classification_input.csv`

3. `0103_find_pac_contributors.Rmd` : Traverses PAC contribution graphs up to 5 levels deep to trace money back to original funders. Produces the PAC review queue.
    - outputs: `01_outputs/pac_classifiability.csv`, `01_outputs/pac_classification_input.csv`

4. `0102b_pac_graph_walk.Rmd` : Traverses the PAC graph and builds `pac_universe.csv` — the list of all PACs (and their contributors) connected to our tracked races. Must run before 0102's second pass. This is used as input for entity resolution. Re-run `0102` after this is updated.
    - output: `01_outputs/pac_universe.csv`

5. `0102c_build_classification_input.Rmd` : Aggregated contributors to determine who crosses the $5k threshold. Combines direct and PAC contributors into the final classification input file.
    - output: `01_outputs/classification_input_combined.csv`

6. `0105_classify_contributors.py` : Applies the classification waterfall for each contributor: prior classifications -> running list (EDD + H1B employers) -> optional EDD lookup -> keyword matching -> identity overrides (PACs, unions, government, retired, etc.)
    - output: `01_outputs/classified_contributors.csv`

7. `0106_match_opensecrets.Rmd` : Matches contributors against OpenSecrets reference data using exact, fuzzy, and token-similarity matching; maps OpenSecrets categories to the custom NAICS schema
    - output: `01_outputs/combined_contributions_os_matches.csv`

8. `0107_update_index.Rmd` : Appends newly classified entities to the persistent classification index; existing entries are never overwritten
    - output: `01_outputs/entity_classification_index.csv`

9. `0108_extend_running_list_from_edd.py` : Promotes EDD-derived classifications from this run into `running_list_alt.csv` so future runs can match them directly
    - output: `../../data/03_input/masterfile/running_list_alt.csv`

---

## Run order for steps 2–5:

`0102` and `0102b` are dependent on each other. 

1. **Run `0102` (first pass)** through the entity resolution and name variant sections. This produces the org and individual name variant review queues and determines which PACs that contributed directly to the races/props meet the $5k threshold.  
2. **Human review:** Review the org and individual name variant queues. Add confirmed name variants to the [2026 name map](https://docs.google.com/spreadsheets/d/1THwQtvw5s9ZO9n7QHncekl4kRMc1Nf3MJHViSYALDps/edit?gid=897118445#gid=897118445). Org decisions also go into `01_inputs/org_name_variant_review_queue_w_review.csv`.
3. **Run `0102b_pac_graph_walk.Rmd`** to traverse the PAC graph and produce `01_outputs/pac_universe.csv`. This grabs all PAC contributors regardless of contribution amount so we can do entity resolution and determine which contributors meet the threshold. 
4. **Re-Run `0102`**  — reads `pac_universe.csv` to do entity resolution over the PAC contributors. 
5. **Human review:** Review the org and individual name variant queues.
6. **Run `0102c_build_classification_input.Rmd`** to aggregated contributors and determine which entities we need to classify. $5k PAC contributors are separate to go through the PAC graph in `0103`. Org/indiviual contributors are ready to be classified directly. 
7. **Run `0103`** to traverse the PAC graph. Now that entity resoltuion is complete, the PAC graph determines which contributors meet the (within-recipient PAC) $5k threshold. 
8. **Human review:** Review the PAC review queue produced by `0103`. Human review asssigns a classification that will be used for the PACs under $5k contributions and processes edge cases. 
9. **Re-run** entity resolution and PAC graph steps if any new PAC ids are added after the manual review of the PACs produced in `0103`


After step 9, continue to `0105_classify_contributors.py`.

---

## Helper files

- `standardization_helpers.R` : Shared R functions used across all R scripts: name standardization, typo fixing, business suffix removal, individual vs. org detection, name parsing

---

## Key inputs

- `01_inputs/power_search_contributions_normalized.csv` : Fetched from Power Search Datasette by `0101`
- `01_inputs/already_classified_contributions.csv` : Prior classified contributions from earlier pipeline runs 
- `01_inputs/entity_registry.csv` : Entity UUID store (used for tracking `Normalized` names which determine entity groups)
- `01_inputs/name_entity_mapping.csv` : Persistent raw name -> UUID mapping; updated by `0102`. This is essentially a copy of the 2026 name map with uuids added. 
- `01_inputs/org_name_variant_review_queue_w_review.csv` : Human-reviewed org name variant decisions (will be multiple review files)
- `01_inputs/FILERNAME_CD.TSV` : CalAccess filer name/ID reference (download from calaccess.californiacivicdata.org)
- `01_inputs/2026 Contributor name mapping and descriptions - Names.csv` : Name variant and normalization map (exported from [2026 name map](https://docs.google.com/spreadsheets/d/1THwQtvw5s9ZO9n7QHncekl4kRMc1Nf3MJHViSYALDps/edit?gid=897118445#gid=897118445))
- `../../data/03_input/masterfile/custom_naics_labels_updated.csv` : Regex rules for identity overrides (PACs, unions, government, retired, etc.)

---

## Key outputs

- `01_outputs/classified_contributors_combined.csv` : Primary output: each contributor entity with NAICS code, label, and classification source
- `01_outputs/combined_contributions_os_matches.csv` : Contributors matched to OpenSecrets level-1/2/3 categories and mapped to custom NAICS codes
- `01_outputs/classification_input_combined.csv` : Input to the LLM and ML classifiers (pipelines 02 and 03)
- `../../data/03_input/masterfile/running_list_alt.csv` : Alternative running list, updated in place after each run
