# Static Data Overview

- `custom_naics_labels.csv`: Assigns custom NAICS labels to donations that exist outside the universe of 2-digit NAICs sectors 
  - (`88` for PACs; `76` for Associations; `77` for Unions; `78` for Defense; `79` for Tech; `99` for Unknown/uncategorized donations)
  - *To be updated* on an as-needed basis. [Reference](https://docs.google.com/document/d/1l6M_IZJ68Y72mSg8bY4K179AxRiz6n6a6d9xSz413PQ/edit?usp=sharing) for the latest guidance on coding donors.
- ` NAICS_OpenSecrets_Crosswalk_Mapped`: A manual cross-walk mapping Level 3 (coarsest) Open-Secrets categories to 2-digit NAICs Sectors. 

## Two-masterfile design (July 2026)

- `running_list.csv` — **Masterfile #1**: real NAICS codes only
  (OpenSecrets/H1B, EDD, crosswalk, manual real-NAICS rows).
- `running_list_custom.csv` — **Masterfile #2**: derived custom-scheme
  view, built by `code/03_aggregating_data/0306_build_custom_masterfile.py`
  from Masterfile #1 + `custom_naics_labels.csv` (EDITORIAL taxonomy) +
  `manual_custom_overrides.csv` (custom-coded manual rows diverted by 0305).
  Regenerate with 0306 whenever the editorial taxonomy changes.

See `code/07_ml_unmatched_classifier/CUSTOM_CODES.md` for the scheme.
