# 05_candidate_industry_affiliations — LEGACY static classification

## TL;DR

**This directory is legacy.** It was the original per-race classification stage — match donors against the masterfile, then keywords, then EDD, then apply custom-label overrides — under the *old* category scheme (single 52 "Finance and Insurance", single 77 "Unions", no 22a/51a/54a/71a/92a/100). It has been superseded by `code/07_ml_unmatched_classifier/0705_classify_race_custom.py`, which does the same job under the current custom scheme with the shared sub-code resolver. Prefer 0705 for all new work; this directory is slated for removal.

**Do not delete it without checking one dependency first** (see below).

## What each file did

| File | Purpose (historical) |
|---|---|
| `0501_classify_donors_with_keywords.py` | Core static classifier: masterfile match → keyword match (`campaign_contribution_keywords.csv`) → custom-label override pass. |
| `0502_classify_donors_with_manual.py` | Merged manually classified sheets into race outputs; contains a hardcoded old-scheme code→label map. |
| `0503_candidate_industry_visualizations.py` | Pie/stacked-bar charts per candidate; hardcoded old-scheme buckets. |
| `0504_classify_other_races.py` | Generalized 0501 to arbitrary race files, adding org/individual entity handling and EDD live-lookup fallback. |

## Data relationships

- **Inputs:** filtered race files from `data/04_output_latest_data_pulls/` (produced by `code/04_donations_data_pull/0401_filter_10k_donations.py`), plus the masterfile and keyword list.
- **Outputs:** `output/05_output/` — classified race CSVs (`*_classified.csv`), run logs, and `figures/`. These reflect the old scheme and should be treated as historical snapshots, not current numbers.

## Before deleting this directory

`code/07_ml_unmatched_classifier/0702_predict_unmatched.py` still dynamically loads `0504_classify_other_races.py` for its donation-level expansion helper (`expand_to_donation_level`). If you remove 05, either move that helper into 07 or stop using 0702's donation-level mode (the newer `0705` has its own donation-level logic and does not depend on this directory).
