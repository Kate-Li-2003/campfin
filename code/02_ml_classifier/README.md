# 02 ML Classifier

## Overview

The 01 contributor pipeline generates static classifications using EDD, H1B, and keyword matching. For entities that cannot be classified by those rule-based methods (`data_source_1` = NA), this pipeline generates classifications using sentence-transformer embeddings of the donor's employer and occupation, then applies additional rule-based corrections.

## Run order

1. **`build_ml_training_data.py`** — builds training data from the masterfile (pre-classified entities from prior runs, OpenSecrets data, H1B + EDD employers)
2. **`0201_train_classifier.py`** — trains three multinomial logit regressions on MiniLM embeddings. Saves model files to `data/07_output_ml_classification/models/`
3. **`run_ml_on_01_output.py`** — runs the trained model on `01_outputs/classification_input_combined.csv` from the 01 pipeline. Output feeds into the 04 pipeline.

## Other scripts

- **`0203_evaluate.py`** — evaluates model accuracy on the training set and an OOD set. Run after `0201` to check for overfitting. Note: the default OOD input paths point to old pipeline outputs — update `DEFAULT_OOD_INPUTS` to a current `01_outputs` classified CSV.
- **`0204_benchmark_reviewed.py`** — benchmarks the full classification stack (ML + keyword priors + custom-code overrides) against a manually reviewed file.

## Modules

- **`text_features.py`** — imported by `0201` and `run_ml_on_01_output`. Weights occupation relative to employer for individuals.  Filters out non-employers populating the employer field (e.g., self-employed, retired).
- **`keyword_priors.py`** — imported by `run_ml_on_01_output`. For classifications with low ML confidence (<0.3), assigns a classification based on keyword matching. Covers employer-affiliated and occupation-affiliated keywords.


## Archived scripts

`archive/0202_predict_unmatched.py` and `archive/0205_classify_race_custom.py` were the old per-race classification approach (reading from `output/05_output/`). Replaced by `run_ml_on_01_output.py`.

## Key inputs

- `../../data/03_input/masterfile/running_list_alt.csv` : Running list for training (updated by 01 pipeline after each run)
- `../01_contributor_pipeline/01_outputs/classification_input_combined.csv` : Contributor data from the 01 pipeline

## Key outputs

- `../../data/07_output_ml_classification/models/` : Trained model files
- `../../data/07_output_ml_classification/01_entities_with_ml_[date].csv` : ML predictions per entity — input to the 04 pipeline
