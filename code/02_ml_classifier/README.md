# 02 ML Classifier

## Overview

The `shared_utils` pipeline generates static classifications using EDD, H1B, and keyword matching. For entities that cannot be classified by those rule-based methods (`data_source_1` = NA), this pipeline generates classifications using sentence-transformer embeddings of the donor's employer and occupation, then applies additional rule-based corrections.

## Script Summary

- **0201: `train_classifier`** — trains three multinomial logit regressions on Microsoft's MiniLM embeddings (trained on `running_list.csv` or the masterfile equivalent). Also trains on occupations. Models and embeddings are saved to `data/07_output_ml_classification/models`.
- **0202: `predict_unmatched`** — takes rows with `data_source_1` = NA and generates level1–level3 category predictions and a NAICS prediction (via an OpenSecrets Level 1 → NAICS custom crosswalk).
- **0203: `evaluate`** — evaluates the model on the masterfile. Used during model training to avoid overfitting.
- **0204: `benchmark_reviewed`** — benchmarks model performance against manually reviewed classifications.
- **0205: `classify_race_custom`** — runs classification for a specific race using the custom NAICS scheme.

## Module Summary

- **`text_features.py`**: imported by `0201`, `0202`. For individuals, occupation is weighted relative to employer. Filters out non-employers populating the employer field (e.g., self-employed, retired).
- **`keyword_priors.py`**: imported by `0202`. For classifications with low ML confidence (<0.3), assigns a classification based on keyword matching. Covers employer-affiliated and occupation-affiliated keywords.

## Integration with the 01 contributor pipeline

- **`build_ml_training_data.py`** builds the ML training data from the masterfile (three sources: pre-classified entities from prior runs, OpenSecrets data, H1B + EDD employers). Output feeds into `0201_train_classifier.py`.
- **`run_ml_on_01_output.py`** runs the trained ML model on the contributor data produced by the 01 pipeline, replacing the earlier per-race `0202` approach.

## Key inputs

- `../../data/03_input/masterfile/running_list.csv` : Main masterfile for training
- `../../data/03_input/masterfile/running_list_alt.csv` : Alternative running list (updated by 01 pipeline) **use this one**
- `../01_contributor_pipeline/01_outputs/classification_input_combined.csv` : Contributor data from the 01 pipeline to run predictions on

## Key outputs

- `../../data/07_output_ml_classification/models/` : Trained model files and embeddings cache
- `../../data/07_output_ml_classification/01_entities_with_ml_[date].csv` : ML predictions per entity — input to the 04 pipeline
