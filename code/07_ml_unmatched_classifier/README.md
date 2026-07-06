# ML Classifier Pipeline Summary

## Overview
The `05` pipeline generates static classifications (e.g., using EDD, H1B, keyword matching), outputting some data file (`05_output/.._classified.csv`). For all donations that cannot be classified under this procedure (`data_source_1` = NA), the 07 pipeline generates classifications using a sentence-transformer embedding of the donor's employer/occupation. It then applies additional rule-based corrections. This directory contains a summary of key scripts and modules used in that process. 

## Script Summary 
- **0701: `train_classifier`** trains three multinomial logit regressions on Microsoft's MiniLM embeddings (used on `running_list.csv`, or the masterfile equivalent). It also trains on occupations. Models and embeddings are saved to `data/07_output_ml_classification/models`.
- **0702: `predict_unmatched`** takes rows with NA `data_source_1` and generates level1 through level3 category predictions and a NAICS prediction (using an OpenSecrets Level 1 - NAICS custom crosswalk).
- **0702: `evaluate`** evaluates the model on the masterfile. Used for model training to avoid overfitting.
## Module Summary
- **text_features**: imported by `0701`, `0702`. For individuals, occupation is weighted relative to employer. Avoids training on non-employers populating the employer field (e.g., self-employed, retired, etc.)
- **keyword_priors.py**: imported by `0702`. For classifications that have a *low ML confidence score* (e.g., <0.3), assign a classification based on keyword matching. Consists of employer-affiliated keywords and occupation-affiliated keywords.

