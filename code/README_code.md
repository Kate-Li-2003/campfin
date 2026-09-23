# Code Overview

This directory contains the full campaign finance classification pipeline, organized into four stages.

## Stages

`01_contributor_pipeline` | Data pull, entity resolution, rule-based classification, OpenSecrets matching 
`02_ml_classifier` | ML classifier for entities not matched by rules 
`03_llm_classifier` | LLM classifier (Gemini web search + Claude/Gemini classification) 
`04_assign_classifications` | Reconcile all classifiers into a final code per entity; compute race/prop breakdowns 
`shared_utils` | Builds and maintains the masterfiles (EDD, H1B, OpenSecrets, manual review) 

## Run order

```
shared_utils/              (one-time or occasional — builds masterfiles)
01_contributor_pipeline/   (each run — pull data, resolve entities, classify)
02_ml_classifier/          (train once; predict on 01 output)
03_llm_classifier/         (web search + LLM classify on 01 output)
04_assign_classifications/ (reconcile all sources; compute breakdowns)
```

See each subdirectory's `README.md` for the script-level run order within each stage.

## Archive

`archive/` contains earlier exploratory code (EDA, older donation analysis, prior classification approaches). Not part of the current pipeline.
