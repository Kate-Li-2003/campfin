# Campaign Finance Classifier

Processes contribution data from California's 2025–2026 election cycle — the Governor's race, Insurance Commissioner's race, and all propositions — and assigns each contributor an industry classification using a combination of static sources and machine learning and LLM classifications.

## Overview

Process:

- Fetch contribution data from Datasette copy of Power Search maintained by CalMatters.
  (https://calmatters-powersearch-2026.fly.dev/powersearch)
- Groups contributors by entity to identify unique contributors above the $5,000 threshold. 
  Only contributors above this threshold are classified.
- Assigns industry classifications to contributors using:
  - **Running list** — a curated database of employer -> NAICS mappings built from H1B and EDD employer data. 
  - **OpenSecrets data** — reference database of campaign finance contributors and their industry
  - **ML classifier** — predicts (custom) NAICS code from employer/occupation text
  - **LLM classifier** — Gemini (web search) and Claude (web search optional) classify all contributors

## Pipeline Structure

```
code/
├── 01_contributor_pipeline/   # Entity resolution, PAC graph traversal, contributor classification based on static sources
├── 02_ml_classifier/          # ML-based NAICS classification for unmatched contributors
├── 03_llm_classifier/         # LLM classification via Gemini (web search) and Claude 
├── 04_assign_classifications/ # Reconciles all classification sources into a final output
└── shared_utils/              # Shared helper scripts
data/
└── 03_input/masterfile/       # Static reference data (running list, OpenSecrets, NAICS labels)
```

See each stage's `README.md` for run instructions, and `code/README_code.md` for the full run order across stages.

### Prerequisites

- Python 3.13+
- R 4.x with RMarkdown
- Virtual environment with packages in `requirements.txt`

### R packages

Install required R packages once:

```r
install.packages(c(
  "tidyverse", "dplyr", "readr", "stringr", "tidyr",
  "fuzzyjoin", "stringdist", "reclin2", "phonics",
  "googlesheets4", "here", "httr", "jsonlite",
  "igraph", "uuid", "digest"
))
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd campfin
```

2. Install Python dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Add API keys for the LLM classifier. Create a `.env` file in `code/03_llm_classifier/`:
```
GEMINI_API_KEY=...
```

## Data

Large input files (raw contribution CSVs, CalAccess reference files) are not tracked in git due to size. Find them in [this folder](https://drive.google.com/drive/u/0/folders/185MF51ba1V25FokzkH5ArEJO1fACWXKk) in Google Drive.
