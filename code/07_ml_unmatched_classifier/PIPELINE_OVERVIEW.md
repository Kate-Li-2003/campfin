# code/07_ml_unmatched_classifier — pipeline overview

## Where 07 sits

The 05 pipeline (static: masterfile lookup → keyword match → EDD → custom
regex rules) classifies most donor entities. Whatever it can't match
(`data_source_1` is null) falls through to 07, which predicts a
classification from a sentence-transformer embedding of the donor's
employer/occupation text, then layers rule-based corrections on top.

```
data/04_output_latest_data_pulls/<race>.csv        (raw donations, 04)
        │
        ▼
code/05 … 0504_classify_other_races.py             (static classification)
        │   masterfile → keyword → EDD → custom_naics_labels.csv overrides
        ▼
output/05_output/<race>_classified.csv             (entity level; unmatched rows have
        │                                           data_source_1 = null)
        ▼
┌──────────────────────────── code/07 ────────────────────────────────┐
│ 0701_train_classifier.py  ──►  models/  (offline, retrain as needed)│
│                                                                     │
│ 0702_predict_unmatched.py  (per race, per entity):                  │
│    1. build embed text        ── text_features.py                   │
│    2. ML predict (LogReg on MiniLM embeddings)  ── models/          │
│    3. promote if level1 conf ≥ --threshold (0.30)                   │
│    4. keyword priors override if naics conf < --prior-threshold     │
│       (0.55)                  ── keyword_priors.py                  │
│    5. custom-code regex overrides (authoritative)                   │
│                               ── custom_naics_labels.csv            │
│    6. 99 fallback for anything still uncoded                        │
│    7. expand entity → donation level (via 0504.assign_entities)     │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
data/07_output_ml_classification/<race>_classified_with_ml.csv
        (one row per qualifying donation)

evaluation:  0703_evaluate.py  (train + OOD accuracy of the raw models)
             0704_benchmark_reviewed.py  (full stack vs reviewed ground truth)
```

## Scripts

### 0701_train_classifier.py (offline training)
Trains three multinomial logistic regressions (level1_category,
level2_category, naics_code) on MiniLM (`all-MiniLM-L6-v2`) embeddings of
`running_list.csv` names. Augmentation (default on; `--no-aug` reverts):
occupation seeds from `occupation_naics_seed.csv` rendered as
`"occupation: {occ}"`, plus labeled individuals harvested from 05 outputs,
both in the exact text format 0702 serves — this teaches the model that
"farmer" ⇒ 11 etc., which bare company names can't. Classes with < 5
examples are dropped. Writes models + `encoder_name.txt` +
`text_format.txt` (v1 = legacy names-only, v2 = augmented format) +
`embeddings_cache.npz` to `data/07_output_ml_classification/models/`.

### 0702_predict_unmatched.py (inference)
For each input CSV, takes rows with `data_source_1` null and runs the
decision stack above. Key outputs per row: `ml_level1_category`,
`ml_level2_category`, `ml_naics_code` (+ `_conf` each) — always written for
unmatched rows regardless of promotion; `prior_naics_code` — whichever
keyword prior fired, even when it didn't override; final
`naics_code`/`naics_label` and `data_source_1` ∈ {`ml prediction`,
`keyword prior`, `custom rule`, `uncategorized fallback`}. Reads
`text_format.txt` so old (v1) models keep receiving v1-format text.
Defaults only cover the lt-governor and insurance-commissioner files — pass
`--inputs` (and a matching `--amount-min`) for other races.

### 0703_evaluate.py (model evaluation)
Top-1/top-3 accuracy of the raw classifiers on (a) the training set
(overfit sanity check) and (b) OOD: real-race entities labeled by non-ML
sources, with training-set name overlaps removed. NAICS is scored at the
normalized 2-digit sector level (31/32/33 ≈ 31-33, 3+ digit codes
truncated). Writes `eval_report.csv`.

### 0704_benchmark_reviewed.py (stack benchmark)
Scores the reviewed ground-truth file (`naics_final_classification` vs the
stack). Uses the file's precomputed `ml_naics_code`/`_conf` columns, so it
measures the rule/prior layers without needing the encoder. Reports
baseline vs stack accuracy, per-layer accuracy, per-code breakdown
(`--per-code`), rows fixed/broken, and remaining confusions. Use
`--entity-level` to dedupe repeat donations. Current: 0.473 → 0.661.

## Modules

### text_features.py
Single source of truth for embed-text construction (both 0701 and 0702
import it). Junk-employer detection (`Self Employed[- ]…`, `Refunded`,
`Retired`, employer = donor's own name), junk-occupation list, and
`build_embed_text`: individuals get `"occupation: {occ}; employer: {emp}"`
(occupation leads — it carries the signal), orgs get the bare name.
`FORMAT_VERSION = "v2"`.

### keyword_priors.py
Ordered high-signal regex → code lists: `EMPLOYER_PRIORS` (checked first;
first match wins; most specific first — oil beats energy, construction
beats power, tech beats education) and `OCCUPATION_PRIORS` (checked when no
employer token fires). Also `CODE_LABELS` (code → label, incl. custom
codes) and `apply_keyword_priors` (the 0702 integration; only touches
previously-unmatched rows, never masterfile/keyword/EDD classifications).
Design rule: general area of business outweighs business type.

## Custom codes

See `CUSTOM_CODES.md` for the 76/77/78/79/88/90/91/99 mapping, precedence
(88 beats 76/77; 90/91 beat 88), and enforcement points.

## Reference / data files

| file | role |
|------|------|
| `data/03_input/masterfile/running_list.csv` | training labels (name → level1/level2/naics); grown over time from masterfile + manual ingests |
| `data/03_input/masterfile/custom_naics_labels.csv` | custom-code regex rules; ordered, later rows win; editable without code changes; used by 0501/0504 and re-applied by 0702 |
| `data/03_input/training data (manual classifications)/occupation_naics_seed.csv` | occupation → NAICS seeds for 0701 augmentation |
| `output/05_output/donors_classified_with_manual.csv` | source of labeled individuals for 0701 augmentation; OOD eval set for 0703 |
| `data/03_input/benchmarks/governor_contribution_classifications.xlsx` | reviewed ground truth for 0704; keep out of training so the benchmark stays honest |
| `data/07_output_ml_classification/models/` | trained artifacts: `*_clf.joblib`, `encoder_name.txt`, `text_format.txt`, `embeddings_cache.npz` |
| `data/04_output_latest_data_pulls/<race>.csv` | raw donation files used by 0702 to expand entities back to donation level |
| `code/07_ml_unmatched_classifier/NAICs, OpenSecrets Crosswalk.xlsx` | reference crosswalk between NAICS sectors and OpenSecrets categories (manual reference, not read by code) |

## Typical workflow

```bash
# 1. retrain after label/seed changes
python code/07_ml_unmatched_classifier/0701_train_classifier.py

# 2. sanity-check the models
python code/07_ml_unmatched_classifier/0703_evaluate.py

# 3. classify a race (defaults cover lt-gov + insurance commissioner)
python code/07_ml_unmatched_classifier/0702_predict_unmatched.py \
  --inputs output/05_output/power_search_governors_data_5k_060626_classified.csv \
  --amount-min 5000

# 4. score the full stack against the reviewed benchmark
python code/07_ml_unmatched_classifier/0704_benchmark_reviewed.py --entity-level --per-code
```
