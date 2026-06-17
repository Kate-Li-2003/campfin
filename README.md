# Campaign Finance ML Classifier
This repo contains three main components: 1) pulling and cleaning election data from CA SOS; 2) matching campaign finance flows to static and dynamic industry classification sources; 3) predicting industry affiliations for remaining unmatched donations.

# Directory layout
After cloning into this repository (`git clone ...`, `cd ... `, `python3 -m venv .venv`, `pip install -r requirements.txt`), run the following: 

## Data Pulling from Cal-Access

To pull **all donations** for a race, run: 
- `.venv/bin/python code/04_donations_data_pull/pull_calaccess.py --races [GOV/LTG/IC]`
- Pulled data will be stored at `data/01CalAccess_CampaignFinance_Data/governor_race_2026.csv` 
To pull all data for the governor's race _above_ a specific donation amount (e.g., 5K), run:
-  `.venv/bin/python code/04_donations_data_pull/0401_filter_10k_donations.py \ --inputs “data/01CalAccess_CampaignFinance_Data/governor_race_2026-04-27.csv” \ --amount-min 5000`
-  Filtered contributions data will be stored at `data/04_output_latest_data_pulls/0401_races_5kfilters/governor_race_2026_over_5k.csv`

## Classifications (Static) 

To generate matches using static/dynamic lookup sources **ONLY**, run:
`python code/05_candidate_industry_affiliations/0504_classify_other_races.py --inputs
  extra/powersearch_2526/power_search_candidates_2526.csv --out-dir output/05_output`
- this routes output to `output/05_output`, under `..._classified.csv`

## Classifications (SVM) 

To generate matches using **classifier** **predictions**, run:
`python code/07_ml_unmatched_classifier/0702_predict_unmatched.py --inputs output/05_output/power_search_candidates_2526_classified.csv`
- this routes output to `data/07_output_ml_classification/`, under `...classified_with_ml.csv`

# other notes
To see how this fits into CM's Campaign Finance Categorization Pipeline, see Steps 2, 3, & 4 [here]([url](https://docs.google.com/document/d/1D_rzSgnsA2Yx8629o9aypV4-XedRAodRtS-tCFN8h-A/edit?usp=sharing)). 
