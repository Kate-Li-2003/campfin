# Campaign Finance ML Classifier
this repo contains three main components: 1) pulling and cleaning election data from CA SOS; 2) matching campaign finance flows to static and dynamic industry classification sources; 3) predicting industry affiliations for remaining unmatched donations.

# directory layout
after cloning into this repository (`git clone ...`, `cd ... `, `python3 -m venv .venv`, `pip install -r requirements.txt`), run the following: 

to pull data from CA SOS, run:

to generate matches using static/dynamic lookup sources **ONLY**, run:
`python code/05_candidate_industry_affiliations/0504_classify_other_races.py --inputs
  extra/powersearch_2526/power_search_candidates_2526.csv --out-dir output/05_output`
- this routes output to `output/05_output`, under `..._classified.csv`

to generate matches using **classifier** **predictions**, run:
`python code/07_ml_unmatched_classifier/0702_predict_unmatched.py --inputs output/05_output/power_search_candidates_2526_classified.csv`
- this routes output to `data/07_output_ml_classification/`, under `...classified_with_ml.csv`

# other notes
to see how this fits into CM's, see Steps 2, 3, & 4 [here]([url](https://docs.google.com/document/d/1D_rzSgnsA2Yx8629o9aypV4-XedRAodRtS-tCFN8h-A/edit?usp=sharing)). 
