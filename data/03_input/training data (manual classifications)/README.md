# Overview

- `Keywords_Manually_Collected.csv`: A list of manually-curated Occupation and Employer keywords that assigns industries based on an entity's occupation or employer. (e.g., an investor at Khosla Ventures is assigned NAICS Code `52`)
- `Manual NAICS Classifications.xlsx`: A list of randomly sampled donations that were *not* first classified, but recieved manual categorization. (e.g., training data). Did not receive editorial review. 
- `manual_reviewed_labels.csv`: A list of randomly sampled donations that were first classified (using this repo's algorithm), and then recieved manual classifications. (Re-training data.) 
- `occupation_naics_seed.csv`: A list of occupations key terms that are associated with a NAICS code (e.g., `farmer` -> `11`)
- `manual_relabeled_data_062826.xlsx`: A list of donations with classifications_that received editorial review_; `naics_code_manual` contains final (John-reviewed) classifications for the donation. Last classified **6/28/26.**
