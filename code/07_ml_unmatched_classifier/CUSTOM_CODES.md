# Custom NAICS-style codes (CalMatters campaign finance)

Custom 2-digit codes layered on top of the standard 2-digit NAICS sectors,
per the "Guidance on coding donors" doc. These are NOT real NAICS codes;
they exist so political entities and a few editorially important groups
don't get lumped into 81 ("Other Services") or misclassified into an
industry sector.

| Code | Meaning                                             |
|------|-----------------------------------------------------|
| 76   | Associations (business & professional)              |
| 77   | Unions (labor, trade, and other)                    |
| 78   | Defense contractors                                 |
| 79   | Technology, software and AI companies (broad — includes startups, e.g. Airbnb, Uber, but also small AI/software shops) |
| 88   | PACs, contributor committees, political action funds|
| 90   | Candidate committees (e.g. "X for Assembly 2026")   |
| 91   | Political parties / party central committees        |
| 99   | Unknown / Uncategorized (retired, unemployed, or undeterminable) |

## Precedence rules

- **88 wins over 76/77**: a union's or association's PAC is coded 88, not
  77/76 (e.g. "Association of California School Administrators PAC" -> 88;
  without the PAC token -> 76). Per guidance: initially code ANY PAC as 88;
  downstream, donors *to* the PAC dictate its eventual categorization.
- **90/91 win over 88**: "Committee to Elect X" / "Democratic Central
  Committee" contain "Committee" but are candidate committees / parties.
- **99** is a fallback, never an override: assigned only when no other
  source (masterfile, keyword, EDD, custom rule, keyword prior, ML above
  threshold) produced a code. Donors coded 99 should be re-attempted on
  future contributions.

## Where each layer is enforced

1. `data/03_input/masterfile/custom_naics_labels.csv` — regex -> code rules
   (rows applied in file order; later rows overwrite earlier matches, so
   the file is ordered lowest -> highest precedence). Applied by
   `0501.apply_custom_label_overrides`, which runs LAST in the 0501/0504
   static pipeline AND again in `0702_predict_unmatched.py` after ML
   promotion, so ML can never final-stamp a political entity.
2. `keyword_priors.py` (this directory) — high-signal employer/occupation
   token -> code priors (incl. 79 tech and 92 government-employer rules).
   Overrides the ML prediction only when ML naics confidence is below
   `--prior-threshold` (default 0.55). The fired prior is always recorded
   in `prior_naics_code` for transparency.
3. `text_features.py` (this directory) — junk-employer detection
   ("Self Employed - <name>", "Refunded", ...) and the shared embed-text
   builder (occupation-led for individuals). Used by 0701 (training
   augmentation) and 0702 (prediction) so train and serve stay in sync
   (`text_format.txt` marker in the models dir).
4. `0702_predict_unmatched.py` — final 99 fallback for rows that remain
   unmatched after all layers (ml_* columns are kept for inspection).

## Notes / open items from the guidance doc

- PAC (88) is not a website-facing category: PACs are to be re-coded from
  the industry mix of their own donors (>= 85% single-industry -> that
  industry; otherwise stays 88 and contributions are apportioned
  proportionally). Not yet implemented in this pipeline.
- Union subtypes (service vs trade) may need a secondary code field later.
