# Custom classification scheme (CalMatters campaign finance)

Custom codes layered on top of the standard 2-digit NAICS sectors, per the
July 2026 editorial category sheet. These are NOT real NAICS codes. The
full taxonomy lives in `data/03_input/masterfile/custom_naics_labels.csv`
(EDITORIAL — edit that file to change categories; no code changes needed).

## Two-masterfile design

- **Masterfile #1** — `data/03_input/masterfile/running_list.csv`.
  Real NAICS codes only, from static sources: OpenSecrets/H1B (0301),
  EDD (0302/0303), crosswalk (0304), manual real-NAICS rows (0305).
  Stable ground truth; the ML classifier (0701) trains on it at PARENT
  level.
- **Masterfile #2** — `data/03_input/masterfile/running_list_custom.csv`.
  Derived editorial view: 0306_build_custom_masterfile.py runs every
  Masterfile #1 entity through the sub-code resolver
  (`code/03_aggregating_data/subcode_resolution.py`). When editorial
  preferences change, edit custom_naics_labels.csv and rerun 0306 only.

## Code table (terminal labels)

| Code | Meaning | Relation to old scheme |
|------|---------|------------------------|
| 11..92 sectors | Standard NAICS sectors with editorial display labels | unchanged codes, some renamed labels |
| 22a  | Renewable energy | new carve-out of 22 |
| 51a  | Media (broadcasters & news) | new carve-out of 51 |
| 52a / 52b | Finance / Insurance | **partition** of 52 (52 no longer terminal) |
| 54a  | Lawyers & law offices | new carve-out of 54 |
| 56a / 56b | Admin & support / Waste mgmt | **partition** of 56 (56 no longer terminal) |
| 71a  | Gambling (gaming/casinos) | new carve-out of 71 |
| 76   | Associations | unchanged |
| 77a / 77b | Private / Public sector unions | **partition** of old 77 |
| 78   | Defense contractors | unchanged |
| 79   | Technology, software and AI | unchanged |
| 88   | PACs / political committees | unchanged |
| 90   | Candidate committees | unchanged |
| 91   | Political parties | unchanged |
| 92a  | Tribes | carve-out of 92 (the old 92 tribal regex moved here) |
| 99   | Unknown / uncategorized | narrowed: undeterminable only |
| 100  | Retired / Homemaker / Student | new, split out of old 99; matched on OCCUPATION |

## Sub-code resolution (subcode_resolution.py)

For one entity (name, occupation, parent naics_code), applied by 0306
(Masterfile #2 build) and 0702/0705 (post-ML):

1. **Employer-name regex pass** — custom_naics_labels.csv rows in file
   order, later rows overwrite earlier (same semantics as the old
   0501.apply_custom_label_overrides). Political rows (88, 90, 91) sit
   last in the file and stay authoritative; 92a (tribes) is ordered after
   71a, so a name carrying both tribal and casino tokens resolves to 92a.
2. **Occupation regex pass** — rows with `applies_to = occupation`
   (currently 100). Runs after (1), so occupation wins for individuals.
3. **Partition fallback** — parent is 52/56/77 and no regex fired: the
   entity text embeds against each child's plain-language `description`
   (same sentence encoder as the 07 models); nearest child wins,
   confidence recorded. The classifier itself stays at parent level.
4. **Passthrough** — parent is itself a terminal code: kept, restamped
   with the scheme's current label.

Provenance: every resolved row carries `resolution_method`
(regex / occupation-rule / embedding / parent-passthrough / manual) and
`resolution_confidence`, so low-confidence embedding calls can be
exported for editorial review.

## Precedence rules (inherited + new)

- **88 wins over 76/77a/77b**: a union's or association's PAC is coded 88.
- **90/91 win over 88**: candidate committees / parties, not generic PACs.
- **92a wins over 92**: tribal entities leave the government bucket.
- **100 wins over employer signals** for individuals (occupation gate).
- **99** is a fallback, never an override; 99 entities should be
  re-attempted on future contributions.

## Where each layer is enforced

1. `custom_naics_labels.csv` — the taxonomy + regex/description rules
   (EDITORIAL).
2. `subcode_resolution.py` — the resolver (PROCEDURAL; order above).
3. `0306_build_custom_masterfile.py` — Masterfile #2 build, applies
   `manual_custom_overrides.csv` (diverted by 0305) last.
4. `keyword_priors.py` — unchanged; still emits PARENT codes, resolution
   happens downstream.
5. `0705_classify_race_custom.py` — end-to-end race classification under
   the custom scheme (Masterfile #2 lookup -> parent-level ML -> resolver
   -> 99 fallback).
6. `0704_benchmark_reviewed.py` — scores at old-scheme parent level
   (sub-codes collapsed; 100 -> 99) until a re-reviewed benchmark exists.

## Open items

- 52/56 sub-splits have no labeled training data; the embedding fallback
  is unvalidated — export low-confidence rows for review and consider a
  hand-labeled ~100-row benchmark of old 52s.
- PAC (88) re-coding from its own donors' industry mix: still not
  implemented.
- 11a (Cannabis) and other non-primary rows in the editorial sheet are
  deliberately excluded until marked primary.
