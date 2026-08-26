# build_race_prop_breakdown_input.R
#
# Builds compute_breakdown.R's input: every contribution to the races/props
# we're tracking, with a NAICS code attached wherever we have one, plus a
# `candidate` column (Recipient Name, falling back to race_prop for ballot
# measures which have no candidate) -- compute_breakdown.R breaks the
#   - $5k+ direct givers to a race/prop -> final_classifications.csv (via
#     classification_input.csv's uuid to contribution_id bridge)
#   - PACs' own direct giving to a race/prop (pac_input.csv) -> split across
#     that PAC's own industry breakdown (pac_industry_breakdown.csv)
#   - everything else (small-dollar, unitemized, unclassified $5k+) -> left
#     with a blank code; compute_breakdown.R buckets these into "Small
#     Dollar" or "Uncategorized" itself
#
# Contribution universe: power_search_contributions_normalized.csv - has all contributions
#
# NOTE: contribution_id is not a unique id -- rows with identical
# contributor/amount/date/recipient info can share one (it's a hash of that
# info). A single contribution_id can map to >1 row in any of these files;
# where that happens for the classification bridge, we collapse to one
# code per contribution_id (preferring a real code over blank/"99"), same
# convention as match_pac_classifications.Rmd's best_cls_per_entity.

library(dplyr)
library(readr)
library(stringr)

# config

POWER_SEARCH_PATH   <- "../08_alternative_pipeline/08_inputs/power_search_contributions_normalized.csv"
CLASSIFICATION_INPUT_PATH <- "../08_alternative_pipeline/08_outputs/classification_input_combined.csv"
FINAL_CLASSIFICATIONS_PATH <- "10_outputs/final_classifications_before_review.csv"
PAC_INPUT_PATH       <- "../08_alternative_pipeline/08_outputs/pac_input.csv"
PAC_INDUSTRY_BREAKDOWN_PATH <- "10_outputs/pac_industry_breakdown.csv"
LABEL_URL <- "https://docs.google.com/spreadsheets/d/11QHvNJsdtMlc1YKo_iNvMB_Jfn5Ui-iYdlWhFYjSm9g/export?format=csv"

OUT_PATH <- "10_outputs/race_prop_breakdown_input.csv"

UNCATEGORIZED_CODE <- "99"

# race/prop filter
# same races the direct-contributor and PAC-graph pipelines were scoped to

race_filter <- function(df) {
  df %>% filter(
    `Ballot Measure(s)` == "OPPOSED: PROPOSITION 040 - IMPOSES ONE-TIME TAX ON CERTAIN INDIVIDUALS AND TRUSTS. INITIATIVE CONSTITUTIONAL AMENDMENT AND STATUTE." |
    `Ballot Measure(s)` == "SUPPORTED: PROPOSITION 040 - IMPOSES ONE-TIME TAX ON CERTAIN INDIVIDUALS AND TRUSTS. INITIATIVE CONSTITUTIONAL AMENDMENT AND STATUTE." |
    `Ballot Measure(s)` == "SUPPORTED: PROPOSITION 041 - REQUIRES AUDITS OF PROGRAMS FUNDED BY NEW STATE SPECIAL TAXES. PROHIBITS NEW STATE TAXES THAT ARE EXCLUDED FROM EXISTING VOTER-APPROVED STATE SPENDING LIMIT..." |
    `Ballot Measure(s)` == "SUPPORTED: PROPOSITION 042 - PROHIBITS NEW STATE PERSONAL PROPERTY TAXES AND CERTAIN RETROACTIVE STATE TAXES. INITIATIVE CONSTITUTIONAL AMENDMENT." |
    `Recipient Name` == "BECERRA, XAVIER" |
    `Recipient Name` == "HILTON, STEVE" |
    Office == "Insurance Commissioner"
  )
}

# load base contribution universe 

base <- read_csv(POWER_SEARCH_PATH, col_types = cols(.default = "c")) %>%
  race_filter() %>%
  mutate(Amount = as.numeric(Amount))

cat(sprintf("Base universe: %d contributions, $%.0f\n", nrow(base), sum(base$Amount, na.rm = TRUE)))

# sector labels (for PAC-split code_label) 

labels <- read.csv(LABEL_URL) %>%
  select(sector, sector_description) %>%
  distinct() %>%
  mutate(sector = as.character(sector))

# direct $5k+ contributor classification 
# bridge: classification_input.csv has both uuid (final_classifications.csv's
# key) and contribution_id (base's key); collapse to one code per
# contribution_id in case of hash collisions.

ci_bridge <- read_csv(CLASSIFICATION_INPUT_PATH, col_types = cols(.default = "c")) %>%
  select(uuid, contribution_id)
final_cls <- read_csv(FINAL_CLASSIFICATIONS_PATH, col_types = cols(.default = "c")) %>%
  select(unit_id, code_final, code_final_description)

direct_lookup <- ci_bridge %>%
  left_join(final_cls, by = c("uuid" = "unit_id")) %>%
  group_by(contribution_id) %>%
  arrange(is.na(code_final) | code_final == UNCATEGORIZED_CODE, .by_group = TRUE) %>%
  slice(1) %>%
  ungroup() %>%
  select(contribution_id, code = code_final, code_label = code_final_description)

cat(sprintf("Direct $5k+ classification bridge: %d contribution_ids, %d with a real code\n",
            nrow(direct_lookup), sum(!is.na(direct_lookup$code) & direct_lookup$code != UNCATEGORIZED_CODE)))

# PAC-to-race contributions 

pac_input <- read_csv(PAC_INPUT_PATH, col_types = cols(.default = "c")) %>%
  mutate(Amount = as.numeric(Amount))
pac_industry_breakdown <- read_csv(PAC_INDUSTRY_BREAKDOWN_PATH, col_types = cols(.default = "c")) %>%
  mutate(amount = as.numeric(amount), total_received = as.numeric(total_received),
         pct_of_total = as.numeric(pct_of_total))

# All rows in pac_input are PACs. Join key: FILER_ID when present (matches
# pac_classifiability$pac_id); fall back to effective_Contributor.ID for the
# propagated IDs from 0802 whose FILER_ID is blank.
pac_input <- pac_input %>%
  mutate(pac_id_key = if_else(
    !is.na(FILER_ID) & trimws(FILER_ID) != "",
    FILER_ID,
    effective_Contributor.ID
  ))

cat(sprintf("pac_input.csv: %d rows, %d with a pac_industry_breakdown entry\n",
            nrow(pac_input), sum(pac_input$pac_id_key %in% pac_industry_breakdown$pac_id)))

# Explode each PAC contribution into one row per industry, scaled by that
# PAC's own pct_of_total. Any shortfall (pct_of_total summing to <1, e.g.
# PACs not yet in pac_review) is left as an uncategorized remainder row.
pac_split_rows <- pac_input %>%
  left_join(
    pac_industry_breakdown %>% select(pac_id, industry, pct_of_total),
    by = c("pac_id_key" = "pac_id"),
    relationship = "many-to-many"  # expected: one PAC contribution explodes into N industry rows
  ) %>%
  mutate(
    industry = coalesce(industry, UNCATEGORIZED_CODE),
    pct_of_total = coalesce(pct_of_total, 0),
    split_amount = Amount * pct_of_total
  )

# top up each original contribution with an uncategorized remainder row so
# split rows sum back to the original amount
pac_split_remainder <- pac_split_rows %>%
  group_by(contribution_id, Amount) %>%
  summarise(allocated = sum(split_amount, na.rm = TRUE), .groups = "drop") %>%
  mutate(remainder = Amount - allocated) %>%
  filter(remainder > 0.01) %>%
  transmute(contribution_id, industry = UNCATEGORIZED_CODE, split_amount = remainder)

pac_split_final <- bind_rows(
  pac_split_rows %>% select(contribution_id, race_prop, Recipient.Name, Contributor.Name, industry, split_amount),
  pac_split_remainder %>%
    left_join(pac_input %>% select(contribution_id, race_prop, Recipient.Name, Contributor.Name), by = "contribution_id")
) %>%
  left_join(labels, by = c("industry" = "sector")) %>%
  transmute(
    race_prop, candidate = coalesce(Recipient.Name, race_prop), Amount = split_amount, Contributor.Name,
    code_final = industry, code_final_description = sector_description
  )

cat(sprintf("PAC-to-race rows exploded: %d original contributions -> %d industry-split rows ($%.0f)\n",
            nrow(pac_input), nrow(pac_split_final), sum(pac_split_final$Amount, na.rm = TRUE)))

# assemble data
# base, enriched with direct-contributor codes, MINUS the pac_input
# contribution_ids (replaced by the exploded/uncategorized pac rows above --
# otherwise that money would be double-counted, once as the PAC's own lump
# transaction and again as its industry split).

base_enriched <- base %>%
  left_join(direct_lookup, by = "contribution_id") %>%
  filter(!contribution_id %in% pac_input$contribution_id) %>%
  transmute(
    race_prop, candidate = coalesce(`Recipient Name`, race_prop), Amount,
    `Contributor.Name` = `Contributor Name`,
    code_final = code, code_final_description = code_label
  )

race_prop_breakdown_input <- bind_rows(base_enriched, pac_split_final)

cat(sprintf("\nFinal input: %d rows, $%.0f (base universe was $%.0f)\n",
            nrow(race_prop_breakdown_input), sum(race_prop_breakdown_input$Amount, na.rm = TRUE),
            sum(base$Amount, na.rm = TRUE)))

dir.create("10_outputs", showWarnings = FALSE)
write_csv(race_prop_breakdown_input, OUT_PATH)
cat(sprintf("Wrote %s\n", OUT_PATH))
