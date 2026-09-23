# build_race_prop_breakdown_input.R
#
# Builds compute_breakdown.R's input: every contribution to the races/props
# we're tracking, with a NAICS code attached wherever we have one, plus a
# `candidate` column (Recipient Name, falling back to race_prop for ballot
# measures which have no candidate) -- compute_breakdown.R breaks the
#   - $5k+ direct givers to a race/prop -> final_classifications.csv (via
#     classification_input.csv's uuid to contribution_id bridge)
#   - PACs own direct giving to a race/prop (pac_input.csv) -> split across
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
today <- format(Sys.Date(), "%m-%d-%y")

#POWER_SEARCH_PATH   <- "../08_alternative_pipeline/08_inputs/power_search_contributions_normalized.csv"
POWER_SEARCH_PATH   <- "../08_alternative_pipeline/08_inputs/power_search_contributions_normalized_08-20-26_updated_entity.csv"
CLASSIFICATION_INPUT_PATH <- "../08_alternative_pipeline/08_outputs/classification_input_combined.csv"
#FINAL_CLASSIFICATIONS_PATH <- "10_outputs/final_classifications_before_review.csv"
FINAL_CLASSIFICATIONS_PATH <- "10_outputs/final_classifications_08-23-25.csv"
PAC_INPUT_PATH       <- "../08_alternative_pipeline/08_outputs/pac_input.csv"
PAC_INDUSTRY_BREAKDOWN_PATH <- "10_outputs/pac_industry_breakdown.csv"
LABEL_URL <- "https://docs.google.com/spreadsheets/d/11QHvNJsdtMlc1YKo_iNvMB_Jfn5Ui-iYdlWhFYjSm9g/export?format=csv"

OUT_PATH <- paste0("10_outputs/race_prop_breakdown_input_", today, ".csv")

UNCATEGORIZED_CODE <- "99"

# Medium-tier codes for contributors not in the $5k+ classification pipeline.
#INDIV_MEDIUM_CODE  <- "indiv_medium"
#INDIV_MEDIUM_LABEL <- "Individual Donors (Mid-Level)"
#ORG_MEDIUM_CODE    <- "org_medium"
#ORG_MEDIUM_LABEL   <- "Organizational Donors (Mid-Level)"

# Assigns medium-tier codes to contributions with no classification (code_final is NA).
# Uses entity_type from the power search data; individuals get INDIV_MEDIUM, all
# others (organizations, unknown) get ORG_MEDIUM. PAC contributions are already
# excluded from base_enriched before this runs.
#assign_medium_tier <- function(df) {
#  df %>% mutate(
#    .unclassified = is.na(code_final),
#    code_final = case_when(
#      !.unclassified              ~ code_final,
#      entity_type == "individual" ~ INDIV_MEDIUM_CODE,
#      TRUE                        ~ ORG_MEDIUM_CODE
#    ),
#    code_final_description = case_when(
#      !.unclassified              ~ code_final_description,
#      entity_type == "individual" ~ INDIV_MEDIUM_LABEL,
#      TRUE                        ~ ORG_MEDIUM_LABEL
#    )
#  ) %>% select(-.unclassified)
#}

# race/prop filter
race_filter <- function(df) {
  df %>% filter(
    `Recipient Name` == "BECERRA, XAVIER" |
    `Recipient Name` == "HILTON, STEVE" |
    (`Recipient Name` == "ALLEN, BEN" & Office == "Insurance Commissioner") |
    `Recipient Name` == "KIM, JANE" | 
     (Ballot.Measure != "" & Ballot.Measure != "SUPPORTED: PROPOSITION 050 - ACA 8 (RIVAS) CONGRESSIONAL REDISTRICTING. (RES. CH. 156, 2025)" & Ballot.Measure != "OPPOSED: PROPOSITION 050 - ACA 8 (RIVAS) CONGRESSIONAL REDISTRICTING. (RES. CH. 156, 2025)")
    
    #`Ballot Measure(s)` == "OPPOSED: PROPOSITION 040 - IMPOSES ONE-TIME TAX ON CERTAIN INDIVIDUALS AND TRUSTS. INITIATIVE CONSTITUTIONAL AMENDMENT AND STATUTE." |
    #`Ballot Measure(s)` == "SUPPORTED: PROPOSITION 040 - IMPOSES ONE-TIME TAX ON CERTAIN INDIVIDUALS AND TRUSTS. INITIATIVE CONSTITUTIONAL AMENDMENT AND STATUTE." |
    #`Ballot Measure(s)` == "SUPPORTED: PROPOSITION 041 - REQUIRES AUDITS OF PROGRAMS FUNDED BY NEW STATE SPECIAL TAXES. PROHIBITS NEW STATE TAXES THAT ARE EXCLUDED FROM EXISTING VOTER-APPROVED STATE SPENDING LIMIT..." |
    #Ballot Measure(s)` == "SUPPORTED: PROPOSITION 042 - PROHIBITS NEW STATE PERSONAL PROPERTY TAXES AND CERTAIN RETROACTIVE STATE TAXES. INITIATIVE CONSTITUTIONAL AMENDMENT." |
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


# Qs
# not sure slice(1) should be used here - what if there's a conflict... 
# may need to transition from using unit_id if there are conflicts
# what level is contribution_id at ? 

ci_joined <- ci_bridge %>%
  left_join(final_cls, by = c("uuid" = "unit_id"))

code_conflicts <- ci_joined %>%
  filter(!is.na(code_final)) %>%
  group_by(contribution_id) %>%
  filter(n_distinct(code_final) > 1) %>%
  ungroup()
if (nrow(code_conflicts) > 0) {
  warning(sprintf(
    "%d contribution_id(s) have conflicting real codes — slice(1) picks arbitrarily. Affected ids: %s",
    n_distinct(code_conflicts$contribution_id),
    paste(unique(code_conflicts$contribution_id), collapse = ", ")
  ))
}

direct_lookup <- ci_joined %>%
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

# all rows in pac_input should be PACs. Join key: FILER_ID when present (matches
# pac_classifiability$pac_id); fall back to effective_Contributor.ID for the
# propagated IDs from 0802 whose FILER_ID is blank
pac_input <- pac_input %>%
  mutate(pac_id_key = if_else(
    !is.na(FILER_ID) & trimws(FILER_ID) != "",
    FILER_ID,
    effective_Contributor.ID
  ))

# restrict to contribution_ids that are actually part of `base` (race_filter()'s
# scope
n_pac_input_before <- nrow(pac_input)
pac_input <- pac_input %>% filter(contribution_id %in% base$contribution_id)
cat(sprintf("pac_input.csv: dropped %d row(s) outside base's race/candidate scope\n",
            n_pac_input_before - nrow(pac_input)))

pac_input %>%
  filter(!pac_id_key %in% pac_industry_breakdown$pac_id) %>%
  select(Contributor.Name, pac_id_key, Amount) %>%
  arrange(desc(as.numeric(Amount)))

cat(sprintf("pac_input.csv: %d rows, %d with a pac_industry_breakdown entry\n",
            nrow(pac_input), sum(pac_input$pac_id_key %in% pac_industry_breakdown$pac_id)))

# stable row identifier needed because contribution_id is a hash and is NOT
# unique — 4 contribution_ids map to 2 rows each. Using .pac_row_id for the
# remainder join keeps it 1:1 and avoids double-counting.
pac_input <- pac_input %>% mutate(.pac_row_id = row_number())

# federal/candidate PACs may have no effective_Contributor.ID and no FILER_ID,
# so pac_id_key = NA and the ID-keyed join below won't find them. Fall back to
# matching by name against pac_industry_breakdown$pac_name (case-insensitive).
# Only entries with a real industry code (not "99") are included — in
# match_pac_classifications, federal/candidate PACs get a real sector code,
# while non-federal no-ID PACs get "99". This naturally restricts the name
# fallback to federal/candidate PACs; all others must be classified by their
# contributors instead.
# Also handles the case where a PAC had its ID coded during manual review:
# pac_industry_breakdown stores it under the new pac_id, but pac_name is still
# populated and the name match will find it.
no_id_pac_input <- pac_input %>%
  filter(is.na(pac_id_key) | trimws(coalesce(pac_id_key, "")) == "") %>%
  mutate(name_key = toupper(trimws(Contributor.Name)))

pac_name_breakdown <- pac_industry_breakdown %>%
  filter(!is.na(pac_name), trimws(pac_name) != "",
         industry != UNCATEGORIZED_CODE) %>%
  mutate(name_key = toupper(trimws(pac_name))) %>%
  select(name_key, industry, pct_of_total)

if (nrow(no_id_pac_input) > 0) {
  name_matched_pacs   <- no_id_pac_input %>% filter( name_key %in% pac_name_breakdown$name_key)
  name_unmatched_pacs <- no_id_pac_input %>% filter(!name_key %in% pac_name_breakdown$name_key)

  if (nrow(name_matched_pacs) > 0) {
    cat(sprintf("Name-based PAC fallback: matched %d row(s) across %d unique PAC name(s):\n",
                nrow(name_matched_pacs), n_distinct(name_matched_pacs$name_key)))
    print(name_matched_pacs %>%
      distinct(Contributor.Name, pac_id_key, Amount) %>%
      arrange(Contributor.Name))
    message("NOTE: if any of the above PACs had an ID coded during manual review, they ",
            "are classified via name match — verify the classification in pac_industry_breakdown.")
  }
  if (nrow(name_unmatched_pacs) > 0) {
    cat(sprintf("Name-based PAC fallback: %d row(s) with no pac_id_key and no federal/candidate name match — left uncategorized:\n",
                nrow(name_unmatched_pacs)))
    print(name_unmatched_pacs %>%
      distinct(Contributor.Name, pac_id_key, Amount) %>%
      arrange(Contributor.Name))
  }
}

# explode each PAC contribution into one row per industry, scaled by the
# PACs pct_of_total. Any shortfall (pct_of_total summing to <1, e.g.
# PACs not yet in pac_review) is left as an uncategorized remainder row.
pac_split_rows <- pac_input %>%
  filter(!is.na(pac_id_key) & trimws(coalesce(pac_id_key, "")) != "") %>%
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

if (nrow(no_id_pac_input) > 0) {
  # name-matched no-ID rows: join against pac_name_breakdown for their industry distribution
  if (nrow(name_matched_pacs) > 0) {
    pac_split_rows <- bind_rows(
      pac_split_rows,
      name_matched_pacs %>%
        left_join(pac_name_breakdown, by = "name_key", relationship = "many-to-many") %>%
        mutate(
          industry     = coalesce(industry, UNCATEGORIZED_CODE),
          pct_of_total = coalesce(pct_of_total, 0),
          split_amount = Amount * pct_of_total
        ) %>%
        select(-name_key)
    )
  }
  # name-unmatched no-ID rows: no federal/candidate match — classify full amount as uncategorized
  if (nrow(name_unmatched_pacs) > 0) {
    pac_split_rows <- bind_rows(
      pac_split_rows,
      name_unmatched_pacs %>%
        mutate(industry = UNCATEGORIZED_CODE, pct_of_total = 1, split_amount = Amount) %>%
        select(-name_key)
    )
  }
}

# top up each original contribution with an uncategorized remainder row so
# split rows sum back to the original amount

# Qs
# what is contribution_id for a pac? 
# what is the point of this step? wasn't it already dealth with? 
# why would this be needed? industires shuld already sum to 100% 
# contribution_id the correct thing to join on? 

pac_split_remainder <- pac_split_rows %>%
  group_by(.pac_row_id, contribution_id, Amount) %>%
  summarise(allocated = sum(split_amount, na.rm = TRUE), .groups = "drop") %>%
  mutate(remainder = Amount - allocated) %>%
  filter(remainder > 0.01) %>%
  transmute(.pac_row_id, contribution_id, industry = UNCATEGORIZED_CODE, split_amount = remainder)

pac_split_final <- bind_rows(
  pac_split_rows %>% select(contribution_id, race_prop, Recipient.Name, Contributor.Name, industry, split_amount),
  pac_split_remainder %>%
    left_join(pac_input %>% select(.pac_row_id, race_prop, Recipient.Name, Contributor.Name), by = ".pac_row_id")
) %>%
  left_join(labels, by = c("industry" = "sector")) %>%
  transmute(
    race_prop, candidate = coalesce(na_if(Recipient.Name, ""), race_prop), Amount = split_amount, Contributor.Name,
    code_final = industry, code_final_description = sector_description
  )

cat(sprintf("PAC-to-race rows exploded: %d original contributions -> %d industry-split rows ($%.0f)\n",
            nrow(pac_input), nrow(pac_split_final), sum(pac_split_final$Amount, na.rm = TRUE)))

# assemble data
# base with direct-contributor codes, NOT including the pac_input
# contribution_ids (replaced by the exploded pac rows above -
# otherwise that money would be double counted)


# race_prop fix here or no? 


base_classified <- base %>%
  left_join(direct_lookup, by = "contribution_id") %>%
  filter(!contribution_id %in% pac_input$contribution_id) %>%
  mutate(
    code_final             = code,
    code_final_description = code_label,
    candidate              = coalesce(na_if(`Recipient Name`, ""), race_prop)
  ) #%>%
  #assign_medium_tier()

base_enriched <- base_classified %>%
  transmute(
    race_prop, candidate, Amount,
    `Contributor.Name` = `Contributor Name`,
    entity_type, code_final, code_final_description
  )

race_prop_breakdown_input <- bind_rows(base_enriched, pac_split_final)

# ── Race/prop contributor audit ───────────────────────────────────────────────
# create more readable version for reviewer to look at what contributes to the
# overall breakdown
# one row per unique contributor × race/prop × industry code
# dedup keys same as assign_final_classification.Rmd's readable output:
#   individuals: name | employer | occupation | zip
#   orgs:        name | employer | occupation
# same dedup key with two different codes appears as two rows (code_conflict = TRUE).

# raw Contributor.ID -> corrected pac_id_key for traversal PACs
# used to dedup PAC contributors by their stable committee ID 
# major donors and individuals (who typically lack a matching entry here) fall through to the name-based key below


# Qs
# consequences of slice(1) here?
# pac_id_key is what? 

pac_id_lookup <- pac_input %>%
  filter(!is.na(Contributor.ID) & trimws(Contributor.ID) != "") %>%
  group_by(Contributor.ID) %>%
  slice(1) %>%
  ungroup() %>%
  select(Contributor.ID, pac_id_key)

base_enriched_audit <- base_classified %>%
  left_join(pac_id_lookup, by = c("Contributor ID" = "Contributor.ID")) %>%
  mutate(
    dedup_key = case_when(
      !is.na(pac_id_key)          ~ pac_id_key,
      entity_type == "individual" ~ paste(`Contributor Name`, processed_employer_name,
                                          processed_occupation, zip_code_processed, sep = "|||"),
      TRUE                        ~ paste(`Contributor Name`, processed_employer_name,
                                          processed_occupation, sep = "|||")
    )
  ) %>%
  group_by(race_prop, candidate, dedup_key) %>%
  mutate(code_conflict = n_distinct(code_final) > 1) %>%
  ungroup() %>%
  group_by(race_prop, candidate, dedup_key, code_final) %>%
  summarise(
    Contributor.Name       = first(`Contributor Name`),
    Contributor.ID         = first(`Contributor ID`),
    entity_type            = first(entity_type),
    code_final_description = first(code_final_description),
    contributor_city       = paste(sort(unique(na.omit(`Contributor City`))),      collapse = " | "),
    contributor_state      = paste(sort(unique(na.omit(`Contributor State`))),     collapse = " | "),
    contributor_zip        = paste(sort(unique(na.omit(zip_code_processed))),      collapse = " | "),
    contributor_employer   = paste(sort(unique(na.omit(processed_employer_name))), collapse = " | "),
    Amount                 = sum(Amount, na.rm = TRUE),
    n_contributions        = n(),
    code_conflict          = first(code_conflict),
    .groups = "drop"
  ) %>%
  mutate(
    contributor_kind = "direct",
    pac_id           = NA_character_,
    pac_pct_of_total = NA_real_
  ) %>%
  select(race_prop, candidate, Contributor.Name, Contributor.ID, contributor_kind,
         pac_id, entity_type, contributor_city, contributor_state, contributor_zip,
         contributor_employer, code_final, code_final_description, Amount,
         n_contributions, pac_pct_of_total, code_conflict)

pac_split_audit_rows <- bind_rows(
  pac_split_rows %>%
    select(contribution_id, race_prop, Recipient.Name, Contributor.Name,
           pac_id_key, industry, split_amount, pct_of_total),
  pac_split_remainder %>%
    left_join(
      pac_input %>% select(.pac_row_id, race_prop, Recipient.Name, Contributor.Name, pac_id_key),
      by = ".pac_row_id"
    ) %>%
    mutate(pct_of_total = NA_real_)
) %>%
  left_join(labels, by = c("industry" = "sector")) %>%
  mutate(candidate = coalesce(Recipient.Name, race_prop))

pac_enriched_audit <- pac_split_audit_rows %>%
  group_by(race_prop, candidate, pac_id_key, industry) %>%
  summarise(
    Contributor.Name       = first(Contributor.Name),
    code_final_description = first(sector_description),
    Amount                 = sum(split_amount, na.rm = TRUE),
    n_contributions        = n_distinct(contribution_id),
    pac_pct_of_total       = first(pct_of_total),
    .groups = "drop"
  ) %>%
  rename(pac_id = pac_id_key, code_final = industry) %>%
  mutate(
    contributor_kind = "seed_pac", Contributor.ID = NA_character_, entity_type = NA_character_,
    contributor_city = NA_character_, contributor_state = NA_character_,
    contributor_zip  = NA_character_, contributor_employer = NA_character_,
    code_conflict    = FALSE
  ) %>%
  select(race_prop, candidate, Contributor.Name, Contributor.ID, contributor_kind,
         pac_id, entity_type, contributor_city, contributor_state, contributor_zip,
         contributor_employer, code_final, code_final_description, Amount,
         n_contributions, pac_pct_of_total, code_conflict)

race_prop_contributor_audit <- bind_rows(base_enriched_audit, pac_enriched_audit) %>%
  arrange(race_prop, candidate, contributor_kind, desc(Amount))

AUDIT_PATH <- paste0("10_outputs/race_prop_contributor_audit_", today, ".csv")
write_csv(race_prop_contributor_audit, AUDIT_PATH)
cat(sprintf("Race/prop contributor audit: %d rows (%d direct, %d seed_pac) → %s\n",
            nrow(race_prop_contributor_audit),
            sum(race_prop_contributor_audit$contributor_kind == "direct"),
            sum(race_prop_contributor_audit$contributor_kind == "seed_pac"),
            AUDIT_PATH))

cat(sprintf("\nFinal input: %d rows, $%.0f (base universe was $%.0f)\n",
            nrow(race_prop_breakdown_input), sum(race_prop_breakdown_input$Amount, na.rm = TRUE),
            sum(base$Amount, na.rm = TRUE)))

dir.create("10_outputs", showWarnings = FALSE)
write_csv(race_prop_breakdown_input, OUT_PATH)
cat(sprintf("Wrote %s\n", OUT_PATH))

