# compute_breakdown.R
#
# Industry/category breakdown of contributions by race/proposition 
# computed against the TOTAL contributions the candidate/prop received
#
# Input: the manually-reviewed classification file.
#
# Categories:
#   - "Small Dollar"  -- unitemized contributions, or any single contribution <= $100
#   - "Uncategorized" -- Contributions >$100 and <$5000 and $5000+ contributors that couldn't be classified
#   - custom NAICS categories -- the contributor's code_final_description (falls
#

library(dplyr)
library(readr)
library(stringr)

# config

INPUT_PATH  <- "manually_reviewed_data.csv"
OUTPUT_PATH <- "10_outputs/industry_breakdown_by_race.csv"

SMALL_DOLLAR_THRESHOLD <- 100     
UNITEMIZED_PATTERN     <- "UNITEMIZED CONTRIBUTIONS" 

# update column names here depending on upstream processing 
COLS <- c(
  race_prop        = "race_prop",
  amount           = "Amount",
  contributor_name = "Contributor.Name",
  code             = "code_final",
  code_label       = "code_final_description"
)


# helpers

require_cols <- function(df, required_actual_names, path) {
  missing <- setdiff(required_actual_names, names(df))
  if (length(missing) > 0) {
    stop(sprintf(
      paste0(
        "%s is missing expected column(s): %s\n",
        "Available columns: %s\n",
        "If the manually-reviewed file uses different column names, update ",
        "COLS at the top of this script to match."
      ),
      path, paste(missing, collapse = ", "), paste(names(df), collapse = ", ")
    ))
  }
}

apply_colmap <- function(df, colmap) {
  for (logical_name in names(colmap)) {
    actual_name <- colmap[[logical_name]]
    if (actual_name %in% names(df)) {
      names(df)[names(df) == actual_name] <- logical_name
    } else {
      df[[logical_name]] <- NA
    }
  }
  df[, names(colmap), drop = FALSE]
}


# load data

raw <- read_csv(INPUT_PATH, col_types = cols(.default = "c"))
require_cols(raw, unname(COLS[c("race_prop", "amount")]), INPUT_PATH)
x <- apply_colmap(raw, COLS)

x <- x %>% mutate(amount = as.numeric(amount))
n_bad_amount <- sum(is.na(x$amount))
if (n_bad_amount > 0) {
  warning(sprintf("%d row(s) have a non-numeric/blank Amount and are dropped from all totals.", n_bad_amount))
}
x <- x %>% filter(!is.na(amount))


n_before <- nrow(x)
x <- x %>% filter(!is.na(race_prop) & race_prop != "")
message(sprintf("Excluded row(s) (no race_prop); %d remain.", n_before - nrow(x), nrow(x)))


# put each contributor into initial bucket

x <- x %>%
  mutate(
    is_unitemized   = str_detect(str_to_upper(coalesce(contributor_name, "")), UNITEMIZED_PATTERN),
    is_small_dollar = is_unitemized | (amount <= SMALL_DOLLAR_THRESHOLD),
    category = case_when(
      is_small_dollar                    ~ "Small Dollar",
      !is.na(code) & code != ""          ~ coalesce(code_label, code),
      TRUE                                ~ "Uncategorized"
    )
  )


# breakdown by race/prop as a % of TOTAL contributions received

breakdown <- x %>%
  group_by(race_prop) %>%
  mutate(race_total = sum(amount)) %>%
  group_by(race_prop, category, code) %>%
  summarise(
    total_amount    = sum(amount),
    n_contributions = n(),
    race_total      = first(race_total),
    .groups = "drop"
  ) %>%
  mutate(pct_of_total = total_amount / race_total * 100) %>%
  arrange(race_prop, desc(total_amount))

# check that percentages within each race_prop sum to 100%
breakdown %>%
  group_by(race_prop) %>%
  summarise(total_pct = sum(pct_of_total), .groups = "drop") %>%
  filter(abs(total_pct - 100) > 0.01)


# write to output file

dir.create(dirname(OUTPUT_PATH), showWarnings = FALSE, recursive = TRUE)
write_csv(breakdown, OUTPUT_PATH)
message(sprintf("Wrote %d row(s) to %s", nrow(breakdown), OUTPUT_PATH))
