# compute_breakdown.R
#
# Industry/category breakdown of contributions, computed against the TOTAL
# contributions each CANDIDATE or prop received. 
#
# candidate_total is every dollar that candidate/prop received folds uncoded money
# into "99" 
#
# Input: the merged race/prop contribution file (see
# build_race_prop_breakdown_input.R).

library(dplyr)
library(readr)

# config

# built by build_race_prop_breakdown_input.R -- combines direct $5k+ givers
# (final_classifications.csv) with PAC-to-race giving split by each PAC's own
# industry breakdown (pac_industry_breakdown.csv)
INPUT_PATH  <- "10_outputs/race_prop_breakdown_input.csv"
OUTPUT_PATH <- "10_outputs/industry_breakdown_by_race.csv"

UNCATEGORIZED_CODE <- "99"

# update column names here depending on upstream processing
COLS <- c(
  race_prop  = "race_prop",
  candidate  = "candidate",
  amount     = "Amount",
  code       = "code_final",
  code_label = "code_final_description"
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
x <- x %>% filter(!is.na(candidate) & candidate != "")
message(sprintf("Excluded %d row(s) (no candidate); %d remain.", n_before - nrow(x), nrow(x)))


# fold anything without a real code into "99" 

x <- x %>%
  mutate(
    code       = if_else(is.na(code) | code == "", UNCATEGORIZED_CODE, code),
    code_label = coalesce(code_label, code)
  )


# breakdown by candidate as a % of TOTAL contributions received -- one row
# per candidate + code

breakdown <- x %>%
  group_by(candidate) %>%
  mutate(candidate_total = sum(amount)) %>%
  group_by(race_prop, candidate, code) %>%
  summarise(
    code_label      = first(code_label),
    total_amount    = sum(amount),
    n_contributions = n(),
    candidate_total = first(candidate_total),
    .groups = "drop"
  ) %>%
  mutate(pct_of_total = total_amount / candidate_total * 100) %>%
  arrange(candidate, desc(total_amount))

# check that percentages within each candidate sum to 100%
breakdown %>%
  group_by(candidate) %>%
  summarise(total_pct = sum(pct_of_total), .groups = "drop") %>%
  filter(abs(total_pct - 100) > 0.01)


# write to output file

dir.create(dirname(OUTPUT_PATH), showWarnings = FALSE, recursive = TRUE)
write_csv(breakdown, OUTPUT_PATH)
message(sprintf("Wrote %d row(s) to %s", nrow(breakdown), OUTPUT_PATH))
