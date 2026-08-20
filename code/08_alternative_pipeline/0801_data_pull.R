
# Summary: 
# Run FIRST
# Loads contribution records from the Datasette database
# Standardizes contribution records including normalization of name, employer, etc.
# Flags whether the contributor is an org or an individual, which race/prop/committee the contribution went to, etc.


# load packages
library(httr)
library(jsonlite)
library(dplyr)
library(stringr)

source("standardization_helpers.R")

# anchor path
library(here)
here::i_am("0801_data_pull.R")


# pull most recent version of contributors data
all_chunks <- list()
next_url <- "https://calmatters-powersearch-2026.fly.dev/powersearch/contributions.json?_size=1000"

while (!is.null(next_url)) {
  resp <- GET(next_url)
  page <- fromJSON(content(resp, "text", encoding = "UTF-8"), simplifyVector = TRUE)
  
  if (is.null(page$rows) || length(page$rows) == 0) break
  
  df <- as.data.frame(page$rows, stringsAsFactors = FALSE)
  names(df) <- page$columns
  all_chunks <- append(all_chunks, list(df))
  
  total <- sum(sapply(all_chunks, nrow))
  if (total %% 50000 == 0) cat("Rows so far:", total, "\n")
  
  next_url <- if (!is.null(page$next_url) && nchar(page$next_url) > 0) page$next_url else NULL
}

contributions_full <- bind_rows(all_chunks)
cat("Total rows:", nrow(contributions_full), "\n")

write.csv(contributions_full,"08_inputs/power_search_contributions_raw.csv",row.names = FALSE)

# Pull IE data
url_ie <- "https://calmatters-powersearch-2026.fly.dev/powersearch/ie.csv?_stream=on&_size=max"
ie_full <- read.csv(url_ie)
write.csv(ie_full,"08_inputs/power_search_ie_raw.csv",row.names = FALSE)


### apply processing functions to data

contributions_full <- contributions_full %>%
  #filter(`Contributor Name` != "Unitemized Contributions") %>%
  mutate(
    standardized_name = standardize_names(`Contributor Name`) %>% fix_typos(),
    processed_name = standardized_name %>%
      remove_pac_info() %>% remove_unit_info() %>%
      str_remove_all("\\s*\\(.*?\\)") %>% replace_business_text() %>% str_squish(),

    standardized_employer_name = standardize_names(`Contributor Employer`) %>% fix_typos(),
    processed_employer_name = standardized_employer_name %>% standardize_occupation_employer() %>%
      remove_pac_info() %>% remove_unit_info() %>% str_remove_all("\\s*\\(.*?\\)") %>%
      replace_business_text() %>%
      str_squish(),

    standardized_occupation = standardize_names(`Contributor Occupation`),
    processed_occupation    = standardize_occupation_employer(standardized_occupation),
    standardized_city       = standardize_names(`Contributor City`),
    zip_code_processed      = substr(`Contributor Zip Code`, 1, 5),

    has_pac_language = has_pac_language(`Contributor Name`),
    entity_type      = if_else(sapply(standardized_name, is_individual), "individual", "organization"),
    race_prop = case_when(
      `Ballot Measure Contribution` == "Y" ~ `Ballot Measure(s)`,
      !is.na(Office) & Office != ""        ~ Office,
      TRUE                                 ~ `Recipient Name`
    ),
    row_id           = row_number(),
    contribution_id  = make_row_hash(`Contributor Name`, `Contributor ID`, Amount, race_prop,
                                     `Contributor City`, `Contributor Employer`,
                                     `Contributor Occupation`, `Contributor Zip Code`, `Start Date`)
  )


write.csv(contributions_full,"08_inputs/power_search_contributions_normalized.csv",row.names = FALSE)
