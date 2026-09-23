

# DEFINE FUNCTIONS TO STANDARDIZE NAMES

pac_keyword_pattern <- regex(
  paste0(
    "\\b(", paste(c(
      "PAC", "POLITICAL ACTION COMMITTEE", "POLITICAL ACTION LEAGUE", "POLITICAL FUND",
      "POLITICAL ACTION FUND", "COMMITTEE", "FPPC", "SCC", "SMALL CONTRIBUTOR",
      "INDEP EXPENDITURE COMMITTEE", "INDEPEDENT EXPENDITURE COMMITTEE", "INDEPENDENT EXPENDITURE COMMITTEE", "IE COMMITTEE",
      # federal PAC indicators
      "FED PAC", "FEC PAC","FEDERAL PAC", "FEC ID", "FEC REPORT", "FED FORM","FEDERAL POLITICAL ACTION"
    ), collapse = "|"), ")\\b",
    "|#C|ID#|COMMITTEE/FEDERAL|IDNUMBER"
  ),
  ignore_case = TRUE
)

# federal-PAC-specific patterns only (subset of pac_keyword_pattern)
fed_pac_pattern <- regex(
  paste(
    "\\b(FED PAC|FEC PAC|PAC ID|FEDERAL PAC|FEC ID|FED ID|FEC REPORT|FED FORM|FEDERAL POLITICAL ACTION)\\b",
    "#C|ID#|ID\\s*NUMBER|IDNUMBER|COMMITTEE/FEDERAL",
    "\\bC00\\d{6}\\b",   # bare FEC committee ID (e.g. C00639229)
    sep = "|"
  ),
  ignore_case = TRUE
)

has_pac_language     <- function(name) str_detect(coalesce(name, ""), pac_keyword_pattern)
has_fed_pac_language <- function(name) str_detect(coalesce(name, ""), fed_pac_pattern)

# candidate_pac_pattern: catches candidate/officeholder committee names.
# Signals: "FOR <office>", ELECT/RE-ELECT, COMMITTEE TO ELECT.
# Excluded: "FRIENDS OF" (too broad — catches non-candidate PACs), "SPONSORED BY" (IE/coalition).
# Double-FOR names (e.g. "X FOR [CANDIDATE] FOR SENATE") are IE committees, not candidate PACs.
candidate_office_terms <- paste(c(
  "GOVERNOR", "SENATE", "ASSEMBLY", "CONGRESS", "SUPERVISOR",
  "MAYOR", "TREASURER", "CONTROLLER", "COMPTROLLER",
  "ATTORNEY\\s+GENERAL", "INSURANCE\\s+COMMISSIONER",
  "SECRETARY\\s+OF\\s+STATE", "CITY\\s+COUNCIL",
  "DISTRICT\\s+ATTORNEY", "STATE\\s+SENATE", "STATE\\s+ASSEMBLY",
  "BOARD\\s+OF\\s+EQUALIZATION", "LIEUTENANT\\s+GOVERNOR",
  "LT\\s+GOVERNOR", "AUDITOR", "JUDGE",
  "SHERIFF", "ASSESSOR", "SCHOOL\\s+BOARD",
  "WATER\\s+BOARD", "PUBLIC\\s+UTILITIES", "COMMUNITY\\s+COLLEGE"
), collapse = "|")

candidate_pac_pattern <- regex(
  paste(
    paste0("\\bFOR\\s+(", candidate_office_terms, ")\\b"),
    "\\b(RE-ELECT|REELECT)\\b",
    "\\bCOMMITTEE\\s+TO\\s+(ELECT|RE-ELECT|REELECT)\\b",
    "\\bCAMPAIGN\\s+COMMITTEE\\b",
    sep = "|"
  ),
  ignore_case = TRUE
)

# IE/coalition committees name a supported candidate in the form "[group] FOR [person] FOR [office]".
# exclude these
candidate_pac_double_for_pat <- regex(
  paste0("\\bFOR\\b.+\\bFOR\\s+(", candidate_office_terms, ")\\b"),
  ignore_case = TRUE
)

is_candidate_pac <- function(name) {
  name_safe  <- coalesce(name, "")
  name_upper <- toupper(str_squish(name_safe))
  str_detect(name_safe, candidate_pac_pattern) &
    !str_detect(name_upper, "\\bSPONSORED\\s+BY\\b") &
    !str_detect(name_upper, candidate_pac_double_for_pat)
}


make_row_hash <- function(...) {
  cols <- list(...)
  n <- length(cols[[1]])
  vapply(seq_len(n), function(i) {
    parts <- vapply(cols, function(col) as.character(col[i]), character(1))
    digest::digest(paste(parts, collapse = "|"), algo = "xxhash32")
  }, character(1))
}

# mirrors `normalize_name` used across other scripts
normalize_name_simple <- function(name) {
  name %>%
    str_to_upper() %>%
    str_replace_all("[^A-Z0-9 ]+", " ") %>%
    str_squish()
}


# convert to uppercase, remove some punctuation and substitute characters with written out version
standardize_names <- function(name) {
  name %>%
    str_to_upper() %>%
    str_replace_all("[.']", "") %>% # want to preserve some punctuation (like commas) for processing names
    str_replace_all(",\\s*,+", ",") %>%
    str_replace_all("\\?\\=s", "\\'") %>% # some special characters were recorded as question marks
    str_replace_all("\\?", "") %>% # could remove some spaces between words
    str_replace_all(" \\,", ",") %>% # remove space before comma
    str_squish()
}

# fix typos found in names
fix_typos <- function(name) { 
  name %>%
    str_to_upper() %>%
    str_replace_all("ENTITITES", "ENTITIES") %>% 
    str_replace_all("ENTITES", "ENTITIES") %>%
    str_replace_all("COMMITTE ", "COMMITTEE ") %>%
    str_replace_all("CAMMITTEE", "COMMITTEE") %>%
    str_replace_all("VACINITY", "VICINITY") %>%
    str_replace_all("AND AFFILIATES ENTITIES", "AND AFFILIATED ENTITIES") %>%
    str_squish()
}


replace_business_text <- function(name){
  name %>%
    # consider adding technologies
    str_replace_all("[^[:alnum:] ()]", " ") %>% # keep parentheses
    str_replace_all("\\,", "") %>%
    str_replace_all(" LTD LLLP$", "") %>%
    str_replace_all(" AND AFFILIATED ENTITIES$", "") %>% # sometimes removing this just leaves a name, can think more about whether to remove
    str_replace_all(" AND AFFILIATED COMPANIES$", "") %>%
    str_replace_all(" AND AFFILIATED ENTITIES INC$", "") %>%
    str_replace_all(" AND AFFILIATED$", "") %>%
    str_replace_all("AND SUBSIDIARIES", "") %>%
    str_replace_all("AND ITS SUBSIDIARIES", "") %>% # could also include 'and its subsidiaries affiliates'
    str_replace_all(" ASSOC ", " ASSOCIATION ") %>% # think it's always association and not associated/associates
    str_replace_all(" LTD LLC$", "") %>%
    str_replace_all(" CORP$", "") %>% # cuts off certain phrases like 'a law corp', but can play around with it
    str_replace_all(" INC$", "") %>%
    str_replace_all(" INCORPORATED$", "") %>% # could include incorporation, corporation
    str_replace_all(" LLC$", "") %>%
    str_replace_all(" LTD$", "") %>%
    str_replace_all(" LLP$", "") %>%
    str_replace_all(" LP$", "") %>%
    str_replace_all(" PA$", "") %>%
    str_replace_all(" AND CO$", "") %>%
    str_replace_all(" CO$", "") %>%
    str_replace_all("COMPANY$", "") %>%
    str_replace_all("MGMT", "MANAGEMENT") %>%
    str_squish()
}



# replace text to do with PAC / committees 
replace_committee_text <- function(name){
  name %>% 
    str_replace_all("POLITICAL ACTION LEAGUE FOR", "") %>%
    str_replace_all("POLITICAL ACTION LEAGUE", "") %>%
    str_replace_all("POLITICAL ACTION COMMITTEE STATE PAC", "") %>%
    str_replace_all("POLITICAL ACTION COMMITTEE", "") %>%
    str_replace_all("POLITICAL ACTION CO$", "") %>%
    str_replace_all("POLITICAL ACTION LEAGUE FOR", "") %>%
    str_replace_all("POLITICAL FUND", "") %>%
    str_replace_all("POLITICAL ACTION FUND", "") %>%
    str_replace_all("PAC ALL PURPOSE", "") %>%
    str_replace_all("PAC ALL PURPOSE ACCOUNT", "") %>%
    str_replace_all("SCC", "") %>%
    str_replace_all("SMALL CONTRIBUTOR COMMITTEE", "") %>%
    str_replace_all("SMALL CONT COMMITTEE", "") %>%
    str_replace_all("SMALL COMMITTEE", "") %>%
    str_replace_all("PAC FED PAC", "") %>% # hoping to get some names that are missing a space e.g. 'CUMMINS INC PAC CIPAC FED PAC ID C00377952'
    str_replace_all(" STATE PAC$", "") %>% # this could also grab 'real estate pac' if space not included 
    str_replace_all("STATEWIDE PAC$", "") %>% 
    str_replace_all("FEDERAL PAC$", "") %>% 
    str_remove("(\\s+(FED\\s+PAC|PAC))+$") %>%
    str_replace_all("MAJOR DONOR ACCOUNT", "") %>%
    str_replace_all("MAJOR DONOR COMMITTEE", "") %>%
    str_replace_all("MAJOR DONOR", "") %>%
    str_replace_all("FPPC", "") 
}


remove_pac_info <- function(name){
  name %>%
    # Parenthesized blocks: (FEC ID C00834291), (Major Donor # 1440663), (#C00084475), etc.
    str_remove_all(
      "(?i)\\s*\\((?:(?:(?:FEDERAL|FED|FEC)(?:\\s+PAC)?|(?:CALIFORNIA\\s+)?MAJOR\\s+DONOR)(?:\\s+(?:ID|ACCOUNT))?[\\s#:]*[A-Z]?\\d*|[#]?[A-Z]\\d{6,})\\)"
    ) %>%
    # Clean up empty parens left behind: () or ( )
    str_remove_all("\\s*\\(\\s*\\)") %>%
    # Bare "ID XXXXXXX" — California Major Donor numeric ID (6–7 digits)
    str_remove("(?i),?\\s*\\bID\\s+\\d{6,7}$") %>%
    # Trailing FEC/Federal/Major Donor references with optional preceding separator
    str_remove(
      "(?i)[\\s,\\-]*\\b(?:(?:STATE\\s+(?:AND|&)\\s+)?(?:FEDERAL|FED|FEC)(?:\\s+PAC)?|(?:CALIFORNIA\\s+)?MAJOR\\s+DONOR)(?:\\s+(?:ID|ACCOUNT))?[\\s#:]*[A-Z]?\\d*$"
    ) %>%
    # Clean up trailing open paren left behind: " ("
    str_remove("\\s*\\($") %>%
    str_squish()
}

remove_unit_info <- function(name) {
  name %>%
    str_replace_all(
      regex("\\s+LOCAL(\\s+UNION)?(\\s+NO|\\s+NUMBER)?\\s*\\d+", ignore_case = TRUE),
      ""
    ) %>%
    str_replace_all(
      regex("\\s+NUMBER\\s*\\d+", ignore_case = TRUE),
      ""
    ) %>%
    str_squish()
}

standardize_occupation_employer <- function(name){ 
  name %>%
    # standardize unemployed text 
    str_replace_all("^N/A$", "NONE") %>% # be wary that these are typically orgs
    str_replace_all("^NA$", "NONE") %>%
    str_replace_all("^N A$", "NONE") %>%
    str_replace_all("^BLANK$",                   "UNKNOWN") %>%
    str_replace_all("^NONE OF YOUR BUSINESS$",   "UNKNOWN") %>%
    str_replace_all("^PREFER NOT TO DISCLOSE$",  "UNKNOWN") %>%
    str_replace_all("^NOT EMPLOYED \\(RETIRED\\)$", "RETIRED") %>%
    str_replace_all("^NONE \\(RETIRED\\)$",      "RETIRED") %>% # think this is here because there was one 'not employed (retired)' where not employed was changed above
    str_replace_all("^NOT EMPLOYED-RETIRED$",    "RETIRED") %>%
    str_replace_all("^NONE-RETIRED$",            "RETIRED") %>%
    str_replace_all("^RETIRED NONE$",            "RETIRED") %>%
    str_replace_all("^RETIRED NOT EMPLOYED",            "RETIRED") %>%
    str_replace_all("^NOT EMPLOYED$",            "NONE") %>%
    str_replace_all("^UNEMPLOYED$",              "NONE") %>%
    str_replace_all("^NO$", "NONE") %>%  # needs to be an exact match
    
    str_replace_all("^NOT EMPOYED$", "NONE") %>%
    str_replace_all("^NOT EMLOYED$", "NONE") %>%
    str_replace_all("^NOT-EMPLOYED$", "NONE") %>%
    str_replace_all("^NOT RMPLOYED$", "NONE") %>%
    str_replace_all("^NOT EMPLOYYED$", "NONE") %>%
    str_replace_all("^A, N \\/$", "NONE") %>% # this is for open secrets data
    
    
    str_replace_all("SELF-EMPLOYED", "SELF EMPLOYED") %>%
    
    # abbreviate common terms. Doesn't account for spelling errors.
    str_replace_all("CHIEF EXECUTIVE OFFICER", "CEO") %>%
    str_replace_all("CHIEF TECHNOLOGY OFFICER", "CTO") %>%
    str_replace_all("CHIEF OPERATING OFFICER", "COO") %>%
    str_replace_all("CHIEF FINANCIAL OFFICER", "CFO") %>%
    
    # misc changes
    str_replace_all("EXEC ", "EXECUTIVE ") %>%
    str_replace_all("EXEC$", "EXECUTIVE") %>%
    
    # sometimes occupation is 'Information Requested' -> looks like it wasn't filled out properly?
    str_replace_all("^INFORMATION REQUESTED$", "UNKNOWN") %>%
    str_replace_all("INFORMATION REQUESTED-?\\s*", "") %>%   # strips it as a prefix (one case was information requested-marketing)
    
    # remove white space
    str_squish()
}

# write functions to standardize city and state names
standardize_city <- function(city) {
  city %>%
    str_replace("\\s+[A-Z]{2}$", "") %>% # remove trailing state abbreviation - check for this before converting to all upper, otherwise gets stuff like Rancho Sante Fe
    str_to_upper() %>%
    str_replace(",.*$", "") %>%       # remove everything after a comma
    str_replace("\\s+\\d{5}(-\\d{4})?$", "") %>%  # remove trailing zip code
    str_replace_all("[^[:alnum:] ]", "") %>% # remove any other punctuation (some have periods in the name)
    str_squish()
}

standardize_state <- function(state) {
  state %>%
    str_to_upper() %>%
    str_replace("\\s+\\d{5}(-\\d{4})?$", "") %>%  # remove trailing zip code
    str_replace_all("[^[:alnum:] ]", "") %>% # remove any other punctuation 
    str_squish()
}


# FUNCTION TO FLAG INDIVIDUALS VS ORGS

org_keywords <- c(
  "INC", "LLC", "INCORPORATED", "CORPORATION", "COMPANY", "LTD", "LLP", "PLC", "OFFICE", "OFFICES","INTERNATIONAL", "CO\\.",
  "BUSINESS", "ORGANIZATION", "PROFESSIONAL", "INVESTMENT", "LAW OFFICE", "LAW OFFICES",
  "AND CO", "PARTNERSHIP", "LTC", "MGMT", "MANAGEMENT", "SERVICES", "ENTITIES", "LP", "AND SON", "AND SONS",
  "ASSOCIATION", "ASSOC", "ASSN", "AFFILIATED", "AFFILIATES", "ASSOCIATES",
  "PAC", "COMMITTEE", "UNION", "POLITICAL", "ACTION","FPPC", "AFL-CIO", "DEMOCRAT","REPUBLICAN", "YES ON MEASURE",
  "EXPENDITURE", "EXPENDITURES","COALITION", "GOVERNMENT", "SPONSOR", "SPONSORED","AGGREGATED","CONTRIBUTOR", "ALLIANCE",
  "INTERMEDIARY","UNITEMIZED", "LABOR",
  "LOCAL", "FUND", "GROUP", "PARTNERS","ALC$", 
  "ENGINEERING", "ARCHITECTS", "CONSULTING", "CONSTRUCTION", "ATTORNEYS", "SOLUTIONS", "INDUSTRIES", "INDUSTRY", "MANUFACTURING", "ACCOUNTANCY",
  "INSTITUTE", "CENTER$", "CHAPTER","FEDERATION", "EMPLOYEE", "INDUSTRY", "CONFERENCE", "CITY COUNCIL", "TRADES COUNIL","CHAMBER", "FOR CONGRESS", "CONGRESS$",
  "JOINT VENTURE", "JV$", "A JV",
  "PROPERTIES", "PROPERTY","REALTY", "INSURANCE","BUILDINGS","APARTMENT", "RESIDENCES",
  "RESORT","HOTEL","MOTEL", "VISIT",'INN AND SPA', "INN AT",
  "PC", "APC", "FIRM",
  "RANCH", "RANCHES", "FARM$", "FARMS", "DAIRY", "DAIRIES", "DRILLING","VINEYARDS","COATINGS",
  "PEDIATRICS", "OPTOMETRY", "OPTOMETRIST","OPTOMETRISTS" , "MEDICAL", "LABORATORY","MEDICAL", "NEUROLOGY", "ACUPUNCTURE", "DISEASE",
  "PAWN SHOP", "JEWELRY", 
  ", THE$", " CORP$", " DBA ",
  "CALIFORNIA", "NAPA VALLEY", "SUGARLAND",
  "CENTENE", "HEALTHNET", "RITZ-CARLTON","T-MOBILE", "TRUCK CENTER"
) #"PA", "CO", "STATE", "CORP","LOAN","LOS ANGELES", "SACRAMENTO",

# shared preprocessing for is_individual():
# uppercase, strip data-entry junk characters, resolve org keywords, strip parens,
# collapse dotted abbreviations, re-check org keywords, normalize "+" to " AND "
prep_name_for_classification <- function(name) {
  name_upper <- str_squish(toupper(name))
  # strip characters that cannot appear in valid names (data-entry typos:
  # backticks, semicolons, etc.)
  name_upper <- str_replace_all(name_upper, "[`@#$%\\^*_=\\[\\]{}|<>;]", "")
  name_upper <- str_squish(name_upper)

  org_pattern <- paste0("\\b(", paste(org_keywords, collapse = "|"), ")\\b")
  if (str_detect(name_upper, org_pattern)) return(list(name_clean = name_upper, is_org_keyword = TRUE))

  # strip anything in parentheses: "(Ret)", "(PhD)", nicknames, etc.
  name_clean <- str_squish(str_remove_all(name_upper, "\\s*\\([^)]*\\)"))

  # collapse dotted abbreviations:
  # "M.D." -> "MD", "J.D." -> "JD", "D.D.S." -> "DDS", "Maj." -> "MAJ"
  name_clean <- str_replace_all(name_clean, "([A-Z])\\.", "\\1")
  name_clean <- str_squish(name_clean)

  if (str_detect(name_clean, org_pattern)) return(list(name_clean = name_clean, is_org_keyword = TRUE))

  # normalize "+" to " AND " for joint contributors (e.g. "REINHART, CHRIS+SUZY")
  name_clean <- str_replace_all(name_clean, "\\+", " AND ")
  name_clean <- str_squish(name_clean)

  list(name_clean = name_clean, is_org_keyword = FALSE)
}

# Employer/Occupation are typically blank/uninformative for orgs but filled in for
# individuals - used below as a tiebreaker for the one pattern that's structurally
# identical between joint individual contributors and multi-partner law firms.
# Checked against RAW employer/occupation (before standardize_occupation_employer
# collapses "NOT EMPLOYED"/"UNEMPLOYED"/"N/A" into "NONE"), so an actually-filled-in
# "not employed" / "retired" response for an individual doesn't get mistaken for a
# blank org field.
blank_occ_emp_values <- c("", "N/A", "NA", "N A", ".")
is_blank_or_uninformative <- function(x) {
  toupper(str_squish(coalesce(x, ""))) %in% blank_occ_emp_values
}

# stricter than is_blank_or_uninformative(): only literal emptiness, excludes "N/A"
is_literally_empty <- function(x) {
  is.na(x) | str_squish(coalesce(x, "")) == ""
}

# TRUE when the final entity_type classification looks inconsistent with the raw
# Employer/Occupation fields: an org with real employer/occupation info, or an
# individual with both fields completely empty. Meant to flag rows for optional
# manual review without blocking the automatic classification.
entity_type_looks_ambiguous <- function(entity_type, employer, occupation) {
  org_with_info   <- entity_type == "organization" &
    (!is_blank_or_uninformative(employer) | !is_blank_or_uninformative(occupation))
  indiv_all_blank <- entity_type == "individual" &
    is_literally_empty(employer) & is_literally_empty(occupation)
  org_with_info | indiv_all_blank
}

is_individual <- function(name, employer = NA_character_, occupation = NA_character_) {

  prep <- prep_name_for_classification(name)
  if (prep$is_org_keyword) return(FALSE)
  name_clean <- prep$name_clean

  # "LAST, FIRST AND FIRST2" - ambiguous between joint individual contributors
  # (e.g. "Smith, Peggy AND Mike") and multi-partner law firms (e.g. "Koszdin, Fields
  # AND Sherry"). When employer/occupation are supplied, use them as a tiebreaker:
  # both blank/uninformative -> org; otherwise -> individual. Without them, default
  # to individual (prior behavior, kept for callers that don't pass this info).
  after_comma <- str_trim(str_split(name_clean, ",")[[1]][2])
  if (!is.na(after_comma) && str_detect(after_comma, "\\bAND\\b")) {
    if (!is.na(employer) || !is.na(occupation)) {
      return(!(is_blank_or_uninformative(employer) && is_blank_or_uninformative(occupation)))
    }
    return(TRUE)
  }

  # normalize slash in compound last names so patterns match: "Friedli/Giono" -> "Friedli-Giono"
  name_clean <- str_replace_all(name_clean, "/", "-")
  
  suffix_title_pat <- paste(
    # standard suffixes
    "JR", "SR", "I", "II", "III", "IV",
    "MR", "MRS", "MS", 
    # academic / professional credentials
    "MD", "PHD", "ESQ", "DDS", "DO", "DR", "MPH",
    "HON", "EDS",
    "OD", "CPA", "DVM", "RN", "NP", "FACS", "TTEE", "JD", "MBA", "CFA","CFP",
    "ND", "CRNA", "NMD", "DMD", "DC",
    "LMFT","HN-BC","CMT",
    # military ranks (abbreviated and spelled-out)
    "MAJ", "COL", "CAPT", "GEN", "LT", "LTC", "ADM", "SGT", "CPT", "CDR", "ENS", "SFC",
    "COMMANDER",
    # military branches / status
    "USAF", "USA", "USN", "USMC", "USCG", "RET",
    sep = "|"
  )
  
  # LAST, FIRST [MIDDLE] [SUFFIX at end]
  pattern_standard <- paste0(
    "^[A-Z'\\-]+(\\s+[A-Z'\\-]+)*",
    ",\\s*",
    "[A-Z'\\-]+",
    "(\\s+[A-Z'\\-\\.]+)*",
    "(\\s+(", suffix_title_pat, "))?$"
  )
  
  # LAST, SUFFIX/TITLE, FIRST [MIDDLE]  e.g. FIORE, JR, MAURO  or  HILL, JD, DR DONALD
  pattern_suffix_middle <- paste0(
    "^[A-Z'\\-]+(\\s+[A-Z'\\-]+)*",
    ",\\s*(", suffix_title_pat, ")",
    ",\\s*",
    "[A-Z'\\-]+",
    "(\\s+[A-Z'\\-\\.]+)*$"
  )
  
  if (str_detect(name_clean, pattern_standard))       return(TRUE)
  if (str_detect(name_clean, pattern_suffix_middle))  return(TRUE)
  
  n_commas <- str_count(name_clean, ",")
  
  if (n_commas == 1) {
    parts <- str_trim(str_split(name_clean, ",")[[1]])
    if (str_detect(parts[1], "^[A-Z'\\- ]+$") &&
        str_detect(parts[2], "^[A-Z'\\-\\. ]+$")) return(TRUE)
  }
  
  # For 2+ commas: handles several formats:
  #   "LAST, CRED, FIRST"       e.g. "Berra, D.D.S., Albert"  "Reiter, USAF RET, Richard"
  #   "LAST, FIRST, SUFFIX"     e.g. "Alvarez, Israel, Jr."
  #   "LAST, FIRST, MIDDLE"     e.g. "Martinez, Javier, Jose"
  # Empty comma segments (data-entry artifacts like "LIN, ,, SOPHIA") are dropped first.
  if (n_commas >= 2) {
    parts <- str_trim(str_split(name_clean, ",")[[1]])
    parts <- parts[nchar(str_squish(parts)) > 0]  # drop blank segments

    cred_pat <- paste0("^(", suffix_title_pat, "|[A-Z])$")

    # each comma-segment may itself be multi-word (e.g. "USMC RET", "O D")
    cred_tokens_ok <- function(seg) {
      toks <- str_split(str_squish(seg), "\\s+")[[1]]
      toks <- toks[nchar(toks) > 0]
      if (length(toks) == 0) return(TRUE)
      all(str_detect(toks, cred_pat))
    }

    # if empty-segment filtering collapsed to two parts, use n_commas==1 logic
    if (length(parts) == 2) {
      if (str_detect(parts[1], "^[A-Z'\\- ]+$") &&
          str_detect(parts[2], "^[A-Z'\\-\\. ]+$")) return(TRUE)
    }

    if (length(parts) >= 3) {
      lp  <- parts[1]
      fp  <- parts[length(parts)]
      mps <- parts[-c(1, length(parts))]

      lp_ok <- str_detect(lp, "^[A-Z'\\- ]+$")
      fp_ok <- str_detect(fp, "^[A-Z'\\-\\. ]+$")

      # Check A: all middle segments are credentials (allows multi-word: "USMC RET", "O D")
      if (lp_ok && fp_ok && all(vapply(mps, cred_tokens_ok, logical(1)))) return(TRUE)

      # Check B: credential/suffix at the END instead of the middle
      # e.g. "ALVAREZ, ISRAEL, JR" — last segment is the suffix
      if (lp_ok && str_detect(fp, cred_pat) &&
          all(str_detect(mps, "^[A-Z'\\-\\. ]+$"))) return(TRUE)

      # Check C: all segments are plain name words (org keywords already filtered above).
      # Guards: (a) ≤ 3 total parts so multi-partner law firms are excluded;
      #         (b) no segment contains "AND" so "&"-converted firm names are excluded
      #             (e.g. "Fitzgerald, Alvarez, AND Ciummo" after standardization);
      #         (c) last-name segment ≤ 2 words so long org openers like
      #             "DRIVE - DEMOCRAT, REPUBLICAN, ..." are excluded.
      all_segs   <- c(lp, mps, fp)
      no_and     <- !any(str_detect(all_segs, "\\bAND\\b"))
      lp_words   <- length(str_split(str_squish(lp), "\\s+")[[1]])
      seg_words  <- vapply(str_split(all_segs, "\\s+"), length, integer(1))
      if (length(parts) <= 3 && lp_ok && fp_ok && no_and && lp_words <= 2 &&
          all(str_detect(all_segs, "^[A-Z'\\-\\. ]+$")) &&
          max(seg_words) <= 3) return(TRUE)
    }
  }
  
  return(FALSE)
}


# Apply contributor ID corrections and federal PAC flags from a review sheet.
# Creates effective_Contributor.ID (corrected value, or original when no correction applies)
# and adds/updates is_fed_pac
#
# corrections columns: Contributor.Name, Contributor.ID (original; "" = missing/NA in data),
#                      corrected_id, is_fed_pac (logical), resolved (logical)
#
# ID correction: only rows with a non-blank corrected_id are applied.
#   Sheet Contributor.ID == "" matches data rows where Contributor.ID is NA/"".
# is_fed_pac: applied only when resolved == TRUE; matched on name only.
# Name matching uses toupper(str_squish()) throughout.
apply_contributor_corrections <- function(df, corrections) {
  df$Contributor.ID           <- as.character(df$Contributor.ID)
  df$effective_Contributor.ID <- df$Contributor.ID
  df$name_key                 <- toupper(str_squish(df$Contributor.Name))

  corrections <- corrections %>%
    mutate(name_key = toupper(str_squish(Contributor.Name)))

  # ── Part 1: apply corrected IDs into effective_Contributor.ID ──────────────
  id_rows <- corrections %>%
    filter(!is.na(corrected_id) & str_trim(as.character(corrected_id)) != "")

  for (i in seq_len(nrow(id_rows))) {
    row      <- id_rows[i, ]
    sheet_id <- str_trim(as.character(coalesce(row$Contributor.ID, "")))
    name_match <- df$name_key == row$name_key

    if (sheet_id == "") {
      id_match <- is.na(df$Contributor.ID) | str_trim(df$Contributor.ID) == ""
    } else {
      id_match <- !is.na(df$Contributor.ID) & str_trim(df$Contributor.ID) == sheet_id
    }

    idx <- which(name_match & id_match)
    if (length(idx) > 0) {
      df$effective_Contributor.ID[idx] <- str_trim(as.character(row$corrected_id))
    }
  }

  # ── Part 2: flag federal PACs ───────────────────────────────────────────────
  fed_pac_name_keys <- corrections %>%
    filter(resolved, is_fed_pac) %>%
    pull(name_key) %>%
    unique()

  if (!"is_fed_pac" %in% names(df)) df$is_fed_pac <- FALSE

  if (length(fed_pac_name_keys) > 0) {
    df$is_fed_pac <- df$is_fed_pac | (df$name_key %in% fed_pac_name_keys)
  }

  df %>% select(-name_key)
}


parse_names_df <- function(name) {
  
  suffixes <- c("JR", "SR", "I","II", "III", "IV", "MD", "PHD", "DDS", "DO", "DVM",
                "ESQ", "CPA", "RN", "NP", "JD", "MBA", "CFA", "OD", "FACS", "TTEE", "EDS",
                "USAF", "USN", "USMC", "USCG", "RET", "RETD")
  prefixes <- c("DR", "MR", "MRS", "MS",
                "MAJ", "COL", "CAPT", "GEN", "LT", "LTC", "ADM", "SGT", "CPT", "CDR")
  
  parse_one <- function(nm) {
    result <- list(last = NA_character_, first = NA_character_,
                   middle = NA_character_, suffix = NA_character_,
                   title = NA_character_)
    
    if (is.na(nm) || str_squish(nm) == "") return(result)
    
    # strip parentheses before splitting: "(RET)" in "REITER, USAF (RET), MAJ RICHARD"
    nm <- str_squish(str_remove_all(nm, "\\s*\\([^)]*\\)"))
    
    parts <- str_trim(str_split(nm, ",")[[1]])
    parts <- parts[parts != ""]
    
    last_tokens <- str_split(str_squish(parts[1]), "\\s+")[[1]]
    if (length(last_tokens) > 1 && tail(last_tokens, 1) %in% suffixes) {
      result$suffix <- tail(last_tokens, 1)
      result$last   <- paste(head(last_tokens, -1), collapse = " ")
    } else {
      result$last <- parts[1]
    }
    
    if (length(parts) == 1) return(result)
    
    remaining <- parts[-1]
    
    is_suffix_token <- vapply(remaining, function(p) {
      all(str_split(str_squish(p), "\\s+")[[1]] %in% suffixes)
    }, logical(1))
    
    if (any(is_suffix_token)) {
      extra_suffix <- paste(remaining[is_suffix_token], collapse = " ")
      result$suffix <- if (is.na(result$suffix)) extra_suffix else paste(result$suffix, extra_suffix)
    }
    
    name_rest <- str_squish(paste(remaining[!is_suffix_token], collapse = " "))
    if (name_rest == "") return(result)
    
    # joint contributor: treat "PEGGY AND MIKE" as a single first-name field
    if (str_detect(name_rest, "\\bAND\\b")) {
      result$first <- name_rest
      return(result)
    }
    
    name_parts <- str_split(name_rest, "\\s+")[[1]]
    
    while (length(name_parts) > 1 && tail(name_parts, 1) %in% suffixes) {
      trailing <- tail(name_parts, 1)
      result$suffix <- if (is.na(result$suffix)) trailing else paste(result$suffix, trailing)
      name_parts <- head(name_parts, -1)
    }
    
    if (length(name_parts) > 1 && name_parts[1] %in% prefixes) {
      result$title <- name_parts[1]
      name_parts   <- name_parts[-1]
    }
    
    result$first  <- name_parts[1]
    result$middle <- if (length(name_parts) > 1) paste(name_parts[-1], collapse = " ") else NA_character_
    result
  }
  
  parsed <- lapply(name, parse_one)
  data.frame(
    last   = vapply(parsed, `[[`, character(1), "last"),
    first  = vapply(parsed, `[[`, character(1), "first"),
    middle = vapply(parsed, `[[`, character(1), "middle"),
    suffix = vapply(parsed, `[[`, character(1), "suffix"),
    title  = vapply(parsed, `[[`, character(1), "title"),
    stringsAsFactors = FALSE
  )
}



