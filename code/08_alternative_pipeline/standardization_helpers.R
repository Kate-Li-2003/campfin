

# DEFINE FUNCTIONS TO STANDARDIZE NAMES

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
    str_replace_all(" \\& ", " AND ") %>% # this may make some names weird e.g. M&D Inc -> MANDD Inc.
    str_replace_all("\\#", "NUMBER") %>%
    str_replace_all(" NO ", " NUMBER ") %>% 
    str_replace_all("[.']", "") %>% # only removing these for now, because there's some punctuation i wanted to preserve, but could try removing all 
    str_replace_all("\\,\\,", "\\,") %>%
    str_replace_all("\\?\\=s", "\\'") %>% # checked these manually
    str_replace_all("\\?", "") %>% # could remove some spaces between words
    str_replace_all(" \\,", ",") %>%
    #str_replace_all("\\ + ", " AND ") %>%
    #str_replace_all("[-]", " ") %>% # don't want to take hyphens out of people's names for now (last names in particular)
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
    str_remove("FED\\s*ID\\s*NUMBER\\s*[C]\\d+") %>% 
    str_remove_all("\\s*\\([^)]*(ID|FPPC|FEC)[^)]*\\)") %>%
    str_remove("\\s*(ID(?:\\s*NUMBER)?|FPPC|FEC)\\s*[A-Z0-9]+$") %>%
    #str_remove("(\\s+(FED\\s+PAC|PAC))+$") %>%
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


# FUNCTION TO FLAG INDIVIDUALS VS ORGS

org_keywords <- c(
  "INC", "LLC", "CORP", "INCORPORATED", "CORPORATION", "COMPANY", "LTD", "LLP", "PLC",
  "AND CO", "PARTNERSHIP", "LTC", "MGMT", "MANAGEMENT", "SERVICES", "ENTITIES", "LP",
  "ASSOCIATION", "ASSOC", "ASSN", "AFFILIATED", "AFFILIATES",
  "PAC", "COMMITTEE", "UNION", "POLITICAL", "ACTION","FPPC",
  "LOCAL", "FUND", "GROUP", "PARTNERS",
  "ENGINEERING", "ARCHITECTS", "CONSULTING", "CONSTRUCTION", "ATTORNEYS", "SOLUTIONS", "INDUSTRIES",
  "PC", "APC", "FIRM"
) #"PA", "CO", "STATE"

is_individual <- function(name) {
  
  name_upper <- str_squish(toupper(name))
  
  # look for org keywords
  org_pattern <- paste0("\\b(", paste(org_keywords, collapse = "|"), ")\\b")
  if (str_detect(name_upper, org_pattern)) return(FALSE)
  
  # strip anything in parentheses: "(Ret)", "(PhD)", nicknames, etc.
  name_clean <- str_squish(str_remove_all(name_upper, "\\s*\\([^)]*\\)"))
  
  # collapse dotted abbreviations:
  # "M.D." -> "MD", "J.D." -> "JD", "D.D.S." -> "DDS", "Maj." -> "MAJ"
  name_clean <- str_replace_all(name_clean, "([A-Z])\\.", "\\1")
  name_clean <- str_squish(name_clean)
  
  # may need to update this -> could be pattern for law firms or for joint contributors
  after_comma <- str_trim(str_split(name_clean, ",")[[1]][2])
  if (!is.na(after_comma) && str_detect(after_comma, "\\bAND\\b")) return(TRUE) # e.g. Smith, Peggy & Mike
  
  #if (str_detect(name_clean, "AND")) return(TRUE)
  
  # normalize slash in compound last names so patterns match: "Friedli/Giono" -> "Friedli-Giono"
  name_clean <- str_replace_all(name_clean, "/", "-")
  
  suffix_title_pat <- paste(
    # standard suffixes
    "JR", "SR", "I", "II", "III", "IV",
    # academic / professional credentials
    "MD", "PHD", "ESQ", "DDS", "DO", "DR", "MR", "MRS", "MS", "HON", "EDS",
    "OD", "CPA", "DVM", "RN", "NP", "FACS", "TTEE", "JD", "MBA", "CFA",
    # military ranks
    "MAJ", "COL", "CAPT", "GEN", "LT", "LTC", "ADM", "SGT", "CPT", "CDR", "ENS", "SFC",
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
  
  # For 2+ commas: first token = last name, last token = first (+ possible title),
  # ALL middle tokens must be credentials, suffixes, military designations, or single-letter initials.
  # Handles: "Berra, D.D.S., M., Albert"  "Mitas, II, M.D., John"  "Reiter, USAF (Ret), Maj. Richard"
  if (n_commas >= 2) {
    parts        <- str_trim(str_split(name_clean, ",")[[1]])
    last_part    <- parts[1]
    first_part   <- parts[length(parts)]
    middle_parts <- parts[-c(1, length(parts))]
    
    cred_or_initial <- paste0("^(", suffix_title_pat, "|[A-Z])$")
    all_middle_ok   <- all(str_detect(middle_parts, cred_or_initial))
    
    if (all_middle_ok &&
        str_detect(last_part,  "^[A-Z'\\- ]+$") &&
        str_detect(first_part, "^[A-Z'\\-\\. ]+$")) return(TRUE)
  }
  
  return(FALSE)
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



