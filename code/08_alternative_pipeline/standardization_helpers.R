

# DEFINE FUNCTIONS TO STANDARDIZE NAMES

pac_keyword_pattern <- regex(
  paste0("\\b(", paste(c(
    "PAC", "POLITICAL ACTION COMMITTEE", "POLITICAL ACTION LEAGUE", "POLITICAL FUND",
    "POLITICAL ACTION FUND", "COMMITTEE", "FPPC", "SCC", "SMALL CONTRIBUTOR","INDEP EXPENDITURE","INDEPEDENT EXPENDITURE"
    #"SMALL CONTRIBUTOR COMMITTEE", "SMALL CONT COMMITTEE"
  ), collapse = "|"), ")\\b"),
  ignore_case = TRUE
)

has_pac_language <- function(name) str_detect(toupper(coalesce(name, "")), pac_keyword_pattern)


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
    str_replace_all(" \\& ", " AND ") %>% # this may make some names weird e.g. M&D Inc -> MANDD Inc.
    str_replace_all("\\#", "NUMBER") %>%
    str_replace_all(" NO ", " NUMBER ") %>% 
    str_replace_all("[.']", "") %>% # only removing these for now, because there's some punctuation i wanted to preserve, but could try removing all 
    str_replace_all(",\\s*,+", ",") %>%
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

  # strip characters that cannot appear in valid names (data-entry typos: backticks, etc.)
  name_upper <- str_replace_all(name_upper, "[`@#$%\\^*_=\\[\\]{}|<>]", "")
  name_upper <- str_squish(name_upper)

  # look for org keywords
  org_pattern <- paste0("\\b(", paste(org_keywords, collapse = "|"), ")\\b")
  if (str_detect(name_upper, org_pattern)) return(FALSE)
  
  # strip anything in parentheses: "(Ret)", "(PhD)", nicknames, etc.
  name_clean <- str_squish(str_remove_all(name_upper, "\\s*\\([^)]*\\)"))
  
  # collapse dotted abbreviations:
  # "M.D." -> "MD", "J.D." -> "JD", "D.D.S." -> "DDS", "Maj." -> "MAJ"
  name_clean <- str_replace_all(name_clean, "([A-Z])\\.", "\\1")
  name_clean <- str_squish(name_clean)

  # re-check org keywords
  if (str_detect(name_clean, org_pattern)) return(FALSE)

  # normalize "+" to " AND " for joint contributors (e.g. "REINHART, CHRIS+SUZY")
  name_clean <- str_replace_all(name_clean, "\\+", " AND ")
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
    "ND", "CRNA", "NMD", "DMD", "DC",
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



