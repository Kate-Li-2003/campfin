
# SIMPLE NAME STANDARDIZATION

standardize_names <- function(name) {
  name %>%
    str_to_upper() %>%
    str_replace_all(" \\& ", " AND ") %>% # this may make some names weird e.g. M&D Inc -> MANDD Inc.
    str_replace_all("\\#", "NUMBER") %>%
    str_replace_all(" NO ", " NUMBER ") %>% 
    str_replace_all("[.']", "") %>% 
    str_replace_all(",\\s*,+", ",") %>%
    str_replace_all("\\?\\=s", "\\'") %>% 
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
  
  name_upper <- str_replace_all(name_upper, "[`@#$%\\^*_=\\[\\]{}|<>]", "")
  name_upper <- str_squish(name_upper)
  
  # look for org keywords
  org_pattern <- paste0("\\b(", paste(org_keywords, collapse = "|"), ")\\b")
  if (str_detect(name_upper, org_pattern)) return(FALSE)
  
  # strip anything in parentheses: "(Ret)", "(PhD)", nicknames, etc.
  name_clean <- str_squish(str_remove_all(name_upper, "\\s*\\([^)]*\\)"))
  
  # collapse dotted abbreviations: "M.D." -> "MD"
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
    # credentials
    "MD", "PHD", "ESQ", "DDS", "DO", "DR", "MR", "MRS", "MS", "HON", "EDS",
    "OD", "CPA", "DVM", "RN", "NP", "FACS", "TTEE", "JD", "MBA", "CFA",
    "ND", "CRNA", "NMD", "DMD", "DC",
    # military ranks
    "MAJ", "COL", "CAPT", "GEN", "LT", "LTC", "ADM", "SGT", "CPT", "CDR", "ENS", "SFC",
    "COMMANDER", "USAF", "USA", "USN", "USMC", "USCG", "RET",
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
  

  if (n_commas >= 2) {
    parts <- str_trim(str_split(name_clean, ",")[[1]])
    parts <- parts[nchar(str_squish(parts)) > 0]  # drop blank segments
    
    cred_pat <- paste0("^(", suffix_title_pat, "|[A-Z])$")
    
    # comma segment can have multiple words (e.g. "USMC RET", "O D")
    cred_tokens_ok <- function(seg) {
      toks <- str_split(str_squish(seg), "\\s+")[[1]]
      toks <- toks[nchar(toks) > 0]
      if (length(toks) == 0) return(TRUE)
      all(str_detect(toks, cred_pat))
    }
    
    # if empty segment filtering collapsed to two parts, use n_commas==1 logic
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
      
      # check middle segments are credentials ("USMC RET", "O D")
      if (lp_ok && fp_ok && all(vapply(mps, cred_tokens_ok, logical(1)))) return(TRUE)
      
      # check for credential/suffix at the end e.g. "ALVAREZ, ISRAEL, JR" 
      if (lp_ok && str_detect(fp, cred_pat) &&
          all(str_detect(mps, "^[A-Z'\\-\\. ]+$"))) return(TRUE)
      
      # check ≤ 3 total parts so multi-partner law firms are excluded
      # check no segment contains "AND" so converted firm names are excluded (e.g. "Fitzgerald, Alvarez, and Ciummo" after standardization)
      #  last-name segment ≤ 2 words (not sure this part is working)
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



