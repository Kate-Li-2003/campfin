# resolve_classifications.R
#
# Logic picking one best NAICS code per contribution `uuid`, given up to four independent
# sources -- rule-based (0805_classify_contributors.py), OpenSecrets match, ML classification, 
# and LLM web-search classification. Sourced by both
# assign_final_classification.Rmd (direct $5k+ contributors) and
# assign_pac_final_classification.Rmd (PAC-graph donor contributors) so the
# two pipelines can't drift apart on how sources are weighted against each
# other.

ML_HIGH_CONF_THRESHOLD <- 0.7

DETERMINISTIC_TIERS <- c("pre_classified", "employer_lookup", "running_list", "keyword match")

# For code resolution: only these tiers are trusted enough to accept without review
# even when LLM disagrees. pre_classified is already locked by code_override().
HIGH_TRUST_CODE_TIERS <- c("pre_classified", "employer_lookup")

NOT_EMPLOYED_CODE <- "100"
FALLBACK_CODE     <- "99"
POLITICAL_CODES   <- c("88", "90") # pacs, candidate committees

# OpenSecrets (0806_match_opensecrets.Rmd) and the LLM (09_llm_classifier) each
# produce their own naics code independently of 0805_classify_contributors.py's
# rule-based path -- the only path that runs codes through 0805's
# build_naics_crosswalk() to translate raw/legacy codes to the current custom
# sector scheme (e.g. raw NAICS "33" -> custom "31-33"). Regenerate with
# build_os_llm_naics_crosswalk.py whenever new OS/LLM output files are added.
OS_LLM_CROSSWALK_PATH <- "10_outputs/os_llm_naics_crosswalk.csv"

apply_os_llm_crosswalk <- function(code_vec, crosswalk_path = OS_LLM_CROSSWALK_PATH) {
  if (!file.exists(crosswalk_path)) {
    warning(sprintf(
      "os/llm NAICS crosswalk not found at %s -- codes used as-is. Regenerate with build_os_llm_naics_crosswalk.py.",
      crosswalk_path
    ))
    return(code_vec)
  }
  cw <- readr::read_csv(crosswalk_path, col_types = readr::cols(.default = "c"), show_col_types = FALSE)
  lookup <- setNames(cw$custom_code, cw$raw_code)
  dplyr::if_else(code_vec %in% names(lookup), unname(lookup[code_vec]), code_vec)
}


# fail if missing a required column
require_cols <- function(df, required_actual_names, path) {
  missing <- setdiff(required_actual_names, names(df))
  if (length(missing) > 0) {
    stop(sprintf(
      paste0(
        "%s is missing expected column(s): %s\n",
        "Available columns: %s\n",
        "If the upstream script renamed these, update the matching *_COLS ",
        "config block at the top of this file to match."
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


conf_is_high <- function(x) {
  if (length(x) == 0 || is.na(x)) return(FALSE)
  if (is.character(x)) return(tolower(x) == "high")
  is.numeric(x) && x >= ML_HIGH_CONF_THRESHOLD
}

resolve_one <- function(rule_val, rule_tier, rule_conf,
                         os_val, os_conf,
                         llm_val, llm_conf,
                         fallback_value = FALLBACK_CODE) {
  is_usable <- function(val) { # make sure its not the "unknown" fallback
    !is.na(val) && val != "" && (is.na(fallback_value) || val != fallback_value)
  }

  raw <- c(rule = rule_val, opensecrets = os_val, llm = llm_val)
  codes <- raw[vapply(raw, is_usable, logical(1))]

  rule_is_deterministic <- !is.na(rule_tier) && rule_tier %in% DETERMINISTIC_TIERS # came from reliable source

  # agreement between 2+ independent pipelines
  if (length(codes) >= 2) {
    tab <- table(codes)
    top <- names(tab)[which.max(tab)]
    if (max(tab) >= 2) {
      agreeing <- names(codes)[codes == top]
      return(list(value = top,
                  source = paste0("agreement (", paste(agreeing, collapse = "+"), ")"),
                  review = FALSE, reason = NA_character_))
    }
  }

  # single deterministic rule source wins outright (ML is excluded — never treated as strong)
  if (is_usable(rule_val) && rule_is_deterministic) {
    return(list(value = rule_val, source = paste0("rule (", rule_tier, ")"), review = FALSE, reason = NA_character_))
  }
  if (is_usable(os_val) && conf_is_high(os_conf)) {
    return(list(value = os_val, source = "opensecrets (high confidence)", review = FALSE, reason = NA_character_))
  }
  if (is_usable(llm_val) && conf_is_high(llm_conf)) {
    return(list(value = llm_val, source = "llm (high confidence)", review = FALSE, reason = NA_character_))
  }

  # exactly one low-confidence source available -- best guess, flag review
  if (length(codes) == 1) {
    src <- names(codes)[1]
    return(list(value = unname(codes[1]), source = paste0(src, " (low confidence, single source)"),
                review = TRUE, reason = paste0("only ", src, " produced a value, and confidence was not high")))
  }

  # 2+ sources but disagree and none is high-confidence -- take the most reliable pipeline available (rule > opensecrets > llm), flag for review
  # make sure this doesn't rank ML > LLM / opensecrets
  if (length(codes) > 1) {
    priority <- c("rule", "opensecrets", "llm")
    winner <- priority[priority %in% names(codes)][1]
    return(list(value = unname(codes[winner]),
                source = paste0(winner, " (disagreement, low confidence)"),
                review = TRUE,
                reason = paste0("sources disagreed: ",
                                 paste(names(codes), codes, sep = "=", collapse = ", "))))
  }

  # nothing reliable to use
  list(value = fallback_value, source = "uncategorized fallback", review = TRUE,
       reason = "no source produced a usable value")
}

# hard overrides
code_override <- function(rule_val, os_val, llm_val, rule_tier) {
  vals <- c(rule = rule_val, opensecrets = os_val, llm = llm_val)

  # pre_classified rows are locked -- this is a specific contributor's own row from
  # already_classified_contributions.csv, already reviewed. Never re-derived from
  # os/llm, even when the recorded code is "99" (that's a deliberate "reviewed and
  # left uncategorized" call, not missing data). employer_lookup/running_list/
  # keyword_match/identity_override are inferred matches, not the same guarantee,
  # so they still go through the normal resolution/override logic below.
  if (!is.na(rule_tier) && rule_tier == "pre_classified" && !is.na(rule_val) && rule_val != "") {
    return(list(value = rule_val, source = "pre_classified (locked)", review = FALSE, reason = NA_character_))
  }

  # not-employed: any source saying so is authoritative
  if (any(vals == NOT_EMPLOYED_CODE, na.rm = TRUE)) {
    return(list(value = NOT_EMPLOYED_CODE, source = "not-employed override", review = FALSE, reason = NA_character_))
  }

  # political entities — also locks identity_override for political codes since
  # PAC/union/govt regex patterns are reliable enough for these specific codes
  if (!is.na(rule_val) && !is.na(rule_tier) &&
      (rule_tier %in% DETERMINISTIC_TIERS || rule_tier == "identity_override") &&
      rule_val %in% POLITICAL_CODES) {
    return(list(value = rule_val, source = "political-entity rule", review = FALSE, reason = NA_character_))
  }

  NULL  # no override -- fall through to resolve_one()
}

# per-axis wrappers

# Code resolution. Priority order:
#   1. code_override (pre_classified locked; not-employed; political entities)
#   2. High-trust rule tiers (employer_lookup) — accept without review even if LLM disagrees
#   3. OS + LLM agree → use their code; flag if rule disagrees
#   4. Rule + OS agree → use rule code, no review
#   5. Rule + LLM agree → use rule code, no review
#   6. ML + LLM agree → use ML/LLM code; flag if rule disagrees OR both low-confidence
#   7. Rule present, no full agreement → use rule, flag if any source disagrees
#   8. No rule: OS → LLM → ML (high-conf before low-conf), all flagged for review
#   9. Nothing usable → fallback
resolve_code_one <- function(rule_val, rule_tier, ml_val, ml_conf, os_val, os_conf, llm_val, llm_conf) {
  override <- code_override(rule_val, os_val, llm_val, rule_tier)
  if (!is.null(override)) return(override)

  is_usable <- function(val) !is.na(val) && nchar(trimws(val)) > 0 && val != FALLBACK_CODE

  rule_usable <- is_usable(rule_val)
  ml_usable   <- is_usable(ml_val)
  llm_usable  <- is_usable(llm_val)
  os_usable   <- is_usable(os_val)

  # All OS matches were manually verified — treat NA confidence as high
  if (os_usable && (is.na(os_conf) || os_conf == "")) os_conf <- "high"

  # High-trust rule tiers: accept without review (pre_classified already locked above)
  if (rule_usable && !is.na(rule_tier) && rule_tier %in% HIGH_TRUST_CODE_TIERS) {
    return(list(value = rule_val, source = paste0("rule (", rule_tier, ")"),
                review = FALSE, reason = NA_character_))
  }

  # OS + LLM agree: two independent reliable sources
  if (os_usable && llm_usable && os_val == llm_val) {
    rule_disagrees <- rule_usable && rule_val != os_val
    return(list(
      value  = os_val,
      source = if (rule_disagrees) "os+llm agreement (rule disagrees)" else "os+llm agreement",
      review = rule_disagrees,
      reason = if (rule_disagrees) "os_llm_agree_rule_disagrees" else NA_character_
    ))
  }

  # Rule + OS agree
  if (rule_usable && os_usable && rule_val == os_val) {
    return(list(value = rule_val, source = "rule+os agreement", review = FALSE, reason = NA_character_))
  }

  # Rule + LLM agree
  if (rule_usable && llm_usable && rule_val == llm_val) {
    return(list(value = rule_val, source = "rule+llm agreement", review = FALSE, reason = NA_character_))
  }

  # ML + LLM agree
  if (ml_usable && llm_usable && ml_val == llm_val) {
    rule_disagrees <- rule_usable && rule_val != ml_val
    both_low_conf  <- !conf_is_high(ml_conf) && !conf_is_high(llm_conf)
    needs_review   <- rule_disagrees || both_low_conf
    reason <- if (rule_disagrees && both_low_conf) {
      "ml_llm_agree_rule_disagrees_low_conf"
    } else if (rule_disagrees) {
      "ml_llm_agree_rule_disagrees"
    } else if (both_low_conf) {
      "ml_llm_agree_low_conf"
    } else {
      NA_character_
    }
    return(list(value = ml_val, source = "ml+llm agreement", review = needs_review, reason = reason))
  }

  # Rule present: use rule; flag only if LLM disagrees (most reliable independent signal).
  # OS/ML disagreement alone does not warrant a review flag here.
  if (rule_usable) {
    llm_disagrees <- llm_usable && llm_val != rule_val
    return(list(
      value  = rule_val,
      source = paste0("rule (", coalesce(rule_tier, "unknown"), ")"),
      review = llm_disagrees,
      reason = if (llm_disagrees) "rule_llm_disagree" else NA_character_
    ))
  }

  # No rule: use best available source (all flagged for review).
  # OS/LLM are preferred over ML — ML is only used as last resort when high-confidence.
  if (os_usable && conf_is_high(os_conf)) {
    return(list(value = os_val, source = "os (high confidence)",
                review = TRUE, reason = "no rule; sources inconsistent"))
  }
  if (llm_usable && conf_is_high(llm_conf)) {
    
    if (ml_usable && !conf_is_high(ml_conf)) {
      return(list(value = llm_val, source = "llm (high confidence)",
                  review = TRUE, reason = "llm only source"))
    }
    
    else{
      return(list(value = FALLBACK_CODE, source = "uncategorized fallback",
                  review = TRUE, reason = "no rule; sources inconsistent"))
    }
    
  }

  if (ml_usable && conf_is_high(ml_conf)) {
    return(list(value = ml_val, source = "ml (high confidence)",
                review = TRUE, reason = "no rule; sources inconsistent"))
  }
  if (os_usable) {
    return(list(value = os_val, source = "os (low confidence)",
                review = TRUE, reason = "no rule; os low confidence"))
  }
  if (llm_usable) {
    return(list(value = llm_val, source = "llm only (low confidence)",
                review = TRUE, reason = "no rule; llm confidence not high"))
  }

  list(value = FALLBACK_CODE, source = "uncategorized fallback",
       review = TRUE, reason = "no source produced a usable value")
}

resolve_category_one <- function(rule_val, rule_tier, ml_conf, os_val, os_conf, llm_val, llm_conf) {
  resolve_one(rule_val, rule_tier, ml_conf, os_val, os_conf, llm_val, llm_conf, fallback_value = NA_character_)
}
