# build_entity_classification_registry.R
#
# Groups org contributor names and individual employer strings into canonical
# entities using three parallel linking tiers, detects classification code
# conflicts, and writes:
#   - entity_classification_registry_[date].csv  (full registry)
#   - entity_classification_conflicts_[date].csv  (conflicts only, by impact)
#   - entity_name_lookup_[date].csv               (name_string → canonical_entity_id)
#
# Run AFTER assign_final_classification.Rmd produces final_classifications_readable_*.csv.
# The entity_name_lookup file is consumed by the optional employer registry patch
# in assign_final_classification.Rmd (requires entity_registry_approved.csv in 10_inputs/).
#
# Three linking tiers (all run in parallel via igraph union-find):
#   1. Names 2026 mapping (name_entity_mapping_rebuilt.csv) — human confirmed, highest priority
#   2. DD org concept map (dd_names_mapped_to_org_concepts_3_18_2026.csv) — human curated
#   3. name_v5 exact match (business suffixes stripped) — automated, no manual review needed

suppressPackageStartupMessages({
  library(tidyverse)
  library(igraph)
})

source("../08_alternative_pipeline/standardization_helpers.R")

today <- format(Sys.Date(), "%m-%d-%y")

# ── Helpers ────────────────────────────────────────────────────────────────────

# name_v5: standardize → fix typos → strip parentheticals → strip business suffixes
make_name_v5 <- function(x) {
  x |>
    standardize_names() |>
    fix_typos() |>
    stringr::str_replace_all("\\s*\\([^)]*\\)\\s*", " ") |>
    stringr::str_squish() |>
    replace_business_text() |>
    stringr::str_squish()
}

# DD bridge standardization — mirrors build_bridge_lookup() in 0802
make_dd_std <- function(x) {
  x |>
    standardize_names() |>
    fix_typos() |>
    replace_business_text() |>
    stringr::str_replace_all("\\s*\\([^)]*\\)\\s*", " ") |>
    stringr::str_squish()
}

UNINFORMATIVE <- c(
  "NONE", "RETIRED", "SELF EMPLOYED", "SELF-EMPLOYED", "SELFEMPLOYED",
  "UNKNOWN", "N/A", "NOT EMPLOYED", "HOMEMAKER", "STUDENT", "UNEMPLOYED",
  "NOT APPLICABLE", "HOME MAKER", "SELF", "NA"
)

is_uninformative <- function(x) {
  trimws(x) %in% UNINFORMATIVE |
    grepl("^(SELF[- ]?EMPLOY|HOMEMAKER|STUDENT|UNEMPLOYED|NOT EMPLOY)", trimws(x))
}

# ── Load data ──────────────────────────────────────────────────────────────────

readable_files <- Sys.glob("10_outputs/final_classifications_readable_*.csv")
if (length(readable_files) == 0) {
  stop(
    "No final_classifications_readable_*.csv in 10_outputs/. ",
    "Run assign_final_classification.Rmd first."
  )
}
readable_path <- readable_files[order(file.mtime(readable_files), decreasing = TRUE)][1]
cat("Reading classifications:", readable_path, "\n")
cls <- read_csv(readable_path, col_types = cols(.default = "c"), show_col_types = FALSE)

# Names 2026 mapping — standardize raw_name to match pipeline's standardized strings
nem_raw <- read_csv(
  "../08_alternative_pipeline/08_inputs/name_entity_mapping_rebuilt.csv",
  col_types = cols(.default = "c"), show_col_types = FALSE
) |>
  select(raw_name, canonical_name, entity_uuid) |>
  filter(!is.na(raw_name), !is.na(entity_uuid)) |>
  mutate(name_std = standardize_names(toupper(trimws(raw_name))))

# DD org concept map — standardize record_name for matching
dd_raw <- read.csv(
  "../08_alternative_pipeline/08_inputs/dd_names_mapped_to_org_concepts_3_18_2026.csv",
  stringsAsFactors = FALSE
) |>
  rename(canonical_name_dd = org_concept_name, alias_group_id = oid) |>
  mutate(
    record_name       = stringr::str_squish(record_name),
    canonical_name_dd = stringr::str_squish(canonical_name_dd),
    record_std        = make_dd_std(record_name)
  ) |>
  filter(!is.na(record_std), record_std != "", !is.na(alias_group_id))

# ── Step 1: Collect name strings ───────────────────────────────────────────────

org_strings <- cls |>
  filter(tolower(entity_type) == "organization") |>
  transmute(
    name_string       = trimws(name),
    role              = "direct_contributor",
    code_final,
    code_final_source,
    rule_code
  ) |>
  filter(!is.na(name_string), name_string != "")

indiv_strings <- cls |>
  filter(tolower(entity_type) == "individual") |>
  transmute(
    name_string       = trimws(employer),
    role              = "employer",
    code_final,
    code_final_source,
    rule_code
  ) |>
  filter(!is.na(name_string), name_string != "", !is_uninformative(name_string))

all_strings <- bind_rows(org_strings, indiv_strings)
unique_names <- tibble(name_string = sort(unique(all_strings$name_string)))
cat(sprintf(
  "Name strings: %d org, %d employer, %d unique total\n",
  n_distinct(org_strings$name_string),
  n_distinct(indiv_strings$name_string),
  nrow(unique_names)
))

# ── Step 2: Three-tier parallel edge generation ────────────────────────────────

# Tier 1: Names 2026 mapping — name_string → "nem_{uuid}"
nem_lookup <- nem_raw |> select(name_std, canonical_name, entity_uuid) |> distinct()

nem_edges <- unique_names |>
  left_join(nem_lookup, by = c("name_string" = "name_std")) |>
  filter(!is.na(entity_uuid)) |>
  transmute(from = name_string, to = paste0("nem_", entity_uuid))

# Tier 2: DD org concept map — name_string → "dd_{alias_group_id}"
dd_lookup <- dd_raw |>
  select(record_std, alias_group_id, canonical_name_dd) |>
  distinct()

dd_std_names <- unique_names |>
  mutate(name_std_dd = make_dd_std(name_string))

dd_edges <- dd_std_names |>
  left_join(dd_lookup, by = c("name_std_dd" = "record_std"), relationship = "many-to-many") |>
  filter(!is.na(alias_group_id)) |>
  transmute(from = name_string, to = paste0("dd_", alias_group_id)) |>
  distinct()

# Tier 3: name_v5 exact match — edges between name strings sharing the same v5 key
v5_names <- unique_names |>
  mutate(v5 = make_name_v5(name_string)) |>
  filter(!is.na(v5), nchar(v5) > 2)  # exclude very short keys (too ambiguous)

v5_edges <- v5_names |>
  inner_join(v5_names, by = "v5", suffix = c("", "_b"), relationship = "many-to-many") |>
  filter(name_string < name_string_b) |>  # deduplicate pairs
  transmute(from = name_string, to = name_string_b)

all_edges <- bind_rows(nem_edges, dd_edges, v5_edges)
cat(sprintf(
  "Edges: %d Names mapping, %d DD concepts, %d name_v5 pairs\n",
  nrow(nem_edges), nrow(dd_edges), nrow(v5_edges)
))

# ── Step 3: Union-find via igraph ──────────────────────────────────────────────

# All graph nodes: name strings + canonical ID pseudo-nodes from tiers 1 & 2
all_nodes <- unique(c(
  unique_names$name_string,
  nem_edges$to,
  dd_edges$to
))

if (nrow(all_edges) > 0) {
  g <- graph_from_data_frame(
    all_edges,
    directed = FALSE,
    vertices = data.frame(name = all_nodes)
  )
  comps    <- components(g)
  comp_tbl <- tibble(
    node    = names(comps$membership),
    comp_id = as.integer(comps$membership)
  )
} else {
  comp_tbl <- tibble(
    node    = all_nodes,
    comp_id = seq_along(all_nodes)
  )
}

# Keep only actual name-string nodes (drop nem_* and dd_* pseudo-nodes)
name_comp_tbl <- comp_tbl |>
  filter(node %in% unique_names$name_string) |>
  rename(name_string = node)

# Assign fresh comp_ids to any singletons that had no edges
max_comp_id <- max(name_comp_tbl$comp_id, 0L)
all_name_comps <- unique_names |>
  left_join(name_comp_tbl, by = "name_string") |>
  mutate(comp_id = coalesce(comp_id, as.integer(row_number() + max_comp_id)))

cat(sprintf(
  "Components: %d total, %d multi-name groups\n",
  n_distinct(all_name_comps$comp_id),
  sum(table(all_name_comps$comp_id) > 1)
))

# ── Step 4: Canonical entity metadata per component ────────────────────────────

# Names 2026 canonical info per component
nem_by_comp <- all_name_comps |>
  left_join(
    unique_names |>
      left_join(nem_lookup |> rename(name_string = name_std), by = "name_string"),
    by = "name_string"
  ) |>
  group_by(comp_id) |>
  summarise(
    nem_id   = first(na.omit(entity_uuid)),
    nem_name = first(na.omit(canonical_name)),
    .groups  = "drop"
  )

# DD canonical info per component
dd_by_comp <- all_name_comps |>
  left_join(dd_std_names, by = "name_string") |>
  left_join(dd_lookup |> rename(name_std_dd = record_std), by = "name_std_dd",
            relationship = "many-to-many") |>
  group_by(comp_id) |>
  summarise(
    dd_id   = first(na.omit(as.character(alias_group_id))),
    dd_name = first(na.omit(canonical_name_dd)),
    .groups = "drop"
  )

comp_meta <- all_name_comps |>
  group_by(comp_id) |>
  summarise(
    raw_name_variants = paste(sort(unique(name_string)), collapse = " | "),
    n_name_variants   = n_distinct(name_string),
    most_freq_name    = names(sort(table(name_string), decreasing = TRUE))[1],
    .groups           = "drop"
  ) |>
  left_join(nem_by_comp, by = "comp_id") |>
  left_join(dd_by_comp,  by = "comp_id") |>
  mutate(
    canonical_entity_id = case_when(
      !is.na(nem_id) ~ paste0("nem_", nem_id),
      !is.na(dd_id)  ~ paste0("dd_",  dd_id),
      TRUE           ~ paste0("v5_",  comp_id)
    ),
    canonical_entity_name = coalesce(nem_name, dd_name, most_freq_name),
    link_tier = case_when(
      !is.na(nem_id) ~ "names_mapping",
      !is.na(dd_id)  ~ "dd_org_concept",
      TRUE           ~ "name_v5"
    )
  ) |>
  select(comp_id, canonical_entity_id, canonical_entity_name, link_tier,
         raw_name_variants, n_name_variants)

# ── Step 5: Aggregate codes per entity × code_final ───────────────────────────

entity_strings <- all_strings |>
  left_join(all_name_comps |> select(name_string, comp_id), by = "name_string") |>
  left_join(comp_meta, by = "comp_id") |>
  filter(!is.na(code_final), code_final != "")

entity_totals <- entity_strings |>
  group_by(canonical_entity_id) |>
  summarise(
    n_as_direct_contributor = sum(role == "direct_contributor"),
    n_as_employer           = sum(role == "employer"),
    .groups = "drop"
  )

registry_base <- entity_strings |>
  group_by(canonical_entity_id, canonical_entity_name, link_tier,
           raw_name_variants, n_name_variants, code_final) |>
  summarise(
    n_contributors       = n(),
    # Distinguishing pipeline-assigned vs. manual_review counts lets reviewers judge
    # whether a minority code is a legitimate employer decision or an individual exception.
    n_from_pipeline      = sum(coalesce(code_final_source, "") != "manual_review"),
    n_from_manual_review = sum(coalesce(code_final_source, "") == "manual_review"),
    code_sources         = paste(sort(unique(na.omit(code_final_source))), collapse = " | "),
    has_direct_role      = any(role == "direct_contributor"),
    has_employer_role    = any(role == "employer"),
    # rule_code_variants: pipeline NAICS codes that were overridden to reach code_final.
    # Non-empty means manual review or another source changed the pipeline result.
    # Even without a code_conflict, this surfaces pipeline/manual disagreement
    # (e.g., ambulance company: rule_code=health, code_final=gov/public_safety).
    rule_code_variants = paste(
      sort(unique(na.omit(rule_code[!is.na(rule_code) & rule_code != code_final]))),
      collapse = " | "
    ),
    .groups = "drop"
  ) |>
  mutate(roles = case_when(
    has_direct_role & has_employer_role ~ "both",
    has_direct_role                     ~ "direct_contributor",
    TRUE                                ~ "employer"
  )) |>
  select(-has_direct_role, -has_employer_role)

# code_conflict = TRUE when the same entity has multiple distinct code_final values,
# whether from pipeline or manual_review. n_from_pipeline and n_from_manual_review
# per row let reviewers assess whether a minority code is an individual exception.
conflict_info <- registry_base |>
  group_by(canonical_entity_id) |>
  mutate(code_conflict = n_distinct(code_final) > 1) |>
  ungroup()

conflict_types <- conflict_info |>
  filter(code_conflict) |>
  group_by(canonical_entity_id) |>
  summarise(
    any_direct   = any(roles %in% c("direct_contributor", "both")),
    any_employer = any(roles %in% c("employer", "both")),
    any_manual   = any(n_from_manual_review > 0),
    any_pipeline = any(n_from_pipeline > 0),
    .groups = "drop"
  ) |>
  mutate(conflict_type = case_when(
    any_direct & any_employer & any_manual & any_pipeline ~ "direct_vs_employer; manual_vs_pipeline",
    any_direct & any_employer                             ~ "direct_vs_employer",
    any_manual & any_pipeline                             ~ "manual_vs_pipeline",
    any_manual & !any_pipeline                            ~ "within_manual_review",
    TRUE                                                  ~ "within_pipeline"
  )) |>
  select(canonical_entity_id, conflict_type)

registry_final <- conflict_info |>
  left_join(conflict_types, by = "canonical_entity_id") |>
  left_join(entity_totals,  by = "canonical_entity_id") |>
  mutate(conflict_resolution_code = NA_character_) |>
  select(
    canonical_entity_id, canonical_entity_name, link_tier,
    raw_name_variants, n_name_variants,
    code_final, n_contributors, n_from_pipeline, n_from_manual_review,
    code_sources, roles, rule_code_variants,
    code_conflict, conflict_type,
    n_as_direct_contributor, n_as_employer,
    conflict_resolution_code
  ) |>
  arrange(desc(n_as_direct_contributor + n_as_employer), canonical_entity_name, code_final)

# ── Step 6: Write outputs ──────────────────────────────────────────────────────

reg_path <- sprintf("10_outputs/entity_classification_registry_%s.csv", today)
write_csv(registry_final, reg_path)
cat(sprintf(
  "Registry: %d entities (%d conflict-free, %d conflicting) → %s\n",
  n_distinct(registry_final$canonical_entity_id),
  n_distinct(registry_final$canonical_entity_id[!coalesce(registry_final$code_conflict, FALSE)]),
  n_distinct(registry_final$canonical_entity_id[coalesce(registry_final$code_conflict, FALSE)]),
  reg_path
))

# Entities with genuine code conflicts — need reviewer resolution
conflicts_path <- sprintf("10_outputs/entity_classification_conflicts_%s.csv", today)
write_csv(
  registry_final |>
    filter(coalesce(code_conflict, FALSE)) |>
    arrange(desc(n_as_direct_contributor + n_as_employer), canonical_entity_name, code_final),
  conflicts_path
)
cat(sprintf(
  "Conflicts: %d conflicting entities → %s\n",
  n_distinct(registry_final$canonical_entity_id[coalesce(registry_final$code_conflict, FALSE)]),
  conflicts_path
))

# Entities where pipeline code was overridden (rule_code != code_final) but no code_conflict.
# Useful for identifying keyword rule gaps — e.g., an ambulance company has one code
# (gov/public_safety, from manual_review) but rule_code was "health". No conflict, but
# this signals that the pipeline's keyword rules should be updated for ambulance-type orgs.
overrides_path <- sprintf("10_outputs/entity_pipeline_overrides_%s.csv", today)
write_csv(
  registry_final |>
    filter(!coalesce(code_conflict, FALSE), rule_code_variants != "", n_from_manual_review > 0) |>
    arrange(desc(n_as_direct_contributor + n_as_employer), canonical_entity_name),
  overrides_path
)
cat(sprintf(
  "Pipeline overrides (no conflict, manual review changed pipeline code): %d entities → %s\n",
  n_distinct((registry_final |>
    filter(!coalesce(code_conflict, FALSE), rule_code_variants != "",
           n_from_manual_review > 0))$canonical_entity_id),
  overrides_path
))

# Flat name lookup consumed by the employer registry patch in assign_final_classification.Rmd
lookup_path <- sprintf("10_outputs/entity_name_lookup_%s.csv", today)
write_csv(
  all_name_comps |>
    select(name_string, comp_id) |>
    left_join(comp_meta |> select(comp_id, canonical_entity_id), by = "comp_id") |>
    select(name_string, canonical_entity_id),
  lookup_path
)
cat(sprintf("Name lookup: %d entries → %s\n",
            nrow(all_name_comps), lookup_path))
