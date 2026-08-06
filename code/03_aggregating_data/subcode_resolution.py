"""Sub-code resolution: map parent-level NAICS codes to the custom scheme.

PROCEDURAL script — the taxonomy itself lives in
data/03_input/masterfile/custom_naics_labels.csv (EDITORIAL). Edit that
file to change categories; this module should not need to change.

The custom scheme (see CUSTOM_CODES.md) keeps most 2-digit NAICS sectors
but (a) partitions some sectors into sub-codes that fully replace the
parent (52 -> 52a/52b, 56 -> 56a/56b, 77 -> 77a/77b), (b) carves editorial
sub-codes out of sectors that remain valid (22a, 51a, 54a, 71a, 92a), and
(c) layers custom codes for political entities etc. (76, 78, 79, 88, 90,
91, 99, 100).

Resolution order for one entity (name, occupation, parent naics_code):

  1. employer-name regex pass  — custom_naics_labels.csv rows in file
     order, later rows overwrite earlier (identical semantics to the old
     0501.apply_custom_label_overrides, so 88/90/91 stay last and
     authoritative).
  2. occupation regex pass     — rows with applies_to == "occupation"
     (e.g. 100 Retired/Homemaker/Student). Runs after (1) so it wins:
     occupation is the stronger signal for individuals.
  3. partition fallback        — entity's parent code is a partitioned
     sector (52/56/77) and no regex fired: pick the sub-code whose
     plain-language `description` embeds closest to the entity text
     (name + occupation). The classifier stays at parent level; this is
     the only ML-ish step and it is confined to choosing among a
     parent's own children.
  4. passthrough               — parent code is itself a terminal label
     in the scheme (sectors, carve-out parents, custom codes): keep it,
     restamped with the scheme's current label.

Every row gets `resolution_method` (regex / occupation-rule / embedding /
parent-passthrough / unresolved) and `resolution_confidence` so
low-confidence embedding calls can be exported for editorial review.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

# Several editorial regexes legitimately use groups for alternation;
# pandas warns that str.contains ignores groups — that's exactly what we want.
warnings.filterwarnings(
    "ignore", message="This pattern is interpreted as a regular expression"
)

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA = REPO_ROOT / "data/03_input/masterfile/custom_naics_labels.csv"

# 3-6 digit real NAICS -> 2-digit parent; NAICS range sectors.
_RANGE_MAP = {"31": "31-33", "32": "31-33", "33": "31-33",
              "44": "44-45", "45": "44-45", "48": "48-49", "49": "48-49"}

STAMP_COLS = ["level1_category", "level2_category", "level3_category"]


# ---------- schema ----------

def load_schema(path: Path | str = DEFAULT_SCHEMA) -> pd.DataFrame:
    """Load the editorial taxonomy. Terminal codes = every row present;
    partition parents (codes appearing only in `parent_code` of
    partition_child rows, e.g. 52) are deliberately absent as rows."""
    schema = pd.read_csv(path, dtype=str)
    required = {"naics_code", "naics_label", "kind", "applies_to"}
    missing = required - set(schema.columns)
    if missing:
        raise SystemExit(f"custom_naics_labels.csv missing columns: {missing}")
    return schema


def partition_parents(schema: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """{parent_code -> child rows} for fully partitioned sectors."""
    kids = schema[schema["kind"] == "partition_child"]
    return {p: g for p, g in kids.groupby("parent_code")}


def normalize_parent(code) -> str:
    """'531210' -> '53', '33' -> '31-33', '52.0' -> '52'; '' if empty."""
    if pd.isna(code):
        return ""
    s = str(code).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if not s or s in ("31-33", "44-45", "48-49"):
        return s
    if re.fullmatch(r"\d{3,6}", s):
        s = s[:2]
    return _RANGE_MAP.get(s, s)


# ---------- encoder (lazy; only needed for the partition fallback) ----------

def load_encoder(models_dir: Path | None = None):
    """Same sentence-transformer the 07 ML stage uses (encoder_name.txt)."""
    from sentence_transformers import SentenceTransformer  # deferred import
    models_dir = models_dir or REPO_ROOT / "data/07_output_ml_classification/models"
    name_file = models_dir / "encoder_name.txt"
    name = name_file.read_text().strip() if name_file.exists() \
        else "sentence-transformers/all-MiniLM-L6-v2"
    return SentenceTransformer(name)


# ---------- main entry point ----------

def resolve_frame(
    df: pd.DataFrame,
    schema: pd.DataFrame,
    name_col: str = "name",
    occ_col: str | None = None,
    parent_col: str = "naics_code",
    encoder=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return `df` with custom_code / custom_label / resolution_method /
    resolution_confidence columns added. Vectorized; regex rules applied
    in file order with later rows overwriting earlier matches."""
    out = df.copy()
    labels = dict(zip(schema["naics_code"], schema["naics_label"]))
    parts = partition_parents(schema)

    code = pd.Series(pd.NA, index=out.index, dtype=object)
    method = pd.Series(pd.NA, index=out.index, dtype=object)
    conf = pd.Series(np.nan, index=out.index, dtype=float)

    # (4-first) parent passthrough as the baseline; regex passes overwrite.
    parent = out[parent_col].map(normalize_parent)
    terminal = parent.isin(set(schema["naics_code"]))
    code[terminal] = parent[terminal]
    method[terminal] = "parent-passthrough"
    conf[terminal] = 1.0

    names = out[name_col].fillna("").astype(str)
    occs = out[occ_col].fillna("").astype(str) if occ_col else None

    # (1) employer-name regex pass, then (2) occupation pass (occupation
    # wins for individuals, so it runs second and overwrites).
    for applies, series in (("employer", names), ("occupation", occs)):
        if series is None:
            continue
        rules = schema[(schema["applies_to"] == applies) & schema["name_regex"].notna()]
        for r in rules.itertuples(index=False):
            hit = series.str.contains(r.name_regex, case=False, regex=True, na=False)
            code[hit] = r.naics_code
            method[hit] = "regex" if applies == "employer" else "occupation-rule"
            conf[hit] = 1.0
            if verbose and int(hit.sum()):
                print(f"  {applies[:3]}-rule {r.naics_code:>4s} ({r.naics_label}): "
                      f"{int(hit.sum()):,} matched")

    # (3) partition fallback: 52/56/77 with no regex hit -> embedding.
    for pcode, kids in parts.items():
        mask = code.isna() & (parent == pcode)
        n = int(mask.sum())
        if not n:
            continue
        if encoder is None:
            method[mask] = "unresolved-partition"
            if verbose:
                print(f"  partition {pcode}: {n:,} rows left unresolved (no encoder)")
            continue
        texts = names[mask]
        if occs is not None:
            texts = (texts + " " + occs[mask]).str.strip()
        emb = encoder.encode(texts.tolist(), normalize_embeddings=True,
                             show_progress_bar=False)
        proto = encoder.encode(kids["description"].fillna(kids["naics_label"]).tolist(),
                               normalize_embeddings=True, show_progress_bar=False)
        sims = emb @ proto.T
        pick = sims.argmax(axis=1)
        # softmax over children as a rough confidence
        ex = np.exp(sims * 10)
        pconf = (ex / ex.sum(axis=1, keepdims=True))[np.arange(len(pick)), pick]
        idx = out.index[mask]
        code[idx] = kids["naics_code"].to_numpy()[pick]
        method[idx] = "embedding"
        conf[idx] = pconf
        if verbose:
            picked = pd.Series(kids["naics_code"].to_numpy()[pick]).value_counts()
            print(f"  partition {pcode}: {n:,} rows via embedding -> "
                  + ", ".join(f"{k}: {v:,}" for k, v in picked.items()))

    out["custom_code"] = code
    out["custom_label"] = code.map(labels)
    out["resolution_method"] = method.fillna("unresolved")
    out["resolution_confidence"] = conf

    # Restamp level1/2/3 from the schema so downstream grouping uses the
    # current editorial hierarchy.
    for col in STAMP_COLS:
        if col in schema.columns:
            out["custom_" + col] = code.map(dict(zip(schema["naics_code"], schema[col])))
    return out
