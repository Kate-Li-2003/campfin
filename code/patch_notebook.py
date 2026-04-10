"""
Patch 01gov_classification_review.ipynb:
  1. Update crosswalk cell (ec22eaa1) to reflect Ag narrowing
  2. Add "Data Processing Changes" section after section 3
  3. Add "Visualizations" section with top-10 candidate pie charts
"""
import json, uuid

NB_PATH = "/Users/kateli/Desktop/CalMatters/campaign finance categorization/code/01gov_classification_review.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

def cell_id(cell):
    return cell.get("id", "")

def make_md(src, cell_id_val=None):
    return {
        "cell_type": "markdown",
        "id": cell_id_val or uuid.uuid4().hex[:8],
        "metadata": {},
        "source": src,
    }

def make_code(src, cell_id_val=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id_val or uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": src,
    }

# ── 1. Update crosswalk cell ─────────────────────────────────────────────────
CROSSWALK_SRC = '''\
# Crosswalk: repo Industry label → our 12 target categories
#
# Changes from initial version:
#   • Beer/Wine/Liquor, Food & Beverage, Food Processing removed from Ag
#     → these labels cover restaurants, distributors & candy companies that
#       are not meaningfully agricultural, and were causing false positives.
REPO_TO_TARGET = {
    # Tech
    "Computer Equipment & Services":          "Tech",
    "Electronics Manufacturing & Services":   "Tech",
    "Telecom Services & Equipment":           "Tech",
    "Telephone Utilities":                    "Tech",
    "Cable TV":                               "Tech",
    "Miscellaneous Communications & Electronics": "Tech",
    # Labor
    "General Trade Unions":                   "Labor",
    "Public Sector Unions":                   "Labor",
    "Transportation Unions":                  "Labor",
    # Oil
    "Oil & Gas":                              "Oil",
    "Miscellaneous Energy":                   "Oil",
    # Utilities
    "Electric Utilities":                     "Utilities",
    "Water Utilities":                        "Utilities",
    "Waste Management":                       "Utilities",
    # Ag — narrowed to genuine farm/ranch/crop labels only
    "Crop Production & Basic Processing":     "Ag",
    "Dairy":                                  "Ag",
    "Livestock":                              "Ag",
    "Poultry & Eggs":                         "Ag",
    "Forestry & Forest Products":             "Ag",
    "Agricultural Services & Products":       "Ag",
    "Miscellaneous Agriculture":              "Ag",
    "Farm Bureau":                            "Ag",
    # Beer/Wine/Liquor  → removed
    # Food & Beverage   → removed
    # Food Processing & Sales → removed
    # Political
    "Candidate Committees":                   "Political",
    "Candidate Contributions":                "Political",
    "Party Committees":                       "Political",
    "Leadership PACs":                        "Political",
    "Joint Candidate Committee":              "Political",
    "Liberal Policy Organization":            "Political",
    "Conservative Policy Organization":       "Political",
    "Christian Conservative":                 "Political",
    # Health Care
    "Health Professionals":                   "Health Care",
    "Hospitals & Nursing Homes":              "Health Care",
    "Pharmaceuticals & Health Products":      "Health Care",
    "Health Services":                        "Health Care",
    "Miscellaneous Health":                   "Health Care",
    "Health & Welfare Policy":                "Health Care",
    # Environment
    "Pro-Environmental Policy":               "Environment",
    "Environmental Services & Equipment":     "Environment",
    "Animal Rights":                          "Environment",
    # Real Estate
    "Real Estate":                            "Real Estate",
    "Home Builders":                          "Real Estate",
    "Construction Services":                  "Real Estate",
    "General Contractors":                    "Real Estate",
    "Special Trade Contractors":              "Real Estate",
    "Building Materials & Equipment":         "Real Estate",
    # Lawyers/Lobbying
    "Lawyers & Lobbyists":                    "Lawyers/Lobbying",
    "Accountants":                            "Lawyers/Lobbying",
}

train_raw["Target_Category"] = train_raw["Industry"].map(REPO_TO_TARGET)
mapped   = train_raw.dropna(subset=["Target_Category"])
unmapped = train_raw[
    train_raw["Target_Category"].isna() & (train_raw["Industry"] != "Uncoded")
]

print(f"Total orgs:           {len(train_raw):,}")
print(f"Mapped to 12 cats:    {len(mapped):,}")
print(f"Uncoded (excluded):   {(train_raw['Industry']=='Uncoded').sum():,}")
print(f"Other (not in scope): {len(unmapped):,}  (Insurance, Automotive, Gambling, etc.)")
print(f"\\nTraining corpus used by KNN: {len(mapped):,} orgs across "
      f"{mapped['Target_Category'].nunique()} categories")
'''

for cell in nb["cells"]:
    if cell_id(cell) == "ec22eaa1":
        cell["source"] = CROSSWALK_SRC
        cell["outputs"] = []
        cell["execution_count"] = None
        break

# ── 2. Insert "Data Processing Changes" section after cell 81f1c061 (§3 markdown) ──
CHANGES_MD = '''\
---
## 3b. Data Processing Changes

This section documents decisions made after auditing classifier output.
Each change is reproducible by re-running `classify_contributors.py`.
'''

CHANGES_CODE = '''\
# Summary of changes made to the classification pipeline after initial audit
changes = [
    {
        "area": "Crosswalk — Ag category",
        "change": "Removed Beer/Wine/Liquor, Food & Beverage, Food Processing & Sales",
        "reason": "These repo labels cover restaurants, distributors and candy companies. "
                  "Including them inflated Ag from ~1,600 to ~4,800 rows and caused "
                  "McDonald's, Harbor Freight Tools and Sanders Candy to appear as Ag.",
        "effect": "Ag: 4,852 → 1,621 rows",
    },
    {
        "area": "Keyword matching — regex",
        "change": "Switched from substring match (kw in text) to word-boundary regex (\\\\bkw\\\\b)",
        "reason": "'farm' was matching inside 'State Farm', 'ranch' inside 'Branch Investment Group', "
                  "etc. Word boundaries require the keyword to be a standalone token.",
        "effect": "Eliminated ~15 false-positive keyword matches per category",
    },
    {
        "area": "Keyword blocklist",
        "change": "Added KEYWORD_BLOCKLIST = ['state farm', 'farm credit']",
        "reason": "Even with word-boundary matching, 'State Farm' (insurance) contains 'farm' "
                  "as a genuine standalone word. The blocklist short-circuits before any keyword "
                  "check fires for known false-positive org names.",
        "effect": "State Farm insurance agents now fall through to KNN rather than keyword match. "
                  "Residual: KNN still votes Ag because no Insurance category exists in taxonomy.",
    },
]

import pandas as pd
df_changes = pd.DataFrame(changes)
display(df_changes.style.set_properties(**{"text-align": "left", "white-space": "pre-wrap"})
        .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}]))
'''

# Find index of cell 81f1c061 and insert after it
insert_after_id = "81f1c061"
insert_idx = next(
    (i for i, c in enumerate(nb["cells"]) if cell_id(c) == insert_after_id), None
)
if insert_idx is not None:
    nb["cells"].insert(insert_idx + 1, make_md(CHANGES_MD))
    nb["cells"].insert(insert_idx + 2, make_code(CHANGES_CODE))

# ── 3. Append "Visualizations" section ───────────────────────────────────────
VIZ_MD = '''\
---
## 6. Visualizations

### Top-10 candidates by total contributions — donation breakdown by industry category

Each pie chart shows the share of total dollars received by that candidate, split by
the contributor's classified industry. "Not Employed / Retired" is shown separately
so individual small-donor giving is visible alongside organised-sector money.
'''

# Consistent color palette — one color per category
VIZ_COLORS_CODE = '''\
# Consistent color map across all charts
CATEGORY_COLORS = {
    "Political":              "#e63946",
    "Labor":                  "#f4a261",
    "Real Estate":            "#2a9d8f",
    "Health Care":            "#457b9d",
    "Lawyers/Lobbying":       "#8338ec",
    "Tech":                   "#3a86ff",
    "Ag":                     "#80b918",
    "Oil":                    "#6d4c41",
    "Crypto/AI":              "#ff006e",
    "Ridesharing":            "#fb5607",
    "Utilities":              "#ffbe0b",
    "Environment":            "#06d6a0",
    "Not Employed / Retired": "#adb5bd",
    "Unknown":                "#dee2e6",
}

# Load results (re-load to be safe after any re-runs)
res = pd.read_csv(RESULT, dtype=str, low_memory=False)
res["Amount"] = pd.to_numeric(res["Amount"], errors="coerce")
print(f"Loaded {len(res):,} classified rows.")
'''

VIZ_PIE_CODE = '''\
# Identify top 10 candidates by total contributions
top10_candidates = (
    res.groupby("Recipient Name")["Amount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)
print("Top 10 candidates:")
for i, c in enumerate(top10_candidates, 1):
    total = res[res["Recipient Name"] == c]["Amount"].sum()
    print(f"  {i:>2}. {c:<55}  ${total:>15,.0f}")
'''

VIZ_CHARTS_CODE = '''\
def pie_for_candidate(ax, candidate, res, color_map, min_pct=1.5):
    """
    Draw a pie chart on `ax` for one candidate.
    Slices below min_pct% are merged into 'Other'.
    """
    sub = (
        res[res["Recipient Name"] == candidate]
        .groupby("Industry_Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )
    total = sub.sum()

    # Merge tiny slices
    main  = sub[sub / total * 100 >= min_pct]
    other = sub[sub / total * 100 <  min_pct].sum()
    if other > 0:
        main["Other"] = other

    colors = [color_map.get(cat, "#cccccc") for cat in main.index]

    wedges, texts, autotexts = ax.pie(
        main.values,
        labels=None,
        colors=colors,
        autopct=lambda p: f"{p:.0f}%" if p >= min_pct else "",
        startangle=90,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(7)

    # Short candidate label (remove party suffixes for readability)
    short = candidate.split(",")[0] if "," in candidate else candidate
    ax.set_title(f"{short}\\n${total/1e6:.1f}M total", fontsize=9, pad=6)
    return main.index.tolist(), colors


# ── Build 2-row × 5-col grid ─────────────────────────────────────────────────
ncols, nrows = 5, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(22, 9))
axes_flat  = axes.flatten()

legend_handles = {}
for ax, candidate in zip(axes_flat, top10_candidates):
    cats, cols = pie_for_candidate(ax, candidate, res, CATEGORY_COLORS)
    for cat, col in zip(cats, cols):
        if cat not in legend_handles:
            legend_handles[cat] = col

# Hide any unused subplots
for ax in axes_flat[len(top10_candidates):]:
    ax.set_visible(False)

# Shared legend below the charts
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=col, label=cat)
           for cat, col in legend_handles.items()]
fig.legend(handles=patches, loc="lower center", ncol=5,
           fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Contribution breakdown by industry — top 10 candidates",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
'''

VIZ_EXCL_CODE = '''\
# Same charts excluding "Not Employed / Retired" to focus on organised-sector money
res_active = res[res["Industry_Category"] != "Not Employed / Retired"].copy()

fig2, axes2 = plt.subplots(nrows, ncols, figsize=(22, 9))
axes2_flat  = axes2.flatten()

legend_handles2 = {}
for ax, candidate in zip(axes2_flat, top10_candidates):
    cats, cols = pie_for_candidate(ax, candidate, res_active, CATEGORY_COLORS)
    for cat, col in zip(cats, cols):
        if cat not in legend_handles2:
            legend_handles2[cat] = col

for ax in axes2_flat[len(top10_candidates):]:
    ax.set_visible(False)

patches2 = [mpatches.Patch(color=col, label=cat)
            for cat, col in legend_handles2.items()]
fig2.legend(handles=patches2, loc="lower center", ncol=5,
            fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig2.suptitle("Contribution breakdown by industry — top 10 candidates  "
              "(excl. Not Employed / Retired)",
              fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
'''

nb["cells"] += [
    make_md(VIZ_MD),
    make_code(VIZ_COLORS_CODE),
    make_code(VIZ_PIE_CODE),
    make_code(VIZ_CHARTS_CODE),
    make_code(VIZ_EXCL_CODE),
]

# ── Save ──────────────────────────────────────────────────────────────────────
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")
print(f"Total cells: {len(nb['cells'])}")
