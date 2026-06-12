"""
Classifier Evaluation
======================
Three complementary error-rate checks for the contributor industry classifier:

  1. Probe test       – classify known orgs with unambiguous industry membership;
                        report hit rate per category and overall.
  2. Leave-one-out CV – for each seed, remove it, retrain KNN, predict it back;
                        measures how well the KNN generalises within the seed set.
  3. Spot-check table – sample rows from the classified CSV (weighted toward
                        low-confidence predictions) for human review.

Run:
  /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 evaluate_classifier.py
"""

import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ── import the taxonomy and helpers from the main classifier ──────────────────
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "classifier",
    "/Users/kateli/Desktop/CalMatters/classify_contributors.py",
)
clf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clf)

INDUSTRY_SEEDS = clf.INDUSTRY_SEEDS
KEYWORD_MAP    = clf.KEYWORD_MAP
OCC_HINTS      = clf.OCC_HINTS
preprocess     = clf.preprocess
keyword_classify = clf.keyword_classify
build_knn_model  = clf.build_knn_model
knn_predict      = clf.knn_predict


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PROBE TEST
#     Well-known orgs whose correct industry is unambiguous.
#     Any reasonable classifier should get these right.
# ─────────────────────────────────────────────────────────────────────────────

PROBES: list[tuple[str, str]] = [
    # Tech
    ("Google LLC",                      "Tech"),
    ("Apple Inc",                       "Tech"),
    ("Microsoft Corporation",           "Tech"),
    ("Salesforce Inc",                  "Tech"),
    ("Oracle Corporation",              "Tech"),
    # Crypto/AI
    ("Coinbase Global Inc",             "Crypto/AI"),
    ("Ripple Labs",                     "Crypto/AI"),
    ("OpenAI",                          "Crypto/AI"),
    ("Anthropic PBC",                   "Crypto/AI"),
    ("Blockchain Capital",              "Crypto/AI"),
    # Ridesharing
    ("Uber Technologies Inc",           "Ridesharing"),
    ("Lyft Inc",                        "Ridesharing"),
    ("DoorDash Inc",                    "Ridesharing"),
    ("Instacart",                       "Ridesharing"),
    # Oil
    ("Chevron Corporation",             "Oil"),
    ("ExxonMobil Corporation",          "Oil"),
    ("California Resources Corporation","Oil"),
    ("ConocoPhillips",                  "Oil"),
    ("Aera Energy LLC",                 "Oil"),
    # Utilities
    ("Pacific Gas and Electric",        "Utilities"),
    ("Southern California Edison",      "Utilities"),
    ("San Diego Gas and Electric",      "Utilities"),
    ("Los Angeles Department of Water and Power", "Utilities"),
    # Labor
    ("SEIU Local 1000",                 "Labor"),
    ("Teamsters Local 396",             "Labor"),
    ("United Auto Workers",             "Labor"),
    ("Western States Regional Council of Carpenters", "Labor"),
    ("AFL-CIO",                         "Labor"),
    # Lawyers/Lobbying
    ("Latham and Watkins LLP",          "Lawyers/Lobbying"),
    ("Gibson Dunn and Crutcher LLP",    "Lawyers/Lobbying"),
    ("Jones Day",                       "Lawyers/Lobbying"),
    ("Brownstein Hyatt Farber Schreck", "Lawyers/Lobbying"),
    # Health Care
    ("Kaiser Permanente",               "Health Care"),
    ("Cedars-Sinai Medical Center",     "Health Care"),
    ("Pfizer Inc",                      "Health Care"),
    ("Blue Shield of California",       "Health Care"),
    ("Planned Parenthood",              "Health Care"),
    # Ag
    ("Paramount Farming Company",       "Ag"),
    ("Trinchero Family Estates",        "Ag"),
    ("Sunsweet Growers Inc",            "Ag"),
    ("Foster Farms",                    "Ag"),
    # Real Estate
    ("CBRE Group Inc",                  "Real Estate"),
    ("Lennar Corporation",              "Real Estate"),
    ("KB Home",                         "Real Estate"),
    ("Irvine Company",                  "Real Estate"),
    # Environment
    ("Sierra Club Foundation",          "Environment"),
    ("NRDC Action Fund",                "Environment"),
    ("The Nature Conservancy",          "Environment"),
    ("Earthjustice",                    "Environment"),
    # Political
    ("Steyer for Governor 2026",        "Political"),
    ("Villaraigosa for Governor 2026",  "Political"),
    ("Democratic State Central Committee", "Political"),
    ("California Republican Party",     "Political"),
]


def run_probe_test(word_vec, char_vec, X_train, train_labels):
    print("\n" + "=" * 70)
    print("1.  PROBE TEST  (known orgs → expected category)")
    print("=" * 70)

    results = []
    for org_name, expected in PROBES:
        processed = preprocess(org_name)
        # keyword first
        pred = keyword_classify(processed) or keyword_classify(org_name.lower())
        method = "keyword"
        conf = 0.90
        if not pred:
            pred, conf = knn_predict(processed, word_vec, char_vec, X_train, train_labels, k=3)
            method = "knn"
        correct = (pred == expected)
        results.append({
            "org": org_name,
            "expected": expected,
            "predicted": pred,
            "correct": correct,
            "method": method,
            "conf": conf,
        })

    df = pd.DataFrame(results)
    total   = len(df)
    correct = df["correct"].sum()
    print(f"\n  Overall hit rate: {correct}/{total}  ({correct/total*100:.1f}%)\n")

    # Per-category breakdown
    print(f"  {'Category':<28}  {'Hits':>4}  {'Total':>5}  {'Acc':>6}  Misses")
    print(f"  {'-'*28}  {'-'*4}  {'-'*5}  {'-'*6}  ------")
    for cat in sorted(df["expected"].unique()):
        sub = df[df["expected"] == cat]
        hits = sub["correct"].sum()
        n = len(sub)
        misses = sub[~sub["correct"]][["org","predicted"]].values
        miss_str = "; ".join(f"{o} → {p}" for o, p in misses) if len(misses) else "—"
        print(f"  {cat:<28}  {hits:>4}  {n:>5}  {hits/n*100:>5.0f}%  {miss_str}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LEAVE-ONE-OUT CROSS-VALIDATION ON SEEDS
#     For each seed: remove it → rebuild vectorizers → predict it back.
#     Measures KNN generalisation across the seed corpus.
# ─────────────────────────────────────────────────────────────────────────────

def build_knn_from_corpus(texts, labels):
    import scipy.sparse as sp
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=5000, min_df=1, max_df=0.95, sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5),
        max_features=3000, min_df=1, max_df=0.95, sublinear_tf=True,
    )
    X_word = word_vec.fit_transform(texts)
    X_char = char_vec.fit_transform(texts)
    X = sp.hstack([X_word, X_char])
    return word_vec, char_vec, X


def run_loo_cv():
    print("\n" + "=" * 70)
    print("2.  LEAVE-ONE-OUT CROSS-VALIDATION ON SEED CORPUS")
    print("=" * 70)

    # Flatten seeds
    all_texts, all_labels = [], []
    for industry, seeds in INDUSTRY_SEEDS.items():
        for s in seeds:
            all_texts.append(preprocess(s))
            all_labels.append(industry)

    n = len(all_texts)
    correct_count = 0
    mistakes = []

    for i in range(n):
        held_out_text  = all_texts[i]
        held_out_label = all_labels[i]

        train_texts  = all_texts[:i] + all_texts[i+1:]
        train_labels = all_labels[:i] + all_labels[i+1:]

        # Need at least 2 examples per class or vectorizer degrades; skip
        # single-class situations gracefully
        from collections import Counter
        counts = Counter(train_labels)
        if counts[held_out_label] == 0:
            continue

        try:
            wv, cv, X = build_knn_from_corpus(train_texts, train_labels)
            pred, conf = knn_predict(held_out_text, wv, cv, X, train_labels, k=3)
        except Exception:
            continue

        if pred == held_out_label:
            correct_count += 1
        else:
            mistakes.append((all_texts[i], held_out_label, pred, round(conf, 2)))

    acc = correct_count / n
    print(f"\n  Seed corpus size : {n} examples")
    print(f"  Correct          : {correct_count}")
    print(f"  LOO accuracy     : {acc*100:.1f}%")

    if mistakes:
        print(f"\n  Misclassified seeds ({len(mistakes)}):")
        print(f"  {'Seed text':<40}  {'True':<22}  {'Predicted':<22}  Conf")
        print(f"  {'-'*40}  {'-'*22}  {'-'*22}  ----")
        for seed, true, pred, conf in sorted(mistakes, key=lambda x: x[1]):
            print(f"  {seed[:40]:<40}  {true:<22}  {pred:<22}  {conf:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CONFIDENCE-STRATIFIED SPOT-CHECK
#     Sample rows from the classified CSV — more heavily from the low-confidence
#     tail — so a human reviewer can scan them efficiently.
# ─────────────────────────────────────────────────────────────────────────────

def run_spot_check(n_per_category: int = 5):
    print("\n" + "=" * 70)
    print("3.  CONFIDENCE-STRATIFIED SPOT-CHECK  (sample for human review)")
    print("=" * 70)

    classified = pd.read_csv(
        "/Users/kateli/Desktop/CalMatters/data/governor_race_classified.csv",
        dtype=str,
    )
    classified["Classification_Confidence"] = pd.to_numeric(
        classified["Classification_Confidence"], errors="coerce"
    )

    # Exclude the special-case bucket (trivially correct) and focus on
    # rows where the model actually had to make a decision
    reviewable = classified[
        classified["Classification_Method"].isin(["keyword", "knn", "knn_low_conf", "occupation_hint"])
    ].copy()

    print(f"\n  Reviewable rows (non-special-case): {len(reviewable):,}")
    print(f"\n  Sampling {n_per_category} rows per category, weighted toward low confidence.\n")
    print(f"  {'#':<4}  {'Category':<22}  {'Conf':>5}  {'Method':<16}  "
          f"{'Org / Employer':<38}  Occupation")
    print(f"  {'-'*4}  {'-'*22}  {'-'*5}  {'-'*16}  {'-'*38}  ----------")

    row_num = 1
    for cat in sorted(reviewable["Industry_Category"].unique()):
        sub = reviewable[reviewable["Industry_Category"] == cat].copy()
        # Sort ascending by confidence so worst predictions surface first
        sub = sub.sort_values("Classification_Confidence")
        sample = sub.head(n_per_category)
        for _, r in sample.iterrows():
            org   = str(r.get("Org_Text_Used", ""))[:38]
            occ   = str(r.get("Contributor Occupation", ""))[:20]
            conf  = r["Classification_Confidence"]
            meth  = str(r.get("Classification_Method", ""))
            print(f"  {row_num:<4}  {cat:<22}  {conf:>5.2f}  {meth:<16}  {org:<38}  {occ}")
            row_num += 1
        print()

    # Also show the overall confidence distribution
    print("\n  CONFIDENCE SCORE DISTRIBUTION (non-special-case rows)")
    print(f"  {'Bucket':<18}  {'Rows':>7}  {'%':>6}  Interpretation")
    print(f"  {'-'*18}  {'-'*7}  {'-'*6}  ---------------")
    bins   = [0.0, 0.40, 0.55, 0.70, 0.85, 1.01]
    labels = ["0.00–0.39 (very low)", "0.40–0.54 (low)", "0.55–0.69 (moderate)",
              "0.70–0.84 (good)", "0.85–1.00 (high)"]
    notes  = ["likely wrong", "uncertain", "plausible", "probably right", "keyword/certain"]
    confs  = reviewable["Classification_Confidence"].dropna()
    cuts   = pd.cut(confs, bins=bins, labels=labels, right=False)
    counts = cuts.value_counts().reindex(labels, fill_value=0)
    total  = len(confs)
    for lbl, note in zip(labels, notes):
        n = counts[lbl]
        print(f"  {lbl:<18}  {n:>7,}  {n/total*100:>5.1f}%  {note}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building KNN model …")
    word_vec, char_vec, X_train, train_labels = build_knn_model(k=3)
    print(f"  {len(train_labels)} seed examples across {len(INDUSTRY_SEEDS)} categories.\n")

    run_probe_test(word_vec, char_vec, X_train, train_labels)
    run_loo_cv()
    run_spot_check(n_per_category=5)
