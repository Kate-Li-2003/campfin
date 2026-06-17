import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def intro(mo):
    mo.md("""
    # Cleaning notebook

    notebook of data cleaning applied as an example to a static data-file:
    **Power Search** export `governor_all_donations_61526.csv`. (note that raw data dumps, for other races / in bulk, will be pulled directly from cal-access)
    """)
    return


@app.cell
def imports():
    import marimo as mo
    import pandas as pd
    from pathlib import Path

    # Project root is two levels up from code/04_donations_data_pull/
    _BASE_DIR = Path(__file__).resolve().parent.parent.parent if "__file__" in dir() else Path.cwd().parent.parent
    DATA_DIR = _BASE_DIR / "data"
    RAW_CSV = DATA_DIR / "governor_all_donations_61526.csv"
    CLEAN_CSV = DATA_DIR / "governor_all_donations_61526_cleaned.csv"
    return CLEAN_CSV, RAW_CSV, mo, pd


@app.cell
def load_raw(RAW_CSV, mo, pd):
    df_raw = pd.read_csv(RAW_CSV, dtype=str, keep_default_na=False)
    mo.vstack([
        mo.md(f"**{len(df_raw):,} rows × {df_raw.shape[1]} columns**"),
        df_raw.head(8),
    ])
    return (df_raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Cleaning
    """)
    return


@app.cell(hide_code=True)
def step1_md(mo):
    mo.md("""
    ## 1. Drop the export footer rows

    - drop all obs that lack a `Recipient Committee ID` (e.g., drops footer rows)
    """)
    return


@app.cell
def step1_drop_footer(df_raw, mo):
    _has_committee = df_raw["Recipient Committee ID"].str.strip() != ""
    df_footer_dropped = df_raw[_has_committee].copy()
    _dropped = df_raw[~_has_committee]

    mo.vstack([
        mo.md(f"Dropped **{len(_dropped)}** footer rows; **{len(df_footer_dropped):,}** contribution rows remain."),
        _dropped[["Transaction Type", "Office", "Recipient Committee ID"]],
    ])
    return (df_footer_dropped,)


@app.cell(hide_code=True)
def step2_md(mo):
    mo.md("""
    ## 2. Convert types

    - **Amount** → float
    - **Start Date / End Date** → datetime
    - **Cycle** → integer year
    """)
    return


@app.cell
def step2_types(df_footer_dropped, mo, pd):
    df_typed = df_footer_dropped.copy()
    df_typed["Amount"] = pd.to_numeric(df_typed["Amount"], errors="coerce")
    df_typed["Start Date"] = pd.to_datetime(df_typed["Start Date"], errors="coerce")
    df_typed["End Date"] = pd.to_datetime(df_typed["End Date"], errors="coerce")
    df_typed["Cycle"] = pd.to_numeric(df_typed["Cycle"], errors="coerce").astype("Int64")

    _report = pd.DataFrame({
        "column": ["Amount", "Start Date", "End Date", "Cycle"],
        "dtype": [str(df_typed[c].dtype) for c in ["Amount", "Start Date", "End Date", "Cycle"]],
        "n_missing_after": [int(df_typed[c].isna().sum()) for c in ["Amount", "Start Date", "End Date", "Cycle"]],
    })

    mo.vstack([
        mo.md(f"Total contributions: **${df_typed['Amount'].sum():,.0f}**"),
        _report,
    ])
    return (df_typed,)


@app.cell(hide_code=True)
def step3_md(mo):
    mo.md("""
    ## 3. Text normalization

    Two cleanups on the string columns:

    1. **Strip** stray leading/trailing whitespace from every text column.
    2. **Unify missing-value placeholders** in Employer / Occupation — e.g., collapse `"n/a"`, `"N/A"`, `"na"`, `"none"` into the same NA value. Preserve real values like *Retired* and *Not Employed* are left untouched.
    """)
    return


@app.cell
def step3_normalize(df_typed, mo):
    PLACEHOLDERS = {"n/a", "na", "none", ""}

    df_norm = df_typed.copy()
    _typed_cols = {"Amount", "Start Date", "End Date", "Cycle"}
    _str_cols = [c for c in df_norm.columns if c not in _typed_cols]
    for _c in _str_cols:
        df_norm[_c] = df_norm[_c].str.strip()

    for _c in ["Contributor Employer", "Contributor Occupation"]:
        _is_ph = df_norm[_c].str.lower().isin(PLACEHOLDERS)
        df_norm.loc[_is_ph, _c] = ""

    _before = df_typed["Contributor Employer"].value_counts().head(5)
    _after = df_norm["Contributor Employer"].replace("", "(blank)").value_counts().head(5)
    mo.hstack([
        mo.vstack([mo.md("**Employer — before**"), _before]),
        mo.vstack([mo.md("**Employer — after**"), _after]),
    ])
    return (df_norm,)


@app.cell(hide_code=True)
def step4_md(mo):
    mo.md("""
    ## 4. Inspect transaction types

    Note that Power Search is more granular than Cal-Access (e.g., Power Search has late contribution/loan/monetary/non-monetary; Cal-Access only has late monetary contribution/non-monetary)
    """)
    return


@app.cell
def step4_txn_types(df_norm):
    _tt = (
        df_norm.groupby("Transaction Type")["Amount"]
        .agg(n="count", total="sum")
        .sort_values("n", ascending=False)
        .reset_index()
    )
    _tt["total"] = _tt["total"].map(lambda v: f"${v:,.0f}")
    _tt
    return


@app.cell(hide_code=True)
def step5_md(mo):
    mo.md("""
    ## 5. Date sanity check

    Contributions should fall within the 2025–2026 cycle. Flag any row dated in
    the future so it can be reviewed — usually a data-entry typo in
    the filing
    """)
    return


@app.cell
def step5_date_sanity(df_norm, mo, pd):
    _today = pd.Timestamp.today().normalize()
    _future = df_norm[df_norm["Start Date"] > _today]
    mo.vstack([
        mo.md(f"**{len(_future)}** row(s) dated after {_today.date()}:"),
        _future[["Start Date", "Recipient Committee", "Contributor Name", "Amount"]],
    ])
    return


@app.cell(hide_code=True)
def step6_md(mo):
    mo.md("""
    ## 6. Potential duplicate contributions

    Rows identical on **date + amount + contributor + committee + type** are
    likely the same gift reported twice (re-filed amendments, overlapping
    exports). They are only *candidates* — two separate $100 gifts from one donor
    on one day are legitimate & aren't dropped for the time being
    """)
    return


@app.cell
def step6_dupes(df_norm, mo):
    KEY_COLS = ["Start Date", "Amount", "Contributor Name", "Recipient Committee ID", "Transaction Type"]
    _dup_mask = df_norm.duplicated(subset=KEY_COLS, keep=False)
    _dups = df_norm[_dup_mask].sort_values(KEY_COLS)
    mo.vstack([
        mo.md(f"**{int(df_norm.duplicated(subset=KEY_COLS).sum()):,}** rows are exact repeats of an earlier row "
              f"({_dup_mask.sum():,} rows involved in a duplicate group)."),
        _dups[["Start Date", "Amount", "Contributor Name", "Recipient Committee", "Transaction Type"]].head(10),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Donation Size Filter
    """)
    return


@app.cell
def _(df_norm):
    large_donations = df_norm[df_norm["Amount"] >= 5000]
    len(large_donations)
    large_donations
    return (large_donations,)


@app.cell(hide_code=True)
def step7_md(mo):
    mo.md("""
    # Export
    """)
    return


@app.cell
def assemble(large_donations, mo):
    df_cleaned_filter = large_donations.copy()

    mo.vstack([
        mo.md(f"### Cleaned dataset — **{len(df_cleaned_filter):,} rows × {df_cleaned_filter.shape[1]} cols**"),
        df_cleaned_filter.head(8),
    ])
    return (df_cleaned_filter,)


@app.cell
def export(CLEAN_CSV, df_cleaned_filter, mo):
    df_cleaned_filter.to_csv(CLEAN_CSV, index=False)
    mo.md(f"Wrote **{len(df_cleaned_filter):,}** rows → `{CLEAN_CSV.name}`")
    return


if __name__ == "__main__":
    app.run()
