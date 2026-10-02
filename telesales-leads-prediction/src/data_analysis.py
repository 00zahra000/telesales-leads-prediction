"""Read-only EDA: basic columns, imbalance and associations by purchase history."""

import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.db import create_db_engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "reports"
CHARTS = PROJECT_ROOT / "charts"
ANALYSIS_REPORT = PROJECT_ROOT / "analysis-report"


def load_raw_leads(connection):
    return pd.read_sql_query(text("SELECT * FROM raw_leads ORDER BY id"), connection)


def markdown_table(frame):
    """Render small tables without an extra Markdown dependency."""
    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            value = f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows.extend("| " + " | ".join(cell(v) for v in row) + " |"
                for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def summarize_dataset(leads):
    """Inspect every column without choosing predictors or changing values."""
    return pd.DataFrame({
        "column": leads.columns,
        "dtype": leads.dtypes.astype(str).values,
        "nonmissing_count": leads.notna().sum().values,
        "missing_count": leads.isna().sum().values,
        "missing_pct": leads.isna().mean().mul(100).values,
        "unique_nonmissing": leads.nunique(dropna=True).values,
    })


def save_eda(leads):
    REPORTS.mkdir(parents=True, exist_ok=True)
    columns = summarize_dataset(leads)
    numeric = leads.select_dtypes(include="number").drop(columns="id", errors="ignore")
    numeric_summary = numeric.describe().T.reset_index().rename(columns={"index": "column"})
    # Surrogate keys and load timestamps would hide duplicated source records.
    source = leads.drop(columns=["id", "loaded_at"], errors="ignore")
    repeated = leads.lead_id.notna() & leads.lead_id.duplicated(keep=False)
    target = leads.completed_purchase.value_counts(dropna=False).rename_axis("completed_purchase").reset_index(name="rows")
    target["pct_rows"] = target.rows / len(leads) * 100
    dates = leads.select_dtypes(include=["datetime", "datetimetz"])
    date_summary = pd.DataFrame({"column": dates.columns,
                                 "earliest": dates.min().values,
                                 "latest": dates.max().values})
    sections = [
        "# Step 1: Basic Exploratory Data Analysis", "",
        "Source: PostgreSQL `raw_leads`, read in a read-only repeatable-read transaction.", "",
        f"Rows: **{len(leads):,}**. Columns: **{len(leads.columns)}**.", "",
        "## Columns", "", markdown_table(columns), "",
        "## First five rows", "", markdown_table(leads.head()), "",
        "## Basic duplicate counts", "",
        f"- Duplicate source rows beyond the first: {int(source.duplicated().sum()):,}.",
        f"- Repeated nonmissing Lead IDs: {leads.loc[repeated, 'lead_id'].nunique():,}.",
        f"- Rows belonging to repeated Lead IDs: {int(repeated.sum()):,}.", "",
        "These are counts only; no records are removed.", "",
        "## Numeric summaries", "",
        "Database surrogate `id` is omitted from numeric statistics; binary fields are included as stored.", "",
        markdown_table(numeric_summary), "",
        "## Target counts", "", markdown_table(target), "",
        "## Timestamp ranges", "", markdown_table(date_summary), "",
        "## Categorical values", "",
        "Show the five most common values per categorical column, excluding the Lead ID identifier. "
        "Counts include missing values; percentages use all rows.", "",
    ]
    for column in leads.select_dtypes(include=["object", "string"]).columns:
        if column == "lead_id":
            continue
        counts = leads[column].value_counts(dropna=False).head(5).rename_axis("value").reset_index(name="rows")
        counts["pct_rows"] = counts.rows / len(leads) * 100
        sections.extend([f"### {column}", "", markdown_table(counts), ""])
    sections.extend(["This step describes the raw dataset only. No imputation, deduplication, "
                     "feature selection, leakage assessment, or model fitting is performed.", ""])
    columns.to_csv(REPORTS / "eda_columns.csv", index=False)
    numeric_summary.to_csv(REPORTS / "eda_numeric_summary.csv", index=False)
    (REPORTS / "eda_report.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"Dataset: {len(leads):,} rows, {len(leads.columns)} columns.")
    print("\nColumn overview:\n" + columns.to_string(index=False))
    print("\nTarget counts:\n" + target.to_string(index=False))
    print("\nTimestamp ranges:\n" + date_summary.to_string(index=False))
    print(f"\nSaved basic EDA to {REPORTS / 'eda_report.md'}")


def cramers_v(first, second):
    """Uncorrected nominal association, ignoring pairwise missing observations.

    Zero means no observed association; one means perfect association. It has
    no sign and is not comparable to the signed numeric correlation scale.
    """
    complete = pd.DataFrame({"first": first, "second": second}).dropna()
    counts = pd.crosstab(complete["first"], complete["second"]).to_numpy(dtype=float)
    if counts.size == 0 or min(counts.shape) < 2:
        return np.nan
    total = counts.sum()
    expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / total
    chi_squared = ((counts - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi_squared / (total * (min(counts.shape) - 1))))


# Review flags, not cleaning rules: legitimate business tails must be preserved.
EXTREME_INTERPRETATIONS = {
    "minutes_since_abandonment": "Long delays may represent stale leads; verify abandonment timestamp and scoring cutoff.",
    "days_to_policy_expiry": "Negative days can mean an expired policy; distant dates need policy-date and unit checks.",
    "price": "Product and coverage mix can explain high quotes; investigate currency/units and compare within product.",
    "discount_percent": "Campaigns can explain unusual discounts; confirm percentage units and pre-score offer terms.",
    "sessions_last_7d": "Heavy engagement, repeat tracking or bots can produce large counts; verify event definitions and window cutoff.",
    "offer_views_last_7d": "Comparison shopping or repeated events may explain high views; investigate tracking and pre-score window.",
    "price_comparisons_last_7d": "Intensive shopping may be legitimate; check repeated tracking and historical window boundaries.",
    "days_since_last_visit": "Long dormancy may be legitimate; verify last known visit and score-time calculation.",
    "expected_margin": "Product economics can explain tails; verify units and quote-time estimation rather than realized profit.",
}


def analyze_outliers(leads):
    """Compute strict 1.5-IQR fences using nonmissing numeric predictor values."""
    excluded = {"id", "lead_id", "completed_purchase", "loaded_at",
                "has_previous_purchase", "visited_offer_page", "incoming_call_last_24h"}
    rows = []
    for column in leads.select_dtypes(include="number").columns:
        if column in excluded:
            continue
        values = leads[column].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        below, above = int((values < lower).sum()), int((values > upper).sum())
        interpretation = EXTREME_INTERPRETATIONS.get(column, "Confirm business meaning and units before interpreting extreme values.")
        if values.empty:
            interpretation += " No observed values: fences are undefined."
        elif iqr == 0:
            interpretation += " IQR is zero: fences collapse; flags indicate departures from the central value."
        rows.append({"feature": column, "observed": len(values), "missing": int(leads[column].isna().sum()),
                     "q1": q1, "q3": q3, "iqr": iqr, "lower_fence": lower, "upper_fence": upper,
                     "below_fence": below, "above_fence": above, "outside_fences": below + above,
                     "outside_pct": 100 * (below + above) / len(values) if len(values) else np.nan,
                     "business_interpretation": interpretation})
    return pd.DataFrame(rows)


def assess_feature_leakage(leads):
    """Explicit provenance assessment; historical presence is not availability proof."""
    # availability, risk, recommendation, reason; conditional inclusion requires confirmation.
    decisions = {
        "id": ("NO", "HIGH", "EXCLUDE", "Database surrogate key and ingestion order, not business information."),
        "lead_id": ("YES: identifier", "HIGH", "EXCLUDE", "Use for joins and entity-safe splits only; repeated IDs must not cross splits."),
        "loaded_at": ("NO", "HIGH", "EXCLUDE", "Ingestion timestamp is created after collection and has no scoring-time meaning."),
        "completed_purchase": ("NO: future outcome", "HIGH", "EXCLUDE", "Prediction target; never a predictor. Confirm the outcome observation horizon."),
        "created_at": ("LIKELY: confirm immutable", "LOW", "INCLUDE", "Lead creation should precede abandonment; use for chronological splitting. Calendar predictors require timestamp confirmation."),
        "channel": ("LIKELY: confirm original attribution", "LOW", "INCLUDE", "Initial acquisition channel should be known before abandonment; verify it is not reassigned after conversion."),
        "partner": ("LIKELY: confirm original attribution", "LOW", "INCLUDE", "Acquisition partner should be known before abandonment; verify attribution is not updated after purchase."),
        "product_type": ("UNKNOWN: funnel snapshot", "MEDIUM", "INVESTIGATE", "Confirm abandoned-funnel intent rather than final purchased product."),
        "device": ("UNKNOWN: event snapshot", "MEDIUM", "INVESTIGATE", "Require pre-score device, not device recorded during subsequent purchase."),
        "city": ("UNKNOWN: profile snapshot", "MEDIUM", "INVESTIGATE", "Confirm city was captured before scoring, not completed during purchase; preserve historical missingness."),
        "insurance_company": ("UNKNOWN: quote snapshot", "HIGH", "INVESTIGATE", "Confirm intended insurer was known before scoring, not the final purchased insurer."),
        "payment_type": ("UNKNOWN: funnel snapshot", "HIGH", "INVESTIGATE", "Payment choice may be assigned at checkout; require the pre-abandonment choice."),
        "minutes_since_abandonment": ("CONDITIONAL: as of score", "MEDIUM", "INVESTIGATE", "Calculate from the last known abandonment to historical scoring time, not extraction time or a later abandonment."),
        "days_to_policy_expiry": ("CONDITIONAL: known policy date", "MEDIUM", "INVESTIGATE", "Use expiry known at scoring and calculate days at that moment; exclude subsequent renewal updates."),
        "price": ("UNKNOWN: quote snapshot", "HIGH", "INVESTIGATE", "Require a quote available before scoring, not final transaction price or later repricing."),
        "discount_percent": ("UNKNOWN: offer snapshot", "HIGH", "INVESTIGATE", "Require pre-score discount; exclude later Telesales incentives and final transaction discounts."),
        "has_previous_purchase": ("CONDITIONAL: strictly historical", "HIGH", "INVESTIGATE", "All counted purchases must precede scoring and exclude this lead's target purchase; confirm identity matching."),
        "visited_offer_page": ("CONDITIONAL: pre-score events", "HIGH", "INVESTIGATE", "Freeze flag at scoring; exclude subsequent revisits, contact responses and purchase activity."),
        "incoming_call_last_24h": ("UNKNOWN: historical window", "HIGH", "INVESTIGATE", "Confirm incoming-call definition and a 24-hour window ending at scoring; exclude calls after prioritization."),
        "sessions_last_7d": ("CONDITIONAL: historical window", "HIGH", "INVESTIGATE", "Seven-day event window must end at scoring; confirm event arrival latency and exclude later sessions."),
        "offer_views_last_7d": ("CONDITIONAL: historical window", "HIGH", "INVESTIGATE", "Seven-day window must contain only views known at scoring, excluding later funnel activity."),
        "price_comparisons_last_7d": ("CONDITIONAL: historical window", "HIGH", "INVESTIGATE", "Seven-day window ends at scoring; exclude subsequent comparisons and account for event arrival latency."),
        "days_since_last_visit": ("CONDITIONAL: as of score", "HIGH", "INVESTIGATE", "Calculate using latest known pre-score visit, not extraction date or a later visit."),
        "expected_margin": ("UNKNOWN: estimation inputs", "HIGH", "INVESTIGATE", "Require pre-score quote-time estimate with historical inputs, not realized profit or completed-policy information. Strong price association suggests redundancy, not proof of leakage."),
    }
    from src.features import ATTRIBUTION_ASSUMPTION, SELECTED_FEATURES

    rows = []
    for column in leads.columns:
        availability, risk, recommendation, reason = decisions.get(
            column, ("UNKNOWN", "HIGH", "INVESTIGATE", "No documented provenance; confirm business meaning and scoring-time availability."))
        if column in SELECTED_FEATURES:
            availability = "ASSUMED: pre-score snapshot (user-authorized experiment)"
            recommendation = "INCLUDE"
            reason = f"{reason} Experimental inclusion: {ATTRIBUTION_ASSUMPTION}"
        rows.append({"feature_name": column, "feature_type": str(leads[column].dtype),
                     "business_meaning": column.replace("_", " "), "available_at_scoring": availability,
                     "leakage_risk": risk, "recommendation": recommendation, "reason": reason})
    return pd.DataFrame(rows)


def save_relationships(leads):
    """Step 2: describe imbalance and compare associations within history groups.

    No resampling or predictor selection. Prior purchase history defines the
    groups; completed_purchase remains the outcome inside each group.
    """
    target, history = "completed_purchase", "has_previous_purchase"
    for column in [target, history]:
        if not leads[column].isin([0, 1]).all():
            raise ValueError(f"{column} requires complete binary values for grouped analysis.")
    CHARTS.mkdir(parents=True, exist_ok=True)
    counts = leads[target].value_counts().reindex([0, 1], fill_value=0)
    distribution = counts.rename_axis(target).reset_index(name="rows")
    distribution["pct_rows"] = distribution.rows / len(leads) * 100
    groups = leads.groupby(history)[target].agg(rows="size", purchasers="sum", purchase_rate="mean").reset_index()
    groups["non_purchasers"] = groups.rows - groups.purchasers
    numeric = leads.select_dtypes(include="number").drop(columns=["id", history], errors="ignore").columns.tolist()
    categorical = [c for c in leads.select_dtypes(include=["object", "string"]).columns
                   if c != "lead_id"]
    associations, category_rates = [], []
    for previous, group in leads.groupby(history):
        active_numeric = [c for c in numeric if group[c].nunique() > 1]
        correlation = group[active_numeric].corr(method="spearman", min_periods=3)
        for first, second in combinations(active_numeric, 2):
            is_target = target in [first, second]
            if first == target:
                first, second = second, first
            value = correlation.loc[first, second]
            associations.append({history: int(previous), "analysis": "feature_target" if is_target else "feature_pair",
                                 "column_1": first, "column_2": second, "method": "Spearman",
                                 "association": value, "paired_rows": int(group[[first, second]].notna().all(axis=1).sum()),
                                 "major_pair": bool(not is_target and abs(value) >= 0.7)})
        for first, second in combinations(categorical + [target], 2):
            value = cramers_v(group[first], group[second])
            associations.append({history: int(previous), "analysis": "feature_target" if second == target else "feature_pair",
                                 "column_1": first, "column_2": second, "method": "Cramer's V",
                                 "association": value, "paired_rows": int(group[[first, second]].notna().all(axis=1).sum()),
                                 "major_pair": bool(second != target and value >= 0.5)})
        for column in categorical:
            rates = group.groupby(column, dropna=False)[target].agg(rows="size", purchasers="sum", purchase_rate="mean").reset_index()
            rates = rates.rename(columns={column: "category"})
            rates.insert(0, "feature", column)
            rates.insert(0, history, int(previous))
            category_rates.append(rates)
        if len(active_numeric) >= 2:
            fig, ax = plt.subplots(figsize=(11, 9))
            colors = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_xticks(range(len(active_numeric)), active_numeric, rotation=90)
            ax.set_yticks(range(len(active_numeric)), active_numeric)
            ax.set_title(f"Spearman correlations: has_previous_purchase={previous} (n={len(group):,})")
            fig.colorbar(colors, ax=ax, label="Spearman correlation")
            fig.tight_layout()
            fig.savefig(CHARTS / f"eda_correlations_previous_purchase_{previous}.png", dpi=150)
            plt.close(fig)
    association_table = pd.DataFrame(associations)
    rates_table = pd.concat(category_rates, ignore_index=True) if category_rates else pd.DataFrame()
    for table, filename in [(distribution, "eda_target_distribution.csv"),
                            (groups, "eda_purchase_history_summary.csv"),
                            (association_table, "eda_associations.csv"),
                            (rates_table, "eda_category_purchase_rates.csv")]:
        table.to_csv(REPORTS / filename, index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["No completed purchase", "Completed purchase"], counts.values)
    ax.set(ylabel="Leads", title="Current purchase outcome distribution")
    for index, row in distribution.iterrows():
        ax.text(index, row.rows, f"{int(row.rows):,} ({row.pct_rows:.2f}%)", ha="center", va="bottom")
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(CHARTS / "eda_target_imbalance.png", dpi=150)
    plt.close(fig)
    positive_rate = float(leads[target].mean())
    ratio = float(counts.loc[0] / counts.loc[1]) if counts.loc[1] else np.nan
    major = association_table.loc[association_table.major_pair].copy()
    sections = ["# Step 2: Imbalance and Associations by Purchase History", "",
                "Groups use `has_previous_purchase`, not `completed_purchase`. "
                "The latter is the current lead's outcome, and is evaluated within each group.", "",
                "## Class distribution", "", markdown_table(distribution), "",
                f"Purchase rate: **{positive_rate:.2%}**. Non-purchase/purchase count ratio: **{ratio:.2f}:1**.", "",
                "These counts quantify imbalance; no weighting, resampling or cleaning is applied.", "",
                "## Purchase-history groups", "", markdown_table(groups), "",
                "## Major feature-pair associations", "",
                "Numeric pairs are flagged at |Spearman| >= 0.7; nominal pairs at Cramer's V >= 0.5. "
                "These are descriptive review thresholds, not significance tests or automatic exclusion rules.", "",
                markdown_table(major) if not major.empty else "No pairs meet those review thresholds.", "",
                "## Associations with the current purchase outcome", "",
                "Compare methods separately: Spearman is signed, while Cramer's V is unsigned. "
                "Category purchase rates are saved separately to show direction and sample sizes.", ""]
    for previous in groups[history]:
        targets = association_table.loc[(association_table[history] == previous) & association_table.analysis.eq("feature_target")].copy()
        targets["absolute_association"] = targets.association.abs()
        targets = targets.sort_values(["method", "absolute_association"], ascending=[True, False]).drop(columns=["analysis", "major_pair", "absolute_association"])
        sections.extend([f"### has_previous_purchase={previous}", "", markdown_table(targets), ""])
    sections.extend(["## Interpretation limits", "",
                     "Identifiers and timestamps are excluded from associations. Purchase history is constant "
                     "inside each group and is not correlated with itself. Missing values are omitted per pair, "
                     "with usable row counts reported; constant columns have undefined association. "
                     "Numeric-nominal feature pairs are not measured here. Cramer's V is uncorrected and descriptive.", "",
                     "Large feature-pair association suggests possible redundancy, not predictive value. "
                     "Outcome associations nominate investigation candidates; weak marginal correlation does not "
                     "rule out nonlinear effects. No feature is selected automatically. Check scoring-time "
                     "availability and leakage before using any candidate, including prior-purchase history. "
                     "Repeated Lead IDs affect independence; these are row-based exploratory findings, not "
                     "causal effects or held-out validation results.", ""])
    ANALYSIS_REPORT.mkdir(parents=True, exist_ok=True)
    outliers = analyze_outliers(leads)
    assessment = assess_feature_leakage(leads)
    outliers.to_csv(ANALYSIS_REPORT / "eda_outliers.csv", index=False)
    assessment.to_csv(REPORTS / "feature_assessment.csv", index=False)
    sections.extend([
        "## Outlier / extreme-value analysis", "",
        "For each nonmissing numeric predictor: IQR = Q3 - Q1; lower fence = Q1 - 1.5 × IQR; "
        "upper fence = Q3 + 1.5 × IQR. Counts use strict inequalities; percentages use observed values. "
        "IDs, target and the three binary flags are excluded. Missing values are neither imputed nor counted as extremes.", "",
        markdown_table(outliers), "",
        "These are investigation flags, not data-error labels. No values are removed, clipped or transformed. "
        "Pooled fences can reflect product mix, skew and discrete counts; investigate context before acting.", "",
        "## Feature leakage assessment", "",
        "Prediction moment: after funnel abandonment, when a lead becomes eligible for Telesales prioritization. "
        "A column being present in historical data does not automatically make it valid for modeling. "
        "Only values already known at that moment may be used. This dataset contains no event-level "
        "snapshots proving those cutoffs, so conditional availability needs business confirmation.", "",
        "INCLUDE means a candidate under the stated assumption, not automatic model selection. "
        "created_at is used for splitting; identifiers, target and ingestion metadata are excluded as predictors. "
        "INVESTIGATE features should remain out of training until their provenance is confirmed. "
        "Existing explicit model feature lists are not changed by this assessment.", "",
        markdown_table(assessment), "",
        "Required confirmations: historical scoring timestamp, frozen quote/profile attribution, event-window "
        "cutoffs and arrival latency, strictly prior purchase history, margin inputs, and target observation horizon. "
        "Correlations with the target cannot establish whether a feature leaks future information.", "",
    ])
    (ANALYSIS_REPORT / "eda_relationships.md").write_text("\n".join(sections), encoding="utf-8")
    print("\nPurchase-history groups:\n" + groups.to_string(index=False))
    print("\nMajor feature-pair associations:\n" + major.to_string(index=False))
    print(f"\nSaved grouped association analysis to {ANALYSIS_REPORT / 'eda_relationships.md'}")


def main():
    engine = None
    try:
        engine = create_db_engine()
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
            connection.execute(text("SET TRANSACTION READ ONLY"))
            leads = load_raw_leads(connection)
        if leads.empty:
            raise ValueError("raw_leads is empty; run src.load_data first.")
        save_eda(leads)
        save_relationships(leads)
    except Exception as exc:
        print(f"EDA failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
