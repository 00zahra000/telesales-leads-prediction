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

from src.logging_config import logger

from src.db import create_db_engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "reports"
CHARTS = PROJECT_ROOT / "charts"


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
    dates = leads.select_dtypes(include=["datetime", "datetimetz"])
    date_summary = pd.DataFrame({"column": dates.columns,
                                 "earliest": dates.min().values,
                                 "latest": dates.max().values})
    sections = [
        "# Step 1: Basic Exploratory Data Analysis", "",
        "Source: PostgreSQL `raw_leads`, read in a read-only repeatable-read transaction.", "",
        f"Rows: **{len(leads):,}**. Columns: **{len(leads.columns)}**.", "",
        "## Columns", "", markdown_table(columns), "",
        "## Basic duplicate counts", "",
        f"- Duplicate source rows beyond the first: {int(source.duplicated().sum()):,}.",
        f"- Repeated nonmissing Lead IDs: {leads.loc[repeated, 'lead_id'].nunique():,}.",
        f"- Rows belonging to repeated Lead IDs: {int(repeated.sum()):,}.", "",
        "These are counts only; no records are removed.", "",
        "## Numeric summaries", "",
        "Database surrogate `id` is omitted from numeric statistics; binary fields are included as stored.", "",
        markdown_table(numeric_summary), "",
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
    (REPORTS / "eda_report.md").write_text("\n".join(sections), encoding="utf-8")


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
                "## Associations with the current purchase outcome", "",
                "Compare methods separately: Spearman is signed, while Cramer's V is unsigned. "
                "Category purchase rates below show direction and sample sizes.", ""]
    for previous in groups[history]:
        targets = association_table.loc[(association_table[history] == previous) & association_table.analysis.eq("feature_target")].copy()
        targets["absolute_association"] = targets.association.abs()
        targets = targets.sort_values(["method", "absolute_association"], ascending=[True, False]).drop(columns=["analysis", "major_pair", "absolute_association"])
        sections.extend([f"### has_previous_purchase={previous}", "", markdown_table(targets), ""])
    sections.extend(["## Interpretation limits", "",
                     "Associations describe observed relationships; they do not establish causation or model usefulness. "
                     "Missing values are omitted per pair, and repeated lead IDs limit independence. "
                     "Spearman is signed; Cramer's V is unsigned; numeric–categorical pairs are not measured.", ""])
    outliers = analyze_outliers(leads)
    if major.empty:
        observation = "No measured feature pairs showed a strong association in either purchase-history group."
    else:
        pairs = []
        for (first, second, method), pair in major.groupby(["column_1", "column_2", "method"]):
            pairs.append(f"`{first}` and `{second}` ({method}: "
                         f"{pair.association.min():.4f}–{pair.association.max():.4f})")
        observation = "Strong associations were observed between " + "; ".join(pairs) + "."
    category_summary = []
    for (previous, feature), rates in rates_table.groupby([history, "feature"]):
        eligible = rates.loc[rates.rows >= 100].sort_values(["purchase_rate", "rows", "category"],
                                                          na_position="last")
        if len(eligible) < 2:
            continue
        low, high = eligible.iloc[0], eligible.iloc[-1]
        category_summary.append({history: previous, "feature": feature,
                                 "lowest_rate_category": low.category, "lowest_rate_rows": low.rows,
                                 "lowest_purchase_pct": low.purchase_rate * 100,
                                 "highest_rate_category": high.category, "highest_rate_rows": high.rows,
                                 "highest_purchase_pct": high.purchase_rate * 100})
    sections.extend(["## Category purchase-rate summary", "",
                     "For each feature and purchase-history group, show the lowest and highest observed "
                     "purchase rates among categories with at least 100 rows; this reporting cutoff reduces "
                     "emphasis on small groups and does not establish statistical significance. Rates are percentages.", "",
                     markdown_table(pd.DataFrame(category_summary)), "",
                     "## Association observations", "", observation + " "
                     "All other measured feature pairs with a defined association fell below the review thresholds "
                     "of absolute Spearman correlation 0.7 or Cramer's V 0.5.", ""])
    sections.extend([
        "## Outlier / extreme-value analysis", "",
        "Values beyond 1.5 × IQR from the lower or upper quartile are flagged for review. "
        "Percentages use nonmissing values; identifiers, the target and binary flags are excluded.", "",
        markdown_table(outliers[["feature", "outside_fences", "outside_pct", "business_interpretation"]]
                       .rename(columns={"outside_fences": "flagged_rows", "outside_pct": "flagged_pct"})), "",
        "These are investigation flags, not data-error labels. No values are removed, clipped or transformed. "
        "Pooled fences can reflect product mix, skew and discrete counts; investigate context before acting.", "",
    ])
    with (REPORTS / "eda_report.md").open("a", encoding="utf-8") as report:
        report.write("\n" + "\n".join(sections))
    logger.success("Saved EDA report for {:,} rows and {} columns to {}.",
                   len(leads), len(leads.columns), REPORTS / "eda_report.md")


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
        logger.error("EDA failed: {}", exc)
        return 1
    finally:
        if engine is not None:
            engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
