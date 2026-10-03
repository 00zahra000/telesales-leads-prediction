"""Probability, capacity-ranking and threshold evaluation; rerun saved model only."""

import json
import math

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

from src.features import prepare_features

CAPACITIES = [0.05, 0.10, 0.20]


def ranking_metrics(y, probability, fraction):
    """Exact ceil(K*n) capacity, averaging random order within a tied boundary.

    With only acquisition categories many leads share a score. Fractional
    captured positives are expected counts, not invented within-category ranking.
    """
    y, probability = np.asarray(y), np.asarray(probability)
    k = max(1, math.ceil(len(y) * fraction))
    cutoff = np.sort(probability)[-k]
    above, tied = probability > cutoff, probability == cutoff
    slots = k - int(above.sum())
    captured = float(y[above].sum() + slots * y[tied].mean())
    precision = captured / k
    return {"contacted_rows": k, "expected_purchasers": captured,
            "precision": precision, "conversion_rate": precision,
            "recall": captured / float(y.sum()), "lift": precision / float(y.mean()),
            "boundary_tied_rows": int(tied.sum()), "boundary_slots": slots}


def calibration_bins(y, probability):
    # Quantile edges avoid a single uninformative 0.0-0.1 bin for rare outcomes.
    table = pd.DataFrame({"target": np.asarray(y), "probability": np.asarray(probability)})
    edges = np.unique(np.quantile(table.probability, np.linspace(0, 1, 9)))
    if len(edges) < 2:
        table["bin"] = "all"
    else:
        table["bin"] = pd.cut(table.probability, edges, include_lowest=True, duplicates="drop")
    return table.groupby("bin", observed=True).agg(
        rows=("target", "size"), mean_probability=("probability", "mean"),
        observed_rate=("target", "mean")).reset_index()


def compute_metrics(y, probability):
    y, probability = np.asarray(y), np.asarray(probability)
    if len(y) == 0 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Evaluation requires both target classes.")
    if probability.shape != y.shape or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Expected one finite probability in [0,1] per row.")
    predicted = probability >= 0.5
    bins = calibration_bins(y, probability)
    calibration_error = np.average(abs(bins.mean_probability - bins.observed_rate), weights=bins.rows)
    return {
        "rows": len(y), "positive_rate": float(y.mean()),
        "mean_probability": float(probability.mean()),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability)),
        "calibration_error_8_quantile_bins": float(calibration_error),
        "threshold_0.5": {"precision": float(precision_score(y, predicted, zero_division=0)),
                          "recall": float(recall_score(y, predicted, zero_division=0)),
                          "f1": float(f1_score(y, predicted, zero_division=0)),
                          "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist()},
        "ranking": {f"{int(k * 100)}%": ranking_metrics(y, probability, k) for k in CAPACITIES},
    }


def comparison_table(metrics):
    return pd.DataFrame([
        {"model": name, "PR-AUC (AP)": m["average_precision"],
         "Log Loss": m["log_loss"], "Brier": m["brier"],
         "Recall@10%": m["ranking"]["10%"]["recall"], "Lift@10%": m["ranking"]["10%"]["lift"]}
        for name, m in metrics.items()
    ])


def ranking_table(metrics):
    return pd.DataFrame([{"capacity": capacity, **values} for capacity, values in metrics["ranking"].items()])


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_evaluation(y, probabilities, partition):
    from src.train import CHARTS, REPORTS
    CHARTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    for kind in ["precision_recall", "calibration"]:
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, probability in probabilities.items():
            if kind == "precision_recall":
                precision, recall, _ = precision_recall_curve(y, probability)
                ap = average_precision_score(y, probability)
                ax.step(recall, precision, where="post",
                        label=f"{name}: PR-AUC (AP) = {ap:.4f}")
            else:
                bins = calibration_bins(y, probability)
                ax.plot(bins.mean_probability, bins.observed_rate, "o-", label=name)
        if kind == "precision_recall":
            ax.axhline(np.mean(y), color="black", linestyle="--",
                       label=f"Random baseline = {np.mean(y):.4f}")
            ax.set(xlabel="Recall", ylabel="Precision")
        else:
            upper = max(0.15, max(float(np.max(p)) for p in probabilities.values()))
            ax.plot([0, upper], [0, upper], "k--", label="perfect calibration")
            ax.set(xlabel="Mean predicted purchase probability", ylabel="Observed purchase rate")
        ax.set_title(f"{partition}: Precision–Recall / PR-AUC (Average Precision)"
                     if kind == "precision_recall" else f"{partition}: {kind.replace('_', ' ')}")
        ax.legend()
        save_figure(fig, CHARTS / f"{partition}_{kind}.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, probability in probabilities.items():
        values = [ranking_metrics(y, probability, k) for k in CAPACITIES]
        axes[0].plot([5, 10, 20], [v["lift"] for v in values], "o-", label=name)
        axes[1].plot([5, 10, 20], [v["recall"] for v in values], "o-", label=name)
    axes[0].axhline(1, color="black", linestyle="--")
    axes[1].plot([5, 10, 20], CAPACITIES, "k--", label="random ranking")
    for ax, label in zip(axes, ["Lift", "Recall"]):
        ax.set(xlabel="Contact capacity (% of leads)", ylabel=label, title=f"{partition}: Top-K {label}")
        ax.legend()
    save_figure(fig, CHARTS / f"{partition}_top_k.png")


def main():
    from src.data_analysis import markdown_table
    from src.train import ARTIFACTS, REPORTS, TARGET, dataset_fingerprint, read_dataset, split_dataset

    metadata_path = ARTIFACTS / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    frame = read_dataset()
    if dataset_fingerprint(frame) != metadata["dataset_fingerprint"]:
        raise ValueError("PostgreSQL data changed since training; retrain before evaluating.")
    partitions, _ = split_dataset(frame, metadata["split"]["validation_start"], metadata["split"]["test_start"])
    model = joblib.load(ARTIFACTS / "model.joblib")
    test = partitions["test"]
    probability = model.predict_proba(prepare_features(test))[:, 1]
    metrics = compute_metrics(test[TARGET], probability)
    # A training-prior reference reveals whether weak acquisition signals add value.
    prior = float(partitions["train"][TARGET].mean())
    reference = compute_metrics(test[TARGET], np.full(len(test), prior))
    test_results = {**metrics, "model_version": metadata["model_version"],
                    "training_prior_test_reference": reference}
    (ARTIFACTS / "test_metrics.json").write_text(
        json.dumps(test_results, indent=2) + "\n", encoding="utf-8")
    # Migrate metadata created before test results had a separate artifact.
    obsolete = [key for key in ("test_metrics", "training_prior_test_reference") if key in metadata]
    if obsolete:
        for key in obsolete:
            del metadata[key]
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    plot_evaluation(test[TARGET], {metadata["model_name"]: probability}, "test")
    comparison = comparison_table(metadata["validation_metrics"])
    final = comparison_table({metadata["model_name"]: metrics, "training_prior_reference": reference})
    print("\nValidation comparison:\n", comparison.to_string(index=False))
    print("\nFinal test metrics:\n", final.to_string(index=False))
    print("\nTop-K test ranking (expected random ordering within score ties):\n",
          ranking_table(metrics).to_string(index=False))
    print("\nThreshold 0.5:", metrics["threshold_0.5"])
    print("\nCalibration: mean predicted probability", metrics["mean_probability"],
          "vs observed rate", metrics["positive_rate"],
          "; quantile-bin absolute calibration error", metrics["calibration_error_8_quantile_bins"])
    report = ["# Modeling results", "", "Features: " + ", ".join(metadata["features"]), "",
              "Conditional timing assumption: " + metadata["timing_assumption"], "",
              "## Split", "", markdown_table(pd.DataFrame(metadata["split"]["summary"])), "",
              f"Cross-period entities purged: {metadata['split']['purged_entities']}; rows: {metadata['split']['purged_rows']}.", "",
              "## Validation comparison", "", markdown_table(comparison), "",
              "Selected: **" + metadata["model_name"] + "**. " + metadata["selection_reason"], "",
              "## Final test (selection already frozen)", "", markdown_table(final), "",
              "## Capacity ranking", "", markdown_table(ranking_table(metrics)), "",
              "Precision and conversion rate are the same quantity. Top-K uses ceil(capacity × rows). "
              "Boundary score ties are averaged over random contact ordering; captured purchasers can be fractional.", "",
              "## Probability and threshold diagnostics", "",
              f"Test rows: {metrics['rows']:,}; ROC-AUC: {metrics['roc_auc']:.4f}.", "",
              f"Mean predicted probability: {metrics['mean_probability']:.4f}; observed: {metrics['positive_rate']:.4f}. "
              f"Eight-quantile-bin absolute calibration error: {metrics['calibration_error_8_quantile_bins']:.4f} "
              "(descriptive, sensitive to bins). No post-hoc calibration was fitted. "
              "Calibration curves and bin counts accompany this report.", "",
              "Threshold=0.5 metrics (reference only; operational selection is by capacity):", "",
              "```json", json.dumps(metrics["threshold_0.5"], indent=2), "```", "",
              "Test stability is a diagnostic, not permission to change the winner. "
              "August prevalence decline and unknown label maturity limit probability transportability. "
              "All-data EDA preceded this holdout: test is withheld from fitting/selection, but not a pristine "
              "prospective evaluation. Validate on newly collected, mature outcomes before deployment.", ""]
    report.extend(["## Validation trials", "",
                   markdown_table(comparison_table(metadata["hyperparameter_trials"]["validation_metrics"])), "",
                   "Feature selection reason: " + metadata["feature_selection_reason"], "",
                   "Excluded after validation: " + ", ".join(metadata["excluded_after_validation"]), ""])
    for name, rows in metadata.get("interpretations", {}).items():
        report.extend([f"## Feature interpretation: {name}", "",
                       "Logistic values are centered log-odds coefficients within each categorical field; "
                       "numeric values use standardized inputs. CatBoost values are feature importances. "
                       "These describe the fitted models, not causal effects.", "",
                       markdown_table(pd.DataFrame(rows)), ""])
    for name, rows in metadata.get("validation_calibration", {}).items():
        report.extend([f"## Validation calibration: {name}", "", markdown_table(pd.DataFrame(rows)), ""])
    report.extend(["## Test calibration", "", markdown_table(calibration_bins(test[TARGET], probability)), "",
                   "## Charts", "",
                   "![Test precision–recall](../charts/test_precision_recall.png)", "",
                   "![Test calibration](../charts/test_calibration.png)", "",
                   "![Test capacity ranking](../charts/test_top_k.png)", ""])
    (REPORTS / "modeling_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
