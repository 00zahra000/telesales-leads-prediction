"""Read PostgreSQL, split chronologically, compare two models, save the winner."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sqlalchemy import text

from src.db import create_db_engine
from src.features import (ATTRIBUTION_ASSUMPTION, CATEGORICAL_FEATURES, SELECTED_FEATURES,
                          EXCLUDED_AFTER_VALIDATION, FEATURE_SELECTION_REASON,
                          make_preprocessor, prepare_features)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
CHARTS = ROOT / "charts"
TARGET = "completed_purchase"
RANDOM_STATE = 42
VALIDATION_START = "2026-07-15"
TEST_START = "2026-08-07"


def read_dataset():
    engine = create_db_engine()
    try:
        with engine.connect() as connection, connection.begin():
            connection.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
            frame = pd.read_sql_query(text("SELECT * FROM raw_leads ORDER BY id"), connection)
    finally:
        engine.dispose()
    if frame.empty or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("Dataset must be nonempty with complete binary targets.")
    if frame.lead_id.isna().any() or frame.created_at.isna().any():
        raise ValueError("Complete lead_id and created_at are required for leakage checks.")
    return frame


def dataset_fingerprint(frame):
    # Includes source fields/labels and row IDs, but excludes reload wall-clock time.
    values = pd.util.hash_pandas_object(frame.drop(columns="loaded_at"), index=False)
    return hashlib.sha256(values.to_numpy().tobytes()).hexdigest()


def split_dataset(frame, validation_start=VALIDATION_START, test_start=TEST_START):
    """Fixed calendar cutoffs; purge all rows of any entity spanning periods.

    Preserve within-period repeats rather than guessing which timestamp is true.
    No entity is moved into an earlier period using a later snapshot.
    """
    validation_start, test_start = pd.Timestamp(validation_start), pd.Timestamp(test_start)
    if validation_start >= test_start:
        raise ValueError("Validation must precede test.")
    split = pd.Series("train", index=frame.index)
    split.loc[frame.created_at >= validation_start] = "validation"
    split.loc[frame.created_at >= test_start] = "test"
    entity_periods = frame.assign(split=split).groupby("lead_id").split.nunique()
    crossing_ids = entity_periods.index[entity_periods > 1]
    keep = ~frame.lead_id.isin(crossing_ids)
    partitions = {name: frame.loc[keep & split.eq(name)].copy()
                  for name in ["train", "validation", "test"]}
    summary = []
    for name, part in partitions.items():
        if part.empty or part[TARGET].nunique() != 2:
            raise ValueError(f"{name} needs observations from both classes; reconsider calendar cutoffs.")
        summary.append({"split": name, "start": str(part.created_at.min()),
                        "end": str(part.created_at.max()), "rows": len(part),
                        "unique_leads": part.lead_id.nunique(),
                        "positive_rate": float(part[TARGET].mean())})
    for earlier, later in [("train", "validation"), ("validation", "test"), ("train", "test")]:
        if set(partitions[earlier].lead_id) & set(partitions[later].lead_id):
            raise ValueError("Entity overlap detected.")
        if partitions[earlier].created_at.max() >= partitions[later].created_at.min():
            raise ValueError("Chronological overlap detected.")
    audit = {"strategy": "chronological calendar split with cross-period entity purge",
             "validation_start": str(validation_start.date()), "test_start": str(test_start.date()),
             "purged_entities": len(crossing_ids), "purged_rows": int((~keep).sum()),
             "repeated_entities": int(frame.lead_id.value_counts().gt(1).sum()),
             "summary": summary}
    return partitions, audit


def build_candidates(trial=1):
    """Three predeclared settings per family, evaluated only on validation."""
    settings = {
        1: (1.0, 200, 3, 0.05, 5),
        2: (0.1, 300, 2, 0.03, 10),
        3: (10.0, 400, 4, 0.03, 20),
    }
    c, iterations, depth, rate, regularization = settings[trial]
    return {
        "logistic_regression": Pipeline([
            ("preprocess", make_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(C=c, max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "catboost": Pipeline([
            ("preprocess", make_preprocessor(scale_numeric=False)),
            ("model", CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=rate,
                                        l2_leaf_reg=regularization, loss_function="Logloss",
                                        random_seed=RANDOM_STATE, thread_count=1,
                                        verbose=False, allow_writing_files=False)),
        ]),
    }


def validation_order(metrics):
    """AP first, then capacity recall and probability quality; never test scores."""
    return (metrics["average_precision"], metrics["ranking"]["10%"]["recall"],
            -metrics["brier"], -metrics["log_loss"])


def select_model(metrics):
    """Predeclared practical gate: modest differences favor the simpler baseline."""
    linear, boost = metrics["logistic_regression"], metrics["catboost"]
    linear_top, boost_top = linear["ranking"]["10%"], boost["ranking"]["10%"]
    wins = (boost["average_precision"] >= linear["average_precision"] + 0.002
            and boost_top["recall"] >= linear_top["recall"]
            and boost_top["lift"] >= linear_top["lift"]
            and boost["brier"] <= linear["brier"] + 0.001
            and boost["log_loss"] <= linear["log_loss"] + 0.01)
    if wins:
        return "catboost", "CatBoost passes the validation AP, Top-10% ranking and probability-quality gate."
    return "logistic_regression", (
        "CatBoost does not pass every validation improvement gate; choose the simpler Logistic Regression. "
        "Gate: AP improvement >=0.002, no lower Top-10% recall/lift, Brier degradation <=0.001, "
        "log-loss degradation <=0.01. These practical tolerances are not significance tests."
    )


def save_interpretation(candidates):
    from src.evaluate import save_figure
    import matplotlib.pyplot as plt

    interpretations = {}
    for name, pipeline in candidates.items():
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        model = pipeline.named_steps["model"]
        values = model.coef_[0] if name == "logistic_regression" else model.feature_importances_
        table = pd.DataFrame({"feature": names, "value": values})
        if name == "logistic_regression":
            # Full one-hot encoding shares a coefficient offset with the intercept.
            # Center within each field to display effects relative to its mean.
            # This changes the interpretation reference, not model predictions.
            table["raw_coefficient"] = values
            for column in CATEGORICAL_FEATURES:
                mask = table.feature.str.startswith(f"categorical__{column}_")
                table.loc[mask, "value"] -= table.loc[mask, "value"].mean()
        table = table.sort_values("value")
        interpretations[name] = table.to_dict(orient="records")
        if name != "logistic_regression":
            continue
        fig, ax = plt.subplots(figsize=(9, max(5, len(table) * 0.28)))
        ax.barh(table.feature, table.value)
        ax.set_xlabel("Centered log-odds coefficient (within field)")
        ax.set_title(name.replace("_", " "))
        save_figure(fig, CHARTS / f"{name}_importance.png")

    return interpretations


def main():
    # Import here to keep the standalone evaluation command independent of __main__.
    from src.evaluate import (calibration_bins, comparison_table, compute_metrics, main as evaluate_saved,
                              plot_evaluation)

    for directory in [ARTIFACTS, REPORTS, CHARTS]:
        directory.mkdir(parents=True, exist_ok=True)
    frame = read_dataset()
    partitions, audit = split_dataset(frame)
    print("Selected features:", SELECTED_FEATURES)
    print("Feature selection:", FEATURE_SELECTION_REASON)
    print("Conditional timing assumption:", ATTRIBUTION_ASSUMPTION)
    print(pd.DataFrame(audit["summary"]).to_string(index=False))
    print("Purged cross-period entities/rows:", audit["purged_entities"], audit["purged_rows"])
    candidates, metrics, probabilities, chosen_trials = {}, {}, {}, {}
    trial_metrics = {}
    for trial in [1, 2, 3]:
        for name, pipeline in build_candidates(trial).items():
            pipeline.fit(prepare_features(partitions["train"]), partitions["train"][TARGET])
            probability = pipeline.predict_proba(prepare_features(partitions["validation"]))[:, 1]
            result = compute_metrics(partitions["validation"][TARGET], probability)
            trial_metrics[f"{name}_trial_{trial}"] = result
            print(f"Trial {trial} {name}: validation AP={result['average_precision']:.6f}, "
                  f"Recall@10%={result['ranking']['10%']['recall']:.6f}", flush=True)
            if name not in metrics or validation_order(result) > validation_order(metrics[name]):
                candidates[name], metrics[name], probabilities[name] = pipeline, result, probability
                chosen_trials[name] = trial
    comparison = comparison_table(metrics)
    print("\nValidation comparison:\n", comparison.to_string(index=False))
    chosen, reason = select_model(metrics)
    print("\nSelected:", chosen, "\n", reason)
    plot_evaluation(partitions["validation"][TARGET], probabilities, "validation")
    interpretations = save_interpretation(candidates)
    joblib.dump(candidates[chosen], ARTIFACTS / "model.joblib")
    # Reload and verify the actual persisted pipeline before evaluating test.
    saved = joblib.load(ARTIFACTS / "model.joblib")
    np.testing.assert_allclose(saved.predict_proba(prepare_features(partitions["validation"]))[:, 1],
                               probabilities[chosen], rtol=0, atol=1e-12)
    metadata = {
        "interpretations": interpretations,
        "validation_calibration": {name: calibration_bins(partitions["validation"][TARGET], probability)
                                   .assign(bin=lambda table: table["bin"].astype(str)).to_dict(orient="records")
                                   for name, probability in probabilities.items()},
        "model_name": chosen, "model_version": datetime.now(timezone.utc).strftime("v2-%Y%m%dT%H%M%S%fZ"),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE, "features": SELECTED_FEATURES,
        "feature_selection_reason": FEATURE_SELECTION_REASON,
        "excluded_after_validation": EXCLUDED_AFTER_VALIDATION,
        "timing_assumption": ATTRIBUTION_ASSUMPTION, "prediction_moment": "after abandonment at Telesales eligibility",
        "dataset_fingerprint": dataset_fingerprint(frame), "source": "PostgreSQL raw_leads",
        "split": audit, "fit_partition": "train only; no train+validation refit",
        "selection_reason": reason, "validation_metrics": metrics,
        "training_prior_validation_reference": compute_metrics(
            partitions["validation"][TARGET], np.full(len(partitions["validation"]),
                                                     partitions["train"][TARGET].mean())),
        "hyperparameter_trials": {"count_per_family": 3, "chosen_trials": chosen_trials,
                                  "selection_order": "validation AP, Top-10% recall, Brier, log loss",
                                  "validation_metrics": trial_metrics},
        "model_parameters": candidates[chosen].named_steps["model"].get_params(),
        "probability_calibration": "Unweighted log-loss models; no post-hoc recalibration. See calibration diagnostics.",
        "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in
                             ["pandas", "numpy", "scikit-learn", "catboost", "joblib", "SQLAlchemy"]},
    }
    (ARTIFACTS / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    evaluate_saved()


if __name__ == "__main__":
    main()
