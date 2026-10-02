"""Explicit score-time features and training-only preprocessing."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Experimental assumption: selected business attributes are pre-score snapshots.
# Outcome, identifiers and ingestion metadata must never become predictors.
NUMERIC_FEATURES = [
    "minutes_since_abandonment", "days_to_policy_expiry", "price",
    "discount_percent", "sessions_last_7d", "offer_views_last_7d",
]
CATEGORICAL_FEATURES = [
    "channel", "city",
    "insurance_company", "payment_type",
]
BINARY_FEATURES = [
    "has_previous_purchase", "visited_offer_page", "incoming_call_last_24h",
]
SELECTED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
EXCLUDED_AFTER_VALIDATION = [
    "device", "product_type", "partner", "price_comparisons_last_7d",
    "days_since_last_visit", "expected_margin",
]
FEATURE_SELECTION_REASON = (
    "Validation ablation selected 13 inputs: removing device, product_type, partner, "
    "price_comparisons_last_7d, days_since_last_visit and expected_margin preserved "
    "PR-AUC and Top-10% recall. Price and expected_margin are nearly redundant; "
    "retain price. See README for the comparison and limitations."
)
ATTRIBUTION_ASSUMPTION = (
    "User-authorized experiment: the 13 selected business attributes are assumed available "
    "at abandonment/Telesales eligibility. Quotes and profile values are frozen "
    "pre-score snapshots; history and event windows exclude the target purchase "
    "and all later events; durations are calculated at scoring time. This is an "
    "assumption, not verified historical provenance."
)


def prepare_features(frame):
    """Same stateless input preparation at training and inference; no engineering.

    Missing categorical SQL values become np.nan for SimpleImputer. Ignore extra
    columns, but fail on absent required columns. Never infer features from types.
    """
    features = frame.loc[:, SELECTED_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].where(features[column].notna(), np.nan)
    return features


def make_preprocessor(scale_numeric):
    transformers = []
    if NUMERIC_FEATURES:
        steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        if scale_numeric:
            steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), NUMERIC_FEATURES))
    if BINARY_FEATURES:
        transformers.append(("binary", SimpleImputer(strategy="most_frequent",
                                                      keep_empty_features=True), BINARY_FEATURES))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers.append(("categorical", categorical, CATEGORICAL_FEATURES))
    return ColumnTransformer(transformers, remainder="drop")
