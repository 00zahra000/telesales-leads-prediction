"""Score one snapshot per Lead ID and atomically save a versioned sales queue."""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.logging_config import logger

from src.db import create_db_engine
from src.features import SELECTED_FEATURES, prepare_features
from src.train import ARTIFACTS

# Also executed here because init.sql runs only when the database volume is new.
SCORING_SCHEMA = """
CREATE TABLE IF NOT EXISTS lead_scores (
    lead_id TEXT NOT NULL,
    source_row_id BIGINT NOT NULL,
    purchase_probability DOUBLE PRECISION NOT NULL
        CHECK (purchase_probability BETWEEN 0 AND 1),
    priority BIGINT NOT NULL CHECK (priority > 0),
    score_rank BIGINT NOT NULL CHECK (score_rank > 0),
    prediction_timestamp TIMESTAMPTZ NOT NULL,
    model_version TEXT NOT NULL,
    PRIMARY KEY (model_version, lead_id),
    UNIQUE (model_version, priority)
)
"""


def build_scores(frame, model, model_version, timestamp):
    """Latest created_at/id breaks duplicate ambiguity without consulting outcomes.

    Priority 1 is highest; tied probabilities use lead_id lexical ordering.
    score_rank shares a rank across probability ties: priority conveys no extra signal.
    """
    if frame.empty or frame.lead_id.isna().any() or frame.created_at.isna().any():
        raise ValueError("Scoring requires nonempty data with lead IDs and creation timestamps.")
    leads = frame.sort_values(["created_at", "id"]).drop_duplicates("lead_id", keep="last").copy()
    probabilities = model.predict_proba(prepare_features(leads))[:, 1]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Model returned invalid purchase probabilities.")
    scores = pd.DataFrame({"lead_id": leads.lead_id.to_numpy(),
                           "source_row_id": leads.id.to_numpy(),
                           "purchase_probability": probabilities})
    scores = scores.sort_values(["purchase_probability", "lead_id"], ascending=[False, True]).reset_index(drop=True)
    scores["priority"] = np.arange(1, len(scores) + 1)
    scores["score_rank"] = scores.purchase_probability.rank(method="dense", ascending=False).astype(int)
    scores["prediction_timestamp"] = timestamp
    scores["model_version"] = model_version
    return scores


def main():
    metadata = json.loads((ARTIFACTS / "model_metadata.json").read_text())
    if metadata["features"] != SELECTED_FEATURES:
        raise ValueError("Artifact features differ from source; use matching code or retrain.")
    model = joblib.load(ARTIFACTS / "model.joblib")
    engine = create_db_engine()
    try:
        with engine.connect() as connection, connection.begin():
            connection.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
            # No outcome column is read or used at inference.
            columns = ["id", "lead_id", "created_at", *SELECTED_FEATURES]
            frame = pd.read_sql_query(text(f"SELECT {', '.join(columns)} FROM raw_leads"), connection)
        scores = build_scores(frame, model, metadata["model_version"], datetime.now(timezone.utc))
        with engine.begin() as connection:
            connection.execute(text(SCORING_SCHEMA))
            connection.execute(text("LOCK TABLE lead_scores IN SHARE ROW EXCLUSIVE MODE"))
            connection.execute(text("DELETE FROM lead_scores WHERE model_version = :version"),
                               {"version": metadata["model_version"]})
            scores.to_sql("lead_scores", connection, if_exists="append", index=False,
                          chunksize=1000, method="multi")
            count = connection.execute(text("SELECT COUNT(*) FROM lead_scores WHERE model_version = :version"),
                                       {"version": metadata["model_version"]}).scalar_one()
            if count != len(scores):
                raise ValueError("Stored scoring count does not match generated scores.")
        logger.success("Saved {:,} unique lead scores to PostgreSQL: version={}; collapsed {:,} repeated rows.",
                       len(scores), metadata["model_version"], len(frame) - len(scores))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
