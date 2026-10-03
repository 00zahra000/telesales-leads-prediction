import json
import unittest

from sqlalchemy import text

from src.db import create_db_engine
from src.load_data import PROJECT_ROOT


class PredictionScoreTests(unittest.TestCase):
    def test_every_lead_has_current_model_score(self):
        metadata = json.loads((PROJECT_ROOT / "artifacts/model_metadata.json").read_text())
        engine = create_db_engine()
        try:
            with engine.connect() as connection, connection.begin():
                connection.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
                version = {"version": metadata["model_version"]}
                total, missing_ids = connection.execute(text(
                    "SELECT COUNT(*), COUNT(*) FILTER (WHERE lead_id IS NULL OR BTRIM(lead_id) = '') "
                    "FROM raw_leads"
                )).one()
                self.assertGreater(total, 0, "No leads available to verify")
                self.assertEqual(missing_ids, 0, "Some source rows have no usable lead ID")
                missing = connection.execute(text("""
                    SELECT COUNT(*) FROM (SELECT DISTINCT lead_id FROM raw_leads) AS leads
                    WHERE NOT EXISTS (
                        SELECT 1 FROM lead_scores AS scores
                        WHERE scores.lead_id = leads.lead_id AND scores.model_version = :version
                    )
                """), version).scalar_one()
                self.assertEqual(missing, 0, "Some leads have no score for the current model")
                expected = connection.execute(text(
                    "SELECT COUNT(DISTINCT lead_id) FROM raw_leads"
                )).scalar_one()
                scored, unique_scored, valid = connection.execute(text("""
                    SELECT COUNT(*), COUNT(DISTINCT lead_id),
                           COUNT(*) FILTER (WHERE purchase_probability BETWEEN 0 AND 1)
                    FROM lead_scores WHERE model_version = :version
                """), version).one()
                self.assertEqual(scored, expected, "Current model must score exactly the unique source leads")
                self.assertEqual(unique_scored, scored, "Current model has duplicate lead scores")
                self.assertEqual(valid, scored, "Some prediction scores are missing or outside [0, 1]")
        finally:
            engine.dispose()
