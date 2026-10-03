import unittest

from sqlalchemy import text

from src.db import create_db_engine
from src.load_data import load_leads


class LoadedDataTests(unittest.TestCase):
    def test_loaded_data_available(self):
        expected = len(load_leads())
        engine = create_db_engine()
        try:
            with engine.connect() as connection:
                actual = connection.execute(text("SELECT COUNT(*) FROM raw_leads")).scalar_one()
            self.assertGreater(actual, 0, "raw_leads contains no data")
            self.assertEqual(actual, expected, "Loaded row count differs from the source CSV")
        finally:
            engine.dispose()
