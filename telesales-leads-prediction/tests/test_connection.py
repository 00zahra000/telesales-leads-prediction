import unittest

from sqlalchemy import text

from src.db import create_db_engine


class ConnectionTests(unittest.TestCase):
    def test_database_connection(self):
        engine = create_db_engine()
        try:
            with engine.connect() as connection:
                self.assertEqual(connection.execute(text("SELECT 1")).scalar_one(), 1)
        finally:
            engine.dispose()
