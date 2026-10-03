import unittest

from src.load_data import load_leads


class SourceDataTests(unittest.TestCase):
    def test_source_data_available(self):
        self.assertFalse(load_leads().empty, "data/leads.csv contains no leads")
