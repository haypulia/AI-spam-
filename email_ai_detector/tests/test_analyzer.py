import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from analyzer import EmailAIAnalyzer
from config import Config

class TestEmailAIAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = EmailAIAnalyzer(api_key=Config.API_KEY)

    def test_load_email(self):
        email = self.analyzer.load_email_from_file("./tests/test_data/test_email.eml")
        self.assertIsNotNone(email)

if __name__ == "__main__":
    unittest.main()