import json
import os
import sys
import unittest
from unittest.mock import patch

# Add src to path
sys.path.append(os.getcwd())


class TestAnalysisTools(unittest.TestCase):
    def setUp(self):
        self.sample_json = {
            "docs": [
                {
                    "title": "Test Regulation",
                    "content": [
                        {
                            "type": "chapter",
                            "title": "Ch1",
                            "children": [{"type": "article", "text": "Sample Article"}],
                        }
                    ],
                    "addenda": [{"type": "addendum_item", "text": "Addendum"}],
                }
            ]
        }
        os.makedirs("tmp_test", exist_ok=True)
        with open("tmp_test/test.json", "w", encoding="utf-8") as f:
            json.dump(self.sample_json, f)

    def tearDown(self):
        import shutil

        if os.path.exists("tmp_test"):
            shutil.rmtree("tmp_test")

    def test_analyze_json_logic(self):
        from src.analyze_json import analyze_json

        with patch("sys.stdout"):
            analyze_json("tmp_test/test.json")

    def test_inspect_json_logic(self):
        from src.inspect_json import analyze_json as inspect_json

        with patch("sys.stdout"):
            inspect_json("tmp_test/test.json")

    def test_refine_json_logic(self):
        from src.refine_json import refine_doc

        doc = {
            "title": "Old Title",
            "preamble": "New Regulation Title\nContent",
            "articles": [{"article_no": "1", "content": ["제1장 장제목", "본문"]}],
        }
        # Use index > 0 to avoid hardcoded title
        refined = refine_doc(doc, 1)
        self.assertEqual(refined["title"], "New Regulation Title")


if __name__ == "__main__":
    unittest.main()
