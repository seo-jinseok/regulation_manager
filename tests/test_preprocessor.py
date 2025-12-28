import unittest

from src.preprocessor import Preprocessor


class TestPreprocessor(unittest.TestCase):
    def test_remove_artifacts(self):
        pre = Preprocessor()
        text = "xml version=1.0\n동의대학교 규정집\n- 1 -\n1-1-1\n PUA\n\n\n\nCollapse"
        cleaned = pre.clean(text)
        self.assertNotIn("xml version", cleaned)
        self.assertNotIn("동의대학교 규정집", cleaned)
        self.assertNotIn("- 1 -", cleaned)
        self.assertIn("· PUA", cleaned)
        self.assertNotIn("\n\n\n", cleaned)

    def test_join_broken_lines(self):
        pre = Preprocessor()
        text = "문장이 끊겨\n있습니다."
        cleaned = pre.clean(text)
        self.assertEqual(cleaned, "문장이 끊겨 있습니다.")

    def test_dont_join_headers(self):
        pre = Preprocessor()
        text = "제1조(목적)\n이 규정은..."
        cleaned = pre.clean(text)
        # Should not join Article header with content
        self.assertIn("제1조(목적)\n", cleaned)


if __name__ == "__main__":
    unittest.main()
