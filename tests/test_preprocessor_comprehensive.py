import unittest

from src.preprocessor import Preprocessor


class TestPreprocessorComprehensive(unittest.TestCase):
    def test_clean_pua_extended(self):
        pre = Preprocessor()
        # Test all mapped chars and random PUA range
        text = "\uf85e \uf09e \uf0fc \ue000 \uf8ff \U000f0000"
        cleaned, count = pre.clean_pua(text)
        self.assertIn("·", cleaned)
        self.assertIn("✓", cleaned)
        self.assertNotIn("\ue000", cleaned)
        self.assertGreater(count, 0)

    def test_join_broken_lines_variations(self):
        pre = Preprocessor()
        cases = [
            ("문장이 끝맺음 없이\n이어집니다.", "문장이 끝맺음 없이 이어집니다."),
            ("제1조(목적)\n이 규정은...", "제1조(목적)\n이 규정은..."),  # Don't join
            ("1. 항목\n내용", "1. 항목\n내용"),  # Don't join
            (
                "| 테이블 | 행 |\n| --- | --- |",
                "| 테이블 | 행 |\n| --- | --- |",
            ),  # Don't join
        ]
        for inp, expected in cases:
            self.assertEqual(pre.clean(inp), expected)

    def test_remove_repetitive_headers(self):
        pre = Preprocessor()
        text = "동의대학교 규정집 (2025)\n본문 내용"
        cleaned = pre.clean(text)
        self.assertNotIn("동의대학교 규정집", cleaned)

    def test_preserve_toc_lines_with_rule_codes(self):
        pre = Preprocessor()
        text = "\n".join(
            [
                "차 례",
                "학칙 1-1-1~1",
                "제3편 행정 3-1-54～1",
                "3. 이 규정집은 번호(예 : 1-1-1～1)를 부여한다.",
            ]
        )
        cleaned = pre.clean(text)
        self.assertIn("학칙 1-1-1~1", cleaned)
        self.assertNotIn("제3편 행정 3-1-54", cleaned)
        self.assertIn("1-1-1～1", cleaned)


if __name__ == "__main__":
    unittest.main()
