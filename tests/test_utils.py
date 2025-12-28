import json
import os
import unittest
from unittest.mock import patch

from src.metadata_extractor import MetadataExtractor
from src.verify_json import verify_json_files


class TestUtils(unittest.TestCase):
    def test_verify_json_logic(self):
        # 1. Valid case
        valid_data = {
            "docs": [
                {"title": "Test", "content": [{"type": "article", "text": "Valid"}]}
            ]
        }
        # 2. Invalid cases
        invalid_data = {"missing_docs": []}
        leak_data = {"docs": [{"content": [{"type": "article", "text": "제1장 본문"}]}]}

        os.makedirs("tmp_test_verify", exist_ok=True)
        try:
            with open("tmp_test_verify/valid.json", "w") as f:
                json.dump(valid_data, f)
            with open("tmp_test_verify/invalid.json", "w") as f:
                json.dump(invalid_data, f)
            with open("tmp_test_verify/leak.json", "w") as f:
                json.dump(leak_data, f)

            with patch("sys.stdout"):
                verify_json_files("tmp_test_verify")
        finally:
            import shutil

            shutil.rmtree("tmp_test_verify")

    def test_metadata_extractor_indexes(self):
        extractor = MetadataExtractor()
        text = """
차 례
학칙 1-1-1
장학규정 2-2-2
찾아보기
<가나다순>
학칙 1-1-1
장학규정 2-2-2
찾아보기
<소관부서별>
기획처
학칙 1-1-1
학생처
장학규정 2-2-2
        """
        result = extractor.extract(text)
        self.assertEqual(len(result["toc"]), 2)
        self.assertEqual(len(result["index_by_alpha"]), 2)
        self.assertIn("기획처", result["index_by_dept"])
        self.assertEqual(len(result["index_by_dept"]["기획처"]), 1)

    def test_metadata_extractor_dept_filters_noise(self):
        extractor = MetadataExtractor()
        text = """
찾아보기
<소관부서별>
학생군사교육단
학생군사교육단운영규정 5-1-24
| --- | --- | --- |
| 제1편 | | |
학교법인동의학원정관
제1편 학칙
        """
        result = extractor.extract(text)
        dept = result["index_by_dept"]
        self.assertIn("학생군사교육단", dept)
        self.assertNotIn("| --- | --- | --- |", dept)
        self.assertNotIn("| 제1편 | | |", dept)
        # Stop before part header
        self.assertNotIn("학교법인동의학원정관", dept)

    def test_metadata_extractor_dept_handles_school_foundation_code(self):
        extractor = MetadataExtractor()
        text = """
찾아보기
<소관부서별>
예산팀
예산결산자문위원회규정【폐지】 4-0-11
학교법인동의학원정관 1-0-1

구매팀
구매업무규정 3-1-125
학교법인동의학원정관
        """
        result = extractor.extract(text)
        dept = result["index_by_dept"]
        self.assertIn("예산팀", dept)
        self.assertIn("구매팀", dept)
        self.assertEqual(dept["예산팀"][-1]["title"], "학교법인동의학원정관")
        self.assertEqual(dept["예산팀"][-1]["rule_code"], "1-0-1")
        self.assertEqual(len(dept["구매팀"]), 1)
        self.assertNotIn("학교법인동의학원정관", dept)


if __name__ == "__main__":
    unittest.main()
