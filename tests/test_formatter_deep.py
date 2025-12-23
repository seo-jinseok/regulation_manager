import unittest
from src.formatter import RegulationFormatter

class TestFormatterDeep(unittest.TestCase):
    def test_chapter_section_subsection_nesting(self):
        text = """
제1장 일반사항
제1절 목적
제1관 정의
제1조(목적) 정의된 목적입니다.
        """.strip()
        formatter = RegulationFormatter()
        docs = formatter.parse(text)
        
        # In current implementation, hierarchy is nested
        # Doc -> Chapter -> Section -> Subsection -> Article
        ch = docs[0]['content'][0]
        self.assertEqual(ch['type'], 'chapter')
        
        sec = ch['children'][0]
        self.assertEqual(sec['type'], 'section')
        
        subsec = sec['children'][0]
        self.assertEqual(subsec['type'], 'subsection')
        
        art = subsec['children'][0]
        self.assertEqual(art['type'], 'article')

    def test_item_subitem_nesting(self):
        text = "제1조(목적) ①항입니다.\n1. 호입니다.\n가. 목입니다."
        formatter = RegulationFormatter()
        docs = formatter.parse(text)
        art = docs[0]['content'][0]
        para = art['children'][0]
        item = para['children'][0]
        sub = item['children'][0]
        self.assertEqual(sub['type'], 'subitem')

    def test_toc_backfill_with_page_range(self):
        text = """
차 례
학칙 1-1-1~1

제1편 학칙
학칙
제1조(목적) 내용
        """.strip()
        formatter = RegulationFormatter()
        docs = formatter.parse(text)

        rule_code = None
        for doc in docs:
            if doc.get("title") == "학칙":
                rule_code = doc.get("metadata", {}).get("rule_code")
                break

        self.assertEqual(rule_code, "1-1-1")

    def test_front_matter_trim_and_index_order(self):
        text = """
별표 양식 1
제3편 행정
차 례
학칙 1-1-1~1
찾아보기
<가나다순>
학칙 1-1-1
찾아보기
<소관부서별>
기획처
학칙 1-1-1
제1편 학칙
학칙
제1조(목적) 내용
        """.strip()
        formatter = RegulationFormatter()
        docs = formatter.parse(text)

        titles = [doc.get("title") for doc in docs[:4]]
        self.assertEqual(titles[0], "차례")
        self.assertEqual(titles[1], "찾아보기")
        self.assertEqual(titles[2], "찾아보기")
        self.assertEqual(titles[3], "학칙")
        self.assertIsNone(docs[0].get("part"))
        self.assertGreater(len(docs[0].get("content", [])), 0)
        self.assertEqual(docs[0]["content"][0]["title"], "학칙")
        self.assertEqual(docs[0]["content"][0]["text"], "1-1-1")
        self.assertEqual(docs[0]["content"][0]["metadata"].get("rule_code"), "1-1-1")
        self.assertGreater(len(docs[1].get("content", [])), 0)
        self.assertGreater(len(docs[2].get("content", [])), 0)

if __name__ == "__main__":
    unittest.main()
