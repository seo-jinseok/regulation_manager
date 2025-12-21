import unittest
from src.refine_json import refine_doc, parse_articles_from_text

class TestRefine(unittest.TestCase):
    def test_parse_articles(self):
        text = "제1조(목적) ... 제2조(명칭) ..."
        articles = parse_articles_from_text(text)
        self.assertGreaterEqual(len(articles), 1)

    def test_refine_doc_basic(self):
        doc = {
            "title": "Old",
            "preamble": "Actual Title\nSome description",
            "articles": [
                {
                    "article_no": "제1조",
                    "content": ["(목적) 이 규정은..."]
                }
            ]
        }
        refined = refine_doc(doc, 1)
        self.assertEqual(refined['title'], "Actual Title")

if __name__ == "__main__":
    unittest.main()
