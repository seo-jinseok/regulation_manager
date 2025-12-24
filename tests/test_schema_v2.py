
import unittest
from src.formatter import RegulationFormatter

class TestSchemaV2(unittest.TestCase):
    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_article_parsing_branch_number(self):
        """Test Article with Branch Number (제29조의2)"""
        text = "제29조의2 (직원의 임용) 직원은 이사장이 임용한다."
        # Note: formatting text input as a list of lines or single string depending on formatter input
        # The formatter parses a full document usually. Let's try parsing a minimal snippet.
        # However, _parse_flat expects document structure. We might test internal helpers if refactored,
        # but for now let's test the main 'parse' output for a single article doc.
        
        # We need a minimal valid structure for `parse`
        doc_text = """
제1편 학칙
제29조의2 (직원의 임용) ① 직원은 이사장이 임용한다.
"""
        result = self.formatter.parse(doc_text)
        self.assertEqual(len(result), 1)
        doc = result[0]
        articles = doc['content'][0]['children'] # root -> article list? No, content hierarchy.
        
        # Regulation -> Article (if no chapter)
        # Check structure:
        # doc['content'] is a list of roots.
        article_node = doc['content'][0]
        
        self.assertEqual(article_node['type'], 'article')
        self.assertEqual(article_node['display_no'], '제29조의2')
        self.assertEqual(article_node['sort_no']['main'], 29)
        self.assertEqual(article_node['sort_no']['sub'], 2)
        self.assertEqual(article_node['title'], '직원의 임용')

    def test_article_parsing_normal(self):
        """Test Article without Branch Number (제29조)"""
        doc_text = "제29조 (직원의 임용) 내용"
        result = self.formatter.parse(doc_text)
        article_node = result[0]['content'][0]
        
        self.assertEqual(article_node['display_no'], '제29조')
        self.assertEqual(article_node['sort_no']['main'], 29)
        self.assertEqual(article_node['sort_no']['sub'], 0)

    def test_paragraph_parsing(self):
        """Test Paragraph Numbering (①)"""
        doc_text = """
제1조 (목적)
① 이 법은...
"""
        result = self.formatter.parse(doc_text)
        article = result[0]['content'][0]
        self.assertTrue(len(article['children']) > 0)
        paragraph = article['children'][0]
        
        self.assertEqual(paragraph['type'], 'paragraph')
        self.assertEqual(paragraph['display_no'], '①')
        self.assertEqual(paragraph['sort_no']['main'], 1)
        self.assertEqual(paragraph['sort_no']['sub'], 0)

    def test_item_parsing(self):
        """Test Item Numbering (1.)"""
        doc_text = """
제1조 (목적)
1. 학교의 설립
"""
        result = self.formatter.parse(doc_text)
        article = result[0]['content'][0]
        # Hierarchy: Article -> Item
        self.assertTrue(len(article['children']) > 0)
        item = article['children'][0]
        
        self.assertEqual(item['type'], 'item') # Or whatever type we define
        self.assertEqual(item['display_no'], '1.')
        self.assertEqual(item['sort_no']['main'], 1)

    def test_subitem_parsing(self):
        """Test SubItem Numbering (가.)"""
        doc_text = """
제1조 (목적)
1. 학교
가. 설립
"""
        result = self.formatter.parse(doc_text)
        article = result[0]['content'][0]
        # Article -> Item -> SubItem
        item = article['children'][0]
        subitem = item['children'][0]
        
        self.assertEqual(subitem['type'], 'subitem')
        self.assertEqual(subitem['display_no'], '가.')
        self.assertEqual(subitem['sort_no']['main'], 1) # '가' should map to 1

if __name__ == '__main__':
    unittest.main()
