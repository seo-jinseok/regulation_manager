
import re
from typing import List, Dict, Any

class RegulationFormatter:
    def parse(self, text: str) -> Dict[str, Any]:
        lines = text.split('\n')
        articles = []
        current_article = None
        current_paragraph = None
        preamble = []
        appendices = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Fixed Regex
            article_match = re.match(r'^(제\s*(\d+)\s*조)\s*(?:\(([^)]+)\))?\s*(.*)', line)
            if article_match:
                if current_article:
                    articles.append(current_article)
                
                article_no = article_match.group(2)
                article_title = article_match.group(3) or ""
                content = article_match.group(4)
                
                current_article = {
                    "article_no": article_no,
                    "title": article_title,
                    "content": [],
                    "paragraphs": []
                }
                
                if content:
                    para_match_inline = re.match(r'^([①-⑮])\s*(.*)', content)
                    if para_match_inline:
                        para_no = para_match_inline.group(1)
                        para_text = para_match_inline.group(2)
                        
                        new_para = {
                            "paragraph_no": para_no,
                            "content": para_text,
                            "items": []
                        }
                        current_article["paragraphs"].append(new_para)
                        current_paragraph = new_para
                    else:
                        current_article["content"].append(content)
                        current_paragraph = None
                else:
                    current_paragraph = None
                continue

            # Detect Appendices
            if re.match(r'^부\s*칙', line):
                if current_article:
                    articles.append(current_article)
                    current_article = None
                appendices.append(line)
                continue
                
            # Detect Paragraph
            paragraph_match = re.match(r'^([①-⑮])\s*(.*)', line)
            if paragraph_match:
                para_no = paragraph_match.group(1)
                content = paragraph_match.group(2)
                
                new_para = {
                    "paragraph_no": para_no,
                    "content": content,
                    "items": []
                }
                
                if current_article:
                    current_article["paragraphs"].append(new_para)
                    current_paragraph = new_para
                else:
                     preamble.append(line)
                continue

            # Detect Item
            item_match = re.match(r'^(\d+\.|[가-하]\.)\s*(.*)', line)
            if item_match:
                item_no = item_match.group(1)
                content = item_match.group(2)
                
                new_item = {
                    "item_no": item_no,
                    "content": content
                }
                
                if current_paragraph:
                    current_paragraph["items"].append(new_item)
                elif current_article:
                     current_article["content"].append(line)
                else:
                    preamble.append(line)
                continue
            
            # General text
            if current_paragraph:
                current_paragraph["content"] += " " + line
            elif current_article:
                current_article["content"].append(line)
            else:
                if appendices:
                    appendices.append(line)
                else:
                    preamble.append(line)

        if current_article:
            articles.append(current_article)

        return {
            "articles": articles
        }

def test_formatter():
    formatter = RegulationFormatter()
    
    # Case 1: Space before title parenthesis
    text1 = "제1조 (목적) 이 법인은..."
    result1 = formatter.parse(text1)
    article1 = result1["articles"][0]
    print(f"Case 1 Title: '{article1['title']}' (Expected: '목적')")
    print(f"Case 1 Content: {article1['content']}")

    # Case 2: Paragraph circled number inline
    text2 = "제4조(주소) ①이 법인의 주사무소는..."
    result2 = formatter.parse(text2)
    article2 = result2["articles"][0]
    print(f"Case 2 Paragraphs count: {len(article2['paragraphs'])} (Expected: 1)")
    print(f"Case 2 Content: {article2['content']}")

if __name__ == "__main__":
    test_formatter()
