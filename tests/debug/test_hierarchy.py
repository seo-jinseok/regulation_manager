from src.formatter import RegulationFormatter

def test_deep_hierarchy():
    text = """
제1조(목적) 이 규정은 ...
① 제1항입니다.
1. 제1호입니다.
가. 제1목입니다.
나. 제2목입니다.
다. 제3목입니다.
2. 제2호입니다.
② 제2항입니다.
    """.strip()
    
    formatter = RegulationFormatter()
    docs = formatter.parse(text)
    
    if not docs:
        print("FAIL: No docs parsed")
        assert False
        
    doc = docs[0]
    content = doc.get('content', [])
    
    if not content:
        print("FAIL: No content")
        assert False
        
    # Check Article
    # The first node might be Chapter/Section if inferred, but here just Article
    # Assuming no chapter headers in text
    
    art = content[0]
    if art['type'] != 'article':
        # Might be wrapped in default chapter? No, current logic doesn't create default chapter.
        print(f"FAIL: First node is {art['type']}")
        assert False
        
    if art['display_no'] != '제1조':
        print(f"FAIL: Article no mismatch: {art['display_no']}")
        assert False
    
    # Check Paragraph 1
    if not art['children']:
        print("FAIL: No paragraphs")
        assert False
        
    para1 = art['children'][0]
    if para1['display_no'] != '①':
        print(f"FAIL: Para 1 no mismatch: {para1['display_no']}")
        assert False
    
    # Check Item 1
    if not para1['children']:
        print("FAIL: No items in Para 1")
        assert False
        
    item1 = para1['children'][0]
    if item1['display_no'] != '1.':
        print(f"FAIL: Item 1 no mismatch: {item1['display_no']}")
        assert False
    
    # Check Subitem 가
    if not item1['children']:
        print("FAIL: No subitems in Item 1")
        assert False
        
    sub1 = item1['children'][0]
    if sub1['display_no'] != '가.':
        print(f"FAIL: Subitem 1 no mismatch: {sub1['display_no']}")
        assert False
        
    if sub1['text'].strip() != '제1목입니다.':
        print(f"FAIL: Subitem text mismatch: {sub1['text']}")
        assert False
    
    print("PASS: Hierarchy parsed correctly.")
    

if __name__ == "__main__":
    test_deep_hierarchy()
