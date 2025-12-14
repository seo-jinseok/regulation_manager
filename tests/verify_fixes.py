import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from formatter import RegulationFormatter
from refine_json import refine_doc, clean_preamble_and_get_title, parse_articles_from_text

def test_title_splitting():
    print("Testing Title Splitting...")
    # Mock text that contains two regulations
    # 1. First reg has appendices that technically end with a title for the NEXT reg.
    # 2. Next reg starts immediately after.
    
    raw_md = """
제1조(목적) 이 규정은 테스트를 목적으로 한다.

부 칙
1. 이 규정은 공포한 날부터 시행한다.

다음 관리 규정
제1조(목적) 이 규정은 두번째 규정이다.
    """.strip()
    
    formatter = RegulationFormatter()
    results = formatter.parse(raw_md)
    
    # We expect 2 regulations
    if len(results) != 2:
        print(f"FAILED: Expected 2 regulations, got {len(results)}")
        return
        
    # Check 2nd reg preamble
    reg2 = results[1]
    preamble = reg2['preamble']
    if "다음 관리 규정" in preamble:
        print("PASSED: Split detected and title line preserved.")
    else:
        print(f"FAILED: Title line missing in 2nd reg preamble. Got: {preamble}")

def test_addenda_structuring():
    print("\nTesting Addenda Structuring...")
    
    appendix_text = """
부 칙
제1조(시행일) 이 규정은 2024년 3월 1일부터 시행한다.
제2조(경과조치) 이 규정 시행 당시...
    """.strip()
    
    # We'll test refine_doc logic implicitly or parse_appendices directly if exposed
    # Since parse_appendices is not exported in __init__ usually, let's use refine_doc on a mock doc
    
    mock_doc = {
        "preamble": "Test",
        "articles": [],
        "appendices": appendix_text
    }
    
    refined = refine_doc(mock_doc, 1)
    
    addenda = refined.get('addenda', [])
    if not addenda:
        print("FAILED: No addenda parsed.")
        return
        
    item = addenda[0]
    articles = item.get('articles')
    
    if not articles:
        print("FAILED: Addenda not structured into articles.")
        print(item)
        return
        
    if len(articles) == 2 and articles[0]['article_no'] == '1':
        print("PASSED: Addenda structured correctly.")
    else:
        print(f"FAILED: Incorrect structure. Got {len(articles)} articles.")

def test_title_cleaning():
    print("\nTesting Title Cleaning...")
    
    # Case 1: Title with revision info as first line
    preamble1 = "<개정 2024.1.1>\n진짜 규정 제목"
    doc1 = {"preamble": preamble1, "articles": [], "appendices": ""}
    
    # refine_doc uses clean_preamble_and_get_title internally
    # But clean_preamble_and_get_title logic was: 
    #   lines = preamble.split('\n')
    #   ... check first line ... 
    #   if first line is <...>, continue loop
    
    title, _ = clean_preamble_and_get_title(doc1, 1)
    
    if title == "진짜 규정 제목":
        print(f"PASSED Case 1: Got '{title}'")
    else:
        print(f"FAILED Case 1: Expected '진짜 규정 제목', got '{title}'")

    # Case 2: Title line has (개정 ...) at end
    preamble2 = "학칙 (개정 2024.1.1)"
    doc2 = {"preamble": preamble2}
    title2, _ = clean_preamble_and_get_title(doc2, 1)
    
    if title2 == "학칙":
        print(f"PASSED Case 2: Got '{title2}'")
    else:
        print(f"FAILED Case 2: Expected '학칙', got '{title2}'")

if __name__ == "__main__":
    test_title_splitting()
    test_addenda_structuring()
    test_title_cleaning()
