
import sys
import os
import json

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from formatter import RegulationFormatter

SAMPLE_MARKDOWN = """
규정집 관리 현황표
...
차 례
제1편 학칙
1. 학칙 ... 10
2. 학칙시행규칙 ... 20
찾아보기
<가나다순>
가나다 ...
제1조 (목적) ...
"""

# HTML imitating TOC header?
# Assuming TOC has a header like "차 례" or just page numbers.
SAMPLE_HTML = """
<html>
<body>
<div class="HeaderArea"><p class="Normal"><span class="lang-ko">차 례</span> 0-0-0 ~ 1</p></div>
<div class="HeaderArea"><p class="Normal"><span class="lang-ko">찾아보기</span> 0-0-0 ~ 5</p></div>
</body>
</html>
"""

def test_toc_index():
    formatter = RegulationFormatter()
    
    print("Parsing sample document with TOC and Index...")
    results = formatter.parse(SAMPLE_MARKDOWN, html_content=SAMPLE_HTML)
    
    found_toc = False
    found_index = False
    
    for i, doc in enumerate(results):
        print(f"Doc {i+1} Title: '{doc['title']}' Metadata: {doc['metadata']}")
        if doc['title'] == '차례':
            found_toc = True
            # verify content
            print(f"  Preamble lines: {len(doc['preamble'])}")
            print(f"  Preamble sample: {doc['preamble'][:2]}")
        if doc['title'] == '찾아보기':
            found_index = True
            
    if found_toc:
        print("SUCCESS: '차례' (TOC) recognized as separate document.")
    else:
        print("FAILURE: '차례' (TOC) NOT found.")

    if found_index:
        print("SUCCESS: '찾아보기' (Index) recognized as separate document.")
    else:
        print("FAILURE: '찾아보기' (Index) NOT found.")

if __name__ == "__main__":
    test_toc_index()
