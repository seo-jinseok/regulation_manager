
import sys
import os
import json

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from formatter import RegulationFormatter

# Simulate User's Data
# '차례' contains the listing.
# Then actual docs follow.
SAMPLE_MARKDOWN = """
규정집 관리 현황표
...

차 례
제1편 학칙
직제규정 3-1-1
사무분장규정 3-1-2

제1편 학칙

직제규정(타이틀)
제1조 (목적)
직제는 ...

사무분장규정(타이틀)
제1조 (목적)
사무는 ...

찾아보기
<가나다순>
"""

# HTML has headers for '차 례' and '찾아보기', but NOT for '직제규정' or '사무분장규정'
SAMPLE_HTML = """
<html>
<body>
<div class="HeaderArea">차 례 0-0-0 ~ 1</div>
<div class="HeaderArea">찾아보기 0-0-0 ~ 5</div>
</body>
</html>
"""

def test_toc_backfill():
    formatter = RegulationFormatter()
    
    print("Parsing sample document to test TOC Backfilling...")
    results = formatter.parse(SAMPLE_MARKDOWN, html_content=SAMPLE_HTML)
    
    found_toc = False
    
    for i, doc in enumerate(results):
        print(f"Doc {i+1} Title: '{doc['title']}' Metadata: {doc['metadata']}")
        
        if doc['title'] == '차례':
            found_toc = True
            # Verify preamble has the lines
            print(f"  TOC Preamble Length: {len(doc['preamble'])}")
            print(f"  TOC Preamble Content: {repr(doc['preamble'])}")
            
        if "직제규정" in str(doc['title']):
            # This doc should have rule_code from TOC even though HTML header is missing
            rc = doc['metadata'].get('rule_code')
            if rc == '3-1-1':
                print("SUCCESS: '직제규정' backfilled rule_code '3-1-1' from TOC.")
            else:
                print(f"FAILURE: '직제규정' rule_code is '{rc}' (Expected '3-1-1')")

        if "사무분장규정" in str(doc['title']):
             rc = doc['metadata'].get('rule_code')
             if rc == '3-1-2':
                 print("SUCCESS: '사무분장규정' backfilled rule_code '3-1-2' from TOC.")
             else:
                 print(f"FAILURE: '사무분장규정' rule_code is '{rc}' (Expected '3-1-2')")

    if not found_toc:
        print("FAILURE: '차례' (TOC) NOT found.")

if __name__ == "__main__":
    test_toc_backfill()
