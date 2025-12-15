
import sys
import os
import json

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from formatter import RegulationFormatter

SAMPLE_MARKDOWN = """
# 교원강의평가운영규정

제1조 (목적) 이 규정은...
"""

SAMPLE_HTML = """
<html>
<body>
<div class="HeaderArea"><p class="Normal parashape-36"><span class="lang-ko charshape-29">제3편  행정  </span><span class="lang-other charshape-30">3—1—97</span><span class="lang-other charshape-29">～</span><span class="autonumbering autonumbering-page">2</span>&#13;</p></div>
<div class="HeaderArea"><p class="Normal parashape-38"><span class="lang-ko charshape-29">교원강의평가운영규정  </span><span class="lang-other charshape-30">3—1—97</span><span class="lang-other charshape-29">～</span><span class="autonumbering autonumbering-page">1</span>&#13;</p></div>
</body>
</html>
"""

def test_integration():
    formatter = RegulationFormatter()
    
    print("Parsing sample document...")
    results = formatter.parse(SAMPLE_MARKDOWN, html_content=SAMPLE_HTML)
    
    if not results:
        print("No results returned!")
        return
    
    doc = results[0]
    print(f"Title: {doc['title']}")
    print(f"Metadata: {json.dumps(doc['metadata'], ensure_ascii=False, indent=2)}")
    
    metadata = doc['metadata']
    if metadata.get('rule_code') == '3-1-97':
        print("SUCCESS: Rule code extracted correctly.")
    else:
        print(f"FAILURE: Rule code mismatch. Expected '3-1-97', got '{metadata.get('rule_code')}'")

    if metadata.get('page_range') == '1~2':
        print("SUCCESS: Page range extracted correctly.")
    else:
        print(f"FAILURE: Page range mismatch. Expected '1~2', got '{metadata.get('page_range')}'")

if __name__ == "__main__":
    test_integration()
