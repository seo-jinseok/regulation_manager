from src.formatter import RegulationFormatter

def test_html_table_extraction():
    # Mock text
    text = """
[ 별표 1 ]
테이블이 포함된 별표입니다.
    """.strip()
    
    # Mock HTML content
    html_content = """
<html><head><style>.s1{color:red;}</style></head>
<body>
<div class="HeaderArea">...</div>
<p><span>[ 별표 1 ]</span></p>
<p><span>테이블이 포함된 별표입니다.</span></p>
<table>
  <tr><td>Row 1 Col 1</td><td>Row 1 Col 2</td></tr>
  <tr><td>Row 2 Col 1</td><td>Row 2 Col 2</td></tr>
</table>
</body></html>
    """
    
    formatter = RegulationFormatter()
    docs = formatter.parse(text, html_content=html_content)
    
    if not docs:
        print("FAIL: No docs parsed")
        exit(1)
        
    doc = docs[0]
    attached = doc.get("attached_files", [])
    
    if not attached:
        print("FAIL: No attached files found")
        exit(1)
        
    af = attached[0]
    print(f"Title: {af.get('title')}")
    html = af.get('html', '')
    
    if "<table>" not in html:
        print("FAIL: Table tag not found in extracted HTML")
        print(f"Extracted: {html}")
        exit(1)
        
    if "Row 1 Col 1" not in html:
        print("FAIL: Table content not found")
        exit(1)
        
    print("PASS: Table extracted successfully.")
    exit(0)

if __name__ == "__main__":
    test_html_table_extraction()
