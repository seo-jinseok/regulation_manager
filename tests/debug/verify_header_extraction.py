
import re
from bs4 import BeautifulSoup

# This is a snippet from tmp_html_debug/index.xhtml
RAW_HTML_SNIPPET = """
<div class="HeaderArea"><p class="Normal parashape-36"><span class="lang-ko charshape-29">제3편  행정  </span><span class="lang-other charshape-30">3—1—97</span><span class="lang-other charshape-29">～</span><span class="autonumbering autonumbering-page">2</span>&#13;</p><p class=""><span class="lang-other">󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏</span>&#13;</p></div>
<div class="HeaderArea"><p class="Normal parashape-38"><span class="lang-ko charshape-29">교원강의평가운영규정  </span><span class="lang-other charshape-30">3—1—97</span><span class="lang-other charshape-29">～</span><span class="autonumbering autonumbering-page">1</span>&#13;</p><p class=""><span class="lang-other">󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏󰠏</span>&#13;</p></div>
<div class="HeaderArea"><p class="Normal parashape-38"><span class="lang-en charshape-29">LINC+</span><span class="lang-ko charshape-29">사업단운영규정  </span><span class="lang-other charshape-30">3—1—98</span><span class="lang-other charshape-29">～</span><span class="autonumbering autonumbering-page">1</span>&#13;</p></div>
"""

def extract_header_metadata(html_content):
    """
    Extracts header metadata including rule codes (e.g., 3-1-97) and page numbers.
    Returns a dictionary of found metadata.
    """
    metadata = {
        "rule_codes": set(),
        "headers": []
    }
    
    if not html_content:
        return metadata

    soup = BeautifulSoup(html_content, 'html.parser')
    header_areas = soup.find_all('div', class_='HeaderArea')
    
    print(f"Found {len(header_areas)} HeaderArea entries")
    
    for div in header_areas:
        original_text = div.get_text(strip=True)
        # Normalize various dash/tilde characters
        normalized_text = re.sub(r'[\u2010-\u2015\u2212\uFF0D]', '-', original_text)
        normalized_text = re.sub(r'[~\uFF5E\u301C]', '~', normalized_text)
        
        # Regex to find rule code like 3-1-97 (~ page part optional)
        # matches: 3-1-97, 3-1-97~1, etc.
        match = re.search(r'(\d+-\d+-\d+)(~\d+)?', normalized_text)
        if match:
            rule_code = match.group(1)
            page_part = match.group(2) # e.g. "~1"
            
            page_number = page_part.replace('~', '') if page_part else None
            
            metadata["rule_codes"].add(rule_code)
            
            # Extract prefix (Regulation Title or Section Name)
            prefix = normalized_text.split(rule_code)[0].strip()
            # Clean up trailing tildes or special chars
            prefix = re.sub(r'[~\u301c\u2053]+$', '', prefix).strip()
            
            # Ignore common garbage line endings or separators
            if "행정" in prefix and "제" in prefix and "편" in prefix:
                 # Likely "제3편 행정" - this is section info, not title
                 section_name = prefix
                 title_candidate = None
            else:
                 section_name = None
                 title_candidate = prefix
            
            metadata["headers"].append({
                "rule_code": rule_code,
                "page": page_number,
                "prefix": prefix,
                "is_likely_title": title_candidate is not None
            })

    return metadata

def test():
    results = extract_header_metadata(RAW_HTML_SNIPPET)
    print("\nExtraction Results:")
    print(f"Unique Rule Codes: {results['rule_codes']}")
    for h in results['headers']:
        print(f" - Code: {h['rule_code']}, Prefix: '{h['prefix']}'")

if __name__ == "__main__":
    test()
