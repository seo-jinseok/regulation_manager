
import re
from bs4 import BeautifulSoup

def test_extraction():
    # Strings from the user's raw file
    # Note: using the exact unicode chars seen in previous file view
    # '—' (U+2014) and '～' (U+FF5E)
    raw_texts = [
        "제2편 학칙 2—1—1～2",
        "동의대학교학칙 2—1—1～5",
        "학교법인동의학원정관 1—0—1",
        "차 례 0—0—0 ~ 1",
        "찾아보기 0—0—0 ~ 5"
    ]
    
    html_content = "<html><body>"
    for text in raw_texts:
        html_content += f'<div class="HeaderArea">{text}</div>'
    html_content += "</body></html>"
    
    print("Testing extraction with raw strings...")
    
    soup = BeautifulSoup(html_content, 'html.parser')
    header_areas = soup.find_all('div', class_='HeaderArea')

    for div in header_areas:
        original_text = div.get_text(strip=True)
        print(f"Original: '{original_text}'")
        
        # Normalize dashes and tildes
        normalized_text = re.sub(r'[\u2010-\u2015\u2212\uFF0D]', '-', original_text)
        normalized_text = re.sub(r'[~\uFF5E\u301C]', '~', normalized_text)
        print(f"Normalized: '{normalized_text}'")

        # Regex for Rule Code (e.g. 3-1-97) and optional Page Number (~1)
        match = re.search(r'(\d+-\d+-\d+)(~\d+)?', normalized_text)
        if match:
            rule_code = match.group(1)
            page_part = match.group(2)
            page_number = page_part.replace('~', '') if page_part else None
            
            # Extract prefix
            prefix = normalized_text.split(rule_code)[0].strip()
            prefix = re.sub(r'[~\u301c\u2053]+$', '', prefix).strip()
            
            print(f"MATCH: RuleCode='{rule_code}', Page='{page_number}', Prefix='{prefix}'")
        else:
            print("NO MATCH")
        print("-" * 20)

if __name__ == "__main__":
    test_extraction()
