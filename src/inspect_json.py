import json
import re

file_path = "data/output/규정집9-349(20251202).json"

def analyze_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"File Name: {data.get('file_name')}")
    print(f"Scan Date: {data.get('scan_date')}")
    
    docs = data.get('docs', [])
    print(f"\nTotal Regulations (docs): {len(docs)}")

    for i, doc in enumerate(docs):
        print(f"\n--- Regulation {i+1} ---")
        # Try to find a title in preamble or first article?
        preamble = doc.get('preamble', '')
        print(f"Preamble length: {len(preamble)} chars")
        print(f"Preamble start: {preamble[:100].replace(chr(10), ' ')}...")
        
        articles = doc.get('articles', [])
        print(f"Total Articles: {len(articles)}")
        
        if articles:
            print(f"First Article: {articles[0].get('article_no')} - {articles[0].get('title')}")
            print(f"Last Article: {articles[-1].get('article_no')} - {articles[-1].get('title')}")

        # Check for bad patterns
        chapter_in_content = 0
        article_no_issues = 0
        
        for art in articles:
            # Check content for "제N장"
            content_str = " ".join(art.get('content', []))
            for p in art.get('paragraphs', []):
                content_str += " " + p.get('content', '')
            
            if re.search(r'제\s*\d+\s*장', content_str):
                chapter_in_content += 1
            
            # Check for Addenda in content
            if "부칙" in content_str:
                # specifically look for "부      칙" or "부 칙" header 
                if re.search(r'부\s*칙', content_str):
                    pass # Just noting it might be in content

            if not art.get('article_no'):
                article_no_issues += 1
            elif not re.match(r'^\d+$', art.get('article_no')):
                 # Check if it's like 9-2 or anything non-digit
                 print(f"  [Info] Non-standard Article No: {art.get('article_no')}")
        
        print(f"Articles with 'Chapter' in content: {chapter_in_content}")
        
        # Check last article for Addenda clues
        if articles:
            last = articles[-1]
            last_content = " ".join(last.get('content', []))
            if "부칙" in last_content or "시행일" in last_content:
                print(f"  [Info] Last article seems to contain Addenda info.")

if __name__ == "__main__":
    analyze_json(file_path)
