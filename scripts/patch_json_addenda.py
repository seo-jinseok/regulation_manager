
import json
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from formatter import RegulationFormatter

def patch_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatter = RegulationFormatter()
    
    modified = False
    
    # Handle root being list of docs or dict with 'docs'
    docs = data if isinstance(data, list) else data.get('docs', [])
    
    for doc in docs:
        # Check if it has addenda
        if 'addenda' in doc and doc['addenda']:
            new_addenda = []
            new_attached = doc.get('attached_files', [])
            
            for ad in doc['addenda']:
                # The 'text' field currently contains the full lump (Addenda + Appendices)
                # We re-run _parse_appendices on this text.
                # Note: _parse_appendices expects "부 칙" header in most cases if it wasn't stripped.
                # In current JSON, ad['title'] is "부 칙", ad['text'] is the content WITHOUT "부 칙" header usually?
                # Let's check the JSON content from previous 'view_file':
                # "title": "부 칙", "text": "1. 이 규정은..." 
                # formatter._parse_appendices splits by "부 칙" or "별표".
                # If we pass just "1. ...", it might validly treat it.
                # However, the regex expects `(?:^|\n)(?:\|\s*)?((?:부\s*칙)|...)`.
                # If we feed it "1. ... [별 표 1]", it won't split "Addenda" structure unless we prepend "부 칙".
                
                # So we verify logic:
                raw_text = ad.get('text', '')
                if not raw_text:
                    new_addenda.append(ad)
                    continue
                    
                # Prepend title to simulate original text flow for the splitter
                simulated_text = f"{ad.get('title', '부 칙')}\n{raw_text}"
                
                # Use the new formatter logic
                split_addenda, split_attached = formatter._parse_appendices(simulated_text)
                
                if not split_addenda:
                    # Should not happen if text exists
                    new_addenda.append(ad)
                else:
                    # The first split addenda is our replacement
                    # We preserve children if they are better in the output, 
                    # but actually the new parser also re-parses children.
                    # Since existing children included appendices text (incorrectly?), 
                    # we prefer the new parsing which might be cleaner.
                    # But if we want to be safe, we just update text and let children be re-parsed?
                    # The `_parse_appendices` returns fully populated dicts with children.
                    new_addenda.extend(split_addenda)
                    
                if split_attached:
                    new_attached.extend(split_attached)
                    modified = True
            
            doc['addenda'] = new_addenda
            doc['attached_files'] = new_attached
            
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Patched {file_path}")
    else:
        print(f"No changes needed for {file_path}")

if __name__ == "__main__":
    target_file = "output/규정집9-343-test1.json"
    patch_json(target_file)
