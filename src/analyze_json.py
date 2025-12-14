import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List

def check_structure(data: Dict[str, Any]) -> List[str]:
    errors = []
    required_keys = ['file_name', 'scan_date', 'docs']
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing root key: {key}")
    return errors

def check_doc(doc: Dict[str, Any], index: int) -> List[str]:
    errors = []
    
    # 1. Title Validation
    title = doc.get('title')
    if title is None:
        errors.append(f"Doc #{index}: Missing 'title' field")
        print(f"DEBUG: Doc #{index} Preamble:\n{doc.get('preamble', '')[:200]}...")
    elif str(title).strip() == "":
        errors.append(f"Doc #{index}: Empty 'title' value")
        print(f"DEBUG: Doc #{index} Preamble:\n{doc.get('preamble', '')[:200]}...")


    # 2. Articles Validation
    articles = doc.get('articles', [])
    if not isinstance(articles, list):
         errors.append(f"Doc #{index}: 'articles' is not a list")
    else:
        for i, article in enumerate(articles):
            if not isinstance(article, dict):
                # errors.append(f"Doc #{index}, Article #{i}: invalid format (not a dict)")
                pass 
            else:
                if 'article_no' not in article:
                    errors.append(f"Doc #{index}, Article #{i}: Missing 'article_no'")
                if 'content' not in article:
                    errors.append(f"Doc #{index}, Article #{i}: Missing 'content'")

    # 3. Addenda & Appendices & Attached Files (Structure Check)
    # We want to catch if they are just strings (unparsed) or empty dicts where content should be
    for key in ['addenda', 'appendices', 'attached_files']:
        items = doc.get(key, [])
        if not isinstance(items, list):
             errors.append(f"Doc #{index}: '{key}' is not a list")
             continue
             
        for j, item in enumerate(items):
            if isinstance(item, str):
                # This explicitly fails if we expect ONLY structured data. 
                # For now, we flag it as a warning or error depending on strictness. 
                # Given user goal "100% accuracy", unstructured might be considered a 'fail' to be fixed.
                errors.append(f"Doc #{index}: {key} item #{j} is unstructured string (needs parsing)")
            elif isinstance(item, dict):
                 # Check for minimal structure if it's a dict
                 # usually expected: "no", "title", "content" or similar
                 if not item:
                     errors.append(f"Doc #{index}: {key} item #{j} is empty dict")
            else:
                 errors.append(f"Doc #{index}: {key} item #{j} has invalid type {type(item)}")

    return errors

def analyze_json(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"File found: {path}")
        return

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON {path}: {e}")
        return

    print(f"Analyzing: {path.name}")
    print(f"Scan Date: {data.get('scan_date', 'Unknown')}")
    
    # Root Structure
    root_errors = check_structure(data)
    if root_errors:
        for err in root_errors:
            print(f"  [CRITICAL] {err}")
        # If root structure is broken, might stop here
    
    docs = data.get('docs', [])
    print(f"Total Documents: {len(docs)}")
    
    total_errors = 0
    
    for i, doc in enumerate(docs):
        doc_errors = check_doc(doc, i)
        if doc_errors:
            # Print Doc Title if available for context
            doc_title = doc.get('title', 'NO_TITLE')
            print(f"  Issue in Doc #{i} ({doc_title}):")
            for err in doc_errors:
                print(f"    - {err}")
            total_errors += len(doc_errors)

    if total_errors == 0:
        print(f"  [SUCCESS] All checks passed for {len(docs)} documents.")
    else:
        print(f"  [FAIL] Found {total_errors} validation issues.")
    print("-" * 40)

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_json.py <path_to_json_or_dir>")
        sys.exit(1)
        
    input_path = Path(sys.argv[1])
    
    if input_path.is_file():
        analyze_json(str(input_path))
    elif input_path.is_dir():
        json_files = sorted(list(input_path.glob("*.json")))
        if not json_files:
            print("No JSON files found in directory.")
        for f in json_files:
            analyze_json(str(f))
    else:
        print(f"Invalid input path: {input_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
