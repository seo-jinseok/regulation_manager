import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List

def check_structure(data: Dict[str, Any]) -> List[str]:
    errors = []
    if 'docs' not in data:
        errors.append("Missing root key: docs")
    if 'file_name' not in data:
        errors.append("Missing root key: file_name")
    return errors

def _validate_nodes(nodes: List[Any], label: str) -> List[str]:
    errors = []
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"{label} #{i}: invalid node (not a dict)")
            continue
        if not node.get('type'):
            errors.append(f"{label} #{i}: missing 'type'")
        children = node.get('children', [])
        if not isinstance(children, list):
            errors.append(f"{label} #{i}: 'children' is not a list")
        else:
            errors.extend(_validate_nodes(children, f"{label} #{i} children"))
        sort_no = node.get('sort_no')
        if sort_no is not None and not isinstance(sort_no, dict):
            errors.append(f"{label} #{i}: 'sort_no' is not a dict")
    return errors

def check_doc(doc: Dict[str, Any], index: int) -> List[str]:
    errors = []
    
    # 1. Title Validation
    title = doc.get('title')
    if title is None:
        errors.append(f"Doc #{index}: Missing 'title' field")
    elif str(title).strip() == "":
        errors.append(f"Doc #{index}: Empty 'title' value")
    
    # 2. Content Validation
    content = doc.get('content', [])
    if not isinstance(content, list):
        errors.append(f"Doc #{index}: 'content' is not a list")
    else:
        errors.extend(_validate_nodes(content, f"Doc #{index} content"))

    # 3. Addenda & Attached Files (Structure Check)
    addenda = doc.get('addenda', [])
    if not isinstance(addenda, list):
        errors.append(f"Doc #{index}: 'addenda' is not a list")
    else:
        errors.extend(_validate_nodes(addenda, f"Doc #{index} addenda"))

    attached = doc.get('attached_files', [])
    if not isinstance(attached, list):
        errors.append(f"Doc #{index}: 'attached_files' is not a list")
    else:
        for j, item in enumerate(attached):
            if not isinstance(item, dict):
                errors.append(f"Doc #{index}: attached_files item #{j} has invalid type {type(item)}")
            elif not item.get("title"):
                errors.append(f"Doc #{index}: attached_files item #{j} missing title")

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
    if data.get("file_name"):
        print(f"Source File: {data.get('file_name')}")
    
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
