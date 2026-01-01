
import json
from src.rag.interface.cli import _find_json_path

def inspect_table():
    json_path = _find_json_path()
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    
    doc = data["docs"][10]
    print(f"Regulation: {doc.get('title')}")
    
    # Path from search: content[1].children[0].children[0].metadata.tables[0]
    try:
        table_node = doc["content"][1]["children"][0]["children"][0]["metadata"]["tables"][0]
        print("\n--- HTML ---")
        print(table_node.get("html", "No HTML"))
        print("\n--- MARKDOWN ---")
        print(table_node.get("markdown", "No Markdown"))
        
        # Also print text of the parent to see context
        print("\n--- Parent Text ---")
        print(doc["content"][1]["children"][0]["children"][0].get("text", "No Text"))
    except Exception as e:
        print(f"Error accessing path: {e}")

if __name__ == "__main__":
    inspect_table()

if __name__ == "__main__":
    inspect_table()
