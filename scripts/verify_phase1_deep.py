import json
import re
import glob

def verify_phase1_deep():
    files = glob.glob("output_verify/*.json")
    if not files:
        print("FAIL: No JSON found")
        exit(1)
        
    path = files[0]
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    docs = data['docs']
    
    max_depth = 0
    deepest_doc = ""
    
    def get_depth(nodes, current_depth):
        max_d = current_depth
        for node in nodes:
            children = node.get('children', [])
            if children:
                d = get_depth(children, current_depth + 1)
                if d > max_d:
                    max_d = d
        return max_d

    for doc in docs:
        d = get_depth(doc.get('content', []), 1) # Start at 1 (Content itself is a list of level 1 nodes)
        if d > max_depth:
            max_depth = d
            deepest_doc = doc.get('title')

    print(f"INFO: Max hierarchy depth found: {max_depth} in '{deepest_doc}'")
    
    # We expect at least depth 4 (Article -> Paragraph -> Item -> Subitem)
    if max_depth < 4:
        print("WARNING: Max depth < 4. Deep nesting might be missing.")
    else:
        print("PASS: Deep hierarchy (Level 4+) detected.")

if __name__ == "__main__":
    verify_phase1_deep()
