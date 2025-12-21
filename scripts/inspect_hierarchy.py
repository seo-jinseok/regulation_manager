import json
import glob

def inspect_hierarchy():
    files = glob.glob("output_verify/*.json")
    if not files: return
    path = files[0]
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    target_title = "학교법인동의학원정관"
    target_doc = next((d for d in data['docs'] if d['title'] == target_title), None)
    
    if not target_doc:
        print(f"Doc '{target_title}' not found.")
        return

    def find_deepest_path(nodes):
        best_path = []
        for node in nodes:
            children = node.get('children', [])
            if not children:
                path = [node]
            else:
                child_path = find_deepest_path(children)
                path = [node] + child_path
            
            if len(path) > len(best_path):
                best_path = path
        return best_path

    path = find_deepest_path(target_doc['content'])
    print(f"Deepest Path (Depth {len(path)}):")
    for i, node in enumerate(path):
        indent = "  " * i
        print(f"{indent}- {node['type']} {node.get('display_no', '')} {node.get('title') or ''} | {node.get('text', '')[:50]}")

if __name__ == "__main__":
    inspect_hierarchy()
