import json
from pathlib import Path

out_dir = Path('data/output')
files = [f for f in out_dir.glob('*.json') if 'dummy' not in f.name and 'metadata' not in f.name]
target_file = sorted(files, key=lambda x: x.stat().st_size, reverse=True)[0]

with open(target_file, 'r') as f:
    data = json.load(f)

docs = data.get('docs', [])
target = next((r for r in docs if '교원인사규정' in r.get('title', '')), None)

if target:
    content = target.get('content', [])
    
    def find_node(nodes, title_pattern):
        for node in nodes:
            if title_pattern in node.get('title', ''):
                return node
            if 'children' in node:
                found = find_node(node['children'], title_pattern)
                if found: return found
        return None

    article = find_node(content, '교원의 자격')
    if article and 'children' in article:
        child0 = article['children'][0]
        meta = child0.get('metadata', {})
        print(f"Child 0 Metadata Keys: {meta.keys()}")
        if 'tables' in meta:
            tables = meta['tables']
            print(f"Tables Count: {len(tables)}")
            # Tables is list or dict? code says `tables[index-1]`, assumes list.
            if isinstance(tables, list) and len(tables) > 0:
                 t1 = tables[0]
                 print(f"Table 1: {str(t1)[:500]}")
            else:
                 print(f"Tables type: {type(tables)}")
        else:
            print("No 'tables' in Child 0 metadata")
            
        # Also check Article node metadata
        meta = article.get('metadata', {})
        print(f"Article Metadata Keys: {meta.keys()}")
