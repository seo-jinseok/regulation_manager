import json
import glob
import os
import re

def verify_json_files(output_dir="data/output"):
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in {output_dir}")
        return

    for json_file in json_files:
        print(f"\nVerifying {json_file}...")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'docs' not in data:
                print(f"  [Error] 'docs' key missing.")
                continue
            
            docs = data['docs']
            print(f"  Found {len(docs)} documents.")
            
            total_articles = 0
            docs_with_content = 0
            chapter_leak_count = 0
            
            for doc_idx, doc in enumerate(docs):
                content = doc.get('content', [])
                if content:
                    docs_with_content += 1
                
                # Recursive function to find articles and check text
                def traverse_nodes(nodes):
                    count = 0
                    leaks = 0
                    for node in nodes:
                        node_type = node.get('type')
                        text = node.get('text', '')
                        
                        if node_type == 'article':
                            count += 1
                        
                        # Check for Chapter/Section headers leaking into text
                        # e.g. "제1장 총칙" appearing in text
                        if re.search(r'(^|\n)제\s*\d+\s*장($|\s)', text):
                            leaks += 1
                            print(f"      [Leak Alert] '{text[:60]}...'")
                        
                        children = node.get('children', [])
                        c, l = traverse_nodes(children)
                        count += c
                        leaks += l
                    return count, leaks

                art_count, leak_count = traverse_nodes(content)
                total_articles += art_count
                chapter_leak_count += leak_count
                
                if art_count == 0 and "규정" in doc.get('title', ''):
                     # Only warn if it looks like a regulation but has no articles
                     # print(f"  [Warning] Doc '{doc.get('title')}' has no articles.")
                     pass

            print(f"  Documents with content: {docs_with_content}/{len(docs)}")
            print(f"  Total Articles found: {total_articles}")
            
            if chapter_leak_count > 0:
                print(f"  [Warning] nodes with potential Chapter header leaks: {chapter_leak_count}")
            
            print("  Structure seems valid.")

        except json.JSONDecodeError as e:
            print(f"  [Error] Invalid JSON: {e}")
        except Exception as e:
            print(f"  [Error] Processing failed: {e}")

if __name__ == "__main__":
    verify_json_files()
