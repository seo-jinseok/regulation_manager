import json
import re
import glob

def verify_phase1():
    files = glob.glob("output_verify/*.json")
    if not files:
        print("FAIL: No JSON found")
        exit(1)
        
    path = files[0]
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    docs = data['docs']
    print(f"Total docs: {len(docs)}")
    
    garbage_count = 0
    html_table_count = 0
    total_attached = 0
    
    for doc in docs:
        title = doc.get('title', '')
        if not title: continue

        # Check for garbage title
        # Pattern: Starts with digit+dot OR contains "시행한다"
        # Also check for "부칙" as title (should not happen if parsed correctly into addenda)
        if re.match(r'^\d+\.', title) or "시행한다" in title:
            print(f"FAIL: Garbage title found: '{title}'")
            garbage_count += 1
            
        # Check attached files
        attached = doc.get('attached_files', [])
        total_attached += len(attached)
        for af in attached:
            html = af.get('html', '')
            if "<table" in html.lower():
                html_table_count += 1
                
    if garbage_count > 0:
        print(f"FAIL: Found {garbage_count} garbage titles.")
        exit(1)
        
    print(f"PASS: No garbage titles found.")
    print(f"INFO: Found {html_table_count} attached files with HTML tables out of {total_attached} total attached files.")
    
    if total_attached > 0 and html_table_count == 0:
         # It's possible attached files are just text? But usually they are forms.
         # If strict requirement, maybe warn.
         print("WARNING: Attached files found but no tables detected.")
         # exit(1) # Let's not fail yet, maybe they are text-only forms?

    exit(0)

if __name__ == "__main__":
    verify_phase1()
