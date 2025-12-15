import sys
import os
import re
from typing import List, Dict, Optional

# Mocking the RegulationFormatter's _extract_html_segments method for testing
class MockFormatter:
    def _extract_html_segments(self, attached_files: List[Dict], html_content: str):
        """
        Extracts HTML segments for each attached file from the full HTML content.
        Adds an 'html' field to each attached_file dictionary.
        """
        print(f"DEBUG: Processing {len(attached_files)} attached files.")
        
        # 1. Extract CSS Style block
        style_match = re.search(r'(<style.*?>.*?</style>)', html_content, re.DOTALL | re.IGNORECASE)
        style_block = style_match.group(1) if style_match else ""
        print(f"DEBUG: Found style block length: {len(style_block)}")
        
        lower_html = html_content.lower()
        
        def make_html_search_key(title):
            escaped = title.replace("<", "&lt;").replace(">", "&gt;")
            return escaped.lower()

        positions = []
        for i, af in enumerate(attached_files):
            title = af["title"]
            search_key = make_html_search_key(title)
            print(f"DEBUG: Search key for '{title}': '{search_key}'")
            
            pos = lower_html.find(search_key)
            print(f"DEBUG: Found '{search_key}' at position: {pos}")
            
            if pos != -1:
                positions.append((pos, i))
        
        positions.sort()
        
        for idx, (start_pos, af_index) in enumerate(positions):
            if idx + 1 < len(positions):
                end_pos = positions[idx+1][0]
            else:
                end_pos = len(html_content)
                
            segment = html_content[start_pos:end_pos]
            print(f"DEBUG: Extracted segment length for '{attached_files[af_index]['title']}': {len(segment)}")
            
            full_html = f"<html><head>{style_block}</head><body>{segment}</body></html>"
            attached_files[af_index]["html"] = full_html

def main():
    # 1. Read the generated HTML file (from previous run)
    # Note: The temp dir is gone, but we can try to recreate or read if we know where hwp5html puts it.
    # Or we can just use the provided hwp file to generate html again.
    
    # Actually, let's use the file we saw in tmp_html_debug if it still exists?
    # No, I should run hwp5html again or assume the file content from my previous read (step 198)
    
    html_path = "tmp_html_debug/index.xhtml"
    if not os.path.exists(html_path):
        print("DEBUG: tmp_html_debug/index.xhtml not found. Please run: uv run hwp5html --output tmp_html_debug 규정/규정집9-343-test1.hwp")
        return

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # 2. Mock attached files
    attached_files = [
        {"title": "<별지 1호 서식>", "text": "dummy"},
        {"title": "<별지 4호 서식>", "text": "dummy"},
    ]
    
    formatter = MockFormatter()
    formatter._extract_html_segments(attached_files, html_content)
    
    for af in attached_files:
        if "html" in af:
            print(f"SUCCESS: '{af['title']}' has HTML content ({len(af['html'])} chars).")
        else:
            print(f"FAILURE: '{af['title']}' missing HTML content.")

if __name__ == "__main__":
    main()
