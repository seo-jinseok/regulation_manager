import os
import glob
from pathlib import Path
import argparse
from typing import List

def merge_markdown_files(output_dir: str):
    """
    Scans the output directory for subdirectories containing split markdown files
    and merges them into a single 'full.md' file.
    
    This is useful if the regulation was processed in chunks/pages and saved in 'splits' subfolder.
    If the current main.py saves as single files (_clean.md), this script can be adapted
    or used to merge multiple related regulations if needed.
    
    Current Logic:
    1. Looks for folders in output_dir.
    2. In each folder, looks for a 'splits' subfolder.
    3. Merges all .md files in 'splits' to 'full.md'.
    """
    target_path = Path(output_dir)
    
    if not target_path.exists():
        print(f"Directory not found: {output_dir}")
        return

    # Check immediate subdirectories (assuming each regulation has its own folder if split)
    # OR, if main.py outputs `filename_clean.md`, we might not need this.
    # But fulfilling the requirement for "merge_regulations.py":
    
    print(f"Scanning {target_path} for split content...")
    
    # Strategy 1: Merge numbered splits in subfolders
    subdirs = [d for d in target_path.iterdir() if d.is_dir()]
    
    for subdir in subdirs:
        splits_dir = subdir / "splits"
        if splits_dir.exists() and splits_dir.is_dir():
            print(f"Processing split files in {subdir.name}...")
            md_files = sorted(list(splits_dir.glob("*.md")))
            
            if not md_files:
                print("  No markdown files found in splits directory.")
                continue
                
            full_content = []
            for md_file in md_files:
                with open(md_file, 'r', encoding='utf-8') as f:
                    full_content.append(f.read())
            
            merged_content = "\n\n".join(full_content)
            
            output_file = subdir / "full.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(merged_content)
                
            print(f"  Created {output_file}")
            
    # Strategy 2: (Optional) If main.py produces independent _partX.md files in the root output
    # (Not implemented as current main.py produces single files).
    
    print("Merge process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge split regulation markdown files.")
    parser.add_argument("--output_dir", type=str, default="data/output", help="Directory containing regulation outputs")
    
    args = parser.parse_args()
    merge_markdown_files(args.output_dir)
