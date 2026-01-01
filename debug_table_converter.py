
from pathlib import Path
from src.parsing.html_table_converter import convert_html_tables_to_markdown
from src.parsing.table_extractor import TableExtractor

def debug_converter():
    output_dir = Path("data/output")
    # Find the files (assuming only one set exists or picking the first)
    xhtml_files = list(output_dir.glob("*.xhtml"))
    md_files = list(output_dir.glob("*_raw.md"))
    
    if not xhtml_files or not md_files:
        print("Raw files not found in data/output")
        return

    xhtml_path = xhtml_files[0]
    md_path = md_files[0]
    
    print(f"Loading {xhtml_path.name}...")
    with open(xhtml_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    print(f"Loading {md_path.name}...")
    with open(md_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    print("Extracting HTML tables...")
    html_tables = convert_html_tables_to_markdown(html_content)
    print(f"HTML Tables found: {len(html_tables)}")

    print("Extracting Markdown tables...")
    extractor = TableExtractor()
    _, markdown_tables = extractor.split_markdown_tables(markdown_content)
    print(f"Markdown Tables found: {len(markdown_tables)}")

    if len(html_tables) != len(markdown_tables):
        print("MISMATCH DETECTED!")
    else:
        print("Counts match.")

if __name__ == "__main__":
    debug_converter()
