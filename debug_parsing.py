
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(".").resolve()))

from src.parsing.regulation_parser import RegulationParser
from src.parsing.table_extractor import TableExtractor
from src.preprocessor import Preprocessor
from src.cache_manager import CacheManager

def debug():
    raw_md_path = Path("data/output/규정집9-343(20250909)_raw.md")
    if not raw_md_path.exists():
        print("Raw MD file not found")
        return

    with open(raw_md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Slice around the interesting part: 18000 to 19000
    chunk_lines = lines[18000:19000]
    text = "".join(chunk_lines)

    print(f"Parsing content length: {len(text)}")
    
    # Run Preprocessor
    print("\nRunning Preprocessor on chunk...")
    class MockCacheManager:
        pass
    
    preprocessor = Preprocessor(llm_client=None, cache_manager=MockCacheManager())
    cleaned_chunk = preprocessor.clean(text)
    
    print(f"Cleaned chunk len: {len(cleaned_chunk)}")

    parser = RegulationParser()
    regs = parser.parse_flat(cleaned_chunk)
    
    print(f"Parsed {len(regs)} regulations in cleaned chunk.")
    
    target_reg = None
    for i, reg in enumerate(regs):
        print(f"Reg {i}:")
        print(f"  Preamble len: {len(reg.get('preamble', []))}")
        
        # Check Title/Identity
        is_kyowon = False
        for line in reg['preamble']:
            if "교원인사규정" in line:
                is_kyowon = True
                break
        
        target_art = None
        for art in reg['articles']:
            if art['article_no'] == "제8조" or art['article_no'] == "제 8 조":
                if "자격" in art.get('title', ''):
                    target_art = art
                    break
        
        if is_kyowon:
            print("  [Identified as 교원인사규정]")
        if target_art:
            print("  [Found Article 8]")
            target_reg = reg
            
        # Check Reserve forces
        has_reserve = False
        for line in reg['preamble']:
            if "예비군" in line:
                has_reserve = True
        for art in reg['articles']:
             if "예비군" in "".join(art.get('content', [])):
                has_reserve = True
             for para in art.get('paragraphs', []):
                if "예비군" in para.get('content', ''):
                    has_reserve = True
        
        if has_reserve:
            print("  [Contains '예비군']")
            
    if not target_reg:
        print("Target regulation (Article 8) not found in cleaned chunk.")
        return

    print("\nExtracting table from target article (Cleaned)...")
    articles = target_reg['articles']
    for art in articles:
        if art['article_no'] == "제8조":
            for para in art['paragraphs']:
                if "|" in para.get('content', ''):
                     print(f"  Para {para.get('paragraph_no')} contains table chars.")
                     # Extract
                     node = {"text": para.get('content'), "metadata": {}}
                     extractor = TableExtractor()
                     extractor._extract_tables_in_nodes([node])
                     if "tables" in node["metadata"]:
                         for t in node["metadata"]["tables"]:
                             print(f"  Table Content Start: {t['markdown'][:50]}...")
                             if "예비군" in t['markdown']:
                                 print("  [CRITICAL] Table content mismatch found! Reserve Forces table in Article 8!")
                             else:
                                 print("  [OK] Table content looks correct.")

if __name__ == "__main__":
    debug()
