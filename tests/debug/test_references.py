from src.formatter import RegulationFormatter

def test_cross_references_extracted():
    text = "제1조(목적) 이 규정은 제5조 및 제10조제1항에 따른다."
    formatter = RegulationFormatter()
    docs = formatter.parse(text)
    
    if not docs or not docs[0]['content']:
        print("FAIL: No content parsed")
        exit(1)
        
    art = docs[0]['content'][0]
    if 'references' not in art:
        print("FAIL: references field missing in node")
        exit(1)
        
    refs = art['references']
    if not isinstance(refs, list):
        print(f"FAIL: references is not a list: {type(refs)}")
        exit(1)
        
    found_texts = [r['text'] for r in refs]
    print(f"Found references: {found_texts}")
    
    if "제5조" not in found_texts:
        print("FAIL: '제5조' not extracted")
        exit(1)
        
    if "제10조제1항" not in found_texts:
        # Depending on regex, it might be separate or combined
        # But we target the combined form if possible
        if not any("제10조" in t for t in found_texts):
            print("FAIL: '제10조' not extracted")
            exit(1)
    
    print("PASS: Cross-references extracted.")
    exit(0)

if __name__ == "__main__":
    test_cross_references_extracted()
