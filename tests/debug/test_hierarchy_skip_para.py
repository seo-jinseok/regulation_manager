from src.formatter import RegulationFormatter

def test_hierarchy_skip_para():
    text = """
제1조(목적)
1. 바로 아이템이 나옵니다.
가. 그리고 서브아이템.
    """.strip()
    
    formatter = RegulationFormatter()
    docs = formatter.parse(text)
    doc = docs[0]
    art = doc['content'][0]
    
    if not art['children']:
        print("FAIL: No children in Article")
        assert False

    para = art['children'][0]
    # Implicit paragraph should have empty display_no
    if para['display_no'] != "":
        print(f"FAIL: Implicit paragraph has display_no '{para['display_no']}'")
        assert False
        
    if not para['children']:
        print("FAIL: No children in Paragraph")
        assert False

    item = para['children'][0]
    if item['display_no'] != '1.':
        print(f"FAIL: Item not found, got {item['display_no']}")
        assert False
        
    print("PASS: Hierarchy skip para handled.")
    

if __name__ == "__main__":
    test_hierarchy_skip_para()
