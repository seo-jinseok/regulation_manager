from src.formatter import RegulationFormatter

def test_split_false_positive():
    # Scenario: Regulation A has an addendum item mentioning "규정".
    # Regulation B follows immediately.
    # The parser should NOT split at the addendum item.
    
    text = """
제1조(목적) A규정의 목적입니다.
부 칙
1. (시행일) 이 규정은 2024년 1월 1일부터 시행한다.
2. (경과조치) ...

제1조(목적) B규정의 목적입니다.
    """.strip()

    formatter = RegulationFormatter()
    docs = formatter.parse(text)
    
    print(f"Parsed {len(docs)} docs.")
    for i, doc in enumerate(docs):
        print(f"Doc {i}: Title='{doc['title']}'")
        
    # We expect 2 docs:
    # Doc 0: Title should be "A규정의 목적입니다" (or inferred, likely empty/fallback if preamble is empty)
    # Doc 1: Title should be inferred from preamble if any, or empty.
    # BUT we specifically check that "이 규정은..." did NOT become a title.
    
    for doc in docs:
        if "이 규정은" in str(doc['title']):
            print("FAIL: '이 규정은...' was incorrectly identified as a title.")
            assert False
            
    print("PASS: No false positive titles found.")
    

if __name__ == "__main__":
    test_split_false_positive()
