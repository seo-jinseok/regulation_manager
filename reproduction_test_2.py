
import re
from src.rag.infrastructure.patterns import REGULATION_ONLY_PATTERN, RULE_CODE_PATTERN
from src.rag.interface.common import decide_search_mode

def test():
    queries = [
        "교원업적평가 규정",
        "교원업적평가규정",
        "3-1-6"
    ]
    
    print(f"Pattern source: {REGULATION_ONLY_PATTERN.pattern}")
    
    for q in queries:
        print(f"Query: '{q}'")
        match = REGULATION_ONLY_PATTERN.match(q)
        print(f"  Regex Match: {bool(match)}")
        if match:
            print(f"  Group 1: '{match.group(1)}'")
            
        mode = decide_search_mode(q)
        print(f"  Decided Mode: {mode}")

if __name__ == "__main__":
    test()
