import re

def check_leak(text):
    # Old regex: r'(^|\s)제\s*\d+\s*장($|\s)'
    # New regex: r'(^|\n)제\s*\d+\s*장($|\s)'
    if re.search(r'(^|\n)제\s*\d+\s*장($|\s)', text):
        return True
    return False

if __name__ == "__main__":
    # 1. False Positive Case (Should return False)
    text_fp = '이 규정은 학교법인동의학원정관(이하 “정관”이라 한다) 제7장 제2절 및 동의대학교(이하 “본 대학교”라 한...'
    
    # 2. True Positive Case (Should return True)
    text_tp = '제1장 총칙\n이 규정은...'
    
    # 3. True Positive Case (Should return True - pure header)
    text_tp2 = '제 5 장 보칙'

    failures = []

    if check_leak(text_fp):
        failures.append("FAIL: False positive detected. The regex incorrectly flagged a reference as a leak.")
    
    if not check_leak(text_tp):
        failures.append("FAIL: True positive missed. The regex failed to flag a real leak (start of line).")

    if not check_leak(text_tp2):
        failures.append("FAIL: True positive missed. The regex failed to flag a real leak (pure header).")

    if failures:
        for f in failures:
            print(f)
        exit(1)
    else:
        print("PASS: All cases handled correctly.")
        exit(0)