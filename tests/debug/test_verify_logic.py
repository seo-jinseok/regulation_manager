import re

def check_leak(text):
    if re.search(r'(^|\s)제\s*\d+\s*장($|\s)', text):
        return True
    return False

if __name__ == "__main__":
    # This text contains "제7장" as a reference, which should NOT be considered a leak header.
    # A header leak would look like "제1장 총칙\n제1조..." where "제1장 총칙" is part of the text.
    text = '이 규정은 학교법인동의학원정관(이하 “정관”이라 한다) 제7장 제2절 및 동의대학교(이하 “본 대학교”라 한...'
    
    if check_leak(text):
        print("FAIL: False positive detected. The regex incorrectly flagged a reference as a leak.")
        exit(1)
    else:
        print("PASS: Correctly identified as not a leak.")
        exit(0)

