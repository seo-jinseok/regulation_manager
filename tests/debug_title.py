import re

def _extract_clean_title(preamble_list):
    candidates = [line.strip() for line in preamble_list if line.strip()]
    
    best_title = ""
    start_debug = False
    
    suffixes = ("규정", "세칙", "지침", "요령", "강령", "내규", "학칙", "헌장", "기준", "수칙", "준칙", "요강", "운영", "정관")

    print(f"Total candidates: {len(candidates)}")
    for i, line in enumerate(reversed(candidates)):
        print(f"Check {i}: {line}")
        
        # Skip if meta info
        if (line.startswith('<') and line.endswith('>')) or (line.startswith('(') and line.endswith(')')):
            print("  -> Skip meta")
            continue
        
        # Clean
        cleaned = re.sub(r'\s*[<\[\(].*?[>\]\)]', '', line).strip()
        cleaned = re.sub(r'\s*제\s*\d+\s*장.*', '', cleaned).strip()
            
        if not cleaned: 
            print("  -> Empty after clean")
            continue
        
        # Verify if it looks like a title
        if cleaned.endswith(suffixes) or "규정" in cleaned:
            print(f"  -> MATCH: {cleaned}")
            best_title = cleaned
            break
        else:
            print(f"  -> No suffix match: {cleaned}")

    return best_title

# Text from the JSON Preamble
text = """「별표 양식 1」
「별표 양식 1」
規 程 集
![](bindata/BIN0001.bmp)
동의대학교
편찬례 및 취급 요령
1. 이 규정집은 ...
...
2019년 2월 1일
기획처 기획팀
규정집 추록 가제 정리대장
| 추록횟수 | 가제 정리 | 정리자 | 부서장 | 비고 |
...
규정집 관리 현황표
...
차 례
찾아보기
<가나다순>
찾아보기
<소관부서별>
...
학생군사교육단
| 제1편 | | |
| 학 교 법 인 | | |
| 제1편 | | |
| 학교법인 | | |
학교법인동의학원정관"""

preamble_list = text.split('\n')
print(f"Extracted Title: {_extract_clean_title(preamble_list)}")
