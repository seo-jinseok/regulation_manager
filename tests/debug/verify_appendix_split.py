
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from formatter import RegulationFormatter

def check_parsing(text_to_parse):
    formatter = RegulationFormatter()
    full_text = text_to_parse
    
    addenda, attached = formatter._parse_appendices(full_text)
    
    print(f"Original Text Length: {len(full_text)}")
    print("-" * 20)
    print(f"Addenda Count: {len(addenda)}")
    for i, a in enumerate(addenda):
        print(f"Addenda {i} Title: {a['title']}")
        if a['children']:
            print(f"Addenda {i} Children Count: {len(a['children'])}")
            print(f"Addenda {i} Text Field: {a['text']}") # Expect None
            for child in a['children']:
                print(f"  - Child Level: {child['level']}, Text: {child['text'][:30]}...")
        else:
            print(f"Addenda {i} Text First Line: {a['text'].splitlines()[0] if a['text'] else 'EMPTY'}")
        
    print("-" * 20)
    print(f"Attached Files Count: {len(attached)}")
    for i, a in enumerate(attached):
        print(f"Attached {i} Title: {a['title']}")
        print(f"Attached {i} Content Start: {a['text'][:50]}...")
    print("-" * 20)


def test_parsing():
    print("=== Test 1: Standard Bracket Style ===")
    text = """
1. 이 규정은 2007년 4월 1일부터 시행한다.
2. 이 변경 규정은 2008년 2월 1일부터 시행한다.
[별 표 1] <삭제 2019. 1. 1>
[별 표 2] 연구실 사고 시 긴급대처 방안과 행동요령 <신설 2017. 7. 1>
| 사고유형 | 긴급대처방안과 행동요령 |
| 일반사항 | - 사고 발생 시 즉시 응급조치를 취한 후... |
    """.strip()
    check_parsing("부 칙\n" + text)

    print("\n=== Test 3: Bracket Style Appendix with Spaces ===")
    text3 = """
부 칙
1. 이 규정은 2024년 1월 1일부터 시행한다.
[별 표 1]
테스트 별표 내용입니다.
    """
    check_parsing(text3)
    
    print("\n=== Test 4: Angle Bracket Style Appendix ===")
    text4 = """
부 칙
1. 이 규정은 시행한다.
<별지 제1호 서식>
서식 내용입니다.
<별지 제2호 서식>
또 다른 서식.
    """
    check_parsing(text4)

    print("\n=== Test 5: Angle Bracket Style Appendix with Spaces ===")
    text5 = """
부 칙
1. 시행일.
< 별 지 1 >
내용.
    """
    check_parsing(text5)

if __name__ == "__main__":
    test_parsing()
