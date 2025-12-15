
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from formatter import RegulationFormatter

def test_parsing():
    formatter = RegulationFormatter()
    
    # Sample text from the issue
    text = """
1. 이 규정은 2007년 4월 1일부터 시행한다.
2. 이 변경 규정은 2008년 2월 1일부터 시행한다.
[별 표 1] <삭제 2019. 1. 1>
[별 표 2] 연구실 사고 시 긴급대처 방안과 행동요령 <신설 2017. 7. 1>
| 사고유형 | 긴급대처방안과 행동요령 |
| 일반사항 | - 사고 발생 시 즉시 응급조치를 취한 후... |
    """.strip()
    
    # Prefix with "부 칙" as the splitter expects it if it's the start
    full_text = "부 칙\n" + text
    
    addenda, attached = formatter._parse_appendices(full_text)
    
    print(f"Original Text Length: {len(full_text)}")
    print("-" * 20)
    print(f"Addenda Count: {len(addenda)}")
    for i, a in enumerate(addenda):
        print(f"Addenda {i} Title: {a['title']}")
        print(f"Addenda {i} Text First Line: {a['text'].splitlines()[0]}")
        print(f"Addenda {i} Text Last Line: {a['text'].splitlines()[-1]}")
        
    print("-" * 20)
    print(f"Attached Files Count: {len(attached)}")
    for i, a in enumerate(attached):
        print(f"Attached {i} Title: {a['title']}")
        print(f"Attached {i} Content Start: {a['text'][:50]}...")

if __name__ == "__main__":
    test_parsing()
