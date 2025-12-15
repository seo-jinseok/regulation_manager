{
  "regulation_id": "REG-2024-001",
  "title": "○○대학교 학칙",
  "metadata": {...},
  "content": [
    {
      "level": "chapter",
      "number": "1",
      "title": "총칙",
      "text": null,
      "children": [
        {
          "level": "article",
          "number": "1",
          "title": "목적",
          "text": "이 학칙은 ○○대학교의...",
          "children": [
            {
              "level": "paragraph",
              "number": "1",
              "title": null,
              "text": "학칙의 세부사항은...",
              "children": [
                {
                  "level": "item",
                  "number": "1",
                  "text": "학사 운영에 관한 사항",
                  "children": []
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

## AI 프롬프트 개선안
```
당신은 한국 법령 및 규정 구조화 전문가입니다.
다음 규정 텍스트를 계층적 JSON으로 변환해주세요.

[중요 원칙]
1. 존재하는 구조만 표현 (장이 없으면 조부터 시작)
2. 계층 레벨: chapter(장) > section(절) > article(조) > paragraph(항) > item(호) > subitem(목)
3. 각 노드는 level, number, title(있는 경우), text, children 속성 포함
4. 빈 children은 빈 배열 []로 표시

[인식 패턴]
- 장: "제N장"
- 절: "제N절"
- 조: "제N조"
- 항: ①, ②, ③
- 호: 1., 2., 3.
- 목: 가., 나., 다.

[특수 케이스]
- 단서 조항: "다만," "단," 등은 같은 레벨의 text에 포함
- 별표/별지: appendix 배열에 별도 저장
- 개정 이력: metadata의 amendments 배열에 저장

[입력 텍스트]
(여기에 HWP 텍스트)

[출력]
위 스키마를 따르는 JSON만 출력 (설명 없이)