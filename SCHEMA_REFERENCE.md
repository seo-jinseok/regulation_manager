# 규정 JSON 스키마 레퍼런스 (Regulation JSON Schema Reference)

이 문서는 `regulation_manager`가 생성하는 JSON 출력 결과의 구조를 정의합니다. 이 스키마는 대한민국 법령(법률, 시행령, 대학 규정)을 데이터베이스에 적재하기 적합하도록, 원문의 계층 구조를 완벽하게 보존하여 표현하도록 설계되었습니다.

## 루트 객체 (Root Object)

JSON 파일의 최상위 루트는 단일 소스 파일에서 파싱된 문서(규정)들의 집합을 나타냅니다.

| 필드명 | 타입 | 설명 |
| :--- | :--- | :--- |
| `file_name` | `string` | 원본 HWP 파일의 이름입니다. |
| `docs` | `Array<Document>` | 파싱된 규정 문서(`Document`)들의 리스트입니다. |

## 문서 객체 (Document Object)

`docs` 배열의 각 객체는 하나의 독립된 규정(예: "학칙", "장학 규정")을 나타냅니다.

| 필드명 | 타입 | 설명 |
| :--- | :--- | :--- |
| `title` | `string` | 규정의 공식 명칭입니다 (예: "동의대학교학칙"). |
| `part` | `string` | *(선택)* 해당 규정이 속한 편(Part) 또는 범주입니다 (예: "제2편 학칙"). |
| `metadata` | `Object` | 처리 메타데이터입니다 (스캔 일시, 소스 파일명, 규정 코드 등). |
| `preamble` | `string` | 제1조 또는 제1장 이전에 나오는 서문 텍스트입니다. |
| `content` | `Array<Node>` | 규정 본문의 계층적 구조 리스트입니다 (장, 조 등). |
| `addenda` | `Array<Node>` | 부칙(Addenda)을 나타내는 구조화된 노드 리스트입니다. |
| `attached_files` | `Array<Object>` | 별표, 별지 서식과 같은 부속 파일 리스트입니다. 각 객체는 `title`, `text`, 그리고 시각적 레이아웃 보존을 위한 `html` 필드를 가질 수 있습니다. |

## 노드 객체 (Node Object - Recursive)

규정 본문(`content`)과 부칙(`addenda`)의 세부 내용은 **노드(Node)** 의 트리 구조로 표현됩니다. 모든 구조적 요소(장, 절, 조, 항, 호, 목)는 하나의 노드가 됩니다.

| 필드명 | 타입 | 설명 |
| :--- | :--- | :--- |
| `id` | `string` | UUID v4 문자열입니다. |
| `type` | `string` | 구조적 레벨 타입입니다. Enum: `chapter`(장), `section`(절), `subsection`(관), `article`(조), `paragraph`(항), `item`(호), `subitem`(목), `addendum`(부칙헤더), `addendum_item`(부칙항목), `text`. |
| `display_no` | `string` | 원문 표시 번호입니다 (예: "제5조", "①", "1.", "가."). 없을 수 있습니다. |
| `sort_no` | `Object` | 정렬용 숫자 키입니다. `{ "main": int, "sub": int }` |
| `title` | `string` | 노드의 제목입니다 (예: "총칙", "목적"). 없을 수 있습니다(null). |
| `text` | `string` | 해당 노드의 본문 텍스트입니다 (자식 노드의 텍스트는 포함하지 않음). |
| `confidence_score` | `float` | 해당 노드 추출의 신뢰도 점수 (0.0 ~ 1.0) 입니다. |
| `references` | `Array<Object>` | 본문 내에서 발견된 다른 조항/항목에 대한 상호 참조 리스트입니다. |
| `children` | `Array<Node>` | 중첩된 하위(자식) 노드들의 리스트입니다. |

### 노드 레벨 및 계층 구조 (Hierarchy)

계층 구조는 아래 순서를 엄격히 따릅니다 (단, 중간 레벨은 생략될 수 있습니다. 예: 장(Chapter) 없이 조(Article)가 바로 나올 수 있음).

1.  **`chapter` (장)**: 가장 상위의 그룹핑 단위입니다.
2.  **`section` (절)**: 장(Chapter)의 하위 구분입니다.
3.  **`subsection` (관)**: 절(Section)의 하위 구분입니다.
4.  **`article` (조)**: 법령의 가장 기본적인 단위입니다 (예: "제1조").
5.  **`paragraph` (항)**: 원문자 번호(①, ②)를 가지거나, 조 하위의 문단입니다.
6.  **`item` (호)**: 숫자 번호 아이템입니다 (1., 2.).
7.  **`subitem` (목)**: 가나다 순의 하위 아이템입니다 (가., 나.).

### 예시 (Examples)

#### 1. 조(Article)와 항(Paragraph)
```json
{
  "type": "article",
  "display_no": "제6조",
  "sort_no": { "main": 6, "sub": 0 },
  "title": "자산의 구분",
  "text": "",
  "children": [
    {
      "type": "paragraph",
      "display_no": "①",
      "sort_no": { "main": 1, "sub": 0 },
      "text": "이 법인의 자산은 기본재산과 보통재산으로 구분한다.",
      "children": []
    }
  ]
}
```

#### 2. 호(Item)와 목(Subitem)
```json
{
  "type": "item",
  "display_no": "1.",
  "sort_no": { "main": 1, "sub": 0 },
  "text": "교원 인사 관련 사항",
  "children": [
    {
      "type": "subitem",
      "display_no": "가.",
      "sort_no": { "main": 1, "sub": 0 },
      "text": "신규 채용에 관한 사항",
      "children": []
    }
  ]
}
```

## 부칙 파싱 (Addenda Parsing)

부칙(Addenda) 또한 노드 리스트로 구조화됩니다.
*   **부칙 항목**: 번호가 있는 개정 사항 (예: "1. (시행일) ...")은 `type: "addendum_item"`으로 처리됩니다.
*   **텍스트 처리**: `children` 노드가 생성된 경우, 상위 부칙 노드의 `text` 필드는 중복을 방지하기 위해 빈 문자열로 설정될 수 있습니다.

```json
"addenda": [
  {
    "type": "addendum",
    "title": "부 칙",
    "text": "", 
    "children": [
      {
        "type": "addendum_item",
        "display_no": "1.",
        "sort_no": { "main": 1, "sub": 0 },
        "text": "이 정관은 1981년 9월 11일부터 시행한다.",
        "children": []
      }
    ]
  }
]
```
