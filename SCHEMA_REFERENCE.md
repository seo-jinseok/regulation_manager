# 규정 JSON 스키마 레퍼런스

이 문서는 규정 관리 시스템이 생성하는 JSON 출력 결과의 구조를 정의합니다.

---

## 문서 개요

### 이 문서의 목적

본 문서는 `regulation convert` 명령으로 생성되는 JSON 파일의 스키마를 명세합니다. 개발자, 데이터 엔지니어, 또는 규정 데이터를 활용하려는 분들을 위한 기술 참조 문서입니다.

### 대상 독자

- RAG(Retrieval-Augmented Generation) 시스템 개발자
- 규정 데이터를 데이터베이스에 적재하려는 데이터 엔지니어
- JSON 데이터를 직접 활용하려는 애플리케이션 개발자

### 스키마 설계 철학

본 스키마는 다음 원칙을 따라 설계되었습니다:

1. **계층 구조 보존**: 한국 법령 체계(편/장/절/조/항/호/목)를 그대로 반영하여 원문의 구조를 손실 없이 표현
2. **RAG 최적화**: 벡터 검색에 필요한 필드(`embedding_text`, `parent_path` 등)를 사전 계산하여 제공
3. **재현성**: 동일한 입력에 대해 동일한 출력(결정적 UUID 등)을 보장하여 증분 동기화 지원
4. **확장성**: 향후 필드 추가에 유연하게 대응할 수 있는 스키마 버전 관리

---

## 목차

- [루트 객체](#루트-객체-root-object)
- [문서 객체](#문서-객체-document-object)
- [노드 객체](#노드-객체-node-object---recursive)
- [부칙 파싱](#부칙-파싱-addenda-parsing)
- [RAG 최적화 필드](#rag-최적화-필드-rag-enhancement-fields)
- [사용 사례](#사용-사례)
- [관련 문서](#관련-문서)

---

## 루트 객체 (Root Object)

JSON 파일의 최상위 루트는 단일 소스 파일에서 파싱된 문서(규정)들의 집합을 나타냅니다.

| 필드명 | 타입 | 필수 | 설명 |
|--------|------|------|------|
| `schema_version` | `string` | ✓ | JSON 출력 스키마 버전입니다 (예: `"v4"`). |
| `generated_at` | `string` | ✓ | 생성 시각(UTC, ISO 8601)입니다. |
| `pipeline_signature` | `string` | ✓ | 캐시 및 재현성을 위한 파이프라인 시그니처입니다. |
| `file_name` | `string` | ✓ | 원본 HWP 파일의 이름입니다. |
| `toc` | `Array<Object>` | ✓ | 규정집 목차 엔트리 리스트입니다. 각 엔트리는 `title`, `rule_code`를 가집니다. |
| `index_by_alpha` | `Array<Object>` | ✓ | 찾아보기(가나다순) 엔트리 리스트입니다. |
| `index_by_dept` | `Object<string, Array<Object>>` | ✓ | 찾아보기(소관부서별) 엔트리 맵입니다. |
| `docs` | `Array<Document>` | ✓ | 파싱된 규정 문서들의 리스트입니다. |
| `rag_enhanced` | `boolean` | - | RAG 후처리 완료 여부입니다 (후처리 시 `true`). |
| `rag_schema_version` | `string` | - | RAG 스키마 버전입니다 (예: `"2.0"`). |

### 예시

```json
{
  "schema_version": "v4",
  "generated_at": "2026-01-03T05:00:00Z",
  "pipeline_signature": "hwp5html-v0.1.16+parser-v2.1",
  "file_name": "규정집9-343(20250909).hwp",
  "toc": [
    { "title": "학칙", "rule_code": "2-1-1" },
    { "title": "장학금규정", "rule_code": "2-2-5" }
  ],
  "index_by_alpha": [...],
  "index_by_dept": { "교무처": [...], "학생처": [...] },
  "docs": [...],
  "rag_enhanced": true,
  "rag_schema_version": "2.0"
}
```

> **팁**: 데이터베이스 적재 시 `rag_enhanced=true`이고 `rag_schema_version`이 존재하면 RAG 필드(예: `keywords`, `chunk_level`)가 포함된 것으로 판단할 수 있습니다.

---

## 문서 객체 (Document Object)

`docs` 배열의 각 객체는 하나의 독립된 규정(예: "학칙", "장학 규정")을 나타냅니다.

| 필드명 | 타입 | 필수 | 설명 |
|--------|------|------|------|
| `doc_type` | `string` | ✓ | 문서 타입. Enum: `regulation`, `toc`, `index_alpha`, `index_dept`, `index`, `note`, `unknown` |
| `title` | `string` | ✓ | 규정의 공식 명칭입니다 (예: "동의대학교학칙"). |
| `part` | `string` | - | 해당 규정이 속한 편(Part) 또는 범주입니다 (예: "제2편 학칙"). |
| `metadata` | `Object` | ✓ | 처리 메타데이터입니다 (스캔 일시, 소스 파일명, 규정 코드 등). |
| `preamble` | `string` | - | 제1조 또는 제1장 이전에 나오는 서문 텍스트입니다. |
| `content` | `Array<Node>` | ✓ | 규정 본문의 계층적 구조 리스트입니다 (장, 조 등). |
| `addenda` | `Array<Node>` | - | 부칙(Addenda)을 나타내는 구조화된 노드 리스트입니다. |
| `attached_files` | `Array<Object>` | - | 별표, 별지 서식과 같은 부속 파일 리스트입니다. |
| `status` | `string` | - | 규정 상태. Enum: `active`, `abolished`. (RAG 후처리 시 추가) |
| `abolished_date` | `string` | - | 폐지 일자 (ISO 8601). (RAG 후처리 시 추가) |

### metadata 객체

| 필드명 | 타입 | 설명 |
|--------|------|------|
| `rule_code` | `string` | 규정 번호 (예: "3-1-24") |
| `source_file` | `string` | 원본 HWP 파일명 |
| `scanned_at` | `string` | 스캔 일시 (ISO 8601) |

---

## 노드 객체 (Node Object - Recursive)

규정 본문(`content`)과 부칙(`addenda`)의 세부 내용은 **노드(Node)** 의 트리 구조로 표현됩니다. 모든 구조적 요소(장, 절, 조, 항, 호, 목)는 하나의 노드가 됩니다.

### 필드 정의

| 필드명 | 타입 | 필수 | 설명 |
|--------|------|------|------|
| `id` | `string` | ✓ | 결정적 UUID(uuid5) 문자열입니다. 동일 입력에 대해 동일한 값이 생성됩니다. |
| `type` | `string` | ✓ | 구조적 레벨 타입입니다. 아래 표 참조. |
| `display_no` | `string` | - | 원문 표시 번호입니다 (예: "제5조", "①", "1.", "가."). |
| `sort_no` | `Object` | ✓ | 정렬용 숫자 키입니다. `{ "main": int, "sub": int }` |
| `title` | `string` | - | 노드의 제목입니다 (예: "총칙", "목적"). |
| `text` | `string` | ✓ | 해당 노드의 본문 텍스트입니다 (자식 노드의 텍스트는 포함하지 않음). |
| `confidence_score` | `float` | - | 해당 노드 추출의 신뢰도 점수 (0.0 ~ 1.0). |
| `references` | `Array<Object>` | - | 본문 내에서 발견된 다른 조항에 대한 상호 참조 리스트입니다. |
| `children` | `Array<Node>` | ✓ | 중첩된 하위(자식) 노드들의 리스트입니다. |
| `metadata` | `Object` | - | 노드별 부가 정보입니다. |

### 노드 타입 (type)

| 타입 | 한글명 | 설명 | 예시 |
|------|--------|------|------|
| `chapter` | 장 | 가장 상위의 그룹핑 단위 | 제1장, 제2장 |
| `section` | 절 | 장(Chapter)의 하위 구분 | 제1절, 제2절 |
| `subsection` | 관 | 절(Section)의 하위 구분 | 제1관, 제2관 |
| `article` | 조 | 법령의 가장 기본적인 단위 | 제1조, 제2조 |
| `paragraph` | 항 | 원문자 번호를 가지는 하위 단위 | ①, ②, ③ |
| `item` | 호 | 숫자 번호 아이템 | 1., 2., 3. |
| `subitem` | 목 | 가나다 순의 하위 아이템 | 가., 나., 다. |
| `addendum` | 부칙헤더 | 부칙 섹션 헤더 | 부 칙 |
| `addendum_item` | 부칙항목 | 부칙 내 개별 항목 | 1., 2. |
| `text` | 텍스트 | 구조화되지 않은 일반 텍스트 | - |

### 계층 구조 (Hierarchy)

계층 구조는 아래 순서를 따릅니다. 중간 레벨은 생략될 수 있습니다 (예: 장 없이 조가 바로 나올 수 있음).

```
chapter (장)
  └─ section (절)
       └─ subsection (관)
            └─ article (조)
                 └─ paragraph (항)
                      └─ item (호)
                           └─ subitem (목)
```

### 예시

#### 조(Article)와 항(Paragraph)

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

#### 호(Item)와 목(Subitem)

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

---

## 부칙 파싱 (Addenda Parsing)

부칙(Addenda) 또한 노드 리스트로 구조화됩니다.

### 구조

- **부칙 헤더**: `type: "addendum"`으로 표현
- **부칙 항목**: 번호가 있는 개정 사항 (예: "1. (시행일) ...")은 `type: "addendum_item"`으로 처리
- **텍스트 처리**: `children` 노드가 생성된 경우, 상위 부칙 노드의 `text` 필드는 중복 방지를 위해 빈 문자열로 설정될 수 있습니다.

### 예시

```json
"addenda": [
  {
    "type": "addendum",
    "title": "부 칙",
    "text": "",
    "metadata": { "has_text": false },
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

---

## RAG 최적화 필드 (RAG Enhancement Fields)

`enhance_for_rag.py` 스크립트로 후처리 시 추가되는 필드들입니다. Hybrid RAG(키워드 + 벡터 검색) 데이터베이스 구축에 최적화되어 있습니다.

### 노드(Node) 레벨 추가 필드

| 필드명 | 타입 | 설명 | 용도 |
|--------|------|------|------|
| `parent_path` | `Array<string>` | 루트부터 현재 노드까지의 계층 경로 | 검색 결과에 문맥 정보 제공, 필터링 |
| `full_text` | `string` | 벡터 임베딩용 self-contained 텍스트 | 임베딩 생성 |
| `embedding_text` | `string` | 임베딩용 컨텍스트 강화 텍스트 (아래 참조) | 임베딩 생성 |
| `chunk_level` | `string` | 검색 청크 레벨 (article, paragraph, item 등) | 청크 필터링 |
| `is_searchable` | `boolean` | 검색 대상 여부 | 검색 대상 필터링 |
| `token_count` | `integer` | 임베딩 텍스트의 추정 토큰 수 | 청크 크기 관리 |
| `keywords` | `Array<Object>` | 본문 핵심 키워드 (`term`, `weight`) | BM25 검색 보조 |
| `amendment_history` | `Array<Object>` | 개정/신설/삭제 이력 (`date`, `type`) | 이력 추적 |
| `effective_date` | `string` | 시행일 (YYYY-MM-DD) | 시간 기반 필터링 |

#### embedding_text 필드

벡터 검색을 위한 컨텍스트 강화 텍스트입니다.

**형식**:
```
규정명 > 장 > 절 > 관 > 조문번호 제목: 본문
```

**예시**:
```
동의대학교학칙 > 제3장 학사 > 제1절 수업 > 제15조 수업일수: 수업일수는 연간 16주 이상으로 한다.
```

**특징**:
- 규정명 (문서 제목) 항상 포함
- 최근 3개 계층 세그먼트 포함 (장 > 절 > 관 또는 절 > 관 > 조)
- 중복 세그먼트 자동 제거 (공백 정규화 후 비교)
- 토큰 수 최적화를 위해 최대 4개 세그먼트로 제한 (규정명 + 3개)

### 예시 (RAG Enhanced Node)

```json
{
  "type": "paragraph",
  "display_no": "①",
  "sort_no": { "main": 1, "sub": 0 },
  "title": "",
  "text": "이사와 감사는 이사회에서 선임하여 관할청의 승인을 받아 취임한다. (개정 2006.11.06., 2022.04.21.)",
  "parent_path": ["학교법인동의학원정관", "제4장 임원", "제24조 임원의 선임방법"],
  "full_text": "[학교법인동의학원정관 > 제4장 임원 > 제24조 임원의 선임방법 > ①] 이사와 감사는 이사회에서 선임하여...",
  "embedding_text": "이사와 감사는 이사회에서 선임하여 관할청의 승인을 받아 취임한다.",
  "chunk_level": "paragraph",
  "is_searchable": true,
  "token_count": 38,
  "keywords": [
    { "term": "이사", "weight": 1.0 },
    { "term": "감사", "weight": 1.0 },
    { "term": "선임", "weight": 0.7 },
    { "term": "승인", "weight": 0.8 }
  ],
  "amendment_history": [
    { "date": "2006-11-06", "type": "개정" },
    { "date": "2022-04-21", "type": "개정" }
  ],
  "children": []
}
```

---

## 사용 사례

### 특정 조항 검색

규정명과 조문 번호로 특정 조항을 찾는 방법입니다.

```python
import json

with open("data/output/규정집.json") as f:
    data = json.load(f)

# 규정명으로 문서 찾기
def find_regulation(title: str):
    for doc in data["docs"]:
        if title in doc["title"]:
            return doc
    return None

# 조문 번호로 노드 찾기 (재귀)
def find_article(node: dict, display_no: str):
    if node.get("display_no") == display_no:
        return node
    for child in node.get("children", []):
        result = find_article(child, display_no)
        if result:
            return result
    return None

# 사용 예시
reg = find_regulation("교원인사규정")
if reg:
    for content_node in reg["content"]:
        article = find_article(content_node, "제15조")
        if article:
            print(article["text"])
```

### RAG 시스템에서의 활용

벡터 데이터베이스에 적재할 청크를 추출하는 방법입니다.

```python
def extract_searchable_chunks(doc: dict) -> list:
    """is_searchable=True인 노드만 추출"""
    chunks = []
    
    def traverse(node: dict, path: list):
        if node.get("is_searchable", False):
            chunks.append({
                "id": node["id"],
                "text": node.get("embedding_text", node["text"]),
                "parent_path": node.get("parent_path", path),
                "rule_code": doc["metadata"]["rule_code"],
                "regulation_name": doc["title"],
            })
        for child in node.get("children", []):
            traverse(child, node.get("parent_path", path))
    
    for content_node in doc.get("content", []):
        traverse(content_node, [doc["title"]])
    
    return chunks

# 사용 예시
chunks = []
for doc in data["docs"]:
    if doc["doc_type"] == "regulation":
        chunks.extend(extract_searchable_chunks(doc))

print(f"총 {len(chunks)}개의 검색 가능한 청크")
```

### 개정 이력 추적

특정 규정의 모든 개정 이력을 추출하는 방법입니다.

```python
def extract_amendment_history(doc: dict) -> list:
    """모든 노드에서 개정 이력 추출"""
    history = []
    
    def traverse(node: dict):
        for amendment in node.get("amendment_history", []):
            history.append({
                "node_id": node["id"],
                "display_no": node.get("display_no"),
                **amendment
            })
        for child in node.get("children", []):
            traverse(child)
    
    for content_node in doc.get("content", []):
        traverse(content_node)
    
    return sorted(history, key=lambda x: x["date"], reverse=True)

# 사용 예시
reg = find_regulation("학칙")
if reg:
    amendments = extract_amendment_history(reg)
    for a in amendments[:10]:
        print(f"{a['date']}: {a['display_no']} ({a['type']})")
```

---

## 관련 문서

| 문서 | 설명 |
|------|------|
| [README.md](./README.md) | 시스템 개요 및 상세 기술 설명 |
| [QUICKSTART.md](./QUICKSTART.md) | 빠른 시작 가이드 |
| [LLM_GUIDE.md](./LLM_GUIDE.md) | LLM 설정 가이드 |
| [AGENTS.md](./AGENTS.md) | 개발자 및 AI 에이전트 가이드 |
