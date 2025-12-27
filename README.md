# 규정 관리 시스템

대학 규정집(HWP)을 구조화된 JSON으로 변환하고, AI 기반 검색을 제공하는 시스템입니다.

---

## 개요

이 시스템은 두 가지 핵심 기능을 제공합니다:

1. **규정 변환**: HWP 파일 → 구조화된 JSON
2. **규정 검색**: 자연어 질의 기반 AI 검색 및 답변 생성

```
[HWP 파일] → [JSON 변환] → [벡터 DB 저장] → [자연어 검색/질문]
```

---

## 사용 방법

### 설치

```bash
git clone <repository-url> && cd regulation_manager
uv venv && source .venv/bin/activate
uv sync
```

### 기본 워크플로우

```bash
# 1. HWP 파일을 JSON으로 변환
uv run regulation-manager "data/input/규정집.hwp"

# 2. 벡터 DB에 저장
uv run regulation-rag sync data/output/규정집.json

# 3. 규정 검색
uv run regulation-rag search "교원 연구년 신청 자격"

# 4. AI에게 질문 (선택)
uv run regulation-rag ask "교원 연구년 신청 자격은?"
```

### 검색 옵션

| 옵션 | 설명 |
|------|------|
| `-n 10` | 검색 결과 개수 지정 |
| `--include-abolished` | 폐지된 규정 포함 |
| `--no-rerank` | AI 재정렬 비활성화 (빠른 검색) |

### LLM 질문 옵션

| 옵션 | 설명 |
|------|------|
| `--provider ollama` | LLM 프로바이더 (ollama, lmstudio, openai 등) |
| `--model gemma2` | 사용할 모델명 |
| `-v`, `--verbose` | 상세 정보 출력 (LLM 설정, 인덱스 구축 등) |
| `--show-sources` | 참고 규정 전문 출력 |

### 웹 UI

비개발자를 위한 통합 웹 인터페이스를 제공합니다.

```bash
uv run regulation-web
```

파일 업로드 → 변환 → DB 동기화 → 질문까지 한 화면에서 진행할 수 있습니다.

### MCP 서버

AI 에이전트(Claude, Cursor 등)에서 규정 검색 기능을 사용할 수 있는 MCP(Model Context Protocol) 서버를 제공합니다.

```bash
# MCP 서버 실행 (stdio 모드)
uv run regulation-mcp
```

**지원 도구 (Tools)**:

| Tool | 설명 |
|------|------|
| `search_regulations` | 규정 검색 (Hybrid + Rerank) |
| `ask_regulations` | AI 질문-답변 |
| `get_sync_status` | 동기화 상태 조회 |

> DB 관리(sync, reset)는 CLI로 수행합니다.

**클라이언트 연결 설정** (Claude Desktop 예시):

```json
{
  "mcpServers": {
    "regulation-rag": {
      "command": "uv",
      "args": ["run", "regulation-mcp"],
      "cwd": "/path/to/regulation_manager"
    }
  }
}
```

---

## 명령어 요약

### 규정 변환

| 명령어 | 설명 |
|--------|------|
| `regulation-manager "파일.hwp"` | HWP → JSON 변환 |
| `regulation-manager "파일.hwp" --use_llm` | LLM 전처리 활성화 (품질 향상) |
| `regulation-manager "파일.hwp" --no-enhance-rag` | RAG 최적화 비활성화 |

### RAG 시스템

| 명령어 | 설명 |
|--------|------|
| `regulation-rag sync <json>` | JSON → 벡터 DB 동기화 |
| `regulation-rag sync <json> --full` | 전체 재동기화 |
| `regulation-rag search "<쿼리>"` | 규정 검색 |
| `regulation-rag ask "<질문>"` | AI 답변 생성 |
| `regulation-rag status` | 동기화 상태 확인 |
| `regulation-rag reset --confirm` | DB 초기화 |

---

## 시스템 개요

### 🎯 일반인을 위한 설명

이 시스템은 **"대학 규정집 검색 도우미"**입니다. 복잡한 규정집에서 원하는 내용을 빠르게 찾아주고, AI가 질문에 대한 답변을 생성해줍니다.

```
📄 HWP 규정집 → 🗃️ 데이터베이스 저장 → 🔍 검색 → 🤖 AI 답변
```

**작동 원리 (비유)**:

1. **책을 스캔해서 정리** (HWP → JSON): 두꺼운 규정집 책을 스캔해서 목차별로 정리하고, 각 조항에 태그를 붙입니다.
2. **도서관에 보관** (벡터 DB): 정리된 내용을 검색하기 쉬운 도서관 서가에 배치합니다.
3. **사서가 검색** (Hybrid Search): 두 명의 사서가 동시에 검색합니다.
   - 첫째 사서: 정확한 단어를 찾습니다 ("제15조" → 정확히 "제15조" 포함된 문서)
   - 둘째 사서: 의미가 비슷한 내용을 찾습니다 ("학교 쉬고 싶어요" → 휴학 관련 규정)
4. **전문가가 재검토** (Reranking): AI 전문가가 두 사서가 찾은 결과를 다시 정밀 검토하여 순서를 재정렬합니다.
5. **답변 작성** (LLM): 최종적으로 AI가 찾은 규정들을 읽고, 질문에 대한 답변을 작성합니다.

---

### 🔬 전공자를 위한 기술 설명

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              전체 처리 파이프라인                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  [1] HWP → JSON 변환                                                            │
│  ┌────────┐    ┌─────────────┐    ┌────────────────┐    ┌──────────────────┐   │
│  │  HWP   │ -> │  hwp5html   │ -> │ RegulationParser│ -> │ enhance_for_rag │   │
│  │  File  │    │ (HTML 변환) │    │ (구조화 파싱)   │    │ (RAG 필드 추가) │   │
│  └────────┘    └─────────────┘    └────────────────┘    └──────────────────┘   │
│                                                                                 │
│  [2] 벡터 DB 동기화                                                              │
│  ┌──────────────┐    ┌─────────────────┐    ┌────────────────┐                  │
│  │ JSONLoader   │ -> │ ChromaVectorStore│ -> │ BGE-M3 Embedding│                  │
│  │ (청크 추출)   │    │ (증분 동기화)    │    │ (다국어 임베딩) │                  │
│  └──────────────┘    └─────────────────┘    └────────────────┘                  │
│                                                                                 │
│  [3] 질문(Ask) 처리 파이프라인                                                    │
│  ┌───────┐    ┌──────────────┐    ┌───────────────────────────────────┐        │
│  │ Query │ -> │ QueryAnalyzer│ -> │        Hybrid Search              │        │
│  └───────┘    │ • 유형 분석   │    │  ┌─────────┐    ┌─────────────┐  │        │
│               │ • 동의어 확장 │    │  │  BM25   │ -> │             │  │        │
│               │ • 불용어 제거 │    │  │(Sparse) │    │  RRF 융합   │  │        │
│               └──────────────┘    │  └─────────┘    │  (k=60)     │  │        │
│                                   │  ┌─────────┐    │             │  │        │
│                                   │  │ Dense   │ -> │             │  │        │
│                                   │  │(BGE-M3) │    └─────────────┘  │        │
│                                   │  └─────────┘                     │        │
│                                   └──────────────────────────────────┘        │
│                                                   ↓                            │
│  ┌────────────────────────┐    ┌──────────────────────────────────────┐       │
│  │   BGE Reranker v2-m3   │ <- │ Top-K 후보 (BM25 60% + Dense 40%)    │       │
│  │   (Cross-Encoder)      │    └──────────────────────────────────────┘       │
│  │   • Query-Doc Pair Scoring                                         │       │
│  │   • 조항 번호 매칭 보너스                                            │       │
│  └────────────────────────┘                                                   │
│                ↓                                                               │
│  ┌──────────────────────────────────────────────────────────────────┐         │
│  │                    LLM 답변 생성                                  │         │
│  │  • Context: Top-N 재정렬 결과 (parent_path 포함)                  │         │
│  │  • Confidence: Reranker Score 기반 신뢰도 계산                    │         │
│  │  • Provider: ollama / lmstudio / openai / gemini                 │         │
│  └──────────────────────────────────────────────────────────────────┘         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 상세 처리 단계

### 1️⃣ HWP → JSON 변환 (문서 구조화)

HWP 파일의 복잡한 규정 내용을 **계층적 JSON 구조**로 변환합니다.

**처리 단계:**

| 단계 | 컴포넌트 | 설명 |
|------|----------|------|
| 1 | `hwp5html` | HWP를 HTML/XHTML로 변환 |
| 2 | `RegulationParser` | 편/장/절/조/항/호/목 계층 구조 파싱 |
| 3 | `ReferenceResolver` | 상호 참조 해석 ("제15조 참조" → 링크) |
| 4 | `enhance_for_rag.py` | RAG 최적화 필드 추가 |

**변환 예시:**

```
[HWP 원본]                      [JSON 출력]
제4조 (용어의 정의)       →     { "display_no": "제4조",
  1. 학과(전공)란...              "title": "용어의 정의",
  6. 교육편제 조정은...           "children": [
    가. 통합이란...                 { "display_no": "6.", "text": "교육편제 조정은...",
    나. 신설이란...                   "children": [
    다. 폐지란...                       { "display_no": "다.", "text": "폐지란..." }
                                      ]}]}
```

**RAG 최적화 필드:**

| 필드 | 타입 | 설명 | 예시 |
|------|------|------|------|
| `parent_path` | `Array<string>` | 계층 경로 (Breadcrumb) | `["학과평가규정", "제4조 용어의 정의", "6. 교육편제 조정"]` |
| `embedding_text` | `string` | 임베딩용 텍스트 | `"제4조 > 6. 교육편제 조정 > 다. 폐지: 폐지란..."` |
| `keywords` | `Array<{term, weight}>` | 핵심 키워드 | `[{"term": "학과", "weight": 0.9}]` |
| `chunk_level` | `string` | 청크 레벨 | `article`, `paragraph`, `item` |
| `is_searchable` | `boolean` | 검색 대상 여부 | `true` |

---

### 2️⃣ 벡터 DB 동기화 (ChromaDB)

변환된 JSON의 각 조항을 **청크(Chunk)** 단위로 분리하여 ChromaDB에 저장합니다.

**기술 사양:**

| 항목 | 값 | 설명 |
|------|---|------|
| **벡터 DB** | ChromaDB | 로컬 영속 저장 (`data/chroma_db/`) |
| **임베딩 모델** | `BAAI/bge-m3` | 1024차원, 다국어 지원, 한국어 최적화 |
| **청크 단위** | 조항(Article) 기준 | 평균 ~50 토큰/청크 |
| **동기화 방식** | 증분 동기화 | 해시 비교로 변경분만 업데이트 |

**메타데이터 스키마:**

```python
{
    "id": "uuid5(...)",           # 결정적 UUID (재생성 시 동일)
    "rule_code": "3-1-24",        # 규정 번호
    "regulation_name": "교원연구년제규정",
    "parent_path": ["교원연구년제규정", "제3조 자격"],
    "status": "active",           # active / abolished
    "effective_date": "2020-04-01",
}
```

---

### 3️⃣ 질문(Ask) 처리 파이프라인

사용자가 `regulation-rag ask "질문"`을 실행하면 다음 단계로 처리됩니다.

#### Step 3-1: 쿼리 분석 (QueryAnalyzer)

```python
# 입력
query = "교원 연구년 신청 자격은 무엇인가요?"

# 분석 결과
{
    "query_type": "natural_question",    # 쿼리 유형
    "bm25_weight": 0.3,                  # Sparse 가중치
    "dense_weight": 0.7,                 # Dense 가중치
    "expanded_query": "교원 연구년 신청 자격 연구년제 자격요건",  # 동의어 확장
    "cleaned_query": "교원 연구년 신청 자격",  # 불용어 제거
}
```

**쿼리 유형별 가중치:**

| 쿼리 유형 | 패턴 예시 | BM25 | Dense |
|-----------|----------|------|-------|
| 조문 번호 | `"제15조"`, `"학칙 제3조"` | 0.6 | 0.4 |
| 규정명 | `"장학금규정"`, `"휴학 학칙"` | 0.5 | 0.5 |
| 자연어 질문 | `"어떻게 휴학하나요?"` | 0.2 | 0.8 |
| 기본값 | 그 외 | 0.3 | 0.7 |

#### Step 3-2: Hybrid Search

두 가지 검색 방식을 결합하여 정확도를 높입니다.

```
Query ──┬── [BM25 Sparse Search] ────────┬── [RRF Fusion] ── 후보 결과
        │   • Okapi BM25 (k1=1.5, b=0.75)│   • k = 60
        │   • TF-IDF 기반 키워드 매칭    │   • rank_score = Σ 1/(k + rank)
        │                                │
        └── [Dense Vector Search] ───────┘
            • BGE-M3 Embedding (1024-dim)
            • Cosine Similarity
            • 의미적 유사도 기반
```

**BM25 수식:**

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

**RRF 융합 수식:**

$$\text{RRF}(d) = \sum_{r \in \text{ranklists}} \frac{1}{k + r(d)}$$

#### Step 3-3: Reranking (BGE Cross-Encoder)

Bi-Encoder 검색 결과를 Cross-Encoder로 재정렬합니다.

| 항목 | 값 |
|------|---|
| **모델** | `BAAI/bge-reranker-v2-m3` |
| **방식** | Cross-Encoder (Query-Document Pair Scoring) |
| **입력** | `[query, document]` 쌍 |
| **출력** | Relevance Score (0~1) |

**Bi-Encoder vs Cross-Encoder:**

```
[Bi-Encoder]                    [Cross-Encoder]
Query  → Encoder → Vec_q        Query + Doc → Encoder → Score
Doc    → Encoder → Vec_d              ↓
         ↓                       더 정밀하지만 느림
    cos(Vec_q, Vec_d)           (Top-K에만 적용)
    빠르지만 덜 정밀
```

**보너스 점수:**

| 조건 | 보너스 | 설명 |
|------|--------|------|
| 조문 번호 정확 매칭 | +0.15 | Query에 `제N조` 포함 & Doc에 동일 조문 |
| 규정명 정확 매칭 | +0.10 | Query에 규정명 포함 |

#### Step 3-4: LLM 답변 생성

Reranking된 상위 문서를 Context로 제공하여 LLM이 답변을 생성합니다.

**Context 구성:**

```
[1] 규정: 교원연구년제규정 (3-1-24)
경로: 교원연구년제규정 > 제3조 자격
내용: ① 첫번째 연구년제를 위한 근무년수: 본 대학교에 6년 이상 재직한 자...
관련도: 0.99

[2] 규정: 교원파견연구에관한규정 (3-1-61)
경로: 교원파견연구에관한규정 > 제3조 자격
내용: 파견연구 대상자는 본 대학교 전임교원으로서...
관련도: 0.95
```

**Prompt 구조:**

```
당신은 대학 규정 전문가입니다. 아래 규정을 참고하여 질문에 답변하세요.

[규정 내용]
{context}

[질문]
{question}

[지시사항]
- 규정에 명시된 내용만 답변하세요.
- 규정에 없는 내용은 "규정에 명시되어 있지 않습니다"라고 답변하세요.
- 출처 규정 번호를 함께 명시하세요.
```

**신뢰도 계산:**

```python
confidence = mean(top_n_reranker_scores) * answer_coverage_ratio
# top_n_reranker_scores: 상위 N개 Reranker 점수
# answer_coverage_ratio: 답변에 인용된 규정 비율
```

---

### 4️⃣ 증분 동기화

월간 규정 업데이트 시 변경된 규정만 동기화하여 처리 시간을 단축합니다.

**동기화 알고리즘:**

```python
for regulation in new_json:
    hash = sha256(regulation_content)
    if hash != stored_hash:
        if regulation_id in store:
            update(regulation)  # 수정
        else:
            add(regulation)     # 신규
    
for stored_id in store:
    if stored_id not in new_json:
        delete(stored_id)       # 삭제
```

**상태 파일 (`sync_state.json`):**

```json
{
  "last_sync": "2025-12-26T10:36:09Z",
  "json_file": "규정집9-343(20250909).json",
  "regulations": {
    "3-1-24": {"hash": "abc123...", "chunk_count": 54},
    "3-1-61": {"hash": "def456...", "chunk_count": 32}
  }
}
```

---

## 환경 설정

```bash
cp .env.example .env
```

**주요 설정:**

```bash
# LLM 기본값
LLM_PROVIDER=ollama
LLM_MODEL=gemma2
LLM_BASE_URL=http://localhost:11434

# API 키 (클라우드 LLM 사용 시)
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
```

실행 시 `.env`를 자동 로드하므로, 위 설정이 코드 기본값보다 우선 적용됩니다.

코드 기본값(옵션/`.env` 미설정 시):

| 사용 위치 | LLM 기본값 |
|-----------|------------|
| `regulation-manager` | provider: `openai` (model: `gpt-4o`) |
| `regulation-rag` / 웹 UI | provider: `ollama` (model: `gemma2`, base_url: `http://localhost:11434`) |

---

## 문서

| 문서 | 설명 |
|------|------|
| [QUICKSTART.md](./QUICKSTART.md) | 빠른 시작 가이드 |
| [docs/LLM_GUIDE.md](./docs/LLM_GUIDE.md) | LLM 설정 가이드 |
| [SCHEMA_REFERENCE.md](./SCHEMA_REFERENCE.md) | JSON 출력 스키마 명세 |
| [AGENTS.md](./AGENTS.md) | 개발자 가이드 |

---

## 문제 해결

| 문제 | 해결 방법 |
|------|----------|
| "데이터베이스가 비어 있습니다" | `sync` 명령 실행 |
| "파일을 찾을 수 없습니다" | 파일 경로 확인 (절대 경로 권장) |
| 검색 결과가 부정확함 | `--no-rerank` 제거하여 AI 재정렬 활성화 확인 |
| 변환 품질이 낮음 | `--use_llm` 옵션으로 LLM 전처리 활성화 |
| "hwp5html 실행 파일을 찾을 수 없습니다" | `hwp5html` 설치 후 다시 실행 (변환은 hwp5html CLI 필요) |

---

# 개발자 가이드

## 프로젝트 구조

```
regulation_manager/
├── src/
│   ├── main.py              # 변환 파이프라인 진입점
│   ├── converter.py         # HWP → Markdown/HTML
│   ├── formatter.py         # Markdown → JSON
│   ├── enhance_for_rag.py   # RAG 최적화 필드 추가
│   ├── parsing/             # 파싱 모듈
│   └── rag/                 # RAG 시스템 (Clean Architecture)
│       ├── interface/       # CLI, Web UI
│       ├── application/     # Use Cases
│       ├── domain/          # 도메인 모델
│       └── infrastructure/  # ChromaDB, Reranker, LLM
├── data/
│   ├── input/               # HWP 파일 입력
│   ├── output/              # JSON 출력
│   └── chroma_db/           # 벡터 DB 저장소
└── tests/                   # pytest 테스트
```

## RAG 아키텍처

```
[Query] → [Hybrid Search] → [BGE Reranker] → [LLM 답변 생성]
                 ↓                  ↓
         BM25 + Dense         Cross-Encoder
        (RRF 융합)             Reranking
```

### 핵심 컴포넌트

| 컴포넌트 | 파일 | 설명 |
|----------|------|------|
| 벡터 저장소 | `chroma_store.py` | ChromaDB 기반 임베딩 저장/검색 |
| Reranker | `reranker.py` | BGE-reranker-v2-m3 Cross-encoder |
| 검색 Use Case | `search_usecase.py` | 검색 로직 및 스코어링 |
| JSON 로더 | `json_loader.py` | 규정 JSON 파싱 및 청크 추출 |

### 임베딩 텍스트 구조

```python
# 기존 (순수 텍스트)
embedding_text = "수업일수는 연간 16주 이상으로 한다."

# 현재 (계층 맥락 포함)
embedding_text = "제3장 학사 > 제1절 수업 > 제15조 수업일수: 수업일수는 연간 16주 이상으로 한다."
```

## 개발 명령어

```bash
# 테스트 실행
uv run pytest

# 특정 테스트
uv run pytest tests/test_enhance_for_rag.py -v

# 의존성 추가
uv add <package>
```

## 요구 사항

- Python 3.11+
- `uv` 패키지 매니저
- `hwp5` 라이브러리 + `hwp5html` CLI (HWP 파일 처리)
