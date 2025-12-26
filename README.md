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

## 시스템 특징

### 1️⃣ HWP 구조화 (JSON 변환)

HWP 파일의 복잡한 규정 내용을 **계층적 JSON 구조**로 변환합니다.

```
[HWP 원본]                      [JSON 출력]
제4조 (용어의 정의)       →     { "display_no": "제4조",
  1. 학과(전공)란...              "title": "용어의 정의",
  2. 입학정원이란...              "children": [
  ...                               { "display_no": "1.", "text": "학과(전공)란..." },
  6. 교육편제 조정은...             { "display_no": "6.", "text": "교육편제 조정은...",
    가. 통합이란...                   "children": [
    나. 신설이란...                     { "display_no": "가.", "text": "통합이란..." },
    다. 폐지란...                       { "display_no": "다.", "text": "폐지란..." }
                                      ]}]}
```

**RAG 최적화 필드 자동 생성** (`enhance_for_rag.py`):

| 필드 | 설명 | 예시 |
|------|------|------|
| `parent_path` | 계층 경로 (Breadcrumb) | `["학과평가규정", "제4조 용어의 정의", "6. 교육편제 조정"]` |
| `embedding_text` | 임베딩용 텍스트 (경로 포함) | `제4조 용어의 정의 > 6. 교육편제 조정 > 다. 학과(전공) 폐지: 폐지란...` |
| `keywords` | 핵심 키워드 (가중치 포함) | `[{"term": "학과", "weight": 0.9}]` |

### 2️⃣ 벡터 DB 저장 (ChromaDB)

변환된 JSON의 각 조항을 **청크(Chunk)** 단위로 분리하여 ChromaDB에 저장합니다.

- **임베딩 모델**: `BAAI/bge-m3` (다국어 지원, 한국어 최적화)
- **증분 동기화**: 변경된 규정만 업데이트 (월간 규정 업데이트 시 처리 시간 단축)
- **메타데이터 저장**: `rule_code`, `parent_path`, `status`, `effective_date` 등

### 3️⃣ 쿼리 가공 (Query Processing)

사용자 질문을 분석하여 **최적의 검색 전략**을 자동 선택합니다 (`QueryAnalyzer`).

| 단계 | 처리 내용 | 예시 |
|------|----------|------|
| **유형 분석** | 쿼리 패턴 감지 | `"제15조"` → 조문 번호 / `"휴학하려면?"` → 자연어 질문 |
| **동적 가중치** | BM25/Dense 비율 자동 조정 | 조문 번호: BM25 60% / 자연어: Dense 60% |
| **동의어 확장** | 유사어 추가 | `"폐과"` → `"폐과 학과폐지 전공폐지"` |
| **불용어 제거** | 검색 노이즈 제거 | `"~하려면"`, `"~인가요"` 등 제거 |

### 4️⃣ Hybrid Search (검색)

**두 가지 검색 방식을 결합**하여 정확도를 높입니다.

```
[Query] ──┬── BM25 (키워드 매칭) ──┬── RRF 융합 ── [결과]
          │                        │
          └── Dense (의미 검색) ───┘
```

- **BM25**: 정확한 키워드 매칭 (예: `"제15조"`, `"휴학규정"`)
- **Dense**: 의미적 유사성 검색 (예: `"학교 쉬고 싶어요"` → 휴학 규정)
- **RRF 융합**: Reciprocal Rank Fusion으로 두 결과 통합

### 5️⃣ Reranking (BGE Cross-Encoder)

검색 결과를 **Cross-Encoder**로 재정렬하여 정확도를 향상시킵니다.

- **모델**: `BAAI/bge-reranker-v2-m3`
- **작동 방식**: 질문-문서 쌍을 직접 비교하여 관련성 점수 계산
- **조항 번호 매칭 보너스**: `제N조`, `제N항` 등 정확히 일치 시 가산점

### 6️⃣ LLM 답변 생성

검색된 규정을 기반으로 **자연어 답변**을 생성합니다.

- **Context 구성**: 상위 N개 검색 결과를 계층 경로와 함께 제공
- **신뢰도 계산**: Reranker 점수 기반으로 답변 신뢰도 표시
- **출처 표시**: 답변에 사용된 규정의 경로와 관련도 표시

```
📍 학과평가규정 > 제4조 용어의 정의 > 6. 교육편제 조정 > 다. 학과(전공) 폐지
   학과(전공) 폐지란 설치된 학과(전공)가 없어지는 것을 말한다.
   📋 관련도: 100% 🟢 매우 높음 | AI 신뢰도: 0.943
```

### 7️⃣ 증분 동기화

월간 규정 업데이트 시 변경된 규정만 동기화하여 처리 시간을 단축합니다.

- 규정별 해시값 비교로 변경 감지
- 신규/수정/삭제된 규정만 처리
- 상태 파일(`sync_state.json`)로 동기화 이력 관리

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
