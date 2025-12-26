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

### 검색 정확도 향상

- **계층적 맥락 임베딩**: `제3장 > 제1절 > 제15조` 형태의 경로 정보가 검색에 반영
- **BGE Reranker**: Cross-encoder 기반 결과 재정렬 (기본 활성화)
- **조항 번호 매칭**: `제N조`, `제N항` 등 정확한 조항 검색 지원
- **Hybrid Search**: 키워드 검색과 의미 검색 결합

### 증분 동기화

월간 규정 업데이트 시 변경된 규정만 동기화하여 처리 시간을 단축합니다.

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
[Query] → [ChromaDB 검색] → [BGE Reranker] → [LLM 답변 생성]
                ↓                  ↓
         Dense + Sparse      Cross-Encoder
           Retrieval           Reranking
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
- `hwp5` 라이브러리 (HWP 파일 처리)
