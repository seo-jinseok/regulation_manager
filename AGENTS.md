# 개발자 가이드

규정 관리 시스템의 개발 환경, 아키텍처, 코딩 표준에 대한 안내입니다.

---

## 프로젝트 구조

```
regulation_manager/
├── src/                        # 핵심 소스 코드
│   ├── main.py                 # 변환 파이프라인 진입점
│   ├── converter.py            # HWP → Markdown/HTML 변환
│   ├── formatter.py            # Markdown → JSON 변환
│   ├── enhance_for_rag.py      # RAG 최적화 필드 추가
│   ├── exceptions.py           # 도메인 예외 클래스
│   ├── llm_client.py           # LLM 전처리 클라이언트
│   ├── cache_manager.py        # 캐시 관리
│   ├── parsing/                # 파싱 모듈
│   │   ├── regulation_parser.py
│   │   ├── reference_resolver.py
│   │   ├── table_extractor.py
│   │   └── id_assigner.py
│   └── rag/                    # RAG 시스템 (Clean Architecture)
│       ├── interface/          # CLI, Web UI
│       ├── application/        # Use Cases (SearchUseCase, SyncUseCase)
│       ├── domain/             # 엔티티, 값 객체, 리포지토리 인터페이스
│       └── infrastructure/     # ChromaDB, Reranker, LLM Adapter
├── scripts/                    # 유틸리티 스크립트
├── tests/                      # pytest 테스트
├── data/
│   ├── input/                  # HWP 입력 파일
│   ├── output/                 # JSON/MD/HTML 출력
│   ├── chroma_db/              # ChromaDB 벡터 DB (gitignore)
│   ├── llm_cache/              # LLM 응답 캐시 (gitignore)
│   └── sync_state.json         # 동기화 상태 파일
└── docs/                       # 추가 문서
```

---

## 환경 설정

```bash
uv venv                          # 가상환경 생성
uv sync                          # 의존성 설치
cp .env.example .env             # 환경변수 설정
```

`.env` 파일에서 LLM 기본값을 설정할 수 있습니다:

```bash
LLM_PROVIDER=ollama
LLM_MODEL=gemma2
LLM_BASE_URL=http://localhost:11434
```

---

## 주요 명령어

### 변환 파이프라인

```bash
# 기본 실행 (RAG 최적화 포함)
uv run regulation-manager "data/input/규정집.hwp"

# LLM 전처리 활성화
uv run regulation-manager "data/input/규정집.hwp" --use_llm --provider ollama

# RAG 최적화 비활성화
uv run regulation-manager "data/input/규정집.hwp" --no-enhance-rag
```

### RAG CLI

```bash
# 동기화
uv run regulation-rag sync data/output/규정집.json
uv run regulation-rag sync data/output/규정집.json --full  # 전체 재동기화

# 검색 (BGE Reranker 기본 활성화)
uv run regulation-rag search "교원 연구년" -n 5
uv run regulation-rag search "교원 연구년" --no-rerank  # Reranker 비활성화

# LLM 질문
uv run regulation-rag ask "교원 연구년 신청 자격은?" --provider ollama
uv run regulation-rag ask "장학금 조건" --show-sources

# 상태 확인 및 초기화
uv run regulation-rag status
uv run regulation-rag reset --confirm
```

### 테스트

```bash
uv run pytest                        # 전체 테스트
uv run pytest tests/test_*.py -v     # 상세 출력
uv run pytest tests/rag/ -v          # RAG 테스트만
```

---

## RAG 아키텍처

```
[Query]
   │
   ▼
[ChromaDB Dense Search] ────────┐
   │                            │
   ▼                            ▼
[Keyword Bonus Scoring] ── [BGE Reranker (Cross-Encoder)]
   │                            │
   ▼                            ▼
[Results]              [Reranked Results]
   │                            │
   └────────────┬───────────────┘
                ▼
         [LLM 답변 생성]
```

### 핵심 컴포넌트

| 컴포넌트 | 파일 | 설명 |
|----------|------|------|
| 벡터 저장소 | `infrastructure/chroma_store.py` | ChromaDB 기반 임베딩 저장/검색 |
| Reranker | `infrastructure/reranker.py` | BGE-reranker-v2-m3 Cross-encoder |
| 검색 Use Case | `application/search_usecase.py` | 검색 로직, 스코어링, 재정렬 |
| 동기화 Use Case | `application/sync_usecase.py` | 증분/전체 동기화 |
| JSON 로더 | `infrastructure/json_loader.py` | 규정 JSON 파싱 및 청크 추출 |

### 임베딩 텍스트 구조

```python
# 계층 맥락이 포함된 임베딩 텍스트
embedding_text = "제3장 학사 > 제1절 수업 > 제15조 수업일수: 수업일수는 연간 16주 이상으로 한다."
```

### 동적 Hybrid 가중치

쿼리 유형에 따라 BM25/Dense 가중치가 자동 조절됩니다:

| 쿼리 유형 | 예시 | BM25 | Dense |
|-----------|------|------|-------|
| 조문 번호 | "제15조", "학칙 제3조" | 0.6 | 0.4 |
| 규정명 | "장학금규정", "휴학 학칙" | 0.5 | 0.5 |
| 자연어 질문 | "어떻게 휴학하나요?" | 0.2 | 0.8 |
| 기본값 | 그 외 | 0.3 | 0.7 |

구현: `infrastructure/hybrid_search.py`의 `QueryAnalyzer` 클래스

---

## 코딩 표준

- Python 3.11+, 4-space 들여쓰기
- 함수/변수: `snake_case`, 클래스: `CamelCase`
- `pathlib.Path` 사용 권장
- `src/` 내부에서는 상대 import 사용

### RAG 관련 필드

| 필드 | 설명 |
|------|------|
| `parent_path` | 계층 경로 (breadcrumb) |
| `embedding_text` | 맥락 포함 임베딩용 텍스트 |
| `full_text` | 표시용 전체 텍스트 |
| `chunk_level` | 청크 레벨 (article, paragraph 등) |
| `is_searchable` | 검색 가능 여부 |
| `keywords` | 추출된 키워드 (term/weight) |
| `token_count` | 토큰 수 (근사값) |
| `effective_date` | 시행일 (부칙용) |
| `amendment_history` | 개정 이력 |

---

## 테스트 가이드라인

- 프레임워크: `pytest`
- 테스트 파일: `tests/test_*.py`, `tests/rag/unit/**`
- 파싱 로직 변경 시 관련 테스트 추가 필수
- 외부 서비스 의존 테스트는 별도 디렉토리 (`tests/debug/`)

---

## 커밋 규칙

- Conventional Commits 형식 사용: `feat:`, `fix:`, `chore:`, `docs:`
- 스코프 사용 권장: `feat(parser):`, `fix(rag):`
- PR 작성 시 테스트 명령어 포함

---

## 데이터 파일

| 파일/디렉토리 | 설명 | Git |
|---------------|------|-----|
| `data/chroma_db/` | ChromaDB 벡터 DB | gitignore |
| `data/llm_cache/` | LLM 응답 캐시 | gitignore |
| `data/sync_state.json` | 동기화 상태 추적 | gitignore |
