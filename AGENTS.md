# AI Agent Context (AGENTS.md)

> 이 파일은 AI 에이전트(Gemini CLI, Cursor, GitHub Copilot, Claude, Codex 등)가 프로젝트를 이해하고 작업할 때 참조하는 컨텍스트입니다.
> Gemini CLI는 이 파일을 `GEMINI.md`로도 읽습니다. 필요시 심볼릭 링크를 생성하세요: `ln -s AGENTS.md GEMINI.md`

---

## 프로젝트 개요

**이름**: 대학 규정 관리 시스템 (Regulation Manager)

**목적**: 대학 규정집(HWP)을 구조화된 JSON으로 변환하고, Hybrid RAG 기반 AI 검색 및 Q&A를 제공

**핵심 기능**:
1. HWP → JSON 변환 (계층 구조 보존, RAG 최적화 필드 자동 생성)
2. 벡터 DB 동기화 (ChromaDB + BGE-M3 임베딩)
3. Hybrid Search (BM25 + Dense) + BGE Reranker
4. LLM 기반 Q&A (다양한 프로바이더 지원)

**인터페이스**:
- CLI: `regulation` (변환, 검색, 질문, 동기화 통합)
- Web UI: `regulation serve --web` (Gradio)
- MCP Server: `regulation serve --mcp` (AI 에이전트 연동)

---

## ⚠️ 필수 개발 원칙

### 1. TDD (Test-Driven Development)

```
RED → GREEN → REFACTOR
```

- **테스트 먼저 작성**: 기능 구현 전 실패하는 테스트를 먼저 작성
- **최소 구현**: 테스트를 통과하는 최소한의 코드만 작성
- **리팩토링**: 테스트가 통과한 후 코드 개선
- **테스트 위치**: `tests/` 디렉토리, `pytest` 프레임워크 사용

**테스트 명령어**:
```bash
uv run pytest                      # 전체 테스트
uv run pytest tests/rag/ -v        # RAG 모듈 테스트
uv run pytest -k "test_search"     # 특정 패턴 매칭
```

### 2. Clean Architecture

```
[Interface] → [Application] → [Domain] ← [Infrastructure]
```

**레이어 규칙**:
- **Domain**: 비즈니스 로직, 엔티티, 인터페이스 정의 (의존성 없음)
- **Application**: Use Cases (Domain에만 의존)
- **Infrastructure**: 외부 시스템 구현 (Domain 인터페이스 구현)
- **Interface**: CLI, Web UI, MCP Server (Application 호출)

**의존성 방향**: 항상 안쪽(Domain)을 향해야 함. Domain은 외부를 모름.

**디렉토리 구조**:
```
src/rag/
├── domain/           # 엔티티, 값 객체, 리포지토리 인터페이스
├── application/      # Use Cases (SearchUseCase, SyncUseCase)
├── infrastructure/   # ChromaDB, Reranker, LLM 구현체
└── interface/        # CLI, Web UI, MCP Server
```

---

## 프로젝트 구조

```
regulation_manager/
├── src/
│   ├── main.py                 # HWP 변환 파이프라인 진입점
│   ├── converter.py            # HWP → HTML 변환 (hwp5html)
│   ├── formatter.py            # HTML → JSON 변환
│   ├── enhance_for_rag.py      # RAG 최적화 필드 추가
│   ├── parsing/                # 파싱 모듈
│   │   ├── regulation_parser.py    # 편/장/절/조/항/호/목 파싱
│   │   └── reference_resolver.py   # 상호 참조 해석
│   └── rag/                    # RAG 시스템 (Clean Architecture)
│       ├── domain/
│       │   ├── entities.py         # Chunk, Regulation, SearchResult
│       │   ├── value_objects.py    # SearchFilter, SyncResult
│       │   └── repositories.py     # IVectorStore, ILLMClient 인터페이스
│       ├── application/
│       │   ├── search_usecase.py   # 검색/질문 로직
│       │   └── sync_usecase.py     # 동기화 로직
│       ├── infrastructure/
│       │   ├── chroma_store.py     # ChromaDB 벡터 저장소
│       │   ├── hybrid_search.py    # BM25 + Dense, QueryAnalyzer
│       │   ├── reranker.py         # BGE Reranker
│       │   ├── llm_adapter.py      # LLM 클라이언트 어댑터
│       │   └── json_loader.py      # JSON → Chunk 변환
│       └── interface/
│           ├── unified_cli.py      # 통합 CLI 진입점
│           ├── cli.py              # CLI 로직 (search, ask, sync, status, reset)
│           ├── gradio_app.py       # Gradio Web UI
│           └── mcp_server.py       # MCP Server (FastMCP)
├── tests/                      # pytest 테스트
│   ├── test_*.py               # 단위 테스트
│   └── rag/                    # RAG 모듈 테스트
└── data/
    ├── input/                  # HWP 입력 파일
    ├── output/                 # JSON 출력 파일
    ├── chroma_db/              # ChromaDB 저장소 (gitignore)
    ├── sync_state.json         # 동기화 상태 (gitignore)
    └── config/                 # 설정 파일
        ├── synonyms.json       # 동의어 사전 (167개 용어)
        └── intents.json        # 인텐트 규칙 (51개 규칙)
```

---

## 코딩 규칙

### Python 스타일
- **버전**: Python 3.11+
- **패키지 관리**: `uv` 사용 (`pip`, `conda` 사용 금지)
- **네이밍**: `snake_case` (함수/변수), `CamelCase` (클래스)
- **경로**: `pathlib.Path` 사용
- **Import**: `src/` 내부에서 상대 import 사용
- **들여쓰기**: 4 스페이스

### 금지 사항 (DO NOT)
- ❌ Domain 레이어에서 외부 라이브러리 import
- ❌ Use Case에서 Infrastructure 직접 참조 (인터페이스 통해서만)
- ❌ 테스트 없이 기능 추가
- ❌ `sync_state.json`, `.env` 수동 수정
- ❌ `data/chroma_db/` 직접 조작

### 권장 사항 (DO)
- ✅ 새 기능 추가 시 테스트 먼저 작성
- ✅ 복잡한 로직은 작은 함수로 분리
- ✅ 타입 힌트 사용
- ✅ Docstring 작성 (Google 스타일)
- ✅ 에러 핸들링은 도메인 예외 사용 (`src/exceptions.py`)

---

## 핵심 컴포넌트

### 검색 파이프라인 (Ask/Search)

```
Query → QueryAnalyzer → Hybrid Search → BGE Reranker → LLM 답변
         ↓                ↓                ↓              ↓
    유형/대상 분석      BM25 + Dense    Penalize Mismatch  Cross-Encoder    Context 구성
    동의어 확장        RRF 융합                            재정렬          답변 생성
```

**핵심 파일**:
- `application/search_usecase.py`: `search()`, `search_unique()`, `ask()`
- `infrastructure/hybrid_search.py`: `HybridSearcher`, `QueryAnalyzer`, `Audience`
- `infrastructure/reranker.py`: `BGEReranker`

### 동의어/인텐트 데이터

| 파일 | 설명 | 수량 |
|------|------|------|
| `data/config/synonyms.json` | 동의어 사전 ("폐과" → "학과 폐지") | 167개 |
| `data/config/intents.json` | 인텐트 규칙 ("학교에 가기 싫어" → "휴직") | 51개 |

### 주요 데이터 구조

```python
# domain/entities.py
@dataclass
class Chunk:
    id: str                     # uuid5 (결정적)
    text: str                   # 본문
    title: str                  # 조항 제목
    rule_code: str              # 규정 번호 (예: "3-1-24")
    parent_path: List[str]      # 계층 경로
    embedding_text: str         # 임베딩용 텍스트

@dataclass
class SearchResult:
    chunk: Chunk
    score: float                # 0.0 ~ 1.0
```

---

## 명령어 레퍼런스

```bash
# 환경 설정
uv venv && uv sync
cp .env.example .env

# HWP 변환
uv run regulation convert "data/input/규정집.hwp"
uv run regulation convert "data/input/규정집.hwp" --use_llm  # LLM 전처리

# DB 동기화
uv run regulation sync data/output/규정집.json
uv run regulation sync data/output/규정집.json --full   # 전체 재동기화

# 검색
uv run regulation search "교원 연구년 자격" -n 5
uv run regulation search "제15조" --no-rerank

# 질문
uv run regulation ask "교원 연구년 신청 자격은?" --provider lmstudio
uv run regulation ask "휴학 절차" --show-sources -v

# 상태/초기화
uv run regulation status
uv run regulation reset --confirm

# 인터페이스
uv run regulation serve --web     # Web UI (Gradio)
uv run regulation serve --mcp     # MCP Server

# 테스트
uv run pytest
uv run pytest tests/rag/ -v
```

---

## 수정 시 주의사항

| 변경 대상 | 영향 범위 | 필수 조치 |
|-----------|----------|----------|
| `SearchUseCase` | CLI, Web UI, MCP Server | 통합 테스트 실행 |
| `QueryAnalyzer` | 검색 품질 | 검색 테스트 케이스 확인 |
| `Reranker` | 재정렬 정확도 | 보너스 점수 로직 검증 |
| `domain/entities.py` | 전체 시스템 | 모든 테스트 실행 |
| `sync_usecase.py` | 데이터 무결성 | 증분 동기화 테스트 |

---

## 환경 변수 (.env)

```bash
# LLM 기본 설정
LLM_PROVIDER=ollama          # ollama, lmstudio, openai, gemini
LLM_MODEL=gemma2             # 모델명 (프로바이더별 상이)
LLM_BASE_URL=http://localhost:11434

# API 키 (클라우드 사용 시)
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...

# 데이터 경로 (선택)
RAG_DB_PATH=data/chroma_db
RAG_JSON_PATH=data/output/규정집.json

# 동의어/인텐트 사전 (기본값 제공, 수정 시 반영)
RAG_SYNONYMS_PATH=data/config/synonyms.json
RAG_INTENTS_PATH=data/config/intents.json
```

---

## 관련 문서

| 문서 | 설명 |
|------|------|
| [README.md](./README.md) | 시스템 개요 및 상세 기술 설명 |
| [QUICKSTART.md](./QUICKSTART.md) | 빠른 시작 가이드 |
| [SCHEMA_REFERENCE.md](./SCHEMA_REFERENCE.md) | JSON 스키마 명세 |
| [LLM_GUIDE.md](./LLM_GUIDE.md) | LLM 설정 가이드 |
| [USAGE_SCENARIOS.md](./USAGE_SCENARIOS.md) | 실제 사용 시나리오 |
| [PRODUCT_BACKLOG.md](./PRODUCT_BACKLOG.md) | 제품 수준 백로그 |
| [UX_FLOWS.md](./UX_FLOWS.md) | 웹/MCP UX 플로우 |
| [UX_COPY.md](./UX_COPY.md) | UX 카피/컴포넌트 문구 |
| [TASKS_BACKLOG.md](./TASKS_BACKLOG.md) | 모듈별 액션 티켓 |

---

## AI 에이전트별 추가 설정

### Gemini CLI
```bash
ln -s AGENTS.md GEMINI.md  # 심볼릭 링크 생성
```

### GitHub Copilot
`.github/copilot-instructions.md` 파일에 이 내용을 참조하도록 설정:
```markdown
See AGENTS.md for project context and coding guidelines.
```

### Cursor
`.cursor/rules/regulations.mdc` 생성 또는 프로젝트 루트의 `AGENTS.md` 자동 인식

### Claude Code
`CLAUDE.md`로 심볼릭 링크 생성:
```bash
ln -s AGENTS.md CLAUDE.md
```
