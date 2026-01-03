# AI 에이전트 컨텍스트 (AGENTS.md)

> 이 문서는 AI 에이전트(Gemini CLI, Cursor, GitHub Copilot, Claude, Codex 등)가 프로젝트를 이해하고 작업할 때 참조하는 컨텍스트입니다.

---

## 빠른 컨텍스트 (10줄 요약)

```
📦 프로젝트: 대학 규정 관리 시스템 (HWP → JSON → RAG 검색)
📁 구조: src/rag/ 아래 Clean Architecture (domain/, application/, infrastructure/, interface/)
🧪 테스트: TDD 필수 - 기능 구현 전 테스트 먼저 작성
🐍 환경: Python 3.11+, uv 패키지 매니저 (pip/conda 금지)
📜 진입점: `regulation` CLI (convert, sync, search, serve)

⚠️ 핵심 제약:
1. Domain 레이어에서 외부 라이브러리 import 금지
2. 테스트 없이 기능 추가 금지
3. data/chroma_db/, sync_state.json 직접 조작 금지
```

---

## 프로젝트 개요

### 시스템 소개

**이름**: 대학 규정 관리 시스템 (Regulation Manager)

**목적**: 대학 규정집(HWP)을 구조화된 JSON으로 변환하고, Hybrid RAG 기반 AI 검색 및 Q&A를 제공합니다.

**핵심 기능**:

1. **HWP → JSON 변환**: 계층 구조(편/장/절/조/항/호/목) 보존, RAG 최적화 필드 자동 생성
2. **벡터 DB 동기화**: ChromaDB + BGE-M3 임베딩 (1024차원, 한국어 최적화)
3. **하이브리드 검색**: BM25 (키워드) + Dense (의미) + BGE Reranker (재정렬)
4. **LLM 기반 Q&A**: 다양한 프로바이더 지원 (Ollama, OpenAI, Gemini 등)

### 인터페이스

| 인터페이스 | 명령어 | 설명 |
|------------|--------|------|
| CLI | `regulation` | 변환, 검색, 질문, 동기화 통합 |
| Web UI | `regulation serve --web` | Gradio 기반 채팅 인터페이스 |
| MCP Server | `regulation serve --mcp` | AI 에이전트(Claude, Cursor) 연동 |

---

## 아키텍처 원칙

### Clean Architecture

본 프로젝트는 **Clean Architecture** 원칙을 따릅니다. 의존성은 항상 안쪽(Domain)을 향해야 합니다.

```
[Interface] → [Application] → [Domain] ← [Infrastructure]
     ↓              ↓             ↑              ↑
   CLI/Web      Use Cases     Entities      ChromaDB/LLM
```

**레이어별 책임**:

| 레이어 | 위치 | 책임 | 의존성 |
|--------|------|------|--------|
| **Domain** | `src/rag/domain/` | 비즈니스 로직, 엔티티, 인터페이스 정의 | 없음 (순수 Python) |
| **Application** | `src/rag/application/` | Use Cases (검색, 동기화 로직) | Domain만 |
| **Infrastructure** | `src/rag/infrastructure/` | 외부 시스템 구현 (DB, LLM, 검색) | Domain 인터페이스 구현 |
| **Interface** | `src/rag/interface/` | CLI, Web UI, MCP Server | Application 호출 |

**왜 이렇게 구성하는가?**

- **테스트 용이성**: Domain과 Application은 외부 의존성이 없어 단위 테스트가 쉽습니다.
- **유연성**: Infrastructure를 교체해도 비즈니스 로직에 영향이 없습니다 (예: ChromaDB → Qdrant).
- **명확한 책임 분리**: 각 레이어의 역할이 명확하여 코드 이해와 유지보수가 용이합니다.

### TDD (Test-Driven Development)

```
RED → GREEN → REFACTOR
```

1. **RED**: 실패하는 테스트를 먼저 작성합니다.
2. **GREEN**: 테스트를 통과하는 최소한의 코드를 작성합니다.
3. **REFACTOR**: 테스트가 통과한 상태에서 코드를 개선합니다.

**테스트 명령어**:

```bash
uv run pytest                      # 전체 테스트
uv run pytest tests/rag/ -v        # RAG 모듈 테스트
uv run pytest -k "test_search"     # 특정 패턴 매칭
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
│           ├── cli.py              # CLI 로직 (search, sync, status, reset)
│           ├── query_handler.py    # 쿼리 처리 통합 핸들러 (CLI/Web/MCP 공용)
│           ├── formatters.py       # 출력 포맷터 (Rich, Markdown)
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

## 개발 표준

### Python 스타일

| 항목 | 규칙 |
|------|------|
| **버전** | Python 3.11+ |
| **패키지 관리** | `uv` 사용 (`pip`, `conda` 사용 금지) |
| **네이밍** | `snake_case` (함수/변수), `CamelCase` (클래스) |
| **경로** | `pathlib.Path` 사용 |
| **Import** | `src/` 내부에서 상대 import 사용 |
| **들여쓰기** | 4 스페이스 |
| **타입 힌트** | 권장 (함수 시그니처에 명시) |
| **Docstring** | Google 스타일 |

### 금지 사항 (DO NOT)

| 규칙 | 이유 |
|------|------|
| ❌ Domain 레이어에서 외부 라이브러리 import | 비즈니스 로직의 순수성 유지. Domain은 Python 표준 라이브러리만 사용해야 테스트와 교체가 용이합니다. |
| ❌ Use Case에서 Infrastructure 직접 참조 | 의존성 역전 원칙(DIP) 준수. 인터페이스를 통해서만 접근해야 구현체 교체가 가능합니다. |
| ❌ 테스트 없이 기능 추가 | TDD 원칙. 테스트가 기능의 명세 역할을 합니다. |
| ❌ `sync_state.json`, `.env` 수동 수정 | 시스템 무결성 보호. CLI 명령어를 통해서만 상태를 변경해야 합니다. |
| ❌ `data/chroma_db/` 직접 조작 | 데이터베이스 무결성 보호. `sync` 및 `reset` 명령어를 사용하세요. |

### 권장 사항 (DO)

| 규칙 | 이유 |
|------|------|
| ✅ 새 기능 추가 시 테스트 먼저 작성 | TDD 원칙. 테스트가 설계를 이끕니다. |
| ✅ 복잡한 로직은 작은 함수로 분리 | 단일 책임 원칙(SRP). 함수당 하나의 역할만 수행합니다. |
| ✅ 타입 힌트 사용 | IDE 지원과 코드 가독성 향상. |
| ✅ Docstring 작성 (Google 스타일) | 함수의 목적, 파라미터, 반환값을 명시합니다. |
| ✅ 에러 핸들링은 도메인 예외 사용 | `src/exceptions.py`에 정의된 예외 클래스를 사용합니다. |

---

## 핵심 컴포넌트

### 검색 파이프라인 (Ask/Search)

```
Query → QueryAnalyzer → Hybrid Search → Audience Filter → BGE Reranker → LLM 답변
         ↓                ↓                  ↓                ↓              ↓
    유형/대상 분석      BM25 + Dense      대상 불일치        Cross-Encoder    Context 구성
    동의어 확장        RRF 융합          감점 처리          재정렬          답변 생성
```

**핵심 파일**:

| 파일 | 주요 함수/클래스 |
|------|-----------------|
| `application/search_usecase.py` | `search()`, `search_unique()`, `ask()` |
| `infrastructure/hybrid_search.py` | `HybridSearcher`, `QueryAnalyzer`, `Audience` |
| `infrastructure/reranker.py` | `BGEReranker` |

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

### 동의어/인텐트 데이터

| 파일 | 설명 | 수량 |
|------|------|------|
| `data/config/synonyms.json` | 동의어 사전 ("폐과" → "학과 폐지") | 167개 |
| `data/config/intents.json` | 인텐트 규칙 ("학교에 가기 싫어" → "휴직") | 51개 |

---

## 명령어 레퍼런스

```bash
# 환경 설정
uv venv && uv sync
cp .env.example .env

# 대화형 모드 (기본값)
uv run regulation                 # 쿼리 예시 표시, 번호로 선택 가능

# HWP 변환
uv run regulation convert "data/input/규정집.hwp"
uv run regulation convert "data/input/규정집.hwp" --use_llm  # LLM 전처리

# DB 동기화
uv run regulation sync data/output/규정집.json
uv run regulation sync data/output/규정집.json --full   # 전체 재동기화

# 검색
uv run regulation search "교원 연구년 자격" -n 5
uv run regulation search "제15조" --no-rerank
uv run regulation search "휴학" --interactive  # 대화형 모드

# 질문
uv run regulation search "교원 연구년 신청 자격은?" -a
uv run regulation search "휴학 절차" --show-sources -v

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

> **상세한 사용법은 [QUICKSTART.md](./QUICKSTART.md)를 참고하세요.**

---

## 수정 시 체크리스트

코드를 수정할 때 아래 체크리스트를 확인하세요.

| 변경 대상 | 영향 범위 | 필수 조치 |
|-----------|----------|----------|
| `SearchUseCase` | CLI, Web UI, MCP Server | 통합 테스트 실행 |
| `QueryAnalyzer` | 검색 품질 | 검색 테스트 케이스 확인 |
| `Reranker` | 재정렬 정확도 | 보너스 점수 로직 검증 |
| `domain/entities.py` | 전체 시스템 | 모든 테스트 실행 |
| `sync_usecase.py` | 데이터 무결성 | 증분 동기화 테스트 |

---

## 환경 설정

### 환경 변수 (.env)

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

# 동의어/인텐트 사전 (기본값 제공)
RAG_SYNONYMS_PATH=data/config/synonyms.json
RAG_INTENTS_PATH=data/config/intents.json
```

> **LLM 설정에 대한 자세한 내용은 [LLM_GUIDE.md](./LLM_GUIDE.md)를 참고하세요.**

---

## 관련 문서

| 문서 | 설명 |
|------|------|
| [README.md](./README.md) | 시스템 개요 및 상세 기술 설명 |
| [QUICKSTART.md](./QUICKSTART.md) | 빠른 시작 가이드 |
| [SCHEMA_REFERENCE.md](./SCHEMA_REFERENCE.md) | JSON 스키마 명세 |
| [LLM_GUIDE.md](./LLM_GUIDE.md) | LLM 설정 가이드 |

---

## AI 에이전트별 설정

### Gemini CLI

```bash
ln -s AGENTS.md GEMINI.md  # 심볼릭 링크 생성
```

### GitHub Copilot

`.github/copilot-instructions.md` 파일 생성:

```markdown
See AGENTS.md for project context and coding guidelines.
```

### Cursor

프로젝트 루트의 `AGENTS.md`를 자동 인식합니다. 또는 `.cursor/rules/regulations.mdc` 파일을 생성하세요.

### Claude Code

```bash
ln -s AGENTS.md CLAUDE.md
```
