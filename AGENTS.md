# Repository Guidelines

## Project Structure & Module Organization

```
regulation_manager/
├── src/                    # 핵심 파이프라인
│   ├── main.py             # 변환 진입점
│   ├── converter.py        # HWP → Markdown/HTML
│   ├── formatter.py        # Markdown → JSON
│   ├── enhance_for_rag.py  # RAG 최적화 필드 추가
│   ├── llm_client.py       # LLM 전처리
│   ├── cache_manager.py    # 캐시 관리
│   └── rag/                # RAG 시스템 (Clean Architecture)
│       ├── interface/      # CLI
│       ├── application/    # Use Cases
│       ├── domain/         # 도메인 모델
│       └── infrastructure/ # ChromaDB, JSON 로더
├── scripts/                # 유틸리티 스크립트
├── tests/                  # pytest 테스트
├── data/
│   ├── input/              # HWP 입력 파일
│   ├── output/             # JSON/MD/HTML 출력
│   ├── chroma_db/          # ChromaDB 벡터 DB
│   ├── sync_state.json     # 동기화 상태 파일
│   └── config/             # 전처리 규칙 설정
└── docs/                   # 추가 문서
```

## Build, Test, and Development Commands

### 환경 설정
```bash
uv venv                              # 가상환경 생성
uv pip install -r requirements.txt   # 의존성 설치
cp .env.example .env                 # 환경변수 설정
```

### 변환 파이프라인
```bash
# 기본 실행 (RAG 최적화 포함)
uv run python -m src.main "data/input/규정집.hwp"

# RAG 최적화 비활성화
uv run python -m src.main "data/input/규정집.hwp" --no-enhance-rag

# LLM 전처리 활성화 (문서 품질 낮은 경우)
uv run python -m src.main "data/input/규정집.hwp" --use_llm --provider ollama

# CLI 엔트리포인트
regulation-manager "data/input/규정집.hwp"
```

### RAG CLI
```bash
# 동기화 (JSON → ChromaDB)
uv run python -m src.rag.interface.cli sync data/output/규정집.json
uv run python -m src.rag.interface.cli sync data/output/규정집.json --full  # 전체 재동기화

# 검색
uv run python -m src.rag.interface.cli search "교원 연구년" -n 5
uv run python -m src.rag.interface.cli search "학칙" --include-abolished

# LLM 질문 (자연어 답변)
uv run python -m src.rag.interface.cli ask "교원 연구년 신청 자격은?" --provider ollama
uv run python -m src.rag.interface.cli ask "장학금 조건" --provider openai --show-sources

# 상태 확인
uv run python -m src.rag.interface.cli status

# DB 초기화 (모든 데이터 삭제)
uv run python -m src.rag.interface.cli reset --confirm
```

### 테스트
```bash
uv run pytest                    # 전체 테스트
uv run pytest tests/test_*.py    # 특정 테스트
uv lock                          # 의존성 변경 후 lock 갱신
```

## Coding Style & Naming Conventions

- Python 3.11+, 4-space indentation
- `snake_case` for functions/vars, `CamelCase` for classes
- Prefer `pathlib.Path` and relative imports inside `src/`
- Output schema fields: `type`, `display_no`, `sort_no`, `children`, `metadata`
- RAG fields (auto-added): `parent_path`, `full_text`, `embedding_text`, `chunk_level`, `is_searchable`, `token_count`, `keywords` (term/weight), `effective_date`, `status`, `amendment_history`
- Avoid non-ASCII in code/comments unless required by domain data

## Testing Guidelines

- Framework: `pytest`. Test files follow `tests/test_*.py`
- Add focused tests when changing parsing logic (`src/formatter.py`, `src/preprocessor.py`)
- Debug tests should not depend on external services or sleep-based timing

## Commit & Pull Request Guidelines

- Commit messages: Conventional Commits (`feat:`, `fix:`, `chore:`, scopes like `feat(parser):`)
- PRs: brief summary, test commands, note on schema/data changes

## Security & Configuration

- Secrets in `.env` (never commit); use `.env.example` as template
- Optional cache controls: `LLM_CACHE_TTL_DAYS`, `LLM_CACHE_MAX_ENTRIES`

## Data Files

| 파일 | 설명 |
|------|------|
| `data/chroma_db/` | ChromaDB 벡터 DB (gitignore) |
| `data/sync_state.json` | 동기화 상태 추적 |
| `data/config/preprocessor_rules.json` | 전처리 규칙 커스터마이즈 |
