# 빠른 시작 가이드

규정 관리 시스템의 설치부터 검색까지 단계별 안내입니다.

> 시스템 작동 원리 및 기술 상세는 [README.md#시스템-개요](./README.md#시스템-개요)를 참고하세요.

---

## 1단계: 설치

```bash
# 저장소 클론
git clone <repository-url>
cd regulation_manager

# 가상환경 생성 및 활성화
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
uv sync
```

필요 시 `.env`를 설정하면 코드 기본값보다 우선 적용됩니다:

```bash
cp .env.example .env
```

---

## 2단계: 규정 변환

HWP 파일을 `data/input/` 폴더에 배치한 후 변환합니다.

```bash
uv run regulation convert "data/input/규정집.hwp"
```

> 변환에는 `hwp5html` CLI가 필요합니다. 설치되어 있지 않으면 변환이 실패합니다.

**출력 결과** (`data/output/`):

| 파일 | 설명 |
|------|------|
| `규정집.json` | 구조화된 JSON (RAG 필드 포함) |
| `규정집_raw.md` | 마크다운 원문 |
| `규정집_metadata.json` | 목차 및 색인 정보 |

---

## 3단계: 벡터 DB 동기화

변환된 JSON을 검색 가능한 형태로 저장합니다.

```bash
uv run regulation sync data/output/규정집.json
```

**성공 시 출력:**

```
✓ 동기화 완료: 추가 15,678 / 수정 0 / 삭제 0
ℹ 총 청크 수: 15,678
```

---

## 4단계: 규정 검색 및 질문

이제 `search` 명령어 하나로 검색과 질문 답변을 모두 수행할 수 있습니다.

```bash
# 1. 문서 검색 (키워드)
uv run regulation search "교원 연구년 신청 자격"

# 2. AI 질문 (자연어)
uv run regulation search "교원 연구년 신청 자격은?"

# 3. 강제 모드 사용 (선택)
uv run regulation search "연구년" -q  # 문서 검색 강제
uv run regulation search "연구년" -a  # AI 답변 강제
```

### 검색 옵션

| 옵션 | 설명 |
|------|------|
| `-a`, `--answer` | AI 답변 생성 강제 |
| `-q`, `--quick` | 문서 검색만 수행 |
| `-n 10` | 검색 결과 개수 지정 |
| `--include-abolished` | 폐지된 규정 포함 (검색 모드) |
| `-v` | 상세 정보 출력 |

---

## 5단계: 웹 UI (선택)

```bash
uv run regulation serve --web
```

브라우저에서 파일 업로드 → 변환 → DB 동기화 → 질문까지 통합 인터페이스로 진행할 수 있습니다.

---

## 6단계: MCP 서버 (선택)

AI 에이전트(Claude, Cursor 등)에서 규정 검색 기능을 사용할 수 있습니다.

```bash
# MCP 서버 실행
uv run regulation serve --mcp
```

**Claude Desktop 연결** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "regulation-rag": {
      "command": "uv",
      "args": ["run", "regulation", "serve", "--mcp"],
      "cwd": "/path/to/regulation_manager"
    }
  }
}
```

---

## 자주 사용하는 명령어

| 작업 | 명령어 |
|------|--------|
| 변환 | `regulation convert "data/input/규정집.hwp"` |
| 동기화 | `regulation sync <json-path>` |
| **검색/질문** | `regulation search "<쿼리>" [-a/-q]` |
| 상태 확인 | `regulation status` |
| DB 초기화 | `regulation reset --confirm` |
| 웹 UI | `regulation serve --web` |
| MCP 서버 | `regulation serve --mcp` |

---

## 문제 해결

### "데이터베이스가 비어 있습니다"

`sync` 명령을 먼저 실행하세요.

### "파일을 찾을 수 없습니다"

파일 경로를 확인하세요. 절대 경로 또는 `data/input/` 상대 경로를 사용합니다.

### 변환 품질이 낮음

LLM 전처리를 활성화하세요:

```bash
uv run regulation convert "규정.hwp" --use_llm --provider ollama --model gemma2
```

LLM 설정에 대한 자세한 내용은 [LLM_GUIDE.md](./LLM_GUIDE.md)를 참고하세요.

---

## 관련 문서

- [README.md](./README.md) - 전체 안내
- [LLM_GUIDE.md](./LLM_GUIDE.md) - LLM 설정 가이드
- [SCHEMA_REFERENCE.md](./SCHEMA_REFERENCE.md) - JSON 스키마 명세
