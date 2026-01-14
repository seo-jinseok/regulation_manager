# 빠른 시작 가이드

규정 관리 시스템의 설치부터 검색까지 단계별로 안내합니다.

> **예상 소요 시간**: 약 10~15분

> **시스템 작동 원리 및 기술 상세는 [README.md](./README.md)를 참고하세요.**

---

## 목차

1. [설치](#1단계-설치)
2. [규정 변환](#2단계-규정-변환)
3. [벡터 DB 동기화](#3단계-벡터-db-동기화)
4. [규정 검색 및 질문](#4단계-규정-검색-및-질문)
5. [웹 UI (선택)](#5단계-웹-ui-선택)
6. [MCP 서버 (선택)](#6단계-mcp-서버-선택)
7. [자주 사용하는 명령어](#자주-사용하는-명령어)
8. [문제 해결](#문제-해결)

---

## 1단계: 설치

### 목적

Python 가상환경을 생성하고 필요한 패키지를 설치합니다.

### 절차

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

### 환경 변수 설정 (선택)

기본값 대신 다른 LLM 설정을 사용하려면 환경 변수를 설정합니다:

```bash
cp .env.example .env
# .env 파일을 편집하여 LLM_PROVIDER, LLM_MODEL 등을 수정
```

> 환경 변수를 설정하면 코드 기본값보다 우선 적용됩니다.

### 성공 확인

```bash
uv run regulation --help
```

다음과 같은 도움말이 출력되면 설치가 완료된 것입니다:

```
usage: regulation [-h] {convert,sync,search,status,reset,serve} ...
```

---

## 2단계: 규정 변환

### 목적

HWP 형식의 규정집 파일을 AI가 이해할 수 있는 **구조화된 JSON**으로 변환합니다. 이 과정에서 편/장/절/조/항/호/목의 계층 구조가 보존됩니다.

### 사전 요구사항

- `hwp5html` CLI가 설치되어 있어야 합니다.
- HWP 파일을 `data/input/` 폴더에 배치합니다.

### 절차

```bash
uv run regulation convert "data/input/규정집.hwp"
```

**옵션: LLM 전처리 활성화** (변환 품질 향상)

```bash
uv run regulation convert "data/input/규정집.hwp" --use_llm
```

> LLM 전처리는 복잡한 문서 구조를 더 정확하게 파싱하는 데 도움이 됩니다.

### 예상 소요 시간

- 일반 변환: 약 1~2분 (문서 크기에 따라 다름)
- LLM 전처리 활성화: 약 10~30분

### 성공 확인

`data/output/` 폴더에 다음 파일이 생성됩니다:

| 파일 | 설명 |
|------|------|
| `규정집.json` | 구조화된 JSON (RAG 필드 포함) |
| `규정집_raw.md` | 마크다운 원문 |
| `규정집_metadata.json` | 목차 및 색인 정보 |

```bash
# 파일 생성 확인
ls -la data/output/
```

---

## 3단계: 벡터 DB 동기화

### 목적

변환된 JSON의 각 조항을 **벡터 데이터베이스**에 저장합니다. 이를 통해 의미 기반 검색(유사한 내용 찾기)이 가능해집니다.

### 절차

```bash
uv run regulation sync data/output/규정집.json
```

### 예상 소요 시간

- 약 3~5분 (첫 동기화 시, 문서 크기에 따라 다름)
- 이후 증분 동기화는 변경분만 처리하여 더 빠릅니다.

### 성공 확인

```
✓ 동기화 완료: 추가 15,678 / 수정 0 / 삭제 0
ℹ 총 청크 수: 15,678
```

동기화 상태를 확인하려면:

```bash
uv run regulation status
```

출력 예시:

```
📊 동기화 상태
  마지막 동기화: 2026-01-03 14:30:00
  JSON 파일: 규정집.json
  총 규정 수: 343
  총 청크 수: 15,678
```

---

## 4단계: 규정 검색 및 질문

이제 자연어로 규정을 검색하고 질문할 수 있습니다.

### 대화형 모드 (권장)

가장 쉬운 방법은 `regulation` 명령만 실행하여 **대화형 모드**로 시작하는 것입니다:

```bash
uv run regulation
```

**대화형 모드 특징**:

- 시작 시 5개의 예시 쿼리 표시 (다양한 기능 소개)
- 번호 입력으로 예시/제안 쿼리 실행
- AI 답변 후 문맥 기반 후속 쿼리 제안
- `/reset`으로 문맥 초기화, `/exit`로 종료

```
$ uv run regulation

ℹ 대화형 모드입니다. 아래 예시 중 번호를 선택하거나 직접 질문하세요.

  [1] 휴학 신청 절차가 어떻게 되나요?
  [2] 교원 연구년
  [3] 교원인사규정 전문
  [4] 학칙 별표 1
  [5] 학교 그만두고 싶어요

>>> 1

... (AI 답변) ...

💡 연관 질문:
  [1] 복학 절차는?
  [2] 휴학 기간 연장은 가능한가요?

>>> 
```

### 직접 검색/질문

대화형 모드 없이 직접 검색이나 질문을 할 수도 있습니다:

```bash
# 1. AI 질문 (자연어) - 질문 형태일 때 자동으로 AI 답변 생성
uv run regulation search "교원 연구년 신청 자격은?"

# 2. 문서 검색 (키워드) - 관련 문서 목록 반환
uv run regulation search "교원 연구년 신청 자격"

# 3. 강제 모드 사용 (선택)
uv run regulation search "연구년" -q  # 문서 검색 강제
uv run regulation search "연구년" -a  # AI 답변 강제
```

> **전문 보기(Full View)**: 웹 UI 또는 MCP에서는 "교원인사규정 전문" / "교원인사규정 원문"과 같은 요청으로 규정 전체 뷰를 확인할 수 있습니다.

### 검색 옵션

| 옵션 | 설명 | 사용 예시 |
|------|------|----------|
| `--interactive` | 대화형 모드 | `search "휴학" --interactive` |
| `-a`, `--answer` | AI 답변 생성 강제 | `search "휴학" -a` |
| `-q`, `--quick` | 문서 검색만 수행 | `search "휴학" -q` |
| `-n 10` | 검색 결과 개수 지정 | `search "휴학" -n 10` |
| `--include-abolished` | 폐지된 규정 포함 | `search "휴학" --include-abolished` |
| `-v` | 상세 정보 출력 | `search "휴학" -v` |

### 성공 확인

검색 결과가 출력되거나 AI 답변이 생성되면 성공입니다. 검색 결과가 없다면 `sync` 명령이 정상적으로 실행되었는지 확인하세요.

---

## 5단계: 웹 UI (선택)

명령줄(CLI) 대신 웹 브라우저에서 시스템을 사용할 수 있습니다.

### 목적

비개발자나 명령줄에 익숙하지 않은 사용자를 위한 그래픽 인터페이스를 제공합니다.

### 절차

```bash
uv run regulation serve --web
```

브라우저에서 `http://localhost:7860`으로 접속합니다.

### 주요 기능

- **ChatGPT 스타일 인터페이스**: 채팅 형식의 직관적인 대화 UI
- **예시 쿼리 카드**: 클릭 한 번으로 다양한 검색 기능 체험
- **전문 보기**: "전문/원문/전체" 요청 시 규정 전체 뷰 제공
- **대상 선택**: 교수/학생/직원 대상이 모호할 때 선택 UI 제공

파일 업로드 → 변환 → DB 동기화 → 질문까지 한 화면에서 진행할 수 있습니다.

---

## 6단계: MCP 서버 (선택)

AI 에이전트(Claude, Cursor 등)에서 규정 검색 기능을 직접 호출할 수 있습니다.

### 목적

MCP(Model Context Protocol) 서버를 실행하여 AI 에이전트가 규정 검색 도구를 사용할 수 있도록 합니다.

### 절차

```bash
uv run regulation serve --mcp
```

### Claude Desktop 연결

`~/Library/Application Support/Claude/claude_desktop_config.json` 파일에 다음을 추가합니다:

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

### 주요 기능

- `audience` 파라미터로 대상(교수/학생/직원) 지정 가능
- 모호한 질의는 `type=clarification` 응답 반환
- "전문/원문/전체" 요청은 `type=full_view` 응답 반환
- `get_regulation_overview`, `view_article`, `view_chapter` 도구로 규정 구조 탐색 가능

---

## 자주 사용하는 명령어

| 작업 | 명령어 |
|------|--------|
| **대화형 모드** | `uv run regulation` |
| 변환 | `uv run regulation convert "data/input/규정집.hwp"` |
| 동기화 | `uv run regulation sync <json-path>` |
| 검색/질문 | `uv run regulation search "<쿼리>" [-a/-q]` |
| 상태 확인 | `uv run regulation status` |
| DB 초기화 | `uv run regulation reset --confirm` |
| 웹 UI | `uv run regulation serve --web` |
| MCP 서버 | `uv run regulation serve --mcp` |

---

## 문제 해결

### "데이터베이스가 비어 있습니다"

**원인**: `sync` 명령이 실행되지 않았습니다.

**해결**:
```bash
uv run regulation sync data/output/규정집.json
```

### "파일을 찾을 수 없습니다"

**원인**: 파일 경로가 잘못되었습니다.

**해결**: 절대 경로를 사용하거나 `data/input/` 또는 `data/output/` 상대 경로를 사용하세요.

```bash
# 절대 경로 예시
uv run regulation convert "/Users/user/Documents/규정집.hwp"

# 상대 경로 예시
uv run regulation convert "data/input/규정집.hwp"
```

### "hwp5html 실행 파일을 찾을 수 없습니다"

**원인**: `hwp5html` CLI가 설치되지 않았습니다.

**해결**: `hwp5html`을 설치하세요. 자세한 설치 방법은 hwp5 프로젝트 문서를 참고하세요.

### 변환 품질이 낮음

**원인**: 복잡한 문서 구조가 제대로 파싱되지 않았습니다.

**해결**: LLM 전처리를 활성화하세요:

```bash
uv run regulation convert "규정.hwp" --use_llm --provider ollama --model gemma2
```

> LLM 설정에 대한 자세한 내용은 [LLM_GUIDE.md](./LLM_GUIDE.md)를 참고하세요.

### 검색 결과가 부정확함

**원인**: AI 재정렬이 비활성화되어 있거나, 동의어/인텐트 사전이 로드되지 않았습니다.

**해결**:
1. `--no-rerank` 옵션이 사용되지 않았는지 확인
2. `-v` 옵션으로 쿼리 분석 과정 확인

```bash
uv run regulation search "휴학" -v
```

---

## 고급 설정

### 성능 최적화

첫 검색이 느린 경우(2~3초) 다음 옵션을 사용해 최적화할 수 있습니다:

#### 1. BM25 인덱스 캐시

BM25 인덱스를 디스크에 저장하여 재시작 시 인덱스 빌드 시간을 단축합니다:

```bash
# .env 파일에 추가
BM25_INDEX_CACHE_PATH=data/bm25_index.pkl
```

#### 2. 서버 모드 자동 Warmup

`serve` 명령은 자동으로 백그라운드에서 컴포넌트를 미리 초기화합니다:

```bash
uv run regulation serve --web   # 자동으로 warmup 활성화
```

#### 3. CLI에서 수동 Warmup

환경변수로 warmup을 활성화할 수 있습니다:

```bash
WARMUP_ON_INIT=true uv run regulation
```

---

## 관련 문서

| 문서 | 설명 |
|------|------|
| [README.md](./README.md) | 시스템 개요 및 상세 기술 설명 |
| [LLM_GUIDE.md](./LLM_GUIDE.md) | LLM 설정 가이드 |
| [SCHEMA_REFERENCE.md](./SCHEMA_REFERENCE.md) | JSON 스키마 명세 |
| [AGENTS.md](./AGENTS.md) | 개발자 및 AI 에이전트 가이드 |
