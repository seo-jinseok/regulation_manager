# 규정 관리 시스템 (Regulation Management System)

대학 규정(HWP 파일)을 구조화된 JSON으로 변환하고, Hybrid RAG 데이터베이스로 검색할 수 있게 해주는 시스템입니다.

---

## 🚀 Quick Start

5분 안에 규정 검색까지 시작해 보세요!

```bash
# 1. 설치
git clone <repository-url> && cd regulation_manager
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. HWP → JSON 변환 (data/input/에 HWP 파일 배치 후)
uv run python -m src.main "data/input/규정집.hwp"

# 3. 벡터 DB 동기화
uv run python -m src.rag.interface.cli sync data/output/규정집.json

# 4. 검색!
uv run python -m src.rag.interface.cli search "교원 연구년 신청 자격"
```

> 💡 **더 자세한 단계별 가이드**: [QUICKSTART.md](./QUICKSTART.md)

---

## 📊 전체 워크플로우

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         규정 관리 시스템 워크플로우                          │
└─────────────────────────────────────────────────────────────────────────┘

  ┌─────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐
  │  HWP   │ ──▶ │   Markdown   │ ──▶ │     JSON     │ ──▶ │  ChromaDB │
  │  파일   │     │   (중간형식)   │     │ (RAG Enhanced)│     │ (벡터 DB) │
  └─────────┘     └──────────────┘     └──────────────┘     └───────────┘
                                             │                     │
       python -m src.main ─────────────────▶│                     │
                                             │                     │
       python -m src.rag.interface.cli sync ───────────────────▶│
                                                                   │
       python -m src.rag.interface.cli search ◀─────────────────┘
```

| 단계 | 입력 | 출력 | 명령어 |
|------|------|------|--------|
| **1. 변환** | `규정집.hwp` | `규정집.json`, `규정집_raw.md` | `python -m src.main` |
| **2. 동기화** | `규정집.json` | ChromaDB (`data/chroma_db/`) | `cli sync` |
| **3. 검색** | 자연어 쿼리 | 관련 규정 조항 | `cli search` |

---

## 📋 주요 기능

| 기능 | 설명 |
|------|------|
| **HWP → JSON 변환** | 표, 이미지, 계층 구조 보존 |
| **계층적 파싱** | 장 > 절 > 관 > 조 > 항 > 호 > 목 |
| **RAG 최적화** | `parent_path`, `full_text`, `keywords`, `amendment_history` 필드 자동 생성 |
| **벡터 검색** | ChromaDB 기반 의미론적 검색 |
| **증분 동기화** | 월간 업데이트 시 변경분만 동기화 |

---

## 💻 사용법

### 1. 규정 변환 (HWP → JSON)

```bash
# 기본 실행 (RAG 최적화 자동 적용)
uv run python -m src.main "data/input/규정집.hwp"

# 출력 디렉토리 지정
uv run python -m src.main "data/input/규정집.hwp" --output_dir ./result

# RAG 최적화 비활성화
uv run python -m src.main "data/input/규정집.hwp" --no-enhance-rag
```

**출력 파일:**
- `규정집.json` - 구조화된 규정 데이터 (RAG 필드 포함)
- `규정집_raw.md` - 변환된 마크다운 원문
- `규정집_raw.xhtml` - HTML 원문 (디버깅용)
- `규정집_metadata.json` - 목차/색인 정보

### 2. RAG 데이터베이스 관리

#### 동기화 (sync)
JSON 파일을 벡터 DB에 적재합니다.

```bash
# 증분 동기화 (기본값 - 변경분만)
uv run python -m src.rag.interface.cli sync data/output/규정집.json

# 전체 재동기화
uv run python -m src.rag.interface.cli sync data/output/규정집.json --full

# DB 경로 지정
uv run python -m src.rag.interface.cli sync data/output/규정집.json --db-path ./my_db
```

**출력 예시:**
```
ℹ 데이터베이스: data/chroma_db
ℹ JSON 파일: 규정집.json
ℹ 증분 동기화 실행 중...
✓ 동기화 완료: 추가 1,234 / 수정 56 / 삭제 12
ℹ 총 청크 수: 15,678
```

#### 검색 (search)
자연어로 규정을 검색합니다.

```bash
# 기본 검색 (상위 5개)
uv run python -m src.rag.interface.cli search "교원 연구년 신청 자격"

# 결과 개수 지정
uv run python -m src.rag.interface.cli search "장학금 지급 기준" -n 10

# 폐지 규정 포함
uv run python -m src.rag.interface.cli search "학칙" --include-abolished
```

**출력 예시:**
```
          검색 결과: '교원 연구년 신청 자격'
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┓
┃ # ┃ 규정                    ┃ 조항                 ┃ 점수 ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━┩
│ 1 │ 동의대학교교원연구년규정 │ 제3조 연구년 신청자격 │ 0.89 │
│ 2 │ 동의대학교교원연구년규정 │ 제4조 연구년 기간    │ 0.76 │
│ 3 │ 동의대학교학칙           │ 제42조 교원의 임무   │ 0.65 │
└───┴─────────────────────────┴─────────────────────┴──────┘
```

#### 상태 확인 (status)
동기화 상태를 확인합니다.

```bash
uv run python -m src.rag.interface.cli status
```

**출력 예시:**
```
        동기화 상태
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ 항목             ┃ 값                  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ 마지막 동기화     │ 2025-12-25 15:30:00│
│ JSON 파일        │ 규정집-test01.json  │
│ 상태 파일 규정 수 │ 234                 │
│ DB 청크 수       │ 15,678              │
│ DB 규정 수       │ 234                 │
└──────────────────┴────────────────────┘
```

#### 초기화 (reset)
데이터베이스의 모든 데이터를 삭제합니다.

```bash
# DB 초기화 (--confirm 필수)
uv run python -m src.rag.interface.cli reset --confirm
```

**출력 예시:**
```
ℹ 데이터베이스: data/chroma_db
ℹ 삭제 예정 청크 수: 15,678
✓ 데이터베이스 초기화 완료! 15,678개 청크 삭제됨
```

> ⚠️ **주의**: 이 명령은 모든 데이터를 삭제합니다. 복구할 수 없습니다.

---

## ⚙️ 고급 옵션

### LLM 전처리 (문서 품질이 낮은 경우)

스캔 품질이 좋지 않아 문장이 부자연스럽게 끊어진 경우 LLM 보정을 사용합니다.

```bash
# Ollama (로컬)
uv run python -m src.main "규정.hwp" --use_llm --provider ollama --model gemma2

# LM Studio (로컬)
uv run python -m src.main "규정.hwp" --use_llm --provider lmstudio --base_url http://127.0.0.1:1234

# OpenAI (클라우드)
uv run python -m src.main "규정.hwp" --use_llm --provider openai --model gpt-4o
```

### 전체 명령어 옵션

| 옵션 | 설명 | 기본값 |
| :--- | :--- | :--- |
| `input_path` | (필수) 입력 HWP 파일 경로 | - |
| `--output_dir` | 결과 파일 저장 경로 | `data/output` |
| `--use_llm` | LLM 전처리 활성화 | False |
| `--provider` | `ollama`, `lmstudio`, `openai`, `gemini` | `openai` |
| `--model` | 사용할 모델 이름 | (Provider별) |
| `--no-enhance-rag` | RAG 최적화 비활성화 | False |

---

## 📁 프로젝트 구조

```
regulation_manager/
├── src/
│   ├── main.py              # 변환 파이프라인 진입점
│   ├── converter.py         # HWP → Markdown/HTML
│   ├── formatter.py         # Markdown → JSON
│   ├── enhance_for_rag.py   # RAG 최적화 필드 추가
│   └── rag/                  # RAG 시스템
│       ├── interface/cli.py  # CLI (sync, search, status)
│       ├── application/      # Use Cases
│       ├── domain/           # 도메인 모델
│       └── infrastructure/   # ChromaDB, JSON 로더
├── data/
│   ├── input/               # HWP 파일 입력
│   ├── output/              # JSON 출력
│   └── chroma_db/           # 벡터 DB 저장소
├── QUICKSTART.md            # 빠른 시작 가이드
├── SCHEMA_REFERENCE.md      # JSON 스키마 상세
└── AGENTS.md                # 개발자 가이드
```

---

## 📚 문서

| 문서 | 설명 |
|------|------|
| [QUICKSTART.md](./QUICKSTART.md) | 5분 빠른 시작 가이드 |
| [SCHEMA_REFERENCE.md](./SCHEMA_REFERENCE.md) | JSON 출력 스키마 상세 명세 |
| [AGENTS.md](./AGENTS.md) | 개발자 가이드 (빌드, 테스트, 코딩 스타일) |

---

## 🔧 설치 상세

### 요구사항
- Python 3.11+
- `uv` (권장) 또는 `pip`
- `hwp5` 라이브러리 (HWP 파일 처리)

### 설치

```bash
cd regulation_manager
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 환경 변수 설정

```bash
cp .env.example .env
```

**`.env` 주요 설정:**
```bash
# 클라우드 LLM API 키 (선택)
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...

# 캐시 설정 (선택)
LLM_CACHE_TTL_DAYS=30
LLM_CACHE_MAX_ENTRIES=5000
```
