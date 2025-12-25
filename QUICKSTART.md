# Quick Start Guide

5분 안에 규정 검색을 시작하세요!

---

## 1️⃣ 설치 (2분)

```bash
# 저장소 클론 및 이동
git clone <repository-url>
cd regulation_manager

# 가상환경 생성 및 활성화
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
uv sync
```

---

## 2️⃣ 규정 변환 (1분)

HWP 파일을 `data/input/` 폴더에 넣고 변환합니다.

```bash
# 변환 실행
uv run python -m src.main "data/input/규정집.hwp"
```

**결과물** (`data/output/`):
- ✅ `규정집.json` - 구조화된 JSON (RAG 필드 포함)
- 📄 `규정집_raw.md` - 마크다운 원문
- 📋 `규정집_metadata.json` - 목차/색인

---

## 3️⃣ 벡터 DB 동기화 (1분)

변환된 JSON을 ChromaDB에 적재합니다.

```bash
uv run python -m src.rag.interface.cli sync data/output/규정집.json
```

**성공 시 출력:**
```
✓ 동기화 완료: 추가 15,678 / 수정 0 / 삭제 0
ℹ 총 청크 수: 15,678
```

---

## 4️⃣ 검색! (바로 사용)

```bash
# 자연어로 검색
uv run python -m src.rag.interface.cli search "교원 연구년 신청 자격"

# 더 많은 결과
uv run python -m src.rag.interface.cli search "장학금" -n 10
```

---

## 5️⃣ LLM 질문 (선택)

```bash
# 로컬 LLM (기본: Ollama)
uv run python -m src.rag.interface.cli ask "교원 연구년 신청 자격은?"

# 다른 프로바이더
uv run python -m src.rag.interface.cli ask "휴학 절차" --provider lmstudio --base-url http://localhost:1234
```

---

## 6️⃣ 웹 UI (선택)

```bash
uv run python -m src.rag.interface.gradio_app
```

브라우저에서 “올인원” 탭을 열고 파일 업로드 → 변환 → DB 동기화 → 질문까지 한 번에 진행하세요.
올인원 탭의 LLM 설정은 전처리와 질문에 함께 적용됩니다.

---

## 📌 자주 쓰는 명령어

| 작업 | 명령어 |
|------|--------|
| 변환 | `uv run python -m src.main "data/input/규정집.hwp"` |
| 동기화 | `uv run python -m src.rag.interface.cli sync <json-path>` |
| 검색 | `uv run python -m src.rag.interface.cli search "<쿼리>"` |
| **LLM 질문** | `uv run python -m src.rag.interface.cli ask "<질문>"` |
| 웹 UI | `uv run python -m src.rag.interface.gradio_app` |
| 상태 확인 | `uv run python -m src.rag.interface.cli status` |
| DB 초기화 | `uv run python -m src.rag.interface.cli reset --confirm` |

---

## ❓ 문제 해결

### "데이터베이스가 비어 있습니다"
→ `sync` 명령을 먼저 실행하세요.

### "파일을 찾을 수 없습니다"
→ 파일 경로를 확인하세요. `data/input/` 또는 절대 경로를 사용합니다.

### 변환 품질이 좋지 않음
→ `--use_llm` 옵션으로 LLM 보정을 활성화하세요:
```bash
uv run python -m src.main "규정.hwp" --use_llm --provider ollama --model gemma2
```
→ 로컬/상용 LLM 설정은 [docs/LLM_GUIDE.md](./docs/LLM_GUIDE.md)를 참고하세요.

---

**더 자세한 정보**: [README.md](./README.md) | [docs/LLM_GUIDE.md](./docs/LLM_GUIDE.md) | [SCHEMA_REFERENCE.md](./SCHEMA_REFERENCE.md)
