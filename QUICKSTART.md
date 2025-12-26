# 빠른 시작 가이드

규정 관리 시스템의 설치부터 검색까지 단계별 안내입니다.

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

---

## 2단계: 규정 변환

HWP 파일을 `data/input/` 폴더에 배치한 후 변환합니다.

```bash
uv run regulation-manager "data/input/규정집.hwp"
```

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
uv run regulation-rag sync data/output/규정집.json
```

**성공 시 출력:**

```
✓ 동기화 완료: 추가 15,678 / 수정 0 / 삭제 0
ℹ 총 청크 수: 15,678
```

---

## 4단계: 규정 검색

```bash
# 기본 검색
uv run regulation-rag search "교원 연구년 신청 자격"

# 검색 결과 개수 지정
uv run regulation-rag search "장학금" -n 10

# 폐지 규정 포함
uv run regulation-rag search "학칙" --include-abolished
```

---

## 5단계: AI 질문 (선택)

LLM을 사용하여 자연어 답변을 생성합니다.

```bash
# 기본 (Ollama)
uv run regulation-rag ask "교원 연구년 신청 자격은?"

# 다른 LLM 프로바이더 사용
uv run regulation-rag ask "휴학 절차" --provider lmstudio --base-url http://localhost:1234
```

---

## 6단계: 웹 UI (선택)

```bash
uv run regulation-web
```

브라우저에서 파일 업로드 → 변환 → DB 동기화 → 질문까지 통합 인터페이스로 진행할 수 있습니다.

---

## 자주 사용하는 명령어

| 작업 | 명령어 |
|------|--------|
| 변환 | `regulation-manager "data/input/규정집.hwp"` |
| 동기화 | `regulation-rag sync <json-path>` |
| 검색 | `regulation-rag search "<쿼리>"` |
| AI 질문 | `regulation-rag ask "<질문>"` |
| 상태 확인 | `regulation-rag status` |
| DB 초기화 | `regulation-rag reset --confirm` |

---

## 문제 해결

### "데이터베이스가 비어 있습니다"

`sync` 명령을 먼저 실행하세요.

### "파일을 찾을 수 없습니다"

파일 경로를 확인하세요. 절대 경로 또는 `data/input/` 상대 경로를 사용합니다.

### 변환 품질이 낮음

LLM 전처리를 활성화하세요:

```bash
uv run regulation-manager "규정.hwp" --use_llm --provider ollama --model gemma2
```

LLM 설정에 대한 자세한 내용은 [docs/LLM_GUIDE.md](./docs/LLM_GUIDE.md)를 참고하세요.

---

## 관련 문서

- [README.md](./README.md) - 전체 안내
- [docs/LLM_GUIDE.md](./docs/LLM_GUIDE.md) - LLM 설정 가이드
- [SCHEMA_REFERENCE.md](./SCHEMA_REFERENCE.md) - JSON 스키마 명세
