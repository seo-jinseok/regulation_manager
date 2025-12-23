# 규정 관리 시스템 (Regulation Management System)

이 소프트웨어는 대학 규정(HWP 파일)을 효율적으로 관리하고 데이터베이스화하기 위해 개발되었습니다.
규정 파일을 Markdown으로 변환하고, 전처리 과정을 거쳐, 대한민국 법률정보 센터 호환 JSON 형식으로 변환합니다.
전처리 과정에서 **로컬 LLM (Ollama, LM Studio)** 또는 상용 클라우드 LLM을 선택적으로 사용할 수 있습니다.

## 주요 기능

1.  **HWP → Markdown 변환**: `hwp5html`과 `markdownify`를 사용하여 표(Table)와 이미지, 문서 구조를 최대한 유지하며 변환합니다.
2.  **지능형 전처리 (Preprocessing)**:
    *   **Regex 기반 (기본값)**: 대부분의 깔끔한 규정 파일은 이 모드만으로도 완벽하게 처리됩니다. 페이지 번호, 아티팩트 제거, 단순한 문장 연결 등을 수행하며 속도가 매우 빠릅니다.
    *   **LLM 기반 (옵션)**: 파일 상태가 좋지 않아 문장이 심하게 끊기거나 의미론적 복구가 필요한 경우에만 사용을 권장합니다. (Ollama, LM Studio 등 지원)
3.  **JSON 구조화**: 변환된 텍스트를 `조(Article)`, `항(Paragraph)`, `호(Item)` 단위의 계층적 JSON 데이터로 파싱합니다.
4.  **자동 정제 (Refinement)**: 변환된 JSON에서 조항 내 챕터 정보를 분리하고, 부칙과 별지/서식을 별도 필드로 구조화합니다.

## 설치 방법 (Installation)

이 프로젝트는 Python 3.11+ 환경에서 동작하며, `uv`를 사용한 의존성 관리를 권장합니다.

```bash
# 디렉토리 이동
cd regulation_manager

# 가상환경 생성 (uv)
uv venv

# 가상환경 활성화
source .venv/bin/activate

# 의존성 설치
uv pip install -r requirements.txt
```

### 환경 변수 설정 (.env)

`cp .env.example .env` 명령어로 설정 파일을 생성하세요.

캐시 동작을 조정하려면 아래 옵션을 추가할 수 있습니다:
* `LLM_CACHE_TTL_DAYS`: 캐시 만료 일수
* `LLM_CACHE_MAX_ENTRIES`: 캐시 최대 항목 수

---

## 사용법 (Usage)

### 1. 기본 사용법
입력 데이터는 `data/input` 폴더에 넣고 실행하는 것을 권장합니다. (대화형 모드에서는 `data/input`을 우선 탐색하고, 없을 경우 `규정` 폴더를 확인합니다.)

```bash
# 가상환경 활성화 상태에서:
python -m src.main "data/input/규정집.hwp"

# 또는 `uv run` 사용 (활성화 불필요):
uv run python -m src.main "data/input/규정집.hwp"

# 설치 후 CLI 엔트리포인트 사용:
regulation-manager "data/input/규정집.hwp"
```

### 2. 고급 사용법 - LLM 사용 (화질이 나쁜 문서용)
변환 결과에서 문장이 부자연스럽게 끊기거나 정규식만으로 처리가 안 될 경우 `--use_llm` 옵션을 켜세요.

#### 2-1. 로컬 LLM (Ollama) 사용
[ollama.com](https://ollama.com)에서 Ollama 설치 및 모델(`gemma2` 등) 다운로드 후:
```bash
python -m src.main "/path/to/규정.hwp" --use_llm --provider ollama --model gemma2
```

#### 2-2. 로컬 LLM (LM Studio) 사용
1. [lmstudio.ai](https://lmstudio.ai)에서 설치 및 모델(`eeve-korean` 등) 다운로드.
2. **Local Server** 탭에서 서버 Start.
3. 아래 명령어로 실행 (모델 ID는 서버 로그나 curl로 확인):
```bash
python -m src.main "/path/to/규정.hwp" --use_llm --provider lmstudio --base_url http://127.0.0.1:1234 --model eeve-korean-instruct-7b-v2.0-preview-mlx
```

#### 2-3. 클라우드 LLM (OpenAI) 사용
```bash
python -m src.main "/path/to/규정.hwp" --use_llm --provider openai --model gpt-4o
```

---

## 전체 명령어 옵션

| 옵션 | 설명 | 기본값 | 예시 |
| :--- | :--- | :--- | :--- |
| `input_path` | (필수) 입력 HWP 파일 또는 디렉토리 경로 | - | `"./규정집.hwp"` |
| `--output_dir` | 결과 파일 저장 경로 | `.../data/output` | `--output_dir ./result` |
| `--use_llm` | LLM 전처리 활성화 플래그 | False | `--use_llm` |
| `--provider` | `ollama`, `lmstudio`, `local`, `openai`, `gemini` | `openai` | `--provider ollama` |
| `--model` | 사용할 모델 이름 (LM Studio는 필수) | (Provider별 기본값) | `--model gemma2` |
| `--base_url` | 로컬 서버 API 주소 (필요 시 변경) | (Provider별 기본값) | `--base_url http://localhost:11434` |
| `--allow_llm_fallback` | LLM 초기화 실패 시 정규식 모드로 계속 진행 | False | `--allow_llm_fallback` |

> 기본 출력 경로는 `data/output`이며, 기존 `output/` 디렉토리는 레거시 경로로 취급됩니다.

### 출력 파일
* `<파일명>.json`: 최종 규정 JSON
* `<파일명>_raw.md`: 변환된 원본 마크다운
* `<파일명>_raw.xhtml`: 변환된 원본 HTML (가능한 경우)
* `<파일명>_metadata.json`: 차례/찾아보기 색인 추출 결과

### 전처리 규칙 커스터마이즈
`data/config/preprocessor_rules.json`을 수정하거나, 환경 변수 `PREPROCESSOR_RULES_PATH`로 규칙 파일 경로를 지정할 수 있습니다.

## 데이터 구조 (JSON Schema)

본 시스템은 규정집을 데이터베이스화하기 용이하도록 엄격한 계층 구조를 가진 JSON으로 변환합니다. 상세한 스펙은 [SCHEMA_REFERENCE.md](./SCHEMA_REFERENCE.md) 문서를 참고하세요.

### 구조 개요
*   **Root**: 파일명과 `docs` 리스트를 포함합니다.
*   **Document**: 개별 규정 단위(예: 학칙, 장학규정). `title`, `part`(편), `metadata`, `content` 등을 포함합니다.
*   **Node (Content)**: `content` 내부는 `type` 필드를 통해 계층적으로 구조화됩니다.
    *   **Type Hierarchy**: `chapter` (장) > `section` (절) > `subsection` (관) > `article` (조) > `paragraph` (항) > `item` (호) > `subitem` (목)
    *   **Addenda**: 부칙 또한 `article`과 `item` 노드로 구조화되어 파싱됩니다.

### 예시
```json
{
  "type": "article",
  "display_no": "제6조",
  "sort_no": { "main": 6, "sub": 0 },
  "title": "자산의 구분",
  "text": "",
  "children": [
    {
      "type": "paragraph",
      "display_no": "①",
      "sort_no": { "main": 1, "sub": 0 },
      "text": "이 법인의 자산은 기본재산과 보통재산으로 구분한다.",
      "children": []
    }
  ]
}
```

---
