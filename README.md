# 규정 관리 시스템 (Regulation Management System)

이 소프트웨어는 대학 규정(HWP 파일)을 효율적으로 관리하고 데이터베이스화하기 위해 개발되었습니다.
규정 파일을 Markdown으로 변환하고, 전처리 과정을 거쳐, 대한민국 법률정보 센터 호환 JSON 형식으로 변환합니다.
전처리 과정에서 **로컬 LLM (Ollama, LM Studio)** 또는 상용 클라우드 LLM을 선택적으로 사용할 수 있습니다.

## 주요 기능

## 주요 기능

1.  **HWP → Markdown 변환**: `hwp5html`과 `markdownify`를 사용하여 표(Table)와 이미지, 문서 구조를 최대한 유지하며 변환합니다.
2.  **지능형 전처리 (Preprocessing)**:
    *   **Regex 기반 (기본값)**: 대부분의 깔끔한 규정 파일은 이 모드만으로도 완벽하게 처리됩니다. 페이지 번호, 아티팩트 제거, 단순한 문장 연결 등을 수행하며 속도가 매우 빠릅니다.
    *   **LLM 기반 (옵션)**: 파일 상태가 좋지 않아 문장이 심하게 끊기거나 의미론적 복구가 필요한 경우에만 사용을 권장합니다. (Ollama, LM Studio 등 지원)
3.  **JSON 구조화**: 변환된 텍스트를 `조(Article)`, `항(Paragraph)`, `호(Item)` 단위의 계층적 JSON 데이터로 파싱합니다.
4.  **자동 정제 (Refinement)**: 변환된 JSON에서 조항 내 챕터 정보를 분리하고, 부칙과 별지/서식을 별도 필드로 구조화합니다.

## 설치 방법 (Installation)

이 프로젝트는 Python 3.11+ 환경에서 동작합니다.

```bash
# 디렉토리 이동
cd regulation_manager

# 가상환경 생성 (uv 사용)
uv venv

# 가상환경 활성화
source .venv/bin/activate

# 의존성 설치
uv pip install -r requirements.txt
```

### 환경 변수 설정 (.env)

`cp .env.example .env` 명령어로 설정 파일을 생성하세요. 로컬 LLM이나 Regex 모드만 사용한다면 API 키 설정 없이 비워두어도 됩니다.

---

## 사용법 (Usage)

### 1. 기본 사용법 (권장) - LLM 미사용
대부분의 경우 **LLM 옵션 없이** 실행하는 것이 빠르고 정확합니다.

```bash
# 가상환경 활성화 상태에서:
python -m src.main "/path/to/규정.hwp"

# 또는 `uv run` 사용 (활성화 불필요):
uv run python -m src.main "/path/to/규정.hwp"
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
Done (6322.32s)
  > Preprocessing (Cleaning & Logic)... Done (0.75s)
  > Formatting to JSON Structure... Done (0.03s)
  > Refining JSON Structure... Done (0.01s)
  > Saving JSON (373 docs found)... Done (0.06s)
[20:53:44] Completed 규정집9-343(20250909).hwp in 6323.18s