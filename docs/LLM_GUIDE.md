# LLM 설정 가이드

규정 관리 시스템에서 LLM을 사용하는 방법에 대한 안내입니다.

---

## LLM 사용 위치

| 기능 | 용도 | 명령어 |
|------|------|--------|
| **전처리** | HWP → Markdown 변환 품질 향상 | `regulation-manager --use_llm` |
| **질문 답변** | 자연어 답변 생성 | `regulation-rag ask` |

---

## 지원 프로바이더

| 프로바이더 | 유형 | API 키 | 비고 |
|------------|------|--------|------|
| `ollama` | 로컬 | 불필요 | 가장 쉬운 로컬 설정 |
| `lmstudio` | 로컬 | 불필요 | GUI 기반 서버 |
| `mlx` | 로컬 | 불필요 | macOS Apple Silicon 전용 |
| `local` | 로컬 | 불필요 | OpenAI 호환 서버 (vLLM 등) |
| `openai` | 클라우드 | 필요 | GPT-4o 등 |
| `gemini` | 클라우드 | 필요 | Gemini Pro 등 |
| `openrouter` | 클라우드 | 필요 | 다양한 모델 통합 |

---

## 로컬 LLM 상세 설정

### Ollama

가장 간단한 로컬 LLM 설정입니다.

**설치:**
```bash
# macOS
brew install ollama

# 모델 다운로드
ollama pull gemma2
ollama pull llama3.1
```

**서버 실행:**
```bash
ollama serve  # 기본 포트: 11434
```

**사용:**
```bash
# 전처리
uv run regulation-manager "data/input/규정집.hwp" --use_llm --provider ollama --model gemma2

# 질문 답변
uv run regulation-rag ask "교원 연구년 신청 자격은?" --provider ollama --model gemma2
```

**권장 모델:**
- `gemma2` - 한국어 성능 우수, 가벼움
- `llama3.1` - 범용 성능
- `qwen2.5` - 다국어 지원

---

### LM Studio

GUI 기반 로컬 LLM 서버입니다.

**설치:**
1. [lmstudio.ai](https://lmstudio.ai) 에서 다운로드
2. 앱 실행 후 모델 다운로드 (예: `llama-3.1-8b-instruct`)
3. "Local Server" 탭에서 서버 시작 (기본 포트: 1234)

**사용:**
```bash
# 전처리
uv run regulation-manager "data/input/규정집.hwp" --use_llm --provider lmstudio --base_url http://127.0.0.1:1234

# 질문 답변
uv run regulation-rag ask "장학금 조건" --provider lmstudio --base-url http://127.0.0.1:1234
```

**주의사항:**
- LM Studio 앱에서 서버가 실행 중이어야 합니다
- 모델이 로드된 상태여야 합니다

---

### MLX (macOS Apple Silicon)

Apple Silicon Mac에서 최적화된 추론을 제공합니다.

**설치:**
```bash
pip install mlx-lm
```

**서버 실행:**
```bash
# OpenAI 호환 서버 시작
mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8080
```

**사용:**
```bash
# 전처리
uv run regulation-manager "data/input/규정집.hwp" --use_llm --provider mlx --base_url http://127.0.0.1:8080

# 질문 답변
uv run regulation-rag ask "휴학 절차" --provider mlx --base-url http://127.0.0.1:8080
```

**권장 모델 (4bit 양자화):**
- `mlx-community/Llama-3.2-3B-Instruct-4bit` - 작고 빠름
- `mlx-community/gemma-2-9b-it-4bit` - 한국어 성능 우수
- `mlx-community/Qwen2.5-7B-Instruct-4bit` - 다국어 지원

---

### 기타 OpenAI 호환 서버

vLLM, llama.cpp 서버 등 OpenAI API 호환 서버를 사용할 수 있습니다.

**vLLM 예시:**
```bash
# 서버 실행
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --port 8000
```

**llama.cpp 예시:**
```bash
# 서버 실행
./llama-server -m model.gguf --host 0.0.0.0 --port 8000
```

**사용:**
```bash
# 전처리
uv run regulation-manager "data/input/규정집.hwp" --use_llm --provider local --base_url http://127.0.0.1:8000

# 질문 답변
uv run regulation-rag ask "등록금 감면" --provider local --base-url http://127.0.0.1:8000
```

---

## 클라우드 LLM

### OpenAI

```bash
# .env 파일에 API 키 설정
OPENAI_API_KEY=sk-...
```

```bash
uv run regulation-manager "data/input/규정집.hwp" --use_llm --provider openai --model gpt-4o
uv run regulation-rag ask "졸업 요건" --provider openai --model gpt-4o
```

### Gemini

```bash
# .env 파일에 API 키 설정
GEMINI_API_KEY=AIza...
```

```bash
uv run regulation-manager "data/input/규정집.hwp" --use_llm --provider gemini --model models/gemini-1.5-pro
uv run regulation-rag ask "장학금 조건" --provider gemini
```

### OpenRouter

```bash
# .env 파일에 API 키 설정
OPENROUTER_API_KEY=sk-or-...
```

```bash
uv run regulation-manager "data/input/규정집.hwp" --use_llm --provider openrouter --model google/gemini-pro-1.5
uv run regulation-rag ask "연구년 요건" --provider openrouter
```

---

## 환경 변수 기본값

CLI 옵션을 생략하면 다음 환경 변수가 사용됩니다:

```bash
# .env 파일
LLM_PROVIDER=ollama
LLM_MODEL=gemma2
LLM_BASE_URL=http://localhost:11434
```

---

## 웹 UI

```bash
uv run regulation-web
```

- **올인원 탭**: 업로드 → 변환 → 동기화 → 질문 통합 워크플로우
- **LLM 설정**: 프로바이더, 모델, Base URL 설정 가능
- **데이터 현황 탭**: HWP/JSON 파일 및 동기화 상태 확인

---

## 문제 해결

| 오류 | 원인 | 해결 방법 |
|------|------|----------|
| `LLM 초기화 실패` | API 키 없음 또는 Base URL 오류 | `.env` 파일 확인 |
| `Connection refused` | 로컬 서버 미실행 | 서버 실행 상태 확인 |
| `model not found` | 모델명 오류 | 올바른 모델명 사용 |
| 응답 속도 느림 | 모델 크기 과다 | 작은 모델 또는 양자화 버전 사용 |

---

## 권장 설정

| 용도 | 프로바이더 | 모델 | 비고 |
|------|------------|------|------|
| **개발/테스트** | `ollama` | `gemma2` | 빠른 응답, 한국어 지원 |
| **macOS 최적화** | `mlx` | `gemma-2-9b-it-4bit` | Apple Silicon 활용 |
| **고품질 답변** | `openai` | `gpt-4o` | 비용 발생 |
| **오프라인 환경** | `lmstudio` | 사용자 선택 | GUI 기반 관리 |
