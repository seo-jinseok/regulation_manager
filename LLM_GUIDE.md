# LLM 설정 가이드

규정 관리 시스템에서 LLM(Large Language Model, 대규모 언어 모델)을 설정하고 사용하는 방법을 안내합니다.

---

## LLM이란?

**LLM(Large Language Model)**은 대량의 텍스트 데이터를 학습하여 인간과 유사한 언어를 이해하고 생성할 수 있는 AI 모델입니다. ChatGPT, Gemini, Llama 등이 대표적인 예입니다.

### 본 시스템에서 LLM의 역할

규정 관리 시스템에서 LLM은 다음 두 가지 주요 역할을 수행합니다:

| 역할 | 설명 | 필수 여부 |
|------|------|----------|
| **전처리** | HWP → JSON 변환 시 복잡한 문서 구조를 더 정확하게 파싱 | 선택 |
| **답변 생성** | 검색된 규정을 바탕으로 자연어 답변 작성 | AI 답변 사용 시 필수 |

> **LLM 없이도 작동하는 기능**: 문서 검색, 전문 보기, 대상 선택(교수/학생/직원), 동기화 등

---

## LLM 사용 위치

| 기능 | 용도 | 명령어 |
|------|------|--------|
| **전처리** | HWP → Markdown 변환 품질 향상 | `regulation convert --use_llm` |
| **질문 답변** | 자연어 답변 생성 | `regulation search -a` |
| **MCP 서버** | AI 에이전트 연동 | `regulation serve --mcp` |

---

## 지원 프로바이더

본 시스템은 다양한 LLM 프로바이더를 지원합니다. 환경과 요구사항에 맞게 선택하세요.

### 로컬 vs 클라우드 비교

| 구분 | 로컬 LLM | 클라우드 LLM |
|------|---------|-------------|
| **비용** | 무료 (전기요금만) | API 호출당 과금 |
| **속도** | 하드웨어에 따라 다름 | 일반적으로 빠름 |
| **프라이버시** | 데이터가 외부로 나가지 않음 | 데이터가 외부 서버로 전송 |
| **설정 난이도** | 서버 실행 필요 | API 키만 설정 |
| **인터넷** | 불필요 | 필요 |

### 프로바이더 목록

| 프로바이더 | 유형 | API 키 | 비고 |
|------------|------|--------|------|
| `ollama` | 로컬 | 불필요 | 가장 쉬운 로컬 설정, 권장 |
| `lmstudio` | 로컬 | 불필요 | GUI 기반 서버 |
| `mlx` | 로컬 | 불필요 | macOS Apple Silicon 전용, M1/M2/M3 최적화 |
| `local` | 로컬 | 불필요 | OpenAI 호환 서버 (vLLM 등) |
| `openai` | 클라우드 | 필요 | GPT-4o 등, 가장 높은 품질 |
| `gemini` | 클라우드 | 필요 | Gemini Pro 등 |
| `openrouter` | 클라우드 | 필요 | 다양한 모델 통합 |

---

## 선택 가이드

환경과 목적에 따른 추천 설정입니다.

### 환경별 추천

| 환경 | 추천 프로바이더 | 추천 모델 | 비고 |
|------|----------------|----------|------|
| **개발/테스트** | `ollama` | `gemma2` | 무료, 한국어 양호, 빠른 응답 |
| **macOS M시리즈** | `mlx` | `gemma-2-9b-it-4bit` | Apple Silicon 최적화 |
| **고품질 답변** | `openai` | `gpt-4o` | 비용 발생, 최고 품질 |
| **오프라인 환경** | `lmstudio` | 사용자 선택 | GUI 기반, 인터넷 불필요 |
| **비용 효율** | `openrouter` | `gemini-flash` | 저렴한 클라우드 |

### 성능/비용 비교

| 프로바이더 | 한국어 품질 | 응답 속도 | 비용 |
|------------|-----------|----------|------|
| `openai` (gpt-4o) | ★★★★★ | ★★★★☆ | $$$$ |
| `gemini` (pro) | ★★★★☆ | ★★★★★ | $$ |
| `ollama` (gemma2) | ★★★☆☆ | ★★★★☆ | 무료 |
| `mlx` (gemma-2-9b) | ★★★★☆ | ★★★★☆ | 무료 |
| `lmstudio` | 모델에 따라 다름 | ★★★☆☆ | 무료 |

---

## 로컬 LLM 상세 설정

### Ollama

가장 간단한 로컬 LLM 설정입니다. macOS, Linux, Windows 모두 지원합니다.

**설치**:

```bash
# macOS
brew install ollama

# 모델 다운로드
ollama pull gemma2
ollama pull llama3.1
```

**서버 실행**:

```bash
ollama serve  # 기본 포트: 11434
```

**사용**:

```bash
# 전처리
uv run regulation convert "data/input/규정집.hwpx" --use_llm --provider ollama --model gemma2

# 질문 답변
uv run regulation search "교원 연구년 신청 자격은?" -a --provider ollama --model gemma2
```

**권장 모델**:

| 모델 | 특징 | 크기 |
|------|------|------|
| `gemma2` | 한국어 성능 우수, 가벼움 | ~5GB |
| `llama3.1` | 범용 성능 | ~8GB |
| `qwen2.5` | 다국어 지원 | ~8GB |

---

### LM Studio

GUI 기반 로컬 LLM 서버입니다. 설정이 직관적이며 다양한 모델을 쉽게 다운로드할 수 있습니다.

**설치**:

1. [lmstudio.ai](https://lmstudio.ai)에서 다운로드
2. 앱 실행 후 모델 다운로드 (예: `llama-3.1-8b-instruct`)
3. "Local Server" 탭에서 서버 시작 (기본 포트: 1234)

**사용**:

```bash
# 전처리
uv run regulation convert "data/input/규정집.hwpx" --use_llm --provider lmstudio --base-url http://127.0.0.1:1234

# 질문 답변
uv run regulation search "장학금 조건" -a --provider lmstudio --base-url http://127.0.0.1:1234
```

**주의사항**:

- LM Studio 앱에서 서버가 실행 중이어야 합니다.
- 모델이 로드된 상태여야 합니다.

---

### MLX (macOS Apple Silicon)

Apple Silicon Mac(M1/M2/M3)에서 최적화된 추론을 제공합니다. GPU 메모리를 효율적으로 활용합니다.

**설치**:

```bash
# 시스템 전역 설치
pip install mlx-lm

# 또는 프로젝트 내 설치 (권장)
uv add mlx-lm
```

**서버 실행**:

```bash
# OpenAI 호환 서버 시작
mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8080
```

**사용**:

```bash
# 전처리
uv run regulation convert "data/input/규정집.hwpx" --use_llm --provider mlx --base-url http://127.0.0.1:8080

# 질문 답변
uv run regulation search "휴학 절차" -a --provider mlx --base-url http://127.0.0.1:8080
```

**권장 모델 (4bit 양자화)**:

| 모델 | 특징 | 메모리 |
|------|------|--------|
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | 작고 빠름 | ~2GB |
| `mlx-community/gemma-2-9b-it-4bit` | 한국어 성능 우수 | ~6GB |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | 다국어 지원 | ~5GB |

---

### 기타 OpenAI 호환 서버

vLLM, llama.cpp 서버 등 OpenAI API 호환 서버를 사용할 수 있습니다.

**vLLM 예시**:

```bash
# 서버 실행
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --port 8000
```

**llama.cpp 예시**:

```bash
# 서버 실행
./llama-server -m model.gguf --host 0.0.0.0 --port 8000
```

**사용**:

```bash
uv run regulation search "등록금 감면" -a --provider local --base-url http://127.0.0.1:8000
```

---

## 클라우드 LLM

### OpenAI

가장 높은 품질의 답변을 제공합니다. GPT-4o가 권장됩니다.

**설정**:

```bash
# .env 파일에 API 키 설정
OPENAI_API_KEY=sk-...
```

**사용**:

```bash
uv run regulation convert "data/input/규정집.hwpx" --use_llm --provider openai --model gpt-4o
uv run regulation search "졸업 요건" -a --provider openai --model gpt-4o
```

---

### Gemini

Google의 Gemini 모델을 사용합니다. 빠른 응답 속도가 특징입니다.

**설정**:

```bash
# .env 파일에 API 키 설정
GEMINI_API_KEY=AIza...
```

**사용**:

```bash
uv run regulation convert "data/input/규정집.hwpx" --use_llm --provider gemini --model models/gemini-1.5-pro
uv run regulation search "장학금 조건" -a --provider gemini
```

---

### OpenRouter

다양한 모델을 단일 API로 접근할 수 있는 서비스입니다.

**설정**:

```bash
# .env 파일에 API 키 설정
OPENROUTER_API_KEY=sk-or-...
```

**사용**:

```bash
uv run regulation convert "data/input/규정집.hwpx" --use_llm --provider openrouter --model google/gemini-pro-1.5
uv run regulation search "연구년 요건" -a --provider openrouter
```

---

## 환경 변수 기본값

CLI 옵션을 생략하면 다음 환경 변수가 사용됩니다:

```bash
# .env 파일
LLM_PROVIDER=ollama
LLM_MODEL=gemma2
LLM_BASE_URL=http://localhost:11434

# (선택) 검색 사전 (기본값 제공)
RAG_SYNONYMS_PATH=data/config/synonyms.json
RAG_INTENTS_PATH=data/config/intents.json
```

`.env`는 실행 시 자동 로드되므로, 위 값이 코드 기본값보다 우선 적용됩니다.

**코드 기본값** (옵션/`.env` 미설정 시):

| 사용 위치 | LLM 기본값 |
|-----------|-----------|
| `regulation convert` | provider: `openai` (model: `gpt-4o`) |
| `regulation search -a` / 웹 UI | provider: `ollama` (model: `gemma2`) |
| `regulation serve --mcp` | provider: `lmstudio` (base_url: `http://127.0.0.1:1234`) |

---

## 웹 UI

```bash
uv run regulation serve --web
```

웹 UI에서는 다음 설정이 가능합니다:

- **설정 패널**: 프로바이더, 모델, Base URL 설정
- **올인원 워크플로우**: 업로드 → 변환 → 동기화 → 질문 통합

---

## 문제 해결

| 오류 | 원인 | 해결 방법 |
|------|------|----------|
| `LLM 초기화 실패` | API 키 없음 또는 Base URL 오류 | `.env` 파일 확인 |
| `Connection refused` | 로컬 서버 미실행 | 서버 실행 상태 확인 (`ollama serve` 등) |
| `model not found` | 모델명 오류 또는 미다운로드 | 올바른 모델명 사용, `ollama pull <model>` |
| 응답 속도 느림 | 모델 크기 과다 | 작은 모델 또는 양자화 버전 사용 |
| 메모리 부족 | 모델 크기가 RAM보다 큼 | 더 작은 모델 사용 또는 양자화 버전 사용 |
| 한국어 품질 낮음 | 모델 선택 문제 | `gemma2` 또는 `qwen2.5` 사용 권장 |

---

## 관련 문서

| 문서 | 설명 |
|------|------|
| [README.md](./README.md) | 시스템 개요 및 상세 기술 설명 |
| [QUICKSTART.md](./QUICKSTART.md) | 빠른 시작 가이드 |
| [SCHEMA_REFERENCE.md](./SCHEMA_REFERENCE.md) | JSON 스키마 명세 |
| [AGENTS.md](./AGENTS.md) | 개발자 및 AI 에이전트 가이드 |
