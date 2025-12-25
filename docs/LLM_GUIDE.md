# LLM Usage Guide

This project uses LLMs in two places:
1) Preprocessing (HWP -> Markdown cleanup) via `src.main`.
2) Q&A answers in RAG CLI via `src.rag.interface.cli ask`.

## Quick Choice

| Goal | Provider | Notes |
|------|----------|-------|
| Local, no API key | `ollama` | Easiest local setup |
| Local, GUI server | `lmstudio` | OpenAI-compatible server |
| macOS MLX | `mlx` | OpenAI-compatible MLX server required |
| Any OpenAI-compatible local server | `local` | vLLM, llama.cpp server, etc. |
| Cloud | `openai`, `gemini`, `openrouter` | API key required |

## Common Options

- `--provider`: `openai`, `gemini`, `openrouter`, `ollama`, `lmstudio`, `local`, `mlx`
- `--model`: provider-specific model name
- `--base_url` (main pipeline) or `--base-url` (RAG CLI):
  - Required for `lmstudio`, `local`, `mlx`
  - Optional for `ollama` (default: `http://localhost:11434`)

API keys live in `.env` (copy from `.env.example`).

### Environment Defaults

If CLI flags are omitted, these environment variables are used:
- `LLM_PROVIDER`
- `LLM_MODEL`
- `LLM_BASE_URL`

## Local LLMs

### Ollama
```bash
# Preprocess
uv run python -m src.main "data/input/규정집.hwp" --use_llm --provider ollama --model gemma2

# RAG ask
uv run python -m src.rag.interface.cli ask "교원 연구년 신청 자격은?" --provider ollama
```

### LM Studio (OpenAI-compatible server)
```bash
# Preprocess
uv run python -m src.main "data/input/규정집.hwp" --use_llm --provider lmstudio --base_url http://127.0.0.1:1234

# RAG ask
uv run python -m src.rag.interface.cli ask "장학금 조건" --provider lmstudio --base-url http://127.0.0.1:1234
```

### MLX (macOS)
If you run an OpenAI-compatible MLX server (for example, via `mlx_lm.server`),
use the `mlx` provider.

```bash
# Preprocess
uv run python -m src.main "data/input/규정집.hwp" --use_llm --provider mlx --base_url http://127.0.0.1:8080

# RAG ask
uv run python -m src.rag.interface.cli ask "휴학 절차" --provider mlx --base-url http://127.0.0.1:8080
```

### Generic OpenAI-compatible server
```bash
# Preprocess
uv run python -m src.main "data/input/규정집.hwp" --use_llm --provider local --base_url http://127.0.0.1:8000

# RAG ask
uv run python -m src.rag.interface.cli ask "등록금 감면" --provider local --base-url http://127.0.0.1:8000
```

## Cloud LLMs

### OpenAI
```bash
uv run python -m src.main "data/input/규정집.hwp" --use_llm --provider openai --model gpt-4o
uv run python -m src.rag.interface.cli ask "졸업 요건" --provider openai --model gpt-4o
```

### Gemini
```bash
uv run python -m src.main "data/input/규정집.hwp" --use_llm --provider gemini --model models/gemini-1.5-pro
uv run python -m src.rag.interface.cli ask "장학금 조건" --provider gemini --model models/gemini-1.5-pro
```

### OpenRouter
```bash
uv run python -m src.main "data/input/규정집.hwp" --use_llm --provider openrouter --model google/gemini-pro-1.5
uv run python -m src.rag.interface.cli ask "연구년 요건" --provider openrouter --model google/gemini-pro-1.5
```

## Gradio UI

The web UI lets you choose provider/model/base URL per question:

```bash
uv run python -m src.rag.interface.gradio_app
```

Open the "질문하기" tab and expand "LLM 설정" to configure local or cloud providers.

## Troubleshooting

- `LLM 초기화 실패`: API key가 없거나 base URL이 잘못되었습니다.
- `Connection refused`: 로컬 서버가 실행 중인지 확인하세요.
- `llama-index is not installed`: `uv pip install -r requirements.txt`를 다시 실행하세요.
