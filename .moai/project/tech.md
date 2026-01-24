# 기술 스택 (Technology Stack)

## 기술 스택 개요

대학 규정 관리 시스템은 Python 3.11+ 기반으로 최신 AI/ML 기술을 활용하여 구축되었습니다. Clean Architecture를 따르는 모듈화된 설계로 유지보수성과 확장성을 확보했습니다.

## 핵심 프레임워크 및 라이브러리

### RAG (검색 증강 생성)
- **llama-index >= 0.14.10**: RAG 파이프라인 프레임워크
  - 통합된 검색 및 생성 워크플로우 지원
  - 다양한 LLM 통합 추상화
  - 모듈화된 컴포넌트 설계

### 벡터 데이터베이스
- **chromadb >= 1.4.0**: 로컬 영속 벡터 데이터베이스
  - 로컬 파일 기반 저장으로 데이터 프라이버시 보호
  - 빠른 검색 속도와 낮은 리소스 사용
  - Python 네이티브 통합

### 임베딩 모델
- **flagembedding >= 1.3.5**: BGE-M3 임베딩 모델
  - 1024차원 다국어 임베딩
  - 한국어 최적화
  - Dense + Sparse + Multi-Vector 혼합 지원

### 형태소 분석
- **konlpy >= 0.6.0**: 한국어 형태소 분석
  - Komoran 형태소 분석기 사용
  - BM25 토큰화 품질 향상
  - 복합어 분리를 통한 검색 정확도 개선

### HWP 처리
- **pyhwp >= 0.1b15**: HWP 파일 처리
  - HWP → HTML 변환
  - 텍스트, 서식, 계층 구조 추출
  - CLI 도구 제공 (hwp5html)

### 웹 UI
- **gradio >= 6.2.0**: 대화형 웹 인터페이스
  - ChatGPT 스타일 채팅 UI
  - 파일 업로드 기능
  - 빠른 프로토타이핑

### MCP (Model Context Protocol)
- **mcp[cli] >= 1.9.0**: AI 에이전트 연동
  - Claude, Cursor 등 AI 도구와 통합
  - 표준 프로토콜 기반 통신
  - 도구 및 리소스 노출

### Apple Silicon 최적화
- **mlx-lm >= 0.29.1**: Apple Silicon GPU 가속
  - M1/M2/M3 칩셋 최적화
  - 로컬 LLM 추론 가속
  - 낮은 메모리 사용량

### CLI 도구
- **questionary >= 2.1.1**: 대화형 CLI
- **rich >= 14.2.0**: 터미널 UI 스타일링

## 프레임워크 및 라이브러리 선정 이유

### llama-index 선택 이유
1. **통합된 RAG 파이프라인**: 검색, 재정렬, 생성을 하나의 프레임워크로 관리
2. **다양한 LLM 지원**: Ollama, OpenAI, Gemini 등 통합 인터페이스
3. **모듈화된 설계**: 각 컴포넌트를 독립적으로 교체 가능
4. **활발한 커뮤니티**: 빠른 버전 업데이트와 버그 수정

### ChromaDB 선택 이유
1. **로컬 우선**: 외부 서버 없이 로컬 파일 기반 저장
2. **데이터 프라이버시**: 민감한 규정 데이터가 기관을 떠나지 않음
3. **설치 간편**: 별도의 DB 서버 설정 불필요
4. **Python 네이티브**: Python 애플리케이션과 원활한 통합

### BGE-M3 선택 이유
1. **다국어 지원**: 한국어를 포함한 100+ 언어 지원
2. **긴 문서 처리**: 최대 8192 토큰 처리 가능
3. **혼합 임베딩**: Dense, Sparse, Multi-Vector 결합
4. **SOTA 성능**: MTEB 벤치마크에서 최상위 성능

### Gradio 선택 이유
1. **빠른 개발**: Python 코드만으로 웹 UI 구축
2. **사용자 친화적**: ChatGPT 스타일의 친숙한 인터페이스
3. **공유 용이**: 공개 링크 생성 및 임베드
4. **반응형**: 모바일 지원

### KoNLPy 선택 이유
1. **한국어 특화**: 한국어 형태소 분석에 최적화
2. **다양한 분석기**: Komoran, Hannanum, Kkma 등 지원
3. **BM25 토큰화**: 형태소 단위 분리로 키워드 검색 개선
4. **간편한 통합**: pip 설치로 바로 사용 가능

## 개발 환경 요구사항

### 필수 사양
| 항목 | 최소 사양 | 권장 사양 |
|------|----------|----------|
| 운영체제 | macOS 12+, Ubuntu 20.04+ | macOS 14+, Ubuntu 22.04+ |
| Python | 3.11 | 3.12 |
| RAM | 8GB | 16GB (Reranker 사용 시) |
| 디스크 | 5GB | 10GB+ |
| GPU | 불필요 | CUDA 지원 시 임베딩 가속 |

### 패키지 매니저
- **uv**: Python 패키지 매니저
  - pip보다 10-100배 빠른 설치 속도
  - 의존성 해결의 정확성
  - 가상환경 관리 통합

### 필수 의존성
```bash
# Python 패키지
uv sync

# HWP 변환 도구
uv add pyhwp
```

### 선택적 의존성
```bash
# 개발 도구
uv add --dev pytest pytest-cov ruff

# LLM (로컬)
# Ollama: https://ollama.com
# LM Studio: https://lmstudio.ai

# LLM (클라우드)
# OpenAI API Key 필요
# Gemini API Key 필요
```

## 빌드 및 배포 설정

### 로컬 개발
```bash
# 가상환경 생성 및 활성화
uv venv && source .venv/bin/activate

# 의존성 설치
uv sync

# 환경 변수 설정
cp .env.example .env
```

### CLI 실행
```bash
# 대화형 모드
uv run regulation

# HWP 변환
uv run regulation convert "data/input/규정집.hwp"

# DB 동기화
uv run regulation sync data/output/규정집.json

# 검색
uv run regulation search "휴학 절차"
```

### Web UI 실행
```bash
uv run regulation serve --web
# 기본 주소: http://localhost:7860
```

### MCP 서버 실행
```bash
uv run regulation serve --mcp
# stdio 기반 통신
```

### 테스트 실행
```bash
# 전체 테스트
uv run pytest

# 커버리지 포함
uv run pytest --cov=src --cov-report=html

# 특정 테스트
uv run pytest tests/test_search.py -v
```

### 코드 품질 검사
```bash
# Rinting
uv run ruff check src/

# Formatting
uv run ruff format src/
```

### 배포 (PyPI)
```bash
# 빌드
uv build

# 퍼블리시 (TestPyPI)
uv publish --index testpypi

# 퍼블리시 (PyPI)
uv publish
```

## LLM 설정

### 지원 프로바이더
| 프로바이더 | 모델 예시 | 용도 |
|-----------|----------|------|
| Ollama | gemma2, llama3 | 로컬 추천 |
| LM Studio | 다양 | 로컬 |
| OpenAI | gpt-4o, gpt-4o-mini | 클라우드 |
| Gemini | gemini-1.5-flash | 클라우드 |
| OpenRouter | 다양 | 다중 |

### 환경 변수 설정
```bash
# 기본 LLM 설정
LLM_PROVIDER=ollama
LLM_MODEL=gemma2
LLM_BASE_URL=http://localhost:11434

# OpenAI
OPENAI_API_KEY=sk-...

# Gemini
GEMINI_API_KEY=AIza...
```

## 고급 RAG 설정

### 환경 변수
```bash
# Self-RAG 활성화
ENABLE_SELF_RAG=true

# HyDE 활성화
ENABLE_HYDE=true

# BM25 토큰화 모드
BM25_TOKENIZE_MODE=konlpy  # konlpy | morpheme | simple

# HyDE 캐시
HYDE_CACHE_ENABLED=true
HYDE_CACHE_DIR=data/cache/hyde
```

### 동의어/인텐트 사전
```bash
# 기본 제공 (커스텀 시 경로 변경)
RAG_SYNONYMS_PATH=data/config/synonyms.json
RAG_INTENTS_PATH=data/config/intents.json
```

## 보안 고려사항

### 데이터 보호
- **로컬 LLM 권장**: Ollama 등 로컬 모델 사용 시 데이터가 외부 서버로 전송되지 않음
- **클라우드 LLM 사용 시**: 해당 서비스의 데이터 처리 정책 확인

### 환경 변수 관리
- `.env` 파일을 `.gitignore`에 포함
- API 키를 코드에 직접 하드코딩 금지
- `.env` 파일 권한 설정 (`chmod 600 .env`)

### 네트워크 보안
- 웹 UI는 기본적으로 `localhost`에서만 접근 가능
- 외부 공개 시 방화벽 및 인증 설정 필요
- MCP 서버는 로컬 통신만 지원

## 성능 최적화

### 임베딩 캐싱
- 임베딩 결과를 캐시하여 중복 계산 방지
- ChromaDB의 내장 캐시 활용

### LLM 캐싱
- 동일한 프롬프트에 대한 응답 캐시
- `data/cache/llm/`에 저장

### HyDE 캐싱
- 생성된 가상 문서를 영구 저장
- 동일 쿼리 재사용 시 LLM 호출 생략

### 증분 동기화
- 해시 비교로 변경된 규정만 업데이트
- 전체 재동기화보다 10-100배 빠름

## 추후 확장 가능성

### 벡터 DB 교체
- ChromaDB → Pinecone, Weaviate, Qdrant 등
- `Repository` 인터페이스 구현만 변경

### LLM 추가
- Anthropic Claude, Cohere 등
- `LLMAdapter`만 추가 구현

### 임베딩 모델 교체
- OpenAI Embeddings, Cohere Embeddings 등
- 설정 변경만으로 가능

### 웹 프레임워크 교체
- Gradio → Streamlit, FastAPI, Next.js 등
- Interface 계층만 수정
