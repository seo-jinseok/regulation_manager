# 프로젝트 구조 (Project Structure)

## 디렉터리 구조

```
regulation_manager/
├── .moai/                          # MoAI-ADK 설정
│   └── project/                    # 프로젝트 문서
│       ├── product.md              # 프로젝트 개요
│       ├── structure.md            # 디렉터리 구조
│       └── tech.md                 # 기술 스택
│
├── src/                            # 소스 코드 (Clean Architecture)
│   ├── converter.py                # HWP → Markdown/HTML 변환
│   ├── formatter.py                # Markdown → JSON 변환
│   ├── enhance_for_rag.py          # RAG 최적화 필드 추가
│   ├── preprocessor.py             # HWP 전처리
│   ├── repair.py                   # JSON 복구 도구
│   ├── refine_json.py              # JSON 정제 도구
│   ├── verify_json.py              # JSON 검증 도구
│   ├── cache_manager.py            # 캐시 관리
│   │
│   ├── parsing/                    # 파싱 모듈
│   │   ├── regulation_parser.py   # 규정 파서
│   │   ├── reference_resolver.py  # 상호 참조 해석
│   │   └── hierarchy_builder.py   # 계층 구조 빌더
│   │
│   ├── rag/                        # RAG 시스템 (Clean Architecture)
│   │   ├── interface/              # 인터페이스 계층 (CLI, Web, MCP)
│   │   │   ├── unified_cli.py     # 통합 CLI 진입점
│   │   │   ├── cli.py             # CLI 인터페이스
│   │   │   ├── gradio_app.py      # Gradio Web UI
│   │   │   ├── mcp_server.py      # MCP 서버
│   │   │   ├── query_handler.py   # 쿼리 처리기
│   │   │   ├── chat_logic.py      # 채팅 로직
│   │   │   ├── formatters.py      # 출력 포맷터
│   │   │   ├── query_suggestions.py # 쿼리 제안
│   │   │   ├── link_formatter.py  # 링크 포맷터
│   │   │   └── common.py          # 공유타입/상수
│   │   │
│   │   ├── application/            # 애플리케이션 계층 (Use Cases)
│   │   │   ├── search_usecase.py  # 검색 유스케이스
│   │   │   ├── sync_usecase.py    # 동기화 유스케이스
│   │   │   ├── full_view_usecase.py # 전체 뷰 유스케이스
│   │   │   ├── synonym_generator_service.py # 동의어 생성
│   │   │   ├── auto_learn.py      # 자동 학습
│   │   │   └── evaluate.py        # 평가 도구
│   │   │
│   │   ├── domain/                 # 도메인 계층 (엔티티, 인터페이스)
│   │   │   ├── entities.py        # 도메인 엔티티
│   │   │   ├── repository.py      # 저장소 인터페이스
│   │   │   └── services.py        # 도메인 서비스 인터페이스
│   │   │
│   │   ├── infrastructure/         # 인프라 계층 (구현)
│   │   │   ├── chroma_store.py    # ChromaDB 저장소
│   │   │   ├── json_loader.py     # JSON 로더
│   │   │   ├── hybrid_search.py   # 하이브리드 검색
│   │   │   ├── query_analyzer.py  # 쿼리 분석기
│   │   │   ├── query_expander.py  # 쿼리 확장기
│   │   │   ├── reranker.py        # 재정렬기
│   │   │   ├── llm_client.py      # LLM 클라이언트
│   │   │   ├── llm_adapter.py     # LLM 어댑터
│   │   │   ├── llm_cache.py       # LLM 캐시
│   │   │   ├── self_rag.py        # Self-RAG
│   │   │   ├── hyde.py            # HyDE
│   │   │   ├── retrieval_evaluator.py # 검색 평가기
│   │   │   ├── fact_checker.py    # 사실 검증기
│   │   │   ├── feedback.py        # 피드백 수집
│   │   │   ├── keyword_extractor.py # 키워드 추출
│   │   │   ├── patterns.py        # 패턴 매칭
│   │   │   └── tool_definitions.py # 도구 정의
│   │   │
│   │   ├── config.py               # RAG 설정
│   │   ├── logging_config.py       # 로깅 설정
│   │   ├── exceptions.py           # 예외 정의
│   │   └── __init__.py
│   │
│   ├── tools/                      # 유틸리티 도구
│   │   └── merger.py               # 병합 도구
│   │
│   └── __init__.py
│
├── data/                           # 데이터 디렉터리
│   ├── input/                      # HWP 파일 입력
│   ├── output/                     # JSON 출력
│   ├── chroma_db/                  # ChromaDB 저장소
│   ├── cache/                      # 캐시 디렉터리
│   │   └── hyde/                   # HyDE 캐시
│   └── config/                     # 설정 파일
│       ├── synonyms.json           # 동의어 사전
│       └── intents.json            # 인텐트 규칙
│
├── tests/                          # 테스트
│   ├── test_converter.py
│   ├── test_formatter.py
│   ├── test_rag_*.py
│   └── fixtures/                   # 테스트 픽스처
│
├── docs/                           # 문서
├── .env.example                    # 환경 변수 예시
├── pyproject.toml                  # 프로젝트 설정
├── README.md                       # 프로젝트 README
├── QUICKSTART.md                   # 빠른 시작 가이드
├── LLM_GUIDE.md                    # LLM 설정 가이드
├── SCHEMA_REFERENCE.md             # JSON 스키마 레퍼런스
├── QUERY_PIPELINE.md               # 쿼리 파이프라인 문서
└── AGENTS.md                       # 개발자 가이드
```

## 주요 디렉터리 목적

### `src/converter.py`
HWP 파일을 Markdown/HTML로 변환하는 파이프라인입니다. hwp5html 도구를 사용하여 HWP의 텍스트, 서식, 계층 구조를 추출합니다.

### `src/formatter.py`
Markdown을 구조화된 JSON으로 변환합니다. 편/장/절/조/항/호/목 계층 구조를 파싱하고 상호 참조를 해석합니다.

### `src/enhance_for_rag.py`
JSON 데이터에 RAG 최적화 필드를 추가합니다. 임베딩용 텍스트, 키워드, 계층 경로 등 검색 품질 향상을 위한 메타데이터를 생성합니다.

### `src/parsing/`
규정 파싱을 위한 전용 모듈입니다. 복잡한 규정의 계층 구조를 분석하고 파싱합니다.

### `src/rag/interface/`
사용자 인터페이스 계층입니다. CLI, Gradio Web UI, MCP 서버 등 다양한 인터페이스를 제공합니다.

### `src/rag/application/`
비즈니스 로직을 담당하는 유스케이스 계층입니다. 검색, 동기화, 평가 등 핵심 기능을 구현합니다.

### `src/rag/domain/`
도메인 엔티티와 저장소 인터페이스를 정의합니다. 비즈니스 규칙과 데이터 구조를 포함합니다.

### `src/rag/infrastructure/`
외부 시스템과의 연동을 담당하는 인프라 계층입니다. ChromaDB, LLM, Reranker 등을 구현합니다.

### `data/`
데이터 저장소입니다. 입력 HWP 파일, 출력 JSON, ChromaDB, 캐시, 설정 파일 등을 포함합니다.

## 핵심 파일 위치

| 파일/디렉터리 | 용도 |
|--------------|------|
| `pyproject.toml` | 프로젝트 메타데이터, 의존성, 도구 설정 |
| `.env.example` | 환경 변수 템플릿 |
| `src/rag/interface/unified_cli.py` | CLI 진입점 (`regulation` 명령) |
| `src/rag/config.py` | RAG 시스템 설정 |
| `src/rag/infrastructure/chroma_store.py` | ChromaDB 저장소 구현 |
| `src/rag/infrastructure/hybrid_search.py` | 하이브리드 검색 구현 |
| `src/rag/infrastructure/llm_client.py` | LLM 클라이언트 |
| `data/config/synonyms.json` | 동의어 사전 (167개 용어) |
| `data/config/intents.json` | 인텐트 규칙 (51개 규칙) |

## 모듈 구성 (Clean Architecture 레이어별)

### Interface Layer (인터페이스 계층)
- **책임**: 사용자 인터페이스와 외부 통신
- **주요 컴포넌트**:
  - CLI: 대화형 쉘, 일회성 명령 처리
  - Web UI: Gradio 기반 채팅 인터페이스
  - MCP Server: AI 에이전트용 프로토콜 서버
- **특징**: 도메인 로직을 포함하지 않고, Application 계층에 위임

### Application Layer (애플리케이션 계층)
- **책임**: 비즈니스 유스케이스 구현
- **주요 컴포넌트**:
  - SearchUseCase: 검색 및 답변 생성
  - SyncUseCase: DB 동기화 및 관리
  - FullViewUseCase: 규정 전체 뷰 제공
  - SynonymGeneratorService: 동의어 자동 생성
- **특징**: 도메인 엔티티를 조작하여 비즈니스 목적 달성

### Domain Layer (도메인 계층)
- **책임**: 도메인 엔티티와 비즈니스 규칙 정의
- **주요 컴포넌트**:
  - Entity: Regulation, Article, Query 등 도메인 모델
  - Repository Interface: 데이터 접근 인터페이스
  - Service Interface: 도메인 서비스 인터페이스
- **특징**: 외부 의존성 없이 순수 비즈니스 로직

### Infrastructure Layer (인프라 계층)
- **책임**: 외부 시스템 연동 및 기술적 구현
- **주요 컴포넌트**:
  - ChromaStore: 벡터 DB 구현
  - HybridSearch: BM25 + Dense 검색
  - QueryAnalyzer: 쿼리 분석 및 확장
  - Reranker: BGE Reranker 래퍼
  - LLMClient: 다중 LLM 어댑터
- **특징**: Domain 계층의 인터페이스를 구현

## Clean Architecture 원칙

### 의존성 방향
```
Interface → Application → Domain ← Infrastructure
```

- 상위 계층은 하위 계층에 의존하지 않음
- 모든 의존성은 도메인 계층을 향함
- 인프라 계층은 도메인 인터페이스를 구현

### 계층별 책임
- **Interface**: 사용자 입력/출력 처리
- **Application**: 유스케이스 오케스트레이션
- **Domain**: 비즈니스 규칙과 엔티티
- **Infrastructure**: 기술적 구현과 외부 연동

### 장점
- **테스트 용이성**: 각 계층을 독립적으로 테스트 가능
- **유연성**: 구현을 쉽게 교체 가능 (예: ChromaDB → Pinecone)
- **유지보수성**: 비즈니스 로직과 기술적 구현의 분리
