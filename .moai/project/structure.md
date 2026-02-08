# 프로젝트 구조 (Project Structure)

## 디렉터리 구조

```
regulation_manager/
├── .moai/                          # MoAI-ADK 설정
│   ├── project/                    # 프로젝트 문서
│   │   ├── product.md              # 프로젝트 개요
│   │   ├── structure.md            # 디렉터리 구조
│   │   └── tech.md                 # 기술 스택
│   ├── specs/                      # SPEC 문서
│   │   ├── SPEC-RAG-001/           # RAG 시스템 종합 개선
│   │   └── SPEC-RAG-002/           # 품질 및 유지보수성 개선
│   └── config/                     # MoAI 설정
│
├── src/                            # 소스 코드 (Clean Architecture)
│   ├── rag/                        # RAG 시스템 (Clean Architecture)
│   │   ├── interface/              # 인터페이스 계층
│   │   │   ├── unified_cli.py      # 통합 CLI 진입점
│   │   │   ├── gradio_app.py       # Gradio Web UI
│   │   │   ├── mcp_server.py       # MCP 서버
│   │   │   ├── query_handler.py    # 쿼리 처리기
│   │   │   ├── chat_logic.py       # 채팅 로직
│   │   │   ├── query_suggestions.py # 쿼리 제안
│   │   │   └── common.py           # 공유 타입/상수
│   │   │
│   │   ├── application/            # 애플리케이션 계층 (Use Cases)
│   │   │   ├── search_usecase.py   # 검색 유스케이스
│   │   │   ├── sync_usecase.py     # 동기화 유스케이스
│   │   │   ├── full_view_usecase.py # 전체 뷰 유스케이스
│   │   │   ├── conversation_service.py # 멀티턴 대화 서비스
│   │   │   ├── synonym_generator_service.py # 동의어 생성
│   │   │   ├── auto_learn.py       # 자동 학습
│   │   │   └── evaluation/         # RAG 품질 평가 (v2.2.0+)
│   │   │       ├── evaluation_service.py # 평가 서비스
│   │   │       └── __init__.py
│   │   │
│   │   ├── domain/                 # 도메인 계층 (엔티티, 인터페이스)
│   │   │   ├── entities.py         # 도메인 엔티티
│   │   │   ├── value_objects.py    # 값 객체
│   │   │   ├── repositories.py     # 저장소 인터페이스
│   │   │   ├── citation/           # 인용 강화 (v2.1.0+)
│   │   │   │   ├── citation_enhancer.py
│   │   │   │   ├── article_number_extractor.py
│   │   │   │   └── citation_validator.py
│   │   │   ├── llm/                # LLM 도메인 서비스 (v2.1.0+)
│   │   │   │   ├── circuit_breaker.py    # 서킷 브레이커
│   │   │   │   ├── ambiguity_classifier.py # 모호성 분류기
│   │   │   │   └── emotional_classifier.py # 감성 분류기
│   │   │   ├── conversation/       # 멀티턴 대화 (v2.1.0+)
│   │   │   │   ├── session.py      # 세션 관리
│   │   │   │   └── dialog.py       # 대화 관리
│   │   │   ├── evaluation/         # 품질 평가 (v2.2.0+)
│   │   │   │   ├── personas.py     # 6가지 사용자 페르소나
│   │   │   │   ├── models.py       # 평가 모델
│   │   │   │   ├── quality_analyzer.py # 품질 분석기
│   │   │   │   └── parallel_evaluator.py # 병렬 평가기
│   │   │   └── experiment/         # A/B 테스트 (v2.1.0+)
│   │   │       └── ab_test.py      # A/B 테스트 프레임워크
│   │   │
│   │   ├── infrastructure/         # 인프라 계층 (구현)
│   │   │   ├── chroma_store.py     # ChromaDB 저장소
│   │   │   ├── json_loader.py      # JSON 로더
│   │   │   ├── hybrid_search_integration.py # 하이브리드 검색
│   │   │   ├── hybrid_search.py    # 하이브리드 검색 (레거시)
│   │   │   ├── query_analyzer.py   # 쿼리 분석기
│   │   │   ├── query_expander.py   # 쿼리 확장기
│   │   │   ├── reranker.py         # 재정렬기
│   │   │   ├── llm_client.py       # LLM 클라이언트
│   │   │   ├── llm_cache.py        # LLM 캐시
│   │   │   ├── self_rag.py         # Self-RAG
│   │   │   ├── hyde.py             # HyDE
│   │   │   ├── retrieval_evaluator.py # 검색 평가기
│   │   │   ├── fact_checker.py     # 사실 검증기
│   │   │   ├── feedback.py         # 피드백 수집
│   │   │   ├── keyword_extractor.py # 키워드 추출
│   │   │   ├── patterns.py         # 패턴 매칭
│   │   │   ├── cache_warming.py    # 캐시 워밍 (v2.1.0+)
│   │   │   ├── llm/                # LLM 어댑터
│   │   │   │   └── judge/           # LLM-as-Judge (v2.2.0+)
│   │   │   └── storage/            # 저장소 (v2.2.0+)
│   │   │       └── evaluation_store.py # 평가 결과 저장
│   │   │
│   │   ├── automation/             # RAG 테스팅 자동화 (v2.0+)
│   │   │   ├── domain/             # 테스트 도메인
│   │   │   │   ├── entities.py     # 테스트 엔티티
│   │   │   │   ├── extended_entities.py # 확장 엔티티
│   │   │   │   ├── repository.py   # 테스트 저장소 인터페이스
│   │   │   │   ├── value_objects.py # 테스트 값 객체
│   │   │   │   └── context_tracker.py # 문맥 추적
│   │   │   ├── application/        # 테스트 유스케이스
│   │   │   │   ├── generate_test_usecase.py # 테스트 생성
│   │   │   │   ├── execute_test_usecase.py # 테스트 실행
│   │   │   │   └── comprehensive_test_usecase.py # 종합 테스트
│   │   │   ├── infrastructure/     # 테스트 인프라
│   │   │   │   ├── llm_query_generator.py # LLM 쿼리 생성기
│   │   │   │   ├── llm_persona_generator.py # LLM 페르소나 생성기
│   │   │   │   ├── mock_llm_client.py # 모의 LLM 클라이언트
│   │   │   │   ├── multi_turn_simulator.py # 멀티턴 시뮬레이터
│   │   │   │   ├── query_cache.py   # 쿼리 캐시
│   │   │   │   ├── json_session_repository.py # JSON 세션 저장소
│   │   │   │   ├── test_report_generator.py # 테스트 리포트 생성기
│   │   │   │   ├── test_scenario_templates.py # 테스트 시나리오 템플릿
│   │   │   │   ├── evaluation_constants.py # 평가 상수
│   │   │   │   ├── evaluation_helpers.py # 평가 헬퍼
│   │   │   │   └── auto_fact_checker.py # 자동 사실 검증기
│   │   │   └── interface/          # 테스트 인터페이스
│   │   │       └── automation_cli.py # 자동화 CLI
│   │   │
│   │   ├── data_generation/        # 데이터 생성 (v2.2.0+)
│   │   ├── config.py               # RAG 설정
│   │   ├── logging_config.py       # 로깅 설정
│   │   ├── exceptions.py           # 예외 정의
│   │   └── __init__.py
│   │
│   ├── parsing/                    # 파싱 모듈
│   │   ├── regulation_parser.py    # 규정 파서
│   │   ├── reference_resolver.py   # 상호 참조 해석
│   │   ├── table_extractor.py      # 표 추출
│   │   └── html_table_converter.py # HTML 표 변환
│   │
│   ├── tools/                      # 유틸리티 도구
│   │   └── merger.py               # 병합 도구
│   │
│   ├── converter.py                # HWP → Markdown/HTML 변환
│   ├── formatter.py                # Markdown → JSON 변환
│   ├── enhance_for_rag.py          # RAG 최적화 필드 추가
│   ├── preprocessor.py             # HWP 전처리
│   ├── repair.py                   # JSON 복구 도구
│   ├── refine_json.py              # JSON 정제 도구
│   ├── verify_json.py              # JSON 검증 도구
│   ├── cache_manager.py            # 캐시 관리
│   └── __init__.py
│
├── data/                           # 데이터 디렉터리
│   ├── input/                      # HWP 파일 입력
│   ├── output/                     # JSON 출력
│   ├── chroma_db/                  # ChromaDB 저장소
│   ├── cache/                      # 캐시 디렉터리
│   │   ├── hyde/                   # HyDE 캐시
│   │   ├── llm/                    # LLM 캐시
│   │   └── bm25/                   # BM25 캐시 (msgpack)
│   ├── test_sessions/              # 테스트 세션 저장소
│   ├── test_reports/               # 테스트 리포트
│   └── config/                     # 설정 파일
│       ├── synonyms.json           # 동의어 사전 (167개 용어)
│       └── intents.json            # 인텐트 규칙 (51개 규칙)
│
├── tests/                          # 테스트
│   ├── rag/                        # RAG 테스트
│   │   ├── unit/                   # 단위 테스트
│   │   ├── integration/            # 통합 테스트
│   │   └── automation/             # 자동화 테스트
│   ├── test_converter.py
│   ├── test_formatter.py
│   └── fixtures/                   # 테스트 픽스처
│
├── docs/                           # 문서
├── scripts/                        # 스크립트
├── .env.example                    # 환경 변수 예시
├── pyproject.toml                  # 프로젝트 설정
├── pytest.ini                      # pytest 설정
├── README.md                       # 프로젝트 README
├── QUICKSTART.md                   # 빠른 시작 가이드
├── LLM_GUIDE.md                    # LLM 설정 가이드
├── SCHEMA_REFERENCE.md             # JSON 스키마 레퍼런스
├── QUERY_PIPELINE.md               # 쿼리 파이프라인 문서
└── AGENTS.md                       # 개발자 가이드
```

## 주요 디렉터리 목적

### `src/rag/interface/`
사용자 인터페이스 계층입니다. CLI, Gradio Web UI, MCP 서버 등 다양한 인터페이스를 제공합니다.

**주요 파일:**
- `unified_cli.py`: 통합 CLI 진입점 (`regulation` 명령)
- `gradio_app.py`: Gradio 기반 웹 UI
- `mcp_server.py`: MCP 서버 (AI 에이전트 연동)

### `src/rag/application/`
비즈니스 로직을 담당하는 유스케이스 계층입니다. 검색, 동기화, 평가 등 핵심 기능을 구현합니다.

**주요 파일:**
- `search_usecase.py`: 검색 유스케이스
- `sync_usecase.py`: DB 동기화 유스케이스
- `conversation_service.py`: 멀티턴 대화 서비스
- `evaluation/evaluation_service.py`: RAG 품질 평가 서비스 (v2.2.0+)

### `src/rag/domain/`
도메인 엔티티와 저장소 인터페이스를 정의합니다. 비즈니스 규칙과 데이터 구조를 포함합니다.

**주요 하위 모듈:**
- `entities.py`: 도메인 엔티티
- `citation/`: 인용 강화 (v2.1.0+)
- `llm/`: LLM 도메인 서비스 (v2.1.0+)
- `conversation/`: 멀티턴 대화 (v2.1.0+)
- `evaluation/`: 품질 평가 (v2.2.0+)
- `experiment/`: A/B 테스트 (v2.1.0+)

### `src/rag/infrastructure/`
외부 시스템과의 연동을 담당하는 인프라 계층입니다. ChromaDB, LLM, Reranker 등을 구현합니다.

**주요 파일:**
- `chroma_store.py`: ChromaDB 저장소 구현
- `hybrid_search_integration.py`: 하이브리드 검색 구현
- `llm_client.py`: LLM 클라이언트
- `reranker.py`: BGE Reranker 래퍼

### `src/rag/automation/` (v2.0+)
RAG 테스팅 자동화 시스템입니다. Clean Architecture로 구현되었습니다.

**구조:**
- `domain/`: 테스트 도메인 엔티티 및 인터페이스
- `application/`: 테스트 생성, 실행, 평가 유스케이스
- `infrastructure/`: LLM, 저장소, 시뮬레이터 구현
- `interface/`: 자동화 CLI

### `data/`
데이터 저장소입니다. 입력 HWP 파일, 출력 JSON, ChromaDB, 캐시, 설정 파일 등을 포함합니다.

**하위 디렉터리:**
- `cache/hyde/`: HyDE 가상 문서 캐시
- `cache/llm/`: LLM 응답 캐시
- `cache/bm25/`: BM25 인덱스 캐시 (msgpack 포맷)
- `test_sessions/`: 테스트 세션 저장소
- `test_reports/`: 테스트 리포트

## 핵심 파일 위치

| 파일/디렉터리 | 용도 |
|--------------|------|
| `pyproject.toml` | 프로젝트 메타데이터, 의존성, 도구 설정 |
| `pytest.ini` | pytest 설정 (asyncio_mode=auto) |
| `.env.example` | 환경 변수 템플릿 |
| `src/rag/interface/unified_cli.py` | CLI 진입점 (`regulation` 명령) |
| `src/rag/config.py` | RAG 시스템 설정 (매직 넘버 상수화) |
| `src/rag/infrastructure/chroma_store.py` | ChromaDB 저장소 구현 |
| `src/rag/infrastructure/hybrid_search_integration.py` | 하이브리드 검색 구현 |
| `src/rag/infrastructure/llm_client.py` | LLM 클라이언트 |
| `src/rag/domain/evaluation/personas.py` | 6가지 사용자 페르소나 정의 |
| `src/rag/application/evaluation/evaluation_service.py` | RAG 품질 평가 서비스 |
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
  - ConversationService: 멀티턴 대화 관리
  - EvaluationService: RAG 품질 평가 (v2.2.0+)
- **특징**: 도메인 엔티티를 조작하여 비즈니스 목적 달성

### Domain Layer (도메인 계층)
- **책임**: 도메인 엔티티와 비즈니스 규칙 정의
- **주요 컴포넌트**:
  - Entity: Regulation, Article, Query 등 도메인 모델
  - Citation: 인용 강화 (v2.1.0+)
  - LLM Domain Services: 서킷 브레이커, 모호성 분류기, 감성 분류기 (v2.1.0+)
  - Conversation: 세션 및 대화 관리 (v2.1.0+)
  - Evaluation: 6가지 페르소나 및 품질 평가 (v2.2.0+)
  - Experiment: A/B 테스트 프레임워크 (v2.1.0+)
- **특징**: 외부 의존성 없이 순수 비즈니스 로직

### Infrastructure Layer (인프라 계층)
- **책임**: 외부 시스템 연동 및 기술적 구현
- **주요 컴포넌트**:
  - ChromaStore: 벡터 DB 구현
  - HybridSearch: BM25 + Dense 검색
  - QueryAnalyzer: 쿼리 분석 및 확장
  - Reranker: BGE Reranker 래퍼
  - LLMClient: 다중 LLM 어댑터
  - Storage: 평가 결과 저장소 (v2.2.0+)
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

## v2.2.0 주요 변경사항

### 새로운 모듈
1. **src/rag/domain/evaluation/** (RAG 품질 평가)
   - `personas.py`: 6가지 사용자 페르소나 정의
   - `models.py`: 평가 모델
   - `quality_analyzer.py`: 품질 분석기
   - `parallel_evaluator.py`: 병렬 평가기

2. **src/rag/application/evaluation/**
   - `evaluation_service.py`: 평가 서비스

3. **src/rag/infrastructure/storage/**
   - `evaluation_store.py`: 평가 결과 저장소

### 코드 품질 개선
- 중복 코드 제거 (self_rag.py, query_analyzer.py, tool_executor.py)
- 매직 넘버 상수화 (config.py)
- 타입 힌트 개선
- 에러 메시지 표준화 (한국어)

### 성능 최적화
- Kiwi 토크나이저 지연 로딩 (싱글톤 패턴)
- BM25 캐싱 msgpack 전환
- 연결 풀 모니터링
- HyDE LRU 캐싱

## 관련 문서

- [프로젝트 개요](product.md) - 프로젝트 개요 및 기능
- [기술 스택](tech.md) - 기술 스택 상세
- [SPEC-RAG-001](.moai/specs/SPEC-RAG-001/spec.md) - RAG 시스템 종합 개선
- [SPEC-RAG-002](.moai/specs/SPEC-RAG-002/spec.md) - 품질 및 유지보수성 개선
