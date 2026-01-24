# SPEC-TEST-001: 구현 계획

## TAG BLOCK
```yaml
spec_id: SPEC-TEST-001
related_spec: spec.md
phase: Plan
created: 2025-01-24
```

## 1. 마일스톤 (우선순위별)

### Priority 1: 핵심 자동화 인프라 (주요 목표)
**목표:** 페르소나 생성, 쿼리 생성, 테스트 실행의 기본 파이프라인 구축

**작업:**
1. AutomationDomain 엔티티 정의 (domain 계층)
   - Persona, Query, TestSession, TestCase 엔티티
   - IntentAnalysis, FactCheck, QualityScore 값 객체
2. PersonaGenerator 구현
   - 10종 페르소나 정의
   - 난이도 분포 로직 (쉬움 30%, 중간 40%, 어려움 30%)
3. QueryGenerator 구현
   - 쿼리 유형별 생성 로직
   - 의도 추론 3단계 분석 포맷
4. AutomationRepository 인터페이스 정의
   - 테스트 세션 저장/로드
   - 결과 검색 쿼리

**완료 기준:**
- 페르소나 10종이 정의되고 생성 가능
- 쿼리 생성 로직이 난이도 분포를 반영
- domain 계층이 순수 Python으로 유지됨

### Priority 2: 테스트 실행 및 평가 (주요 목표)
**목적:** 단일 쿼리 테스트 실행, 팩트체크, 품질 평가 구현

**작업:**
1. TestExecutor 구현 (application 계층)
   - SearchUseCase 연동
   - 쿼리 실행 및 결과 캡처
2. AutoFactChecker 구현
   - 핵심 주장 추출 (LLM 활용)
   - 검증 쿼리 생성
   - 팩트체크 수행 및 결과 기록
3. QualityEvaluator 구현
   - 6항목 평가 로직
   - 성공/실패 판정
4. MultiTurnSimulator 구현 (기본)
   - 후속 질문 생성 (최소 3턴)
   - 맥락 유지 검증

**완료 기준:**
- 단일 쿼리 테스트 실행이 자동화됨
- 팩트체크가 100% 수행됨
- 품질 평가 점수가 산출됨

### Priority 3: 멀티턴 대화 시뮬레이션 (주요 목표)
**목적:** 5턴 이상의 멀티턴 시나리오 자동화

**작업:**
1. MultiTurnSimulator 고도화
   - 8종 후속 질문 유형 구현
   - 맥락 윈도우 관리
   - 각 Turn별 의도 추론 검증
2. ContextTracker 구현
   - 대화 히스토리 관리
   - 암묵적 정보 추출
   - 의도 진화 추적
3. 맥락 유지 평가 로직
   - 맥락 연결 검증
   - 정보 일관성 확인

**완료 기준:**
- 5턴 대화 시나리오가 자동 생성됨
- 각 Turn별 의도 추론이 검증됨
- 맥락 유지율이 측정됨

### Priority 4: RAG 컴포넌트 분석 (주요 목표)
**목적:** 각 RAG 컴포넌트의 기여도를 자동으로 분석

**작업:**
1. ComponentAnalyzer 구현
   - RAG 파이프라인 로그 파싱
   - 컴포넌트별 동작 감지
   - 기여도 점수 산출
2. ComponentContributionMapper 구현
   - 실패 유형별 컴포넌트 원인 매핑
   - 개선 방향 도출
3. 컴포넌트별 기여도 리포트 생성

**완료 기준:**
- 8개 RAG 컴포넌트 동작이 추적됨
- 기여도 점수가 산출됨
- 실패 원인과 컴포넌트가 연결됨

### Priority 5: 개선 적용 자동화 (보조 목표)
**목적:** 실패 분석 결과를 바탕으로 자동 개선

**작업:**
1. FailureAnalyzer 구현
   - 5-Why 분석 자동화
   - RAG 컴포넌트별 원인 분석
2. ImprovementApplier 구현
   - intents.json 자동 패치
   - synonyms.json 자동 패치
   - 코드 수정 제안 생성
3. RegressionTester 구현
   - 개선 후 회귀 검증

**완료 기준:**
- 실패 케이스가 5-Why 분석됨
- intent/synonym 패치가 자동화됨
- 코드 수정 제안이 생성됨

### Priority 6: 리포트 생성 (보조 목표)
**목적:** 테스트 결과를 종합 리포트로 생성

**작업:**
1. TestReportGenerator 구현
   - 마크다운 리포트 생성
   - 9개 항목 종합 보고
2. 결과 시각화 (선택사항)
   - 차트/그래프 생성
   - HTML 리포트 (선택)

**완료 기준:**
- 마크다운 리포트가 생성됨
- 모든 항목이 포함됨

## 2. 기술 접근 방식

### Clean Architecture 준수

**Domain 계층 (src/rag/automation/domain/):**
```python
# entities.py
@dataclass
class Persona:
    id: str
    name: str
    characteristics: List[str]
    interests: List[str]
    query_style: str  # formal, informal, mixed

@dataclass
class TestCase:
    persona_id: str
    query: str
    difficulty: DifficultyLevel  # EASY, MEDIUM, HARD
    query_type: QueryType
    intent_analysis: IntentAnalysis

@dataclass
class TestSession:
    id: str
    timestamp: datetime
    test_cases: List[TestCase]
    results: List[TestResult]
```

**Application 계층 (src/rag/automation/application/):**
```python
# generate_test_usecase.py
class GenerateTestUseCase:
    def execute(self, config: TestConfig) -> TestSession:
        # PersonaGenerator로 페르소나 생성
        # QueryGenerator로 쿼리 생성
        # TestSession 조립

# execute_test_usecase.py
class ExecuteTestUseCase:
    def execute(self, session: TestSession) -> TestResult:
        # SearchUseCase로 쿼리 실행
        # AutoFactChecker로 팩트체크
        # QualityEvaluator로 품질 평가
```

**Infrastructure 계층 (src/rag/automation/infrastructure/):**
```python
# llm_persona_generator.py
class LLMPersonaGenerator(PersonaGenerator):
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate(self, persona_type: str) -> Persona:
        # Claude API로 페르소나 생성

# json_session_repository.py
class JSONSessionRepository(SessionRepository):
    def save(self, session: TestSession):
        # data/output/test_sessions/에 JSON 저장
```

**Interface 계층 (src/rag/automation/interface/):**
```python
# automation_cli.py
@click.group()
def automation():
    """RAG 테스트 자동화 CLI"""
    pass

@automation.command()
@click.option("--personas", default=3)
@click.option("--queries", default=9)
def run(personas, queries):
    """테스트 세션 실행"""
```

### 기존 RAG 시스템과의 통합

**SearchUseCase 연동:**
```python
# application/execute_test_usecase.py
from src.rag.application.search_usecase import SearchUseCase

class ExecuteTestUseCase:
    def __init__(self, search_usecase: SearchUseCase, ...):
        self.search = search_usecase

    def execute_query(self, query: str) -> QueryResult:
        result = self.search.execute(
            query=query,
            top_k=5,
            enable_answer=True
        )
        return result
```

**기존 LLMClient 활용:**
```python
# infrastructure/llm_persona_generator.py
from src.rag.infrastructure.llm_client import LLMClient

class LLMPersonaGenerator:
    def generate_queries(self, persona: Persona) -> List[str]:
        prompt = self._build_prompt(persona)
        response = self.llm_client.generate(prompt)
        return self._parse_queries(response)
```

### LLM API 통합 (Claude AI)

**의존성 추가:**
```bash
uv add anthropic
```

**환경 변수:**
```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

**LLM 클라이언트 래퍼:**
```python
# infrastructure/claude_client.py
import anthropic

class ClaudeLLMClient:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
```

## 3. 아키텍처 설계 방향

### 계층별 책임

**Domain 계층:**
- 순수 Python만 유지 (외부 라이브러리 금지)
- 엔티티와 값 객체 정의
- 비즈니스 규칙 인터페이스

**Application 계층:**
- 유스케이스 구현
- 기존 RAG UseCase와 연동
- 워크플로우 오케스트레이션

**Infrastructure 계층:**
- LLM API 연동
- 파일 I/O
- 기존 RAG 컴포넌트와의 통합

**Interface 계층:**
- CLI 명령어 제공
- 테스트 실행 인터페이스

### 파일 구조

```
src/rag/automation/
├── domain/
│   ├── entities.py           # Persona, TestCase, TestSession
│   ├── value_objects.py      # IntentAnalysis, FactCheck, QualityScore
│   └── repository.py         # Repository 인터페이스
│
├── application/
│   ├── generate_test_usecase.py
│   ├── execute_test_usecase.py
│   ├── analyze_result_usecase.py
│   └── apply_improvement_usecase.py
│
├── infrastructure/
│   ├── llm_persona_generator.py
│   ├── llm_query_generator.py
│   ├── json_session_repository.py
│   ├── fact_checker.py
│   └── claude_client.py
│
└── interface/
    └── automation_cli.py
```

## 4. 위험 완화 계획

### 기술적 위험

**위험 1: LLM API 비용 과다**
- 완화 조치: 캐싱 전략 도입, 배치 처리, 최소화된 프롬프트
- 영향도: Medium

**위험 2: domain 계층 오염**
- 완화 조치: 엄격한 코드 리뷰, 의존성 방향 확인
- 영향도: High

**위험 3: 기존 RAG 시스템과의 호환성**
- 완화 조치: 통합 테스트, 점진적 롤아웃
- 영향도: High

**위험 4: 팩트체크 자동화의 정확성**
- 완화 조치: 사람 검증 단계 유지, 신뢰도 점수 도입
- 영향도: Medium

### 운영적 위험

**위험 1: 테스트 실행 시간**
- 완화 조치: 병렬 실행, 선택적 테스트
- 영향도: Low

**위험 2: 결과 저장소 크기**
- 완화 조치: 압축, 오래된 결과 아카이빙
- 영향도: Low

## 5. 다음 단계

**SPEC-TEST-001 승인 후:**
1. /moai:2-run SPEC-TEST-001 실행
2. Priority 1 구현 시작
3. 각 마일스톤 완료 후 검증
