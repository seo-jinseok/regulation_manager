# API Documentation v2.1.0

RAG 시스템의 새로운 컴포넌트 API 레퍼런스입니다.

## Table of Contents

- [Circuit Breaker](#circuit-breaker)
- [Ambiguity Classifier](#ambiguity-classifier)
- [Citation Enhancement](#citation-enhancement)
- [Emotional Classifier](#emotional-classifier)
- [Multi-turn Conversation](#multi-turn-conversation)
- [Performance Optimization](#performance-optimization)
- [A/B Testing Framework](#ab-testing-framework)

---

## Circuit Breaker

LLM 프로바이더의 장애를 자동으로 감지하고 복구하는 서킷 브레이커 패턴입니다.

### CircuitBreaker

`src/rag/domain/llm/circuit_breaker.py`

```python
class CircuitBreaker:
    """서킷 브레이커: LLM 프로바이더 장애 자동 감지 및 복구"""

    def __init__(self, provider_name: str, config: CircuitBreakerConfig):
        """
        Args:
            provider_name: LLM 프로바이더 이름 (예: "openai", "ollama")
            config: 서킷 브레이커 설정
        """
```

#### Methods

##### `call(func, *args, **kwargs)`

서킷 브레이커를 통해 함수를 실행합니다.

```python
def call(self, func: Callable, *args, **kwargs) -> Any:
    """
    Args:
        func: 실행할 함수 (LLM 생성 함수 등)
        *args: 함수 인자
        **kwargs: 함수 키워드 인자

    Returns:
        함수 실행 결과

    Raises:
        CircuitBreakerOpenError: 서킷이 열린 상태일 때
    """
```

**사용 예시**:
```python
breaker = CircuitBreaker("openai", config)
try:
    result = breaker.call(llm_client.generate, prompt, system_msg)
except CircuitBreakerOpenError:
    result = fallback_client.generate(prompt, system_msg)
```

##### `get_state() -> CircuitState`

현재 서킷 상태를 반환합니다.

```python
def get_state(self) -> CircuitState:
    """
    Returns:
        CircuitState: CLOSED, OPEN, 또는 HALF_OPEN
    """
```

##### `get_metrics() -> CircuitBreakerMetrics`

서킷 브레이커 메트릭을 반환합니다.

```python
def get_metrics(self) -> CircuitBreakerMetrics:
    """
    Returns:
        CircuitBreakerMetrics:
            - request_count: 총 요청 수
            - failure_count: 총 실패 수
            - success_count: 총 성공 수
            - failure_rate: 실패율 (0.0-1.0)
            - last_failure_time: 마지막 실패 시간
            - last_state_change: 마지막 상태 변경 시간
    """
```

#### Configuration

```python
@dataclass
class CircuitBreakerConfig:
    """서킷 브레이커 설정"""
    failure_threshold: int = 3      # 연속 실패 임계값
    recovery_timeout: float = 60.0  # 복구 타임아웃 (초)
    success_threshold: int = 2      # 성공 임계값 (HALF_OPEN → CLOSED)
    timeout: float = 30.0           # 요청 타임아웃 (초)
```

#### States

```python
class CircuitState(Enum):
    """서킷 브레이커 상태"""
    CLOSED = "closed"           # 정상: 요청 통과
    OPEN = "open"               # 장애: 요청 거부
    HALF_OPEN = "half_open"     # 복구 테스트: 단일 요청 허용
```

#### Exceptions

```python
class CircuitBreakerOpenError(Exception):
    """서킷이 열린 상태일 때 발생"""
    def __init__(self, provider_name: str, recovery_time: float):
        self.provider_name = provider_name
        self.recovery_time = recovery_time
```

---

## Ambiguity Classifier

사용자 쿼리의 모호성을 분류하고 명확화 대화를 생성합니다.

### AmbiguityClassifier

`src/rag/domain/llm/ambiguity_classifier.py`

```python
class AmbiguityClassifier:
    """쿼리 모호성 분류 및 명확화 대화 생성"""

    def __init__(self, config: Optional[AmbiguityClassifierConfig] = None):
        """
        Args:
            config: 분류기 설정 (선택 사항)
        """
```

#### Methods

##### `classify(query: str) -> AmbiguityClassificationResult`

쿼리 모호성을 분류합니다.

```python
def classify(self, query: str) -> AmbiguityClassificationResult:
    """
    Args:
        query: 사용자 쿼리

    Returns:
        AmbiguityClassificationResult:
            - level: AmbiguityLevel (CLEAR, AMBIGUOUS, HIGHLY_AMBIGUOUS)
            - score: 모호성 점수 (0.0-1.0)
            - factors: 분석 요소
                * audience: 감지된 대상 (학생/교수/직원)
                * regulation_type: 규정 유형
                * article_reference: 조문 참조 여부
            - suggested_audiences: 제안된 대상 목록
            - suggested_regulations: 제안된 규정 목록
    """
```

**사용 예시**:
```python
classifier = AmbiguityClassifier()
result = classifier.classify("휴학 규정")

if result.level == AmbiguityLevel.AMBIGUOUS:
    print(f"모호성 점수: {result.score:.2f}")
    print(f"제안된 대상: {result.suggested_audiences}")
```

##### `generate_disambiguation_dialog(result: AmbiguityClassificationResult) -> DisambiguationDialog`

명확화 대화를 생성합니다.

```python
def generate_disambiguation_dialog(
    self,
    result: AmbiguityClassificationResult
) -> DisambiguationDialog:
    """
    Args:
        result: 분류 결과

    Returns:
        DisambiguationDialog:
            - message: 사용자 메시지
            - options: 옵션 목록 (최대 5개)
                * text: 옵션 텍스트
                * clarified_query: 명확화된 쿼리
                * score: 관련도 점수
    """
```

**사용 예시**:
```python
dialog = classifier.generate_disambiguation_dialog(result)
print(dialog.message)
for i, option in enumerate(dialog.options, 1):
    print(f"[{i}] {option.text}")
```

##### `apply_user_selection(original_query: str, selected_option: DisambiguationOption) -> str`

사용자 선택을 적용하여 쿼리를 명확화합니다.

```python
def apply_user_selection(
    self,
    original_query: str,
    selected_option: DisambiguationOption
) -> str:
    """
    Args:
        original_query: 원본 쿼리
        selected_option: 사용자가 선택한 옵션

    Returns:
        명확화된 쿼리
    """
```

##### `learn_from_selection(original_query: str, audience: str)`

사용자 선택에서 학습합니다.

```python
def learn_from_selection(self, original_query: str, audience: str) -> None:
    """
    Args:
        original_query: 원본 쿼리
        audience: 사용자가 선택한 대상
    """
```

#### Configuration

```python
@dataclass
class AmbiguityClassifierConfig:
    """모호성 분류기 설정"""
    clear_threshold: float = 0.3        # CLEAR 분류 임계값
    ambiguous_threshold: float = 0.7    # HIGHLY_AMBIGUOUS 임계값
    max_options: int = 5                # 최대 옵션 수
    audience_keywords: Dict[str, List[str]] = field(default_factory=dict)
    regulation_keywords: Dict[str, List[str]] = field(default_factory=dict)
```

#### Data Classes

```python
@dataclass
class AmbiguityClassificationResult:
    level: AmbiguityLevel
    score: float
    factors: ClassificationFactors
    suggested_audiences: List[str]
    suggested_regulations: List[str]

@dataclass
class ClassificationFactors:
    audience: Optional[str]
    regulation_type: Optional[str]
    has_article_reference: bool
    missing_audience_penalty: float = 0.0
    missing_regulation_penalty: float = 0.0

@dataclass
class DisambiguationDialog:
    message: str
    options: List[DisambiguationOption]

@dataclass
class DisambiguationOption:
    text: str
    clarified_query: str
    score: float
```

---

## Citation Enhancement

정확한 조항 번호 추출 및 검증으로 답변 신뢰도를 높입니다.

### CitationEnhancer

`src/rag/domain/citation/citation_enhancer.py`

```python
class CitationEnhancer:
    """인용 강화: 조항 번호 추출, 검증, 포맷팅"""

    def __init__(self, validator: Optional[CitationValidator] = None):
        """
        Args:
            validator: 인용 검증기 (선택 사항)
        """
```

#### Methods

##### `enhance_citations(chunks: List[Chunk], confidences: List[float]) -> List[EnhancedCitation]`

청크에서 인용을 강화합니다.

```python
def enhance_citations(
    self,
    chunks: List[Chunk],
    confidences: List[float]
) -> List[EnhancedCitation]:
    """
    Args:
        chunks: 검색된 청크 목록
        confidences: 각 청크의 관련도 점수

    Returns:
        EnhancedCitation 목록:
            - regulation_name: 규정 이름
            - article_number: 조항 번호
            - display_no: 표시 번호 (예: "제15조")
            - path: 계층 경로
            - confidence: 관련도 점수
            - is_valid: 검증 여부
            - is_special: 별표/서식 여부
    """
```

**사용 예시**:
```python
enhancer = CitationEnhancer()
citations = enhancer.enhance_citations(chunks, confidences)
for citation in citations:
    if citation.is_valid:
        print(f"「{citation.regulation_name}」 {citation.display_no}")
```

##### `format_citations(citations: List[EnhancedCitation]) -> str`

인용을 포맷팅합니다.

```python
def format_citations(self, citations: List[EnhancedCitation]) -> str:
    """
    Args:
        citations: 강화된 인용 목록

    Returns:
        포맷팅된 인용 문자열
        예: "「직원복무규정」 제26조, 「학칙」 제15조"
    """
```

##### `consolidate_citations(citations: List[EnhancedCitation]) -> List[ConsolidatedCitation]`

동일 규정의 인용을 통합합니다.

```python
def consolidate_citations(
    self,
    citations: List[EnhancedCitation]
) -> List[ConsolidatedCitation]:
    """
    Args:
        citations: 강화된 인용 목록

    Returns:
        ConsolidatedCitation 목록:
            - regulation_name: 규정 이름
            - articles: 통합된 조항 목록
            - range_str: 범위 문자열 (예: "제15조-제20조")
    """
```

#### Supporting Classes

##### ArticleNumberExtractor

`src/rag/domain/citation/article_number_extractor.py`

```python
class ArticleNumberExtractor:
    """청크 메타데이터에서 조항 번호 추출"""

    def extract(self, chunk: Chunk) -> Optional[str]:
        """
        Args:
            chunk: 검색 청크

        Returns:
            조항 번호 (예: "제15조", "제26조제3항")
        """
```

##### CitationValidator

`src/rag/domain/citation/citation_validator.py`

```python
class CitationValidator:
    """인용 검증: 규정 구조 기반 검증"""

    def validate(self, citation: str, regulation: Regulation) -> bool:
        """
        Args:
            citation: 인용 문자열
            regulation: 규정 객체

        Returns:
            검증 결과 (True/False)
        """
```

---

## Emotional Classifier

사용자의 정서적 상태를 감지하고 공감 어조로 응답합니다.

### EmotionalClassifier

`src/rag/domain/llm/emotional_classifier.py`

```python
class EmotionalClassifier:
    """감정 상태 분류 및 프롬프트 조정"""

    def __init__(self, config: Optional[EmotionalClassifierConfig] = None):
        """
        Args:
            config: 분류기 설정 (선택 사항)
        """
```

#### Methods

##### `classify(query: str) -> EmotionalClassificationResult`

쿼리의 감정 상태를 분류합니다.

```python
def classify(self, query: str) -> EmotionalClassificationResult:
    """
    Args:
        query: 사용자 쿼리

    Returns:
        EmotionalClassificationResult:
            - state: EmotionalState (NEUTRAL, SEEKING_HELP, DISTRESSED, FRUSTRATED)
            - confidence: 신뢰도 점수 (0.0-1.0)
            - detected_keywords: 감지된 키워드
            - has_urgency: 긴급 여부
    """
```

**사용 예시**:
```python
classifier = EmotionalClassifier()
result = classifier.classify("학교에 가기 너무 힘들어요 어떡해요")

if result.state == EmotionalState.DISTRESSED:
    print(f"감지된 감정: {result.state.value}")
    print(f"신뢰도: {result.confidence:.2f}")
    print(f"키워드: {result.detected_keywords}")
```

##### `generate_empathy_prompt(result: EmotionalClassificationResult, base_prompt: str) -> str`

감정 상태에 맞는 프롬프트를 생성합니다.

```python
def generate_empathy_prompt(
    self,
    result: EmotionalClassificationResult,
    base_prompt: str
) -> str:
    """
    Args:
        result: 감정 분류 결과
        base_prompt: 기본 프롬프트

    Returns:
        감정 상태에 맞게 조정된 프롬프트
    """
```

**사용 예시**:
```python
base_prompt = "규정에 기반하여 답변하세요"
adapted_prompt = classifier.generate_empathy_prompt(result, base_prompt)
# 결과: "사용자의 어려운 상황을 공감하며 따뜻한 어조로, 규정에 기반하여 답변하세요"
```

#### Configuration

```python
@dataclass
class EmotionalClassifierConfig:
    """감정 분류기 설정"""
    neutral_threshold: float = 0.3          # NEUTRAL 분류 임계값
    seeking_help_threshold: float = 0.5     # SEEKING_HELP 임계값
    frustrated_threshold: float = 0.6       # FRUSTRATED 임계값
    distressed_threshold: float = 0.7       # DISTRESSED 임계값
    keywords: Dict[str, List[str]] = field(default_factory=dict)
```

#### Emotional States

```python
class EmotionalState(Enum):
    """감정 상태"""
    NEUTRAL = "neutral"             # 중립: 일반적인 질문
    SEEKING_HELP = "seeking_help"   # 도움 요청: 방법, 절차 문의
    DISTRESSED = "distressed"       # 곤란: 힘듦, 포기 표현
    FRUSTRATED = "frustrated"       # 좌절: 안됨, 이해 안됨 표현
```

---

## Multi-turn Conversation

대화 맥락을 유지하여 연속 질문의 정확도를 높입니다.

### ConversationSession

`src/rag/domain/conversation/session.py`

```python
@dataclass
class ConversationSession:
    """대화 세션: 문맥 추적 및 관리"""
    session_id: str
    user_id: Optional[str]
    turns: List[ConversationTurn]
    context_summary: str
    created_at: datetime
    last_activity: datetime
    status: SessionStatus
```

#### Methods

##### `create(user_id: Optional[str], timeout_minutes: int = 30) -> ConversationSession`

새 세션을 생성합니다.

```python
@staticmethod
def create(user_id: Optional[str], timeout_minutes: int = 30) -> ConversationSession:
    """
    Args:
        user_id: 사용자 ID (선택 사항)
        timeout_minutes: 세션 타임아웃 (분)

    Returns:
        새 ConversationSession 객체
    """
```

**사용 예시**:
```python
session = ConversationSession.create(user_id="user123", timeout_minutes=30)
```

##### `add_turn(query: str, response: str, metadata: Optional[Dict] = None)`

대화 턴을 추가합니다.

```python
def add_turn(
    self,
    query: str,
    response: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Args:
        query: 사용자 쿼리
        response: 시스템 응답
        metadata: 추가 메타데이터 (선택 사항)
    """
```

##### `get_context_window(max_turns: int = 10) -> List[ConversationTurn]`

문맥 창을 반환합니다.

```python
def get_context_window(self, max_turns: int = 10) -> List[ConversationTurn]:
    """
    Args:
        max_turns: 최대 턴 수

    Returns:
        최근 턴 목록 (오래된 턴은 요약됨)
    """
```

**사용 예시**:
```python
session.add_turn("휴학 방법", "휴학은 다음 절차...")
turns = session.get_context_window(max_turns=10)
# 다음 검색에 turns의 맥락 포함
```

##### `is_expired() -> bool`

세션 만료 여부를 확인합니다.

```python
def is_expired(self) -> bool:
    """
    Returns:
        세션이 만료되었으면 True
    """
```

##### `detect_topic_change(current_query: str, threshold: float = 0.3) -> bool`

주제 변경을 감지합니다.

```python
def detect_topic_change(self, current_query: str, threshold: float = 0.3) -> bool:
    """
    Args:
        current_query: 현재 쿼리
        threshold: 유사도 임계값

    Returns:
        주제가 변경되었으면 True
    """
```

#### Data Classes

```python
@dataclass
class ConversationTurn:
    """대화 턴"""
    turn_id: str
    query: str
    response: str
    timestamp: datetime
    metadata: Optional[Dict] = None

class SessionStatus(Enum):
    """세션 상태"""
    ACTIVE = "active"
    EXPIRED = "expired"
    ARCHIVED = "archived"
```

### ConversationService

`src/rag/application/conversation_service.py`

```python
class ConversationService:
    """대화 서비스: 세션 관리 및 비즈니스 로직"""

    def __init__(self, session_repository: SessionRepository):
        """
        Args:
            session_repository: 세션 저장소
        """
```

#### Methods

##### `create_session(user_id: Optional[str]) -> ConversationSession`

세션을 생성합니다.

```python
def create_session(self, user_id: Optional[str]) -> ConversationSession:
    """
    Args:
        user_id: 사용자 ID

    Returns:
        새 세션
    """
```

##### `get_session(session_id: str) -> Optional[ConversationSession]`

세션을 조회합니다.

```python
def get_session(self, session_id: str) -> Optional[ConversationSession]:
    """
    Args:
        session_id: 세션 ID

    Returns:
        세션 객체 (존재하지 않으면 None)
    """
```

##### `add_turn(session_id: str, query: str, response: str) -> None`

세션에 턴을 추가합니다.

```python
def add_turn(self, session_id: str, query: str, response: str) -> None:
    """
    Args:
        session_id: 세션 ID
        query: 사용자 쿼리
        response: 시스템 응답
    """
```

---

## Performance Optimization

연결 풀링과 캐시 워밍으로 리소스 활용을 최적화합니다.

### ConnectionPoolManager

`src/rag/infrastructure/cache/pool.py`

```python
class ConnectionPoolManager:
    """연결 풀 관리자: Redis 및 HTTP 연결 풀"""

    def __init__(self, config: Optional[ConnectionPoolConfig] = None):
        """
        Args:
            config: 연결 풀 설정
        """
```

#### Methods

##### `get_redis_pool() -> redis.ConnectionPool`

Redis 연결 풀을 반환합니다.

```python
def get_redis_pool(self) -> redis.ConnectionPool:
    """
    Returns:
        Redis 연결 풀 (최대 50개 연결)
    """
```

##### `get_http_pool() -> httpx.AsyncHTTPTransport`

HTTP 연결 풀을 반환합니다.

```python
def get_http_pool(self) -> httpx.AsyncHTTPTransport:
    """
    Returns:
        HTTP 연결 풀 (최대 100개 연결, 20개 keep-alive)
    """
```

##### `get_pool_health() -> PoolHealth`

연결 풀 건전성 상태를 반환합니다.

```python
def get_pool_health(self) -> PoolHealth:
    """
    Returns:
        PoolHealth:
            - redis_pool_size: Redis 풀 크기
            - http_pool_size: HTTP 풀 크기
            - redis_available: Redis 사용 가능 연결
            - http_available: HTTP 사용 가능 연결
    """
```

#### Configuration

```python
@dataclass
class ConnectionPoolConfig:
    """연결 풀 설정"""
    redis_max_connections: int = 50
    http_max_connections: int = 100
    http_max_keepalive: int = 20
    retry_on_timeout: bool = True
    socket_keepalive: bool = True
```

### CacheWarmer

`src/rag/infrastructure/cache/warming.py`

```python
class CacheWarmer:
    """캐시 워밍: 자주 사용되는 규정 사전 임베딩"""

    def __init__(self, embedding_service: EmbeddingService, redis_client: redis.Redis):
        """
        Args:
            embedding_service: 임베딩 서비스
            redis_client: Redis 클라이언트
        """
```

#### Methods

##### `warm_top_regulations(top_n: int = 100) -> CacheWarmingResult`

상위 N개 규정을 워밍합니다.

```python
def warm_top_regulations(self, top_n: int = 100) -> CacheWarmingResult:
    """
    Args:
        top_n: 워밍할 규정 수

    Returns:
        CacheWarmingResult:
            - success_count: 성공한 수
            - failed_count: 실패한 수
            - duration: 소요 시간
    """
```

**사용 예시**:
```python
warmer = CacheWarmer(embedding_service, redis_client)
result = warmer.warm_top_regulations(top_n=100)
print(f"성공: {result.success_count}, 실패: {result.failed_count}")
```

##### `schedule_warming(top_n: int, schedule: str) -> None`

워밍을 예약합니다.

```python
def schedule_warming(self, top_n: int, schedule: str) -> None:
    """
    Args:
        top_n: 워밍할 규정 수
        schedule: 크론 표현식 (예: "0 2 * * *" = 매일 새벽 2시)
    """
```

**사용 예시**:
```python
warmer.schedule_warming(top_n=100, schedule="0 2 * * *")
```

##### `incremental_warm(query_frequency: Dict[str, int]) -> None`

쿼리 빈도에 따른 점진적 워밍입니다.

```python
def incremental_warm(self, query_frequency: Dict[str, int]) -> None:
    """
    Args:
        query_frequency: 쿼리 빈도 맵 (규정 ID -> 빈도)
    """
```

---

## A/B Testing Framework

데이터 기반 최적화를 위한 통계적 실험 관리입니다.

### ExperimentService

`src/rag/application/experiment_service.py`

```python
class ExperimentService:
    """A/B 테스트 서비스: 실험 관리 및 분석"""

    def __init__(self, repository: ExperimentRepository):
        """
        Args:
            repository: 실험 저장소
        """
```

#### Methods

##### `create_experiment(...) -> ExperimentConfig`

새 실험을 생성합니다.

```python
def create_experiment(
    self,
    experiment_id: str,
    name: str,
    control_config: Dict[str, Any],
    treatment_configs: List[Dict[str, Any]],
    target_sample_size: int,
    description: Optional[str] = None,
    significance_level: float = 0.05
) -> ExperimentConfig:
    """
    Args:
        experiment_id: 실험 ID (고유)
        name: 실험 이름
        control_config: 대조군 설정
        treatment_configs: 실험군 설정 목록
        target_sample_size: 목표 표본 크기
        description: 설명 (선택 사항)
        significance_level: 유의 수준 (기본값: 0.05)

    Returns:
        ExperimentConfig
    """
```

**사용 예시**:
```python
service = ExperimentService(repository)
config = service.create_experiment(
    experiment_id="reranker_comparison",
    name="Reranker Model Comparison",
    control_config={"model": "bge-reranker-v2-m3"},
    treatment_configs=[{"model": "cohere-rerank-3"}],
    target_sample_size=1000
)
```

##### `start_experiment(experiment_id: str) -> None`

실험을 시작합니다.

```python
def start_experiment(self, experiment_id: str) -> None:
    """
    Args:
        experiment_id: 실험 ID
    """
```

##### `assign_variant(experiment_id: str, user_id: str) -> str`

사용자에게 실험 버전을 할당합니다.

```python
def assign_variant(self, experiment_id: str, user_id: str) -> str:
    """
    Args:
        experiment_id: 실험 ID
        user_id: 사용자 ID

    Returns:
        할당된 버전 ID (예: "control", "treatment_1")
    """
```

**사용 예시**:
```python
variant = service.assign_variant("reranker_comparison", "user123")
print(f"할당된 버전: {variant}")
```

##### `record_conversion(experiment_id: str, user_id: str, variant: str) -> None`

전환을 기록합니다.

```python
def record_conversion(
    self,
    experiment_id: str,
    user_id: str,
    variant: str
) -> None:
    """
    Args:
        experiment_id: 실험 ID
        user_id: 사용자 ID
        variant: 버전 ID
    """
```

**사용 예시**:
```python
service.record_conversion("reranker_comparison", "user123", "control")
```

##### `analyze_results(experiment_id: str) -> ExperimentAnalysisResult`

실험 결과를 분석합니다.

```python
def analyze_results(self, experiment_id: str) -> ExperimentAnalysisResult:
    """
    Args:
        experiment_id: 실험 ID

    Returns:
        ExperimentAnalysisResult:
            - is_significant: 통계적 유의성 여부
            - winner: 승자 버전 ID
            - confidence: 신뢰도 (0-100)
            - p_value: p-value
            - uplift: 향상률
            - variant_metrics: 각 버전의 메트릭
    """
```

**사용 예시**:
```python
result = service.analyze_results("reranker_comparison")
if result.is_significant:
    print(f"승자: {result.winner}")
    print(f"신뢰도: {result.confidence:.1f}%")
    print(f"향상률: {result.uplift:.2f}%")
```

#### Data Classes

```python
@dataclass
class ExperimentConfig:
    """실험 설정"""
    experiment_id: str
    name: str
    description: Optional[str]
    variants: List[Variant]
    traffic_allocation: Dict[str, float]  # variant_id -> percentage
    target_sample_size: int
    significance_level: float
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

@dataclass
class Variant:
    """실험 버전"""
    variant_id: str
    config: Dict[str, Any]
    traffic_percentage: float

@dataclass
class ExperimentMetrics:
    """실험 메트릭"""
    impressions: int  # 노출 수
    conversions: int  # 전환 수
    conversion_rate: float  # 전환율

@dataclass
class ExperimentAnalysisResult:
    """실험 분석 결과"""
    is_significant: bool
    winner: Optional[str]
    confidence: float  # 0-100
    p_value: float
    uplift: Optional[float]
    variant_metrics: Dict[str, ExperimentMetrics]
    recommendation: str
```

---

## Integration Examples

### Full Pipeline Integration

```python
from src.rag.domain.llm.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.rag.domain.llm.ambiguity_classifier import AmbiguityClassifier
from src.rag.domain.llm.emotional_classifier import EmotionalClassifier
from src.rag.domain.citation.citation_enhancer import CitationEnhancer
from src.rag.domain.conversation.session import ConversationSession
from src.rag.application.conversation_service import ConversationService

# 초기화
circuit_config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60.0)
breaker = CircuitBreaker("openai", circuit_config)

ambiguity_classifier = AmbiguityClassifier()
emotional_classifier = EmotionalClassifier()
citation_enhancer = CitationEnhancer()
conversation_service = ConversationService(session_repository)

# 쿼리 처리
query = "휴학 규정"

# 1. 모호성 분류
ambiguity_result = ambiguity_classifier.classify(query)
if ambiguity_result.level == AmbiguityLevel.HIGHLY_AMBIGUOUS:
    dialog = ambiguity_classifier.generate_disambiguation_dialog(ambiguity_result)
    # 사용자 선택 대기...

# 2. 감정 분류
emotional_result = emotional_classifier.classify(query)

# 3. 세션 문맥 가져오기
session = conversation_service.get_session(session_id)
context_turns = session.get_context_window(max_turns=10)

# 4. 검색 (서킷 브레이커 통해)
try:
    chunks = breaker.call(search_service.search, query, context_turns)
except CircuitBreakerOpenError:
    chunks = search_service.search_fallback(query, context_turns)

# 5. 인용 강화
citations = citation_enhancer.enhance_citations(chunks, confidences)
formatted_citations = citation_enhancer.format_citations(citations)

# 6. 응답 생성
base_prompt = "규정에 기반하여 답변하세요"
if emotional_result.state != EmotionalState.NEUTRAL:
    adapted_prompt = emotional_classifier.generate_empathy_prompt(
        emotional_result, base_prompt
    )
else:
    adapted_prompt = base_prompt

response = llm.generate(adapted_prompt, chunks)

# 7. 응답에 인용 추가
final_response = f"{response}\n\n출처: {formatted_citations}"

# 8. 대화 턴 저장
conversation_service.add_turn(session_id, query, final_response)
```

---

## Error Handling

### Common Exceptions

| Exception | Description | Handling |
|-----------|-------------|----------|
| `CircuitBreakerOpenError` | 서킷이 열린 상태 | 폴백 프로바이더 사용 |
| `AmbiguityException` | 모호성 분류 실패 | 기본 검색 실행 |
| `CitationExtractionError` | 인용 추출 실패 | 일반 인용 포맷 사용 |
| `SessionExpiredError` | 세션 만료 | 새 세션 생성 |
| `ExperimentNotFoundError` | 실험이 존재하지 않음 | 기본 버전 사용 |

---

## Version History

- v2.1.0 (2026-01-28): SPEC-RAG-001 구현 완료
  - Circuit Breaker
  - Ambiguity Classifier
  - Citation Enhancement
  - Emotional Query Support
  - Multi-turn Conversation
  - Performance Optimization
  - A/B Testing Framework

---

## Related Documentation

- [CHANGELOG.md](../CHANGELOG.md)
- [README.md](../README.md)
- [SPEC-RAG-001](../.moai/specs/SPEC-RAG-001/spec.md)
