# SPEC-RAG-001: RAG System Comprehensive Improvements

## TAG BLOCK

```yaml
spec_id: SPEC-RAG-001
title: RAG System Comprehensive Improvements
status: Planned
priority: High
created: 2025-01-27
assigned: manager-spec
lifecycle: spec-first
estimated_effort: 6 weeks
labels: [rag, llm, stability, quality, enhancement]
```

## Environment

### Current System Context

**Project**: University Regulation Manager (대학 규정 관리 시스템)

**Technology Stack**:
- Python 3.11+
- llama-index >= 0.14.10 (RAG framework)
- chromadb >= 1.4.0 (Vector database)
- flagembedding >= 1.3.5 (BGE-M3 embeddings)
- FastAPI (REST API)
- Redis (Caching)

**Current Architecture**:
- Clean Architecture with domain-driven design
- Hybrid RAG (BM25 + Dense retrieval)
- Multiple LLM providers (Ollama, OpenAI, Gemini)
- CLI, Web UI, and MCP interfaces

### System Scope

**In Scope**:
- RAG pipeline components (retrieval, generation, citation)
- LLM provider integration and connection management
- Query processing and disambiguation
- Multi-turn conversation support
- Performance optimization (caching, connection pooling)
- A/B testing infrastructure

**Out of Scope**:
- HWP file processing improvements
- New interface development (CLI/Web/MCP)
- Database schema changes
- New regulation parsing logic

## Assumptions

### Technical Assumptions

- **High Confidence**: Current RAG pipeline uses llama-index with modular component design
- **High Confidence**: ChromaDB is the vector database with local file-based storage
- **Medium Confidence**: LLM providers have inconsistent availability and latency
- **Medium Confidence**: Current citation logic extracts article numbers from chunk metadata
- **Evidence**: Codebase analysis shows `LLMProvider` class in `infrastructure/llm/` directory

### Business Assumptions

- **High Confidence**: Users require consistent and reliable RAG system performance
- **High Confidence**: Ambiguous queries are common in regulation searches (audience, regulation type)
- **Medium Confidence**: Emotional queries indicate user distress requiring empathetic responses
- **Medium Confidence**: Multi-turn conversations improve search accuracy through context
- **Evidence**: Product.md shows student, faculty, and administrative user scenarios

### Integration Assumptions

- **High Confidence**: Existing pytest framework with async testing capabilities
- **High Confidence**: Redis is available for caching and session management
- **Medium Confidence**: Circuit breaker pattern can be integrated without breaking changes
- **Risk if Wrong**: Redis availability impacts all caching features
- **Validation Method**: Verify Redis connectivity in integration tests

## Requirements

### Priority 1: LLM Connection Stability (Weeks 1-2)

#### Ubiquitous Requirements

**REQ-LLM-001**: The system shall monitor LLM provider health continuously during operation.

**REQ-LLM-002**: The system shall log all LLM provider connection attempts with timestamps and latency metrics.

**REQ-LLM-003**: The system shall maintain provider-specific failure counters for circuit breaker state transitions.

#### Event-Driven Requirements

**REQ-LLM-004**: WHEN an LLM provider fails to respond within 30 seconds, the system SHALL mark the provider as degraded and trigger health check.

**REQ-LLM-005**: WHEN a provider fails 3 consecutive requests, the system SHALL open the circuit and route traffic to fallback providers.

**REQ-LLM-006**: WHEN a circuit-open provider health check succeeds, the system SHALL transition to half-open state and allow single test request.

**REQ-LLM-007**: WHEN a half-open provider completes a successful request, the system SHALL close the circuit and restore normal traffic.

**REQ-LLM-008**: WHEN all primary providers are unavailable, the system SHALL return cached responses if available with clear stale indicator.

#### State-Driven Requirements

**REQ-LLM-009**: WHILE a provider circuit is closed (normal), the system SHALL route requests according to load balancing policy.

**REQ-LLM-010**: WHILE a provider circuit is open (failed), the system SHALL immediately reject requests without connection attempt timeout.

**REQ-LLM-011**: WHILE a provider circuit is half-open (recovering), the system SHALL allow single request to validate recovery.

**REQ-LLM-012**: IF no fallback providers are available and cache misses, the system SHALL return graceful error with 503 status and retry-after header.

#### Optional Requirements

**REQ-LLM-013**: Where possible, the system MAY implement adaptive timeout based on historical provider latency percentiles (P50, P95, P99).

**REQ-LLM-014**: Where possible, the system MAY provide provider performance dashboard via metrics endpoint.

#### Unwanted Behavior Requirements

**REQ-LLM-015**: The system shall NOT cascade failures from one provider to affect other providers.

**REQ-LLM-016**: The system shall NOT allow circuit breaker state transitions without proper logging and monitoring.

---

### Priority 1: Ambiguous Query Handling (Weeks 1-2)

#### Ubiquitous Requirements

**REQ-AMB-001**: The system shall classify all incoming queries for ambiguity level (clear, ambiguous, highly ambiguous).

**REQ-AMB-002**: The system shall detect audience ambiguity (student vs faculty vs staff) and regulation type ambiguity.

**REQ-AMB-003**: The system shall provide disambiguation suggestions ranked by relevance score.

#### Event-Driven Requirements

**REQ-AMB-004**: WHEN a query is classified as ambiguous, the system SHALL present clarification dialog to user with top 3 disambiguation options.

**REQ-AMB-005**: WHEN user selects a disambiguation option, the system SHALL re-execute search with clarified context and cache the mapping.

**REQ-AMB-006**: WHEN user rejects all disambiguation options, the system SHALL execute broad search with explicit ambiguity warning in response.

**REQ-AMB-007**: WHEN query contains emotional keywords, the system SHALL detect emotional state before ambiguity classification.

#### State-Driven Requirements

**REQ-AMB-008**: IF query audience is unclear (e.g., "휴학" without user type), the system SHALL request user role clarification.

**REQ-AMB-009**: IF query regulation type is unclear (e.g., "규정" without specific regulation), the system SHALL suggest relevant regulation categories.

**REQ-AMB-010**: IF ambiguity score exceeds threshold (0.7), the system SHALL require clarification before search execution.

**REQ-AMB-011**: IF ambiguity score is moderate (0.4-0.7), the system SHALL execute search with disambiguation suggestions in response.

#### Optional Requirements

**REQ-AMB-012**: Where possible, the system MAY learn from user disambiguation selections to improve future classification.

**REQ-AMB-013**: Where possible, the system MAY provide "skip clarification" option for experienced users.

#### Unwanted Behavior Requirements

**REQ-AMB-014**: The system shall NOT execute expensive LLM generation for highly ambiguous queries without clarification attempt.

**REQ-AMB-015**: The system shall NOT present more than 5 disambiguation options to avoid decision paralysis.

---

### Priority 2: Source Citation Enhancement (Weeks 3-4)

#### Ubiquitous Requirements

**REQ-CIT-001**: The system shall extract article numbers from retrieved chunks with character-level precision.

**REQ-CIT-002**: The system shall validate citation references against actual regulation document structure.

**REQ-CIT-003**: The system shall format citations with specific article references (편/장/절/조/항/호).

#### Event-Driven Requirements

**REQ-CIT-004**: WHEN generating response, the system SHALL include inline citations with article numbers for all referenced regulations.

**REQ-CIT-005**: WHEN citation format is invalid (e.g., missing article number), the system SHALL log warning and use generic citation format.

**REQ-CIT-006**: WHEN multiple chunks from same regulation are referenced, the system SHALL consolidate citations with article ranges (e.g., 제3조-제5조).

**REQ-CIT-007**: WHEN user clicks citation in Web UI, the system SHALL scroll to or highlight specific article in regulation viewer.

#### State-Driven Requirements

**REQ-CIT-008**: IF chunk metadata contains article number field, the system SHALL extract and format citation directly.

**REQ-CIT-009**: IF chunk metadata lacks article number, the system SHALL infer citation from hierarchical path or content regex.

**REQ-CIT-010**: IF citation validation fails, the system SHALL mark citation as "unverified" in response metadata.

#### Optional Requirements

**REQ-CIT-011**: Where possible, the system MAY provide direct link to regulation document with article anchor.

**REQ-CIT-012**: Where possible, the system MAY include page numbers for PDF-based regulations.

#### Unwanted Behavior Requirements

**REQ-CIT-013**: The system shall NOT generate responses without any citation references.

**REQ-CIT-014**: The system shall NOT infer article numbers without confidence threshold (minimum 0.8).

---

### Priority 2: Emotional Query Support (Weeks 3-4)

#### Ubiquitous Requirements

**REQ-EMO-001**: The system shall classify query emotional intent (neutral, seeking help, distressed, frustrated).

**REQ-EMO-002**: The system shall maintain empathy level configuration for response generation.

**REQ-EMO-003**: The system shall provide emotional state metrics for monitoring and optimization.

#### Event-Driven Requirements

**REQ-EMO-004**: WHEN query is classified as distressed, the system SHALL prepend empathetic acknowledgment to response.

**REQ-EMO-005**: WHEN query is classified as frustrated, the system SHALL use calming language and provide clear step-by-step guidance.

**REQ-EMO-006**: WHEN query is classified as seeking help, the system SHALL prioritize clarity over brevity in response formatting.

**REQ-EMO-007**: WHEN emotional query is detected, the system SHALL adjust tone in prompt without modifying factual content.

#### State-Driven Requirements

**REQ-EMO-008**: IF query contains emotional keywords (e.g., "힘들어요", "어떡해요", "답답해요"), the system SHALL classify as distressed.

**REQ-EMO-009**: IF query contains urgency indicators (e.g., "급해요", "빨리", "지금"), the system SHALL prioritize relevant results and provide immediate action items.

**REQ-EMO-010**: IF emotional state is unclear, the system SHALL default to neutral classification.

**REQ-EMO-011**: IF multiple emotional signals conflict, the system SHALL use highest intensity signal for classification.

#### Optional Requirements

**REQ-EMO-012**: Where possible, the system MAY provide resource suggestions (e.g., counseling contacts) for highly distressed queries.

**REQ-EMO-013**: Where possible, the system MAY adapt response complexity based on detected user expertise level.

#### Unwanted Behavior Requirements

**REQ-EMO-014**: The system shall NOT sacrifice factual accuracy for empathetic tone.

**REQ-EMO-015**: The system shall NOT make assumptions about user emotional state without keyword evidence.

---

### Priority 3: Multi-turn Conversation Support (Weeks 5-6)

#### Ubiquitous Requirements

**REQ-MUL-001**: The system shall maintain conversation session state across multiple user turns.

**REQ-MUL-002**: The system shall track conversation context including previous queries, responses, and user feedback.

**REQ-MUL-003**: The system shall implement session timeout with configurable expiration (default 30 minutes).

#### Event-Driven Requirements

**REQ-MUL-004**: WHEN user submits follow-up query, the system SHALL interpret query in context of previous conversation turns.

**REQ-MUL-005**: WHEN conversation exceeds context window (10 turns), the system SHALL summarize early turns and maintain key context.

**REQ-MUL-006**: WHEN user explicitly starts new conversation, the system SHALL clear session context and initialize new state.

**REQ-MUL-007**: WHEN session expires from timeout, the system SHALL persist session for audit and optionally resume on user confirmation.

#### State-Driven Requirements

**REQ-MUL-008**: WHILE conversation is active, the system SHALL include conversation summary in search retrieval queries.

**REQ-MUL-009**: IF follow-up query contains pronouns (e.g., "그거", "그 규정", "그 방법"), the system SHALL resolve references from previous turns.

**REQ-MUL-010**: IF conversation topic shifts significantly (semantic similarity < 0.3), the system SHALL detect topic change and prompt for new conversation.

**REQ-MUL-011**: IF user provides negative feedback on response, the system SHALL adjust search strategy and re-query with modified parameters.

#### Optional Requirements

**REQ-MUL-012**: Where possible, the system MAY provide conversation history export in markdown format.

**REQ-MUL-013**: Where possible, the system MAY suggest related follow-up questions based on conversation context.

#### Unwanted Behavior Requirements

**REQ-MUL-014**: The system shall NOT leak conversation context between different users or sessions.

**REQ-MUL-015**: The system shall NOT retain session data beyond retention period (configurable, default 24 hours).

---

### Priority 3: Performance Optimization (Weeks 5-6)

#### Ubiquitous Requirements

**REQ-PER-001**: The system shall implement connection pooling for Redis and HTTP client connections.

**REQ-PER-002**: The system shall maintain cache hit rate metrics for all cache layers (LLM, embedding, HyDE).

**REQ-PER-003**: The system shall implement cache warming strategy for frequently accessed regulations.

#### Event-Driven Requirements

**REQ-PER-004**: WHEN cache hit rate drops below 60%, the system SHALL log warning and trigger cache warming for top queries.

**REQ-PER-005**: WHEN connection pool is exhausted, the system SHALL queue requests and return 503 status if queue exceeds threshold.

**REQ-PER-006**: WHEN cache warming executes, the system SHALL pre-generate embeddings for top 100 most searched regulations.

**REQ-PER-007**: WHEN cache expires, the system SHALL implement write-through caching for immediate refresh.

#### State-Driven Requirements

**REQ-PER-008**: IF Redis connection is unavailable, the system SHALL degrade gracefully to in-memory caching with reduced TTL.

**REQ-PER-009**: IF system memory is constrained (available < 20%), the system SHALL reduce cache size and evict least recently used entries.

**REQ-PER-010**: IF cache warming is enabled, the system SHALL execute warming during low-traffic periods (configurable schedule).

**REQ-PER-011**: IF connection pool health check fails, the system SHALL close unhealthy connections and re-establish pool.

#### Optional Requirements

**REQ-PER-012**: Where possible, the system MAY implement adaptive cache TTL based on access frequency and regulation update rate.

**REQ-PER-013**: Where possible, the system MAY provide cache performance dashboard via metrics endpoint.

#### Unwanted Behavior Requirements

**REQ-PER-014**: The system shall NOT block on cache operations (all cache operations must be asynchronous with fallback).

**REQ-PER-015**: The system shall NOT cache responses with personal user information or sensitive data.

---

### Priority 3: A/B Testing Framework (Weeks 5-6)

#### Ubiquitous Requirements

**REQ-AB-001**: The system shall implement generic A/B testing service for RAG component comparison.

**REQ-AB-002**: The system shall maintain experiment configuration with variant definitions and traffic allocation.

**REQ-AB-003**: The system shall track experiment metrics including impressions, conversions, and statistical significance.

#### Event-Driven Requirements

**REQ-AB-004**: WHEN user is assigned to experiment variant, the system SHALL record assignment in metrics store.

**REQ-AB-005**: WHEN experiment reaches statistical significance (p-value < 0.05), the system SHALL log result and notify administrators.

**REQ-AB-006**: WHEN experiment is stopped, the system SHALL generate final report with winner recommendation and confidence interval.

**REQ-AB-007**: WHEN user completes conversion event (e.g., positive feedback, successful citation), the system SHALL record conversion for assigned variant.

#### State-Driven Requirements

**REQ-AB-008**: IF experiment is not configured, the system SHALL route all traffic to default variant.

**REQ-AB-009**: IF user has existing experiment assignment, the system SHALL maintain consistent assignment across session.

**REQ-AB-010**: IF experiment traffic allocation is modified, the system SHALL rebalance assignments without disrupting ongoing sessions.

**REQ-AB-011**: IF experiment fails to reach significance after target sample size, the system SHALL extend experiment or mark as inconclusive.

#### Optional Requirements

**REQ-AB-012**: Where possible, the system MAY implement multi-armed bandit algorithm for automatic traffic optimization.

**REQ-AB-013**: Where possible, the system MAY provide experiment configuration API for dynamic management.

#### Unwanted Behavior Requirements

**REQ-AB-014**: The system shall NOT expose experiment assignment or metrics in user-facing responses.

**REQ-AB-015**: The system shall NOT allow experiment configuration changes without proper authentication and authorization.

## Specifications

### Architecture Design

#### Circuit Breaker Pattern (REQ-LLM-001 to REQ-LLM-016)

**Component**: `CircuitBreaker` class in `infrastructure/llm/circuit_breaker.py`

**States**:
- CLOSED (Normal): Requests pass through, failure counter resets on success
- OPEN (Failed): Requests immediately rejected, timeout before half-open transition
- HALF_OPEN (Recovering): Single test request allowed, success closes circuit

**Configuration**:
```python
@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 3  # Consecutive failures to open circuit
    recovery_timeout: float = 60.0  # Seconds before half-open transition
    success_threshold: int = 2  # Consecutive successes to close circuit
    timeout: float = 30.0  # Request timeout in seconds
```

**Metrics**:
- Circuit state transitions
- Provider failure rates
- Request latency percentiles (P50, P95, P99)
- Fallback provider usage

#### Ambiguity Classification (REQ-AMB-001 to REQ-AMB-015)

**Component**: `AmbiguityClassifier` in `domain/query/ambiguity_classifier.py`

**Classification Levels**:
- CLEAR (0.0-0.3): Direct query with clear intent
- AMBIGUOUS (0.4-0.7): Some ambiguity, present suggestions
- HIGHLY_AMBIGUOUS (0.8-1.0): Require clarification before search

**Detection Rules**:
```python
AMBIGUITY_RULES = {
    "audience": ["학생", "교수", "교직원", "교원"],
    "regulation": ["규정", "시행세칙", "지침"],
    "absence_article": [lambda q: not bool(re.search(r'제\d+조', q))]
}
```

**Disambiguation Dialog**:
- Top 3 ranked suggestions
- User selection or "skip" option
- Caching of user selections for learning

#### Citation Enhancement (REQ-CIT-001 to REQ-CIT-014)

**Component**: `CitationExtractor` in `domain/retrieval/citation.py`

**Extraction Pipeline**:
1. Extract from chunk metadata (preferred)
2. Infer from hierarchical path
3. Regex extraction from content
4. Fallback to generic citation

**Validation**:
```python
def validate_citation(citation: str, regulation: Regulation) -> bool:
    pattern = r'제\d+조[의항호목]*'
    return bool(re.search(pattern, citation))
```

**Formatting**:
- Inline citations in response text
- Consolidated citation ranges
- Clickable links in Web UI

#### Emotional Query Support (REQ-EMO-001 to REQ-EMO-015)

**Component**: `EmotionalClassifier` in `domain/query/emotional_classifier.py`

**Emotional States**:
- NEUTRAL: Standard factual response
- SEEKING_HELP: Detailed explanation with examples
- DISTRESSED: Empathetic acknowledgment + factual content
- FRUSTRATED: Step-by-step guidance + calming language

**Keyword Dictionary**:
```python
EMOTIONAL_KEYWORDS = {
    "distressed": ["힘들어요", "어떡해요", "답답해요", "포기"],
    "frustrated": ["안돼요", "왜 안돼요", "너무 복잡해요", "이해 안돼요"],
    "seeking_help": ["어떻게 해요", "방법 알려주세요", "절차가 뭐예요"],
    "urgency": ["급해요", "빨리", "지금", "당장"]
}
```

**Prompt Adaptation**:
```python
def adapt_prompt_for_emotion(base_prompt: str, emotion: EmotionalState) -> str:
    if emotion == EmotionalState.DISTRESSED:
        return "공감해 주는 따뜻한 어조로 답변하고, " + base_prompt
    elif emotion == EmotionalState.FRUSTRATED:
        return "단계별로 명확하게 설명하고, " + base_prompt
    return base_prompt
```

#### Multi-turn Conversation (REQ-MUL-001 to REQ-MUL-015)

**Component**: `ConversationManager` in `domain/conversation/manager.py`

**Session Structure**:
```python
@dataclass
class ConversationSession:
    session_id: str
    user_id: Optional[str]
    turns: List[ConversationTurn]
    context_summary: str
    created_at: datetime
    last_activity: datetime
    status: SessionStatus  # ACTIVE, EXPIRED, ARCHIVED
```

**Context Window**:
- Maximum 10 turns in active context
- Automatic summarization of early turns
- Reference resolution for pronouns

**Topic Change Detection**:
```python
def detect_topic_change(current_query: str, previous_turns: List[Turn]) -> bool:
    semantic_similarity = compute_similarity(current_query, previous_turns[-1].query)
    return semantic_similarity < 0.3
```

#### Performance Optimization (REQ-PER-001 to REQ-PER-015)

**Connection Pooling**:
```python
# Redis connection pool
redis_pool = redis.ConnectionPool(
    max_connections=50,
    retry_on_timeout=True,
    socket_keepalive=True
)

# HTTP connection pool for LLM providers
http_pool = httpx.AsyncHTTPTransport(
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
)
```

**Cache Warming Strategy**:
- Pre-compute embeddings for top 100 regulations
- Schedule during low-traffic periods (2:00 AM default)
- Incremental warming based on query frequency

**Cache Layers**:
1. L1: In-memory cache (fastest, limited size)
2. L2: Redis cache (distributed, persistent)
3. L3: ChromaDB cache (vector similarity)

**Monitoring**:
- Cache hit rate per layer
- Eviction rate
- Memory usage
- Warming job execution time

#### A/B Testing Framework (REQ-AB-001 to REQ-AB-015)

**Component**: `ExperimentService` in `domain/experiment/service.py`

**Experiment Configuration**:
```python
@dataclass
class ExperimentConfig:
    experiment_id: str
    name: str
    description: str
    variants: List[Variant]
    traffic_allocation: Dict[str, float]  # variant_id -> percentage
    target_metrics: List[str]  # e.g., ["citation_accuracy", "user_satisfaction"]
    start_date: datetime
    end_date: Optional[datetime]
    sample_size: int
    significance_level: float = 0.05
```

**Assignment Algorithm**:
- Consistent hashing for user assignment
- Traffic allocation percentage enforcement
- Session consistency maintenance

**Statistical Testing**:
- Two-proportion z-test for conversion rates
- Confidence interval calculation
- Winner recommendation based on uplift

### File Structure

```
src/
├── domain/
│   ├── query/
│   │   ├── ambiguity_classifier.py       # NEW: Ambiguity detection
│   │   ├── emotional_classifier.py        # NEW: Emotional state
│   │   └── disambiguation_dialog.py       # NEW: User interaction
│   ├── retrieval/
│   │   ├── citation.py                    # NEW: Citation extraction
│   │   └── citation_validator.py          # NEW: Citation validation
│   ├── conversation/
│   │   ├── manager.py                     # NEW: Session management
│   │   ├── session.py                     # NEW: Session entity
│   │   └── context_tracker.py             # NEW: Context tracking
│   ├── experiment/
│   │   ├── service.py                     # NEW: A/B testing
│   │   ├── config.py                      # NEW: Experiment config
│   │   └── metrics.py                     # NEW: Statistical analysis
│   └── llm/
│       └── circuit_breaker.py             # NEW: Circuit breaker
├── infrastructure/
│   ├── cache/
│   │   ├── pool.py                        # NEW: Connection pooling
│   │   ├── warming.py                     # NEW: Cache warming
│   │   └── metrics.py                     # NEW: Cache metrics
│   └── monitoring/
│       ├── health_check.py                # NEW: Provider health
│       └── metrics_exporter.py            # NEW: Metrics endpoint
├── application/
│   ├── rag/
│   │   ├── enhanced_pipeline.py           # MODIFIED: Integration
│   │   └── response_generator.py          # MODIFIED: Citation support
│   └── services/
│       └── search_service.py              # MODIFIED: Ambiguity handling
└── interfaces/
    ├── web/
    │   ├── routes/
    │   │   ├── chat.py                    # MODIFIED: Multi-turn
    │   │   └── citation.py                # NEW: Citation links
    │   └── templates/
    │       └── chat.html                  # MODIFIED: Conversation UI
    └── cli/
        └── interactive.py                 # MODIFIED: Clarification dialog
```

### Dependencies

**New Dependencies**:
```toml
[tool.poetry.dependencies]
redis = {version = "^5.0.0", extras = ["hiredis"]}
httpx = {version = "^0.27.0", extras = ["http2"]}
tenacity = "^8.2.0"  # Retry logic
prometheus-client = "^0.20.0"  # Metrics
```

**Existing Dependencies Used**:
- pytest-asyncio (async testing)
- fakeredis (Redis testing)
- freezegun (time testing)

## Traceability

### Requirements to Components Mapping

| Requirement ID | Component | File |
|---------------|-----------|------|
| REQ-LLM-001 ~ REQ-LLM-016 | CircuitBreaker, HealthCheck | infrastructure/llm/circuit_breaker.py |
| REQ-AMB-001 ~ REQ-AMB-015 | AmbiguityClassifier, DisambiguationDialog | domain/query/ambiguity_classifier.py |
| REQ-CIT-001 ~ REQ-CIT-014 | CitationExtractor, CitationValidator | domain/retrieval/citation.py |
| REQ-EMO-001 ~ REQ-EMO-015 | EmotionalClassifier, PromptAdapter | domain/query/emotional_classifier.py |
| REQ-MUL-001 ~ REQ-MUL-015 | ConversationManager, ContextTracker | domain/conversation/manager.py |
| REQ-PER-001 ~ REQ-PER-015 | ConnectionPool, CacheWarmer | infrastructure/cache/pool.py |
| REQ-AB-001 ~ REQ-AB-015 | ExperimentService, MetricsCollector | domain/experiment/service.py |

### Components to Test Cases Mapping

| Component | Test File | Test Coverage Target |
|-----------|-----------|---------------------|
| CircuitBreaker | tests/unit/test_circuit_breaker.py | 90% |
| AmbiguityClassifier | tests/unit/test_ambiguity_classifier.py | 85% |
| CitationExtractor | tests/unit/test_citation_extractor.py | 85% |
| EmotionalClassifier | tests/unit/test_emotional_classifier.py | 85% |
| ConversationManager | tests/integration/test_conversation.py | 85% |
| CacheWarmer | tests/integration/test_cache_warming.py | 80% |
| ExperimentService | tests/unit/test_experiment_service.py | 85% |

### Dependencies

**External Dependencies**:
- Redis 5.0+ (caching, session management)
- Existing LlamaIndex, ChromaDB, BGE-M3 stack

**Internal Dependencies**:
- `domain/query/query_classifier.py` (extend for ambiguity/emotion)
- `infrastructure/llm/llm_provider.py` (extend for circuit breaker)
- `application/rag/pipeline.py` (modify for citation integration)
- `interfaces/web/routes/chat.py` (extend for multi-turn)

---

## Appendix

### Glossary

- **Circuit Breaker**: Design pattern to prevent cascading failures by stopping requests to failing services
- **Ambiguity**: Uncertainty in query intent, audience, or regulation type
- **Disambiguation**: Process of clarifying ambiguous queries through user interaction
- **Citation**: Reference to specific regulation articles (편/장/절/조/항/호)
- **Emotional Query**: Query containing emotional indicators requiring empathetic response
- **Multi-turn Conversation**: Extended dialogue context across multiple user interactions
- **Connection Pooling**: Reusing database and HTTP connections for performance
- **Cache Warming**: Pre-populating cache with frequently accessed data
- **A/B Testing**: Controlled experiment comparing two variants for performance
- **Statistical Significance**: Probability that observed differences are not due to chance

### References

- Circuit Breaker Pattern: [Microsoft Cloud Design Patterns](https://docs.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker)
- EARS Format: [Requirements Engineering Styles](https://www REQ.ivargo MBA requirements/)
- RAG Techniques: [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- A/B Testing: [Google Optimize Documentation](https://support.google.com/optimize/)

### Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-27 | manager-spec | Initial SPEC creation with all 7 improvement areas |

---

**SPEC Status**: Planned
**Next Phase**: /moai:2-run SPEC-RAG-001 (Implementation with DDD)
**Estimated Completion**: 2025-03-10 (6 weeks from start)
