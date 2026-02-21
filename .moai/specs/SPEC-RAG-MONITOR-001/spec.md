# SPEC-RAG-MONITOR-001: RAG Real-time Monitoring System

## Metadata

```yaml
SPEC ID: SPEC-RAG-MONITOR-001
Title: RAG Real-time Interaction Monitoring System
Created: 2026-02-21
Status: Planned
Priority: High
Assigned: manager-spec
Related SPECs: SPEC-RAG-QUALITY-008 (Faithfulness), SPEC-RAG-Q-001 (Quality)
Epic: RAG System Enhancement
Labels: monitoring, observability, cli, dashboard
Lifecycle Level: spec-anchored
```

---

## Problem Statement

### Context

This is a university regulation RAG system that processes natural language queries about academic regulations and policies. Users interact with the system via CLI using `uv run regulation` command.

### Current Pain Points

1. **No Visibility into Internal Processing**: Users cannot observe what happens during RAG pipeline execution - query rewriting, vector search, reranking, and LLM generation remain opaque.

2. **Debugging Difficulty**: When unexpected answers occur, there is no way to trace which step caused the issue without modifying code or checking logs manually.

3. **User Experience Disconnect**: System maintainers cannot observe real usage scenarios in the same environment as end users.

### User Requirements (Korean)

1. "이 프로젝트를 사용하는 사용자와 동일한 환경(CLI)에서 사용하는 장면을 내가 직접 상황을 보고 싶어"
   - Want to see the actual usage scenario in the same CLI environment as users

2. "어떤 질의가 갔고 어떤 답변이 나왔는지 등을 직접 모니터링 하고 싶어"
   - Want to directly monitor what queries were sent and what answers were returned

3. "실행할 때는 사용자와 동일한 명령(uv run regulation)을 사용해야 해"
   - Must use the same command as users: `uv run regulation`

---

## EARS Requirements

### Ubiquitous Requirements

**REQ-001**: The system SHALL log all RAG interactions with timestamps, inputs, outputs, and latency metrics.

**REQ-002**: The system SHALL emit events for every stage of the RAG pipeline execution.

**REQ-003**: All logged events SHALL include a correlation ID to trace end-to-end request flow.

### Event-Driven Requirements

**REQ-010**: WHEN a query is received, the system SHALL emit a `QueryReceived` event containing timestamp, query text, and correlation ID.

**REQ-011**: WHEN query rewriting occurs, the system SHALL emit a `QueryRewritten` event containing original query, rewritten query, and rewriting strategy used.

**REQ-012**: WHEN vector search completes, the system SHALL emit a `SearchCompleted` event containing results, scores, and search latency.

**REQ-013**: WHEN reranking completes, the system SHALL emit a `RerankingCompleted` event containing reordered results with new scores and reranking latency.

**REQ-014**: WHEN LLM generation starts, the system SHALL emit an `LLMGenerationStarted` event containing prompt metadata and model configuration.

**REQ-015**: WHEN LLM generation produces tokens, the system SHALL emit `TokenGenerated` events for streaming output.

**REQ-016**: WHEN answer generation completes, the system SHALL emit an `AnswerGenerated` event containing full response, token count, and generation latency.

### State-Driven Requirements

**REQ-020**: WHEN the `--trace` flag is provided, the system SHALL output detailed processing steps to the terminal in real-time.

**REQ-021**: WHEN the `--trace` flag is provided, the system SHALL display query analysis including intent classification and entity extraction.

**REQ-022**: WHEN the `--trace` flag is provided, the system SHALL display search results with similarity scores and chunk previews.

**REQ-023**: WHEN the `--trace` flag is provided, the system SHALL display reranking scores and ranking changes.

**REQ-024**: WHEN the `--trace` flag is provided, the system SHALL display LLM prompt and streaming generation progress.

**REQ-030**: WHEN the `--monitor` flag is provided, the system SHALL launch a Gradio web dashboard accessible via localhost URL.

**REQ-031**: WHEN the `--monitor` flag is provided, the system SHALL automatically open a browser tab to the dashboard URL.

**REQ-032**: WHEN the dashboard is active, the system SHALL display a live event stream showing all RAG events in real-time.

### Optional Requirements

**REQ-040**: WHERE interaction history persistence is enabled, the system MAY store interactions for later replay and analysis.

**REQ-041**: WHERE performance profiling is requested, the system MAY display detailed latency breakdown for each pipeline stage.

### Unwanted Behavior Requirements

**REQ-050**: The monitoring system SHALL NOT add more than 10% overhead to query response time.

**REQ-051**: The system SHALL NOT log sensitive data (credentials, PII) without explicit user consent.

**REQ-052**: The monitoring system SHALL NOT interfere with normal CLI output when no monitoring flags are provided.

**REQ-053**: The `--monitor` dashboard SHALL NOT block the CLI process - it SHALL run in background or as a separate thread.

---

## Specifications

### Component 1: RAGInteractionLogger (Infrastructure Layer)

**Location**: `src/rag/infrastructure/logging/interaction_logger.py`

**Purpose**: Central event emission and logging for all RAG pipeline stages.

**Event Types**:
- `QueryReceived`: timestamp, query_text, correlation_id
- `QueryRewritten`: original_query, rewritten_query, strategy
- `SearchCompleted`: results, scores, latency_ms
- `RerankingCompleted`: reordered_results, score_changes, latency_ms
- `LLMGenerationStarted`: prompt_preview, model_name, max_tokens
- `TokenGenerated`: token, accumulated_length
- `AnswerGenerated`: full_response, token_count, latency_ms

**Integration Points**:
- QueryRewriter service
- VectorSearchService
- RerankerService
- LLMClient (OpenAI/Anthropic)

### Component 2: CLI --trace Flag

**Location**: `src/rag/interface/cli.py`

**Command**: `uv run regulation search "query" --trace`

**Output Format**:
```
🔍 [TRACE] Query Analysis
   Original: "휴학 절차"
   Rewritten: "대학 휴학 신청 방법 절차"
   Intent: policy_inquiry
   Entities: [휴학]

🔎 [TRACE] Vector Search
   Found 5 results (latency: 45ms)
   1. [0.89] 대학규정 제25조 휴학에 관한 사항...
   2. [0.85] 휴학신청서 제출 안내...
   ...

📊 [TRACE] Reranking
   Reranked 5 results (latency: 12ms)
   1. [0.95] 대학규정 제25조... (+0.06)
   ...

🤖 [TRACE] LLM Generation
   Model: gpt-4o-mini
   Prompt tokens: 1,245
   Streaming answer...
```

### Component 3: CLI --monitor Flag

**Command**: `uv run regulation search "query" --monitor`

**Behavior**:
1. Starts Gradio dashboard on localhost:7860
2. Prints URL: `📊 Monitor dashboard: http://localhost:7860`
3. Opens browser automatically
4. Displays live event stream in web UI
5. Continues normal CLI execution

### Component 4: Live Monitor Dashboard (Web UI)

**Location**: `src/rag/interface/web/live_monitor.py`

**Features**:
- Real-time event stream display
- Query input field for testing
- Search result visualization with scores
- LLM generation streaming display
- Performance metrics panel (latency breakdown)
- Event history log with timestamps

**Dashboard Tabs**:
1. **Live Stream**: Real-time event visualization
2. **Query Test**: Input queries directly from dashboard
3. **Performance**: Latency metrics and statistics
4. **History**: Past queries with replay capability

---

## Constraints

### Technical Constraints

- Must use existing Gradio setup in `src/rag/interface/web/`
- Must integrate with existing CLI without breaking changes
- Must use Python's built-in libraries or existing dependencies where possible
- Event emission must be non-blocking

### Performance Constraints

- Monitoring overhead: < 10% of total response time
- Trace output latency: < 50ms per event
- Dashboard refresh rate: 100ms minimum interval

### Compatibility Constraints

- Must work with existing `uv run regulation` command
- Must support both streaming and non-streaming LLM responses
- Must not conflict with existing Rich output formatting

---

## Dependencies

### Internal Dependencies

- `src/rag/interface/cli.py` - CLI entry point
- `src/rag/application/services/` - Service layer integration
- `src/rag/interface/web/app.py` - Existing Gradio application

### External Dependencies

- `blinker` (optional) - Event signaling (or use custom EventEmitter)
- `structlog` - Structured logging (already in project)
- `gradio` - Dashboard UI (already in project)

---

## Out of Scope

- Multi-user dashboard (single user monitoring only)
- Persistent storage optimization for large-scale logging
- Remote dashboard access (localhost only)
- Authentication for dashboard (development tool)
