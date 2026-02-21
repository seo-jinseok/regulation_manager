# Implementation Plan: SPEC-RAG-MONITOR-001

## Overview

**Goal**: Implement real-time RAG monitoring system with CLI trace output and web dashboard.

**Development Mode**: Hybrid (TDD for new components, DDD for existing integration)

---

## Milestones

### Priority High: Core Event System

**Milestone 1.1: Event Infrastructure**

Tasks:
1. Create `src/rag/infrastructure/logging/events.py`
   - Define EventType enum
   - Define event dataclasses for each event type
   - Create EventEmitter singleton class
   - Implement subscribe/unsubscribe pattern

2. Create `src/rag/infrastructure/logging/interaction_logger.py`
   - Implement RAGInteractionLogger class
   - Add correlation ID generation
   - Integrate with structlog for JSON logging
   - Add performance timestamp helpers

3. Write unit tests for event system
   - Test event emission and subscription
   - Test correlation ID uniqueness
   - Test logging output format

**Deliverables**:
- Event emission infrastructure
- Unit tests with 85%+ coverage
- JSON log format specification

---

**Milestone 1.2: Service Integration**

Tasks:
1. Integrate with QueryRewriter service
   - Add event emission for QueryRewritten event
   - Add latency measurement

2. Integrate with VectorSearchService
   - Add event emission for SearchCompleted event
   - Add result count and top scores to event

3. Integrate with RerankerService
   - Add event emission for RerankingCompleted event
   - Add score change tracking

4. Integrate with LLMClient
   - Add event emission for LLMGenerationStarted
   - Add TokenGenerated events for streaming
   - Add AnswerGenerated event on completion

**Deliverables**:
- Instrumented RAG pipeline
- Integration tests for event flow
- Performance overhead measurement (<10%)

---

### Priority High: CLI --trace Flag

**Milestone 2.1: Trace Output Handler**

Tasks:
1. Create `src/rag/interface/cli/trace_handler.py`
   - Implement TraceOutputHandler class
   - Subscribe to all RAG events
   - Format output for terminal using Rich

2. Add --trace flag to CLI
   - Modify `src/rag/interface/cli.py`
   - Add click.option for --trace
   - Initialize TraceOutputHandler when flag present

3. Implement formatted output for each event type
   - QueryReceived: Query analysis display
   - SearchCompleted: Results with scores
   - RerankingCompleted: Ranking changes
   - LLMGenerationStarted: Model info
   - TokenGenerated: Streaming progress

**Deliverables**:
- Working --trace flag
- Formatted terminal output
- CLI integration tests

---

### Priority Medium: CLI --monitor Flag

**Milestone 3.1: Dashboard Backend**

Tasks:
1. Create `src/rag/interface/web/live_monitor.py`
   - Implement LiveMonitor class
   - Create event buffer for web display
   - Implement SSE (Server-Sent Events) endpoint
   - Add query submission endpoint

2. Integrate with existing Gradio app
   - Modify `src/rag/interface/web/app.py`
   - Add "Live Monitor" tab
   - Add event stream component

3. Implement background server startup
   - Add --monitor flag to CLI
   - Start Gradio in background thread
   - Print and open URL

**Deliverables**:
- Live Monitor tab in dashboard
- Real-time event streaming
- Background server integration

---

**Milestone 3.2: Dashboard Frontend**

Tasks:
1. Create Live Stream component
   - Event timeline visualization
   - Auto-scroll to latest events
   - Event filtering by type

2. Create Query Test component
   - Query input field
   - Submit button with live results
   - Result display panel

3. Create Performance panel
   - Latency breakdown chart
   - Average metrics display
   - Historical statistics

**Deliverables**:
- Complete dashboard UI
- Interactive query testing
- Performance visualization

---

### Priority Low: Optional Features

**Milestone 4.1: History and Replay**

Tasks:
1. Implement interaction history storage
   - Create `data/logs/rag_interactions/` structure
   - Implement daily log rotation
   - Add JSON log format for replay

2. Add History tab to dashboard
   - List past queries
   - Replay button for each query
   - Search/filter history

**Deliverables**:
- Persistent interaction logs
- History tab in dashboard
- Query replay functionality

---

## Technical Approach

### Event System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline Services                     │
├─────────────────────────────────────────────────────────────┤
│  QueryRewriter │ VectorSearch │ Reranker │ LLMClient        │
│       │              │             │            │            │
│       ▼              ▼             ▼            ▼            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              RAGInteractionLogger                    │   │
│  │  - emit_event() - log_interaction() - measure()     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                EventEmitter                          │   │
│  │  - subscribe() - unsubscribe() - emit()             │   │
│  └─────────────────────────────────────────────────────┘   │
│                    │              │                         │
│         ┌─────────┴────────┐─────┴─────────┐               │
│         ▼                  ▼               ▼               │
│  TraceOutputHandler  DashboardHandler  LogHandler          │
└─────────────────────────────────────────────────────────────┘
```

### CLI Integration Pattern

```python
# cli.py
@click.command()
@click.argument('query')
@click.option('--trace', is_flag=True, help='Enable detailed trace output')
@click.option('--monitor', is_flag=True, help='Launch web dashboard')
def search(query: str, trace: bool, monitor: bool):
    if monitor:
        start_dashboard_background()

    if trace:
        enable_trace_output()

    # Normal RAG pipeline execution
    result = rag_pipeline.query(query)
    print(result)
```

### Dashboard Communication Pattern

```
CLI Process                    Dashboard (Gradio)
    │                               │
    ├── QueryReceived ──────────────┼──▶ Display in Live Stream
    ├── QueryRewritten ────────────┼──▶ Update Query Analysis
    ├── SearchCompleted ───────────┼──▶ Show Results
    ├── RerankingCompleted ────────┼──▶ Update Rankings
    ├── LLMGenerationStarted ──────┼──▶ Show Model Info
    ├── TokenGenerated ────────────┼──▶ Stream to Output
    └── AnswerGenerated ───────────┼──▶ Finalize Display
```

---

## File Structure

```
src/rag/
├── infrastructure/
│   └── logging/
│       ├── __init__.py
│       ├── events.py              # NEW: Event types and EventEmitter
│       └── interaction_logger.py  # NEW: RAGInteractionLogger
├── interface/
│   ├── cli.py                     # MODIFY: Add --trace, --monitor flags
│   ├── cli/
│   │   └── trace_handler.py       # NEW: TraceOutputHandler
│   └── web/
│       ├── app.py                 # MODIFY: Add Live Monitor tab
│       └── live_monitor.py        # NEW: Dashboard backend

data/
└── logs/
    └── rag_interactions/          # NEW: Interaction log storage
        └── .gitkeep
```

---

## Testing Strategy

### Unit Tests

- `test_events.py`: Event emission, subscription, correlation IDs
- `test_interaction_logger.py`: Logging format, latency measurement
- `test_trace_handler.py`: Output formatting for each event type

### Integration Tests

- `test_pipeline_instrumentation.py`: End-to-end event flow
- `test_cli_flags.py`: --trace and --monitor flag behavior
- `test_dashboard.py`: Event streaming to web UI

### Performance Tests

- Overhead measurement: Compare response time with/without monitoring
- Target: < 10% overhead for --trace, < 5% for normal execution

---

## Risk Assessment

### Risk 1: Performance Overhead

**Probability**: Medium
**Impact**: High
**Mitigation**:
- Use async event emission
- Buffer events before logging
- Performance testing before merge

### Risk 2: CLI Output Interference

**Probability**: Low
**Impact**: Medium
**Mitigation**:
- Separate trace output from normal output
- Use Rich Live display for non-interference
- Test with existing output formats

### Risk 3: Dashboard Port Conflicts

**Probability**: Low
**Impact**: Low
**Mitigation**:
- Auto-select available port
- Add --port option for manual override
- Clear error message on port conflict

---

## Implementation Sequence

1. **Phase 1** (Events Core): Milestone 1.1
2. **Phase 2** (Integration): Milestone 1.2
3. **Phase 3** (CLI Trace): Milestone 2.1
4. **Phase 4** (Dashboard): Milestone 3.1, 3.2
5. **Phase 5** (Optional): Milestone 4.1

Estimated effort: 2-3 development sessions
