# Acceptance Criteria: SPEC-RAG-MONITOR-001

## Overview

This document defines the test scenarios and acceptance criteria for the RAG Real-time Monitoring System.

---

## Test Scenarios

### Scenario 1: CLI Trace Output

**Given** the RAG system is properly configured
**And** the user runs `uv run regulation search "휴학 절차" --trace`
**When** the query is processed through the RAG pipeline
**Then** the terminal displays:
- Query analysis section with original and rewritten query
- Vector search results with similarity scores
- Reranking results with score changes
- LLM generation progress with model information
- Final answer with generation statistics

```
Feature: CLI trace output for RAG pipeline monitoring

  Scenario: Display complete trace for standard query
    Given the RAG pipeline is operational
    When I run "uv run regulation search '휴학 절차' --trace"
    Then I should see "[TRACE] Query Analysis" section
    And I should see the original query "휴학 절차"
    And I should see "[TRACE] Vector Search" section
    And I should see search results with similarity scores
    And I should see "[TRACE] Reranking" section
    And I should see "[TRACE] LLM Generation" section
    And I should see the final answer
    And the trace output should not exceed 10% overhead
```

---

### Scenario 2: CLI Monitor Dashboard Launch

**Given** the RAG system is properly configured
**And** port 7860 is available
**When** the user runs `uv run regulation search "휴학 절차" --monitor`
**Then** a Gradio dashboard starts on localhost:7860
**And** the URL is printed to the terminal
**And** a browser tab opens automatically
**And** the dashboard shows a "Live Monitor" tab

```
Feature: CLI monitor dashboard launch

  Scenario: Launch dashboard with --monitor flag
    Given port 7860 is available
    When I run "uv run regulation search '휴학 절차' --monitor"
    Then a Gradio server should start
    And I should see "Monitor dashboard: http://localhost:7860"
    And a browser should open to the dashboard URL
    And the "Live Monitor" tab should be visible

  Scenario: Dashboard shows real-time events
    Given the monitor dashboard is running
    When I submit a query through the dashboard
    Then I should see events appear in the Live Stream panel
    And events should appear within 100ms of occurrence
    And the event stream should be ordered by timestamp
```

---

### Scenario 3: Event Emission and Logging

**Given** the RAGInteractionLogger is initialized
**When** a query flows through the RAG pipeline
**Then** the following events are emitted in order:
1. QueryReceived
2. QueryRewritten (if rewriting occurs)
3. SearchCompleted
4. RerankingCompleted
5. LLMGenerationStarted
6. TokenGenerated (multiple, during streaming)
7. AnswerGenerated

```
Feature: Event emission for RAG pipeline stages

  Scenario: Emit all required events in correct order
    Given the event system is initialized
    And a trace listener is subscribed
    When I process the query "등록금 납부 기간"
    Then I should receive event "QueryReceived"
    And I should receive event "QueryRewritten"
    And I should receive event "SearchCompleted"
    And I should receive event "RerankingCompleted"
    And I should receive event "LLMGenerationStarted"
    And I should receive at least one "TokenGenerated" event
    And I should receive event "AnswerGenerated"
    And all events should have the same correlation_id

  Scenario: Include correlation ID for traceability
    Given a query is being processed
    When any event is emitted
    Then the event should include a correlation_id
    And all events for the same query should share the correlation_id
    And the correlation_id should be unique per query
```

---

### Scenario 4: Performance Overhead

**Given** the RAG pipeline processes a query
**When** monitoring is enabled (--trace or --monitor)
**Then** the total response time increases by no more than 10%

```
Feature: Minimal performance overhead

  Scenario: Trace flag overhead under 10%
    Given I measure baseline response time without trace
    When I run with --trace flag
    Then the response time should not exceed baseline * 1.10

  Scenario: Monitor flag overhead under 10%
    Given I measure baseline response time without monitor
    When I run with --monitor flag
    Then the response time should not exceed baseline * 1.10

  Scenario: Normal execution has no overhead
    Given I measure baseline response time
    When I run without any monitoring flags
    Then the response time should be within 5% of baseline
```

---

### Scenario 5: JSON Log Output

**Given** the RAGInteractionLogger is configured for JSON output
**When** a query is processed
**Then** events are logged to `data/logs/rag_interactions/` in JSON format
**And** each log entry contains timestamp, event_type, correlation_id, and event_data

```
Feature: Structured JSON logging

  Scenario: Log events in JSON format
    Given the logging directory exists
    When I process any query
    Then a JSON log file should be created or appended
    And each entry should have "timestamp" in ISO 8601 format
    And each entry should have "event_type" matching the event name
    And each entry should have "correlation_id" as UUID string
    And each entry should have "event_data" as an object

  Scenario: Daily log rotation
    Given log files exist for previous days
    When a new query is processed
    Then logs should be written to a file named with today's date
    And previous day logs should remain accessible
```

---

### Scenario 6: Dashboard Live Stream

**Given** the Live Monitor dashboard is open
**When** a query is submitted (via CLI or dashboard)
**Then** the Live Stream panel updates in real-time
**And** events appear with timestamps and formatted content

```
Feature: Dashboard live stream visualization

  Scenario: Display query received event
    Given the dashboard is open
    When a "QueryReceived" event occurs
    Then the Live Stream should show "Query Received"
    And it should display the query text
    And it should show the timestamp

  Scenario: Display search results
    Given the dashboard is open
    When a "SearchCompleted" event occurs
    Then the Live Stream should show "Search Completed"
    And it should display the result count
    And it should display top 3 results with scores

  Scenario: Display streaming generation
    Given the dashboard is open
    When "TokenGenerated" events occur
    Then the output panel should update with each token
    And the text should appear in a streaming fashion
```

---

### Scenario 7: No Monitoring Flag (Baseline Behavior)

**Given** the user runs `uv run regulation search "query"` without flags
**When** the query is processed
**Then** normal output is displayed
**And** no trace information is shown
**And** no dashboard is launched
**And** performance is not affected by monitoring code

```
Feature: Normal CLI behavior preserved

  Scenario: No flags means normal output
    Given I run "uv run regulation search '휴학 규정'"
    When the command completes
    Then I should see only the answer
    And I should NOT see "[TRACE]" markers
    And no browser should open
    And no dashboard server should start
```

---

## Quality Gates

### TRUST 5 Validation

| Pillar | Criteria | Verification |
|--------|----------|--------------|
| **Tested** | 85%+ code coverage for new modules | pytest --cov |
| **Readable** | Clear naming, English comments | ruff check |
| **Unified** | Consistent formatting | ruff format |
| **Secured** | No sensitive data in logs | Manual review + automated scan |
| **Trackable** | SPEC reference in commits | Conventional commits |

### Coverage Requirements

- `src/rag/infrastructure/logging/events.py`: 90%+
- `src/rag/infrastructure/logging/interaction_logger.py`: 90%+
- `src/rag/interface/cli/trace_handler.py`: 85%+
- `src/rag/interface/web/live_monitor.py`: 80%+

### LSP Quality Gates

- Zero errors
- Zero type errors (mypy strict)
- Zero lint errors (ruff)

---

## Definition of Done

### Functional Completeness

- [ ] --trace flag displays all pipeline stages
- [ ] --monitor flag launches dashboard with Live Monitor tab
- [ ] All 7 event types are emitted correctly
- [ ] Correlation IDs link all events for a query
- [ ] JSON logs are written to correct location
- [ ] Dashboard shows real-time event stream

### Performance Requirements

- [ ] --trace overhead < 10%
- [ ] --monitor overhead < 10%
- [ ] Normal execution overhead < 5%
- [ ] Event emission latency < 50ms

### Quality Requirements

- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] Code coverage >= 85%
- [ ] Zero ruff errors
- [ ] Zero mypy errors

### Documentation

- [ ] README updated with --trace and --monitor usage
- [ ] Code comments for event types
- [ ] Dashboard usage instructions

---

## Test Commands

```bash
# Run unit tests for monitoring modules
pytest tests/unit/infrastructure/logging/ -v

# Run integration tests for CLI flags
pytest tests/integration/cli/test_monitoring.py -v

# Run coverage check
pytest --cov=src/rag/infrastructure/logging --cov=src/rag/interface/cli --cov-report=term-missing

# Run type check
mypy src/rag/infrastructure/logging/ --strict

# Run lint check
ruff check src/rag/infrastructure/logging/ src/rag/interface/cli/

# Manual test: trace output
uv run regulation search "휴학 절차" --trace

# Manual test: monitor dashboard
uv run regulation search "휴학 절차" --monitor
```

---

## Known Limitations

1. **Single User**: Dashboard designed for single-user local monitoring, not multi-user production use.

2. **Localhost Only**: Dashboard accessible only on localhost for security reasons.

3. **No Authentication**: Dashboard has no authentication as it's a development/debugging tool.

4. **Memory Usage**: Event history in dashboard is limited to last 100 events to prevent memory issues.

5. **Log Rotation**: Logs are rotated daily; older logs must be manually archived or deleted.
