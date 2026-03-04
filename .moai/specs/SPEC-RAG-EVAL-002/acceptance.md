# SPEC-RAG-EVAL-002 Acceptance Criteria

## Module 1: Dynamic Query Universe

### AC-1.1: Regulation-Based Query Synthesis (EARS-U-001)

**Given** regulation JSON files exist in `data/output/`
**When** the evaluator runs with `--generate` flag
**Then**
- At least 200 unique queries are generated
- Each query links to a source regulation file
- Query categories include: definition, procedure, eligibility, cross-reference, calculation
- Generated queries are cached in `data/evaluations/generated_queries.json`
- Subsequent runs reuse cached queries unless `--regenerate` is specified

### AC-1.2: Cross-Regulation Queries (EARS-U-002)

**Given** the evaluator generates queries
**When** cross-regulation queries are created
**Then**
- At least 10% of total queries require information from 2+ regulation files
- Each cross-regulation query specifies the expected source regulations
- Difficulty tier is L3 or higher for all cross-regulation queries

### AC-1.3: Adversarial Queries (EARS-U-003)

**Given** the evaluator generates queries
**When** adversarial queries are created
**Then**
- Queries include: non-existent regulation references, contradictory premises, ambiguous terms, out-of-scope topics
- The system correctly identifies unanswerable queries (expected: "no relevant regulation found" type response)
- Difficulty tier is L4 or L5 for all adversarial queries

---

## Module 2: Extended Quality Metrics

### AC-2.1: Latency Tracking (EARS-U-004)

**Given** a query evaluation completes
**When** latency is measured
**Then**
- Response time is recorded for each query in milliseconds
- Summary includes p50, p95, p99 percentiles
- Queries exceeding 10 seconds are flagged as SLOW

### AC-2.2: Answer Consistency (EARS-U-005)

**Given** the `--consistency` flag is provided
**When** the same query is submitted 3 times
**Then**
- Semantic similarity is computed between all response pairs
- Consistency score = minimum pairwise similarity
- Score ≥ 0.85 is PASS, < 0.85 is FAIL
- Inconsistent queries are reported with diff highlights

### AC-2.3: Citation Verification (EARS-U-006)

**Given** a response includes article citations (e.g., "제3조", "제15조의2")
**When** citation verification runs
**Then**
- Each cited article is checked against ChromaDB for existence
- Verification rate ≥ 0.95 is PASS
- Non-existent citations (hallucinated) are explicitly listed
- Score = (verified citations / total citations)

### AC-2.4: Readability Score (EARS-U-007)

**Given** a response is generated
**When** readability is evaluated
**Then**
- Structure score: has clear paragraphs, uses formatting (headers, lists)
- Completeness score: addresses the query directly, provides context
- Score ≥ 0.70 is PASS
- Specific feedback provided when score < 0.70

---

## Module 3: System Health Radar

### AC-3.1: Code Quality Scan (EARS-U-008)

**Given** the `--health` flag is provided
**When** code quality scan runs
**Then**
- Detects bare `except: pass` blocks via AST parsing
- Counts TODO/FIXME comments
- Identifies hardcoded magic numbers in core logic
- Reports file:line locations for each finding
- Known findings list prevents regression (new findings = WARNING)

### AC-3.2: Test Coverage Delta (EARS-U-009)

**Given** `coverage.json` exists in project root
**When** coverage analysis runs
**Then**
- Current overall coverage percentage is reported
- Delta from last evaluation is computed
- Files with coverage < 85% are listed
- Coverage decrease triggers WARNING

### AC-3.3: Config Drift Detection (EARS-U-010)

**Given** environment and configuration files exist
**When** config drift analysis runs
**Then**
- Required environment variables are validated as present
- Known defaults (temperature, top_k, etc.) are checked against expected ranges
- Missing or out-of-range configs are reported as DRIFT_DETECTED

---

## Module 4: Adaptive Difficulty

### AC-4.1: Difficulty Tier System (EARS-U-011)

**Given** queries have been generated or loaded
**When** tier assignment runs
**Then**
- All queries have a tier L1-L5 assigned
- L1: Single-fact lookup queries
- L2: Multi-fact synthesis within one regulation
- L3: Cross-regulation comparison queries
- L4: Edge cases and ambiguous queries
- L5: Adversarial and unanswerable queries
- Tier distribution is balanced (no tier < 10% of total unless insufficient data)

### AC-4.2: Auto-Escalation (EARS-U-012)

**Given** evaluation results for a tier show ≥ 95% pass rate
**When** the difficulty manager runs
**Then**
- The next higher tier is unlocked for testing
- Mastery status is displayed: "L1 ✓ | L2 ✓ | L3 78% | L4 -- | L5 --"
- Previous tier mastery is persisted in `data/evaluations/difficulty_state.json`
- Re-running without `--regenerate` uses persisted state

---

## Module 5: Improvement Roadmap

### AC-5.1: Failure Pattern Clustering (EARS-U-013)

**Given** evaluation results include failed queries
**When** failure clustering runs
**Then**
- Failures are grouped by root cause pattern (e.g., "retrieval_miss", "citation_hallucination", "wrong_audience")
- Each cluster has ≥ 2 example queries
- Cluster names are human-readable

### AC-5.2: Prioritized Roadmap (EARS-U-014)

**Given** failure clusters and system health data are available
**When** roadmap generation runs
**Then**
- Improvement items are sorted by priority score
- Each item includes: description, affected file:line, estimated complexity (S/M/L), suggested SPEC reference
- Top 5 improvements are highlighted in summary

### AC-5.3: Quality Trend Analysis (EARS-U-015)

**Given** ≥ 3 historical evaluation JSONs exist in `data/evaluations/`
**When** trend analysis runs with `--trend` flag
**Then**
- Per-metric trend direction (improving/stable/declining) is computed
- Statistical significance is noted when ≥ 5 data points exist
- Trend chart data exported in JSON format for visualization

---

## Non-Functional Requirements

### AC-NF-1: Backward Compatibility (EARS-E-001)

**Given** a user runs the evaluator with only `--quick` flag (no new flags)
**When** the evaluation completes
**Then**
- Behavior is identical to current codebase
- Same 30 queries, same 4 metrics, same JSON output format
- No new dependencies are required for basic mode

### AC-NF-2: Execution Time (EARS-E-002)

**Given** the evaluator runs with `--full` flag
**When** all modules execute
**Then**
- Total wall-clock time < 30 minutes
- Progress indicator shows current phase
- Partial results saved if interrupted

### AC-NF-3: Persistent Results (EARS-E-003)

**Given** an evaluation completes (any mode)
**When** results are saved
**Then**
- JSON file saved to `data/evaluations/eval_YYYYMMDD_HHMMSS.json`
- JSON schema includes all metric types from all modules
- Historical files are preserved (never overwritten)

### AC-NF-4: Never Nothing To Improve (EARS-E-004)

**Given** the evaluation completes with ALL metrics passing at 100%
**When** the improvement roadmap is generated
**Then**
- At least one improvement recommendation is always produced
- Recommendations may come from: system health, untested tiers, trend analysis, consistency gaps, new regulation coverage
- The report NEVER contains "nothing to improve" or equivalent

---

## Edge Case Scenarios

### EC-1: No Regulation Files

**Given** `data/output/` is empty
**When** `--generate` is used
**Then** evaluator falls back to hardcoded 30 queries with WARNING message

### EC-2: No Historical Data

**Given** `data/evaluations/` has no historical JSONs
**When** `--trend` is used
**Then** trend section shows "Insufficient data for trend analysis (need ≥ 3 runs)"

### EC-3: ChromaDB Unavailable

**Given** ChromaDB is not initialized
**When** citation verification runs
**Then** citation verification is skipped with WARNING, other metrics continue

### EC-4: LLM API Failure

**Given** OpenRouter API returns error during query generation
**When** query synthesis runs
**Then** synthesizer uses cached queries if available, else falls back to hardcoded queries
