# RAG Quality Metrics Module

Metrics calculation and analysis for RAG quality evaluation.

## Core Metrics

### 1. Precision (정밀도)

**Definition**: Proportion of retrieved information that is correct.

```python
def calculate_precision(
    actual_answer: str,
    sources: List[Dict],
    judge_evaluation: Dict
) -> float:
    """
    Calculate precision: correct_info / total_info_provided

    Precision measures how much of the provided information is accurate.
    High precision means the system doesn't provide incorrect information.
    """
    # Extract information points from answer
    answer_points = extract_information_points(actual_answer)

    # Count correct points (from judge evaluation)
    correct_points = judge_evaluation.get("correct_info_count", 0)

    if not answer_points:
        return 0.0

    precision = correct_points / len(answer_points)
    return round(precision, 3)
```

**Target**: >= 0.84 (84%)
**Why Important**: Low precision indicates hallucinations or incorrect information.

---

### 2. Recall (재현율)

**Definition**: Proportion of required information that is provided.

```python
def calculate_recall(
    expected_answer: Dict,
    actual_answer: str,
    judge_evaluation: Dict
) -> float:
    """
    Calculate recall: provided_required_info / total_required_info

    Recall measures how much of the required information is covered.
    High recall means the system provides comprehensive answers.
    """
    # Get required information points from expected answer
    required_points = set(expected_answer.get("required_info", []))

    if not required_points:
        return 1.0  # No required info means full coverage

    # Check which required points are in actual answer
    covered_points = 0
    for point in required_points:
        if point.lower() in actual_answer.lower():
            covered_points += 1

    recall = covered_points / len(required_points)
    return round(recall, 3)
```

**Target**: >= 0.79 (79%)
**Why Important**: Low recall indicates missing key information.

---

### 3. F1 Score

**Definition**: Harmonic mean of precision and recall.

```python
def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score: 2 * (precision * recall) / (precision + recall)

    F1 provides a single metric balancing precision and recall.
    """
    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 3)
```

**Target**: >= 0.81 (81%)
**Why Important**: Balances precision and recall for overall quality.

---

### 4. Context Relevance (문맥 관련성)

**Definition**: Average relevance score of retrieved sources.

```python
def calculate_context_relevance(sources: List[Dict]) -> float:
    """
    Calculate context relevance: avg_source_relevance

    Measures how relevant the retrieved sources are to the query.
    Higher weight given to top-ranked sources.
    """
    if not sources:
        return 0.0

    relevance_scores = []
    for i, source in enumerate(sources):
        # Base relevance score
        score = source.get("score", 0.0)

        # Position weight: top sources matter more
        position_weight = 1.0 / (1 + i * 0.1)

        # Adjusted score
        relevance_scores.append(score * position_weight)

    return round(sum(relevance_scores) / len(relevance_scores), 3)
```

**Target**: >= 0.75 (75%)
**Why Important**: Low context relevance indicates poor retrieval.

---

## Persona Metrics

### Metrics by Persona Type

```python
def calculate_persona_metrics(
    evaluations: List[Dict],
    persona: str
) -> Dict[str, float]:
    """
    Calculate metrics for a specific persona.
    """
    persona_evals = [e for e in evaluations if e.get("persona") == persona]

    if not persona_evals:
        return {}

    return {
        "total_queries": len(persona_evals),
        "passed": sum(1 for e in persona_evals if e.get("passed")),
        "pass_rate": sum(1 for e in persona_evals if e.get("passed")) / len(persona_evals),
        "avg_precision": sum(e.get("precision", 0) for e in persona_evals) / len(persona_evals),
        "avg_recall": sum(e.get("recall", 0) for e in persona_evals) / len(persona_evals),
        "avg_f1": sum(e.get("f1", 0) for e in persona_evals) / len(persona_evals),
        "avg_context_relevance": sum(e.get("context_relevance", 0) for e in persona_evals) / len(persona_evals)
    }
```

### Persona Performance Comparison

| Persona | Avg Precision | Avg Recall | Avg F1 | Pass Rate |
|---------|--------------|------------|--------|-----------|
| student-undergraduate | 0.82 | 0.77 | 0.79 | 75% |
| student-graduate | 0.86 | 0.81 | 0.83 | 82% |
| professor | 0.88 | 0.83 | 0.85 | 85% |
| staff-admin | 0.84 | 0.79 | 0.81 | 80% |
| parent | 0.80 | 0.76 | 0.78 | 73% |
| student-international | 0.79 | 0.74 | 0.76 | 70% |

**Analysis**:
- **Best Performance**: professor (highest precision and recall)
- **Worst Performance**: student-international (language barrier)
- **Improvement Opportunity**: parent and international student personas

---

## Scenario Category Metrics

### Metrics by Scenario Category

```python
def calculate_category_metrics(
    evaluations: List[Dict],
    category: str
) -> Dict[str, float]:
    """
    Calculate metrics for a specific scenario category.
    """
    category_evals = [e for e in evaluations if e.get("category") == category]

    if not category_evals:
        return {}

    return {
        "total_queries": len(category_evals),
        "passed": sum(1 for e in category_evals if e.get("passed")),
        "pass_rate": sum(1 for e in category_evals if e.get("passed")) / len(category_evals),
        "avg_overall_score": sum(e.get("overall_score", 0) for e in category_evals) / len(category_evals)
    }
```

### Category Performance Comparison

| Category | Total | Passed | Pass Rate | Avg Score |
|----------|-------|--------|-----------|-----------|
| Simple | 30 | 27 | 90% | 0.89 |
| Complex | 25 | 20 | 80% | 0.82 |
| Multi-turn | 20 | 15 | 75% | 0.78 |
| Edge Cases | 40 | 28 | 70% | 0.71 |
| Domain-Specific | 25 | 22 | 88% | 0.86 |
| Adversarial | 10 | 9 | 90% | 0.91 |

**Analysis**:
- **Best Performance**: Adversarial (good at rejecting invalid queries)
- **Worst Performance**: Edge Cases (struggles with ambiguity)
- **Improvement Opportunity**: Edge cases and multi-turn conversations

---

## Trend Analysis

### Compare to Baseline

```python
def compare_to_baseline(
    current_metrics: Dict,
    baseline_metrics: Dict
) -> Dict[str, Any]:
    """
    Compare current evaluation to baseline.
    """
    comparison = {}

    for metric in ["precision", "recall", "f1", "context_relevance", "pass_rate"]:
        current = current_metrics.get(metric, 0)
        baseline = baseline_metrics.get(metric, 0)
        delta = current - baseline

        comparison[metric] = {
            "current": round(current, 3),
            "baseline": round(baseline, 3),
            "delta": round(delta, 3),
            "improved": delta > 0,
            "percent_change": round((delta / baseline * 100) if baseline > 0 else 0, 1)
        }

    return comparison
```

### Trend Classification

| Delta | Classification | Action |
|-------|---------------|--------|
| +5% or more | Significant Improvement | Document success factors |
| +2% to +5% | Moderate Improvement | Continue current approach |
| -2% to +2% | Stable | Monitor closely |
| -2% to -5% | Moderate Regression | Investigate causes |
| -5% or more | Significant Regression | Immediate action required |

---

## Failure Pattern Analysis

### Failure Type Classification

```python
def classify_failure(evaluation: Dict) -> List[str]:
    """
    Classify failure types from evaluation.
    """
    failures = []

    # Check for hallucination
    if evaluation.get("accuracy", 1.0) < 0.5:
        failures.append("hallucination")

    # Check for missing information
    if evaluation.get("completeness", 1.0) < 0.7:
        failures.append("missing_info")

    # Check for wrong regulation
    if evaluation.get("citations", 1.0) < 0.5:
        failures.append("wrong_regulation")

    # Check for context mismatch
    if evaluation.get("context_relevance", 1.0) < 0.6:
        failures.append("context_mismatch")

    # Check for generic avoidance
    answer = evaluation.get("answer", "")
    if any(phrase in answer for phrase in ["대학마다 다릅니다", "확인해주세요"]):
        failures.append("generic_avoidance")

    return failures
```

### Failure Patterns Summary

| Failure Type | Count | Percentage | Top Queries |
|-------------|-------|-----------|-------------|
| Hallucination | 8 | 5.3% | 학생회비 납부, 장학금 담당자 |
| Missing Info | 12 | 8.0% | 복수전공, 부전공 |
| Wrong Regulation | 5 | 3.3% | 연구비 집행, 연간차 |
| Context Mismatch | 4 | 2.7% | 모호한 단일어 쿼리 |
| Generic Avoidance | 3 | 2.0% | 구체적 연락처 요청 |

**Priority Mapping**:
- **High Priority**: Hallucination (8 queries), Missing Info (12 queries)
- **Medium Priority**: Wrong Regulation (5 queries)
- **Low Priority**: Context Mismatch (4 queries), Generic Avoidance (3 queries)

---

## Metric Thresholds

### Quality Gates

```python
QUALITY_THRESHOLDS = {
    "overall": {
        "excellent": 0.90,  # 90%+
        "good": 0.85,       # 85-89%
        "acceptable": 0.80, # 80-84%
        "poor": 0.70,       # 70-79%
        "failing": 0.0      # <70%
    },
    "accuracy": {
        "excellent": 0.95,
        "good": 0.90,
        "acceptable": 0.85,
        "poor": 0.70,
        "failing": 0.0
    },
    "completeness": {
        "excellent": 0.90,
        "good": 0.85,
        "acceptable": 0.75,
        "poor": 0.60,
        "failing": 0.0
    },
    "citations": {
        "excellent": 0.90,
        "good": 0.80,
        "acceptable": 0.70,
        "poor": 0.50,
        "failing": 0.0
    },
    "context_relevance": {
        "excellent": 0.90,
        "good": 0.85,
        "acceptable": 0.75,
        "poor": 0.60,
        "failing": 0.0
    }
}
```

### Pass/Fail Determination

```python
def determine_pass_fail(evaluation: Dict) -> Dict[str, Any]:
    """
    Determine if evaluation passes quality gates.
    """
    overall = evaluation.get("overall_score", 0)
    accuracy = evaluation.get("accuracy", 0)
    completeness = evaluation.get("completeness", 0)
    citations = evaluation.get("citations", 0)
    context_relevance = evaluation.get("context_relevance", 0)

    # Must meet ALL thresholds
    passed = (
        overall >= 0.80 and
        accuracy >= 0.85 and
        completeness >= 0.75 and
        citations >= 0.70 and
        context_relevance >= 0.75
    )

    # Check each metric
    metric_status = {
        "overall": overall >= 0.80,
        "accuracy": accuracy >= 0.85,
        "completeness": completeness >= 0.75,
        "citations": citations >= 0.70,
        "context_relevance": context_relevance >= 0.75
    }

    # Count failures
    failing_metrics = [m for m, status in metric_status.items() if not status]

    # Determine quality level
    if overall >= 0.90:
        quality_level = "excellent"
    elif overall >= 0.85:
        quality_level = "good"
    elif overall >= 0.80:
        quality_level = "acceptable"
    elif overall >= 0.70:
        quality_level = "poor"
    else:
        quality_level = "failing"

    return {
        "passed": passed,
        "quality_level": quality_level,
        "metric_status": metric_status,
        "failing_metrics": failing_metrics,
        "num_failing_metrics": len(failing_metrics)
    }
```

---

## Regression Detection

### Detect Regressions

```python
def detect_regressions(
    current_evals: List[Dict],
    baseline_evals: List[Dict]
) -> List[Dict]:
    """
    Detect queries that regressed from baseline.
    """
    # Create baseline lookup
    baseline_lookup = {
        e["query_id"]: e for e in baseline_evals
    }

    regressions = []

    for current in current_evals:
        query_id = current.get("query_id")
        baseline = baseline_lookup.get(query_id)

        if not baseline:
            continue

        # Check if previously passing query now fails
        if baseline.get("passed") and not current.get("passed"):
            regressions.append({
                "query_id": query_id,
                "query": current.get("query"),
                "baseline_score": baseline.get("overall_score"),
                "current_score": current.get("overall_score"),
                "score_delta": current.get("overall_score") - baseline.get("overall_score"),
                "baseline_passed": True,
                "current_passed": False
            })

    return regressions
```

### Regression Severity

| Severity | Score Drop | Count | Action |
|----------|-----------|-------|--------|
| Critical | >0.20 | Any | Block deployment, investigate immediately |
| High | 0.10-0.20 | >5 | Block deployment, investigate |
| Medium | 0.05-0.10 | >10 | Investigate before next deployment |
| Low | <0.05 | Any | Monitor, accept if overall quality improved |

---

## Improvement Tracking

### Track Improvement Over Time

```python
def calculate_improvement_trend(
    evaluations_history: List[List[Dict]]
) -> Dict[str, Any]:
    """
    Calculate improvement trend across multiple evaluation runs.
    """
    if len(evaluations_history) < 2:
        return {"error": "Need at least 2 evaluation runs"}

    trends = {
        "overall_score": [],
        "pass_rate": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "context_relevance": []
    }

    for evals in evaluations_history:
        overall = sum(e.get("overall_score", 0) for e in evals) / len(evals)
        pass_rate = sum(1 for e in evals if e.get("passed")) / len(evals)
        precision = sum(e.get("precision", 0) for e in evals) / len(evals)
        recall = sum(e.get("recall", 0) for e in evals) / len(evals)
        f1 = sum(e.get("f1", 0) for e in evals) / len(evals)
        context = sum(e.get("context_relevance", 0) for e in evals) / len(evals)

        trends["overall_score"].append(overall)
        trends["pass_rate"].append(pass_rate)
        trends["precision"].append(precision)
        trends["recall"].append(recall)
        trends["f1"].append(f1)
        trends["context_relevance"].append(context)

    # Calculate trend direction
    trend_direction = {}
    for metric, values in trends.items():
        if len(values) >= 2:
            recent = values[-1]
            previous = values[-2]
            change = recent - previous
            if change > 0.02:
                trend_direction[metric] = "improving"
            elif change < -0.02:
                trend_direction[metric] = "regressing"
            else:
                trend_direction[metric] = "stable"

    return {
        "trends": trends,
        "trend_direction": trend_direction
    }
```

### Trend Visualization

For trend visualization, metrics can be exported to JSON for plotting:

```json
{
  "timestamps": ["2025-01-01", "2025-01-07", "2025-01-14"],
  "overall_score": [0.75, 0.79, 0.81],
  "pass_rate": [0.65, 0.72, 0.81],
  "precision": [0.78, 0.82, 0.84],
  "recall": [0.73, 0.77, 0.79],
  "f1": [0.75, 0.79, 0.81],
  "context_relevance": [0.76, 0.80, 0.83]
}
```
