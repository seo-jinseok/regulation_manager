# RAG Quality Evaluation Module

LLM-as-Judge evaluation system for RAG quality assessment.

## Evaluation Pipeline

### Step 1: Query Execution

Execute queries through the RAG CLI:

```bash
# Standard query execution
regulation ask "{query}" --format json --output /tmp/rag_response.json

# With specific options
regulation ask "{query}" --top-k 5 --use-rerank --format json
```

Expected response format:
```json
{
  "answer": "휴학은 학칙 제15조에 따라 신청할 수 있습니다...",
  "sources": [
    {
      "regulation": "학칙",
      "article": "제15조",
      "title": "휴학",
      "content": "제15조(휴학) ① 학생은 질병...",
      "score": 0.89
    }
  ],
  "confidence": 0.87,
  "metadata": {
    "model": "gpt-4o",
    "timestamp": "2025-01-07T14:05:30Z"
  }
}
```

### Step 2: LLM-as-Judge Evaluation

#### Judge Prompt Template

```python
judge_prompt = f"""
You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Evaluate the quality of the following RAG response based on the criteria below.

## Query
{query}

## Expected Answer
{expected_answer}

## Actual Answer
{actual_answer}

## Retrieved Sources
{sources}

## Evaluation Criteria

### 1. Accuracy (0.0-1.0)
Is the answer factually correct without hallucination?
- Score 1.0: Completely accurate, no errors or hallucinations
- Score 0.8-0.9: Minor inaccuracies, no significant hallucinations
- Score 0.5-0.7: Some factual errors or minor hallucinations
- Score 0.3-0.4: Significant factual errors or hallucinations
- Score 0.0-0.2: Major hallucinations or completely incorrect

**Hallucination Indicators**:
- Fake phone numbers (02-XXXX-XXXX pattern)
- Information not in sources
- Wrong university regulations (e.g., "서울대", "한국외대")
- Generic avoidance: "대학마다 다릅니다", "확인해주세요"

### 2. Completeness (0.0-1.0)
Does the answer cover all key information points?
- Score 1.0: All required information present
- Score 0.8-0.9: Most information present, minor omissions
- Score 0.5-0.7: Significant information missing
- Score 0.3-0.4: Major gaps in information
- Score 0.0-0.2: Very incomplete

**Required Information**:
{required_info}

### 3. Citations (0.0-1.0)
Are regulation references accurate and properly formatted?
- Score 1.0: Accurate "규정명 + 제X조" format, multiple relevant citations
- Score 0.8-0.9: Generally accurate citations, minor formatting issues
- Score 0.5-0.7: Some citations accurate, others incorrect or missing
- Score 0.3-0.4: Few or inaccurate citations
- Score 0.0-0.2: No citations or completely wrong

**Expected Citation Format**: "학칙 제15조", "등록금 규정 제8조"

### 4. Context Relevance (0.0-1.0)
Are the retrieved sources relevant to the query?
- Score 1.0: All sources highly relevant
- Score 0.8-0.9: Most sources relevant
- Score 0.5-0.7: Mixed relevance
- Score 0.3-0.4: Few relevant sources
- Score 0.0-0.2: No relevant sources

## Output Format

Provide your evaluation in the following JSON format:

```json
{{
  "accuracy": <score 0.0-1.0>,
  "completeness": <score 0.0-1.0>,
  "citations": <score 0.0-1.0>,
  "context_relevance": <score 0.0-1.0>,
  "reasoning": {{
    "accuracy": "<brief explanation>",
    "completeness": "<brief explanation>",
    "citations": "<brief explanation>",
    "context_relevance": "<brief explanation>"
  }},
  "issues": ["<list of specific issues found>"],
  "strengths": ["<list of specific strengths>"]
}}
```

## Special Considerations

1. **Hallucination Detection**: If you detect fake phone numbers, wrong university names, or information not in sources, assign a low accuracy score.

2. **Regulation Format**: Citations should follow "규정명 + 제X조" format. Informal citations like "관련 규정에 따르면" should receive lower scores.

3. **Persona Appropriateness**: Consider if the answer matches the expected persona expertise level (simple language for beginners, formal for professors).

4. **Edge Cases**: For vague queries, clarify if the system asked for clarification or provided a reasonable interpretation.

5. **Multi-turn**: For conversations, check if context was maintained across turns.

Provide your evaluation now.
"""
```

### Step 3: Judge Model Configuration

```python
judge_config = {
    "model": "gpt-4o",  # or os.getenv("RAG_JUDGE_MODEL", "gpt-4o")
    "temperature": 0.0,  # Deterministic evaluation
    "max_tokens": 1000,  # Sufficient for reasoning
    "response_format": {"type": "json_object"}  # Force JSON output
}

# Alternative: Use cheaper model for initial filtering
if cost_sensitive:
    judge_config["model"] = "gpt-4o-mini"  # 10x cheaper
```

### Step 4: Response Parsing

```python
import json
from typing import Dict, Any

def parse_judge_response(response: str) -> Dict[str, Any]:
    """Parse LLM-as-Judge response."""
    try:
        # Extract JSON from response
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()

        evaluation = json.loads(json_str)

        # Validate scores
        for metric in ["accuracy", "completeness", "citations", "context_relevance"]:
            if metric not in evaluation:
                raise ValueError(f"Missing metric: {metric}")
            if not 0.0 <= evaluation[metric] <= 1.0:
                raise ValueError(f"Invalid score for {metric}: {evaluation[metric]}")

        return evaluation

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse judge response: {e}")
```

---

## Evaluation Criteria Details

### Accuracy Scoring Rubric

| Score | Description | Examples |
|-------|-------------|----------|
| 1.0 | Perfect accuracy | All information correct, no hallucinations |
| 0.9 | Minor issues | Insignificant detail errors, no impact on utility |
| 0.8 | Acceptable | Some minor inaccuracies, overall correct |
| 0.7 | Noticeable errors | Several factual errors but core info correct |
| 0.6 | Significant issues | Multiple errors, some misinformation |
| 0.5 | Poor accuracy | Many errors, questionable reliability |
| 0.4 | Major errors | Significant hallucinations or wrong information |
| 0.3 | Very poor | Substantial misinformation |
| 0.2 | Severe issues | Almost entirely incorrect |
| 0.1 | Near failure | Minimal correct information |
| 0.0 | Complete failure | All information wrong or hallucinated |

**Automatic Failure Conditions:**
- Fake phone number detected: Accuracy = 0.0
- Wrong university name (e.g., "서울대", "한국외대"): Accuracy = 0.0
- Generic avoidance without helpful info: Accuracy ≤ 0.3

### Completeness Scoring Rubric

| Score | Coverage | Examples |
|-------|----------|----------|
| 1.0 | 100% | All required points covered |
| 0.9 | 90-99% | Minor omission, non-critical detail |
| 0.8 | 80-89% | One or two minor omissions |
| 0.7 | 70-79% | Several minor omissions or one major omission |
| 0.6 | 60-69% | Noticeable gaps |
| 0.5 | 50-59% | Half of required info missing |
| 0.4 | 40-49% | Significant gaps |
| 0.3 | 30-39% | Major information missing |
| 0.2 | 20-29% | Very incomplete |
| 0.1 | 10-19% | Minimal information |
| 0.0 | 0-9% | Almost no useful information |

**Required Information Calculation:**
```python
required_points = set(expected_answer["required_info"])
covered_points = set()

for point in required_points:
    if point.lower() in actual_answer.lower():
        covered_points.add(point)

completeness = len(covered_points) / len(required_points)
```

### Citations Scoring Rubric

| Score | Description | Examples |
|-------|-------------|----------|
| 1.0 | Perfect | "학칙 제15조에 따라", "등록금 규정 제8조" |
| 0.9 | Excellent | Multiple accurate citations, minor format issues |
| 0.8 | Good | Accurate citations present |
| 0.7 | Acceptable | Some citations, others missing |
| 0.6 | Fair | Few or generic citations |
| 0.5 | Poor | Inaccurate citation format |
| 0.4 | Weak | Wrong citation references |
| 0.3 | Very weak | Minimal citation attempt |
| 0.2 | Bad | Incorrect citations |
| 0.1 | Near failure | Single generic mention |
| 0.0 | No citations | No regulation references |

**Citation Quality Checks:**
```python
def check_citation_quality(answer: str) -> float:
    """Check citation format and accuracy."""
    # Perfect: "규정명 + 제X조"
    if re.search(r'\w+规[정|程]\s*제\d+조', answer):
        return 1.0

    # Good: Has both regulation name and article
    if '규정' in answer and '제' in answer and '조' in answer:
        return 0.8

    # Fair: Has regulation name or article
    if '규정' in answer or ('제' in answer and '조' in answer):
        return 0.5

    # Poor: Generic mention
    if '관련' in answer and ('규정' in answer or '조' in answer):
        return 0.3

    # No citation
    return 0.0
```

### Context Relevance Scoring

```python
def calculate_context_relevance(sources: List[Dict]) -> float:
    """Calculate average source relevance."""
    if not sources:
        return 0.0

    relevance_scores = []
    for source in sources:
        score = source.get("score", 0.0)
        # Weight by position (top sources matter more)
        position_weight = 1.0 / (1 + sources.index(source) * 0.1)
        relevance_scores.append(score * position_weight)

    return sum(relevance_scores) / len(relevance_scores)
```

---

## Overall Quality Calculation

```python
def calculate_overall_quality(evaluation: Dict[str, float]) -> Dict[str, Any]:
    """Calculate overall quality score with weighted metrics."""

    # Weights
    weights = {
        "accuracy": 0.35,
        "completeness": 0.25,
        "citations": 0.20,
        "context_relevance": 0.20
    }

    # Calculate weighted score
    overall = sum(
        evaluation[metric] * weight
        for metric, weight in weights.items()
    )

    # Pass determination
    passed = overall >= 0.80

    # Individual metric pass/fail
    metric_thresholds = {
        "accuracy": 0.85,
        "completeness": 0.75,
        "citations": 0.70,
        "context_relevance": 0.75
    }

    metric_status = {
        metric: evaluation[metric] >= threshold
        for metric, threshold in metric_thresholds.items()
    }

    return {
        "overall_score": round(overall, 3),
        "passed": passed,
        "metric_scores": evaluation,
        "metric_status": metric_status,
        "weights": weights
    }
```

---

## Pass/Fail Logic

### Pass Conditions
All of the following must be met:
1. Overall score >= 0.80 (80%)
2. Accuracy >= 0.85 (no hallucinations)
3. Completeness >= 0.75 (key information present)
4. Citations >= 0.70 (regulation references)
5. Context Relevance >= 0.75 (relevant sources)

### Fail Conditions
Any of the following:
1. Overall score < 0.80
2. Any metric below its threshold
3. Hallucination detected (fake phone, wrong university)
4. Generic avoidance without helpful information

### Edge Case Handling

```python
def handle_edge_cases(query: str, answer: str) -> Optional[str]:
    """Handle special edge cases."""

    # Vague single-word queries
    if len(query.split()) == 1:
        if "무엇" in answer or "어떤" in answer or "관련" in answer:
            # System asked for clarification - this is good
            return None  # No penalty
        # If system answered without clarification, check relevance
        return "check_relevance"

    # Adversarial queries (wrong university)
    if "서울대" in query or "한국외대" in query or "연세대" in query:
        if query.replace("서울대", "").replace("한국외대", "").replace("연세대", "") in answer:
            # System answered about wrong university - FAIL
            return "hallucination_detected"

    # Contact queries
    if "연락처" in query or "전화" in query or "번호" in query:
        if re.search(r'02-\d{3,4}-\d{4}', answer):
            # Check if number is fake (common patterns)
            return "check_hallucination"

    return None
```
