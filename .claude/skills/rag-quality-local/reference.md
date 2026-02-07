# RAG Quality Local - Reference Guide

External resources, related work, and references for RAG quality evaluation.

## Table of Contents

1. [Related Projects](#related-projects)
2. [Evaluation Frameworks](#evaluation-frameworks)
3. [Academic References](#academic-references)
4. [Tools and Libraries](#tools-and-libraries)
5. [Best Practices](#best-practices)
6. [Community Resources](#community-resources)

---

## Related Projects

### rag-quality (Existing Skill)

**Location**: `.claude/skills/rag-quality/SKILL.md`

**Relationship**: Complementary

**Key Differences**:
- `rag-quality`: Basic RAG evaluation, RAGAS integration, basic personas
- `rag-quality-local`: Advanced skill with sub-agent personas, 150+ scenarios, LLM-as-Judge, SPEC generation

**When to Use Each**:
- Use `rag-quality` for: Quick evaluations, basic quality checks, dashboard monitoring
- Use `rag-quality-local` for: Comprehensive testing, persona simulation, improvement SPEC generation

### test_scenarios/rag_quality_evaluator.py

**Location**: `test_scenarios/rag_quality_evaluator.py`

**Purpose**: Standalone evaluation script with fixed test queries

**Key Features**:
- 30 predefined test queries
- Basic scoring (intent, answer, UX)
- Command-line execution
- Markdown report generation

**Integration with rag-quality-local**:
- Can import test queries as baseline scenarios
- Scoring logic can inform LLM-as-Judge prompts
- Report format compatible with evaluation storage

---

## Evaluation Frameworks

### RAGAS

**Version**: ragas>=0.4.3

**Key Metrics**:
- **Faithfulness**: Measures factual consistency of generated answer against retrieved context
- **Answer Relevancy**: Assesses how relevant the answer is to the query
- **Context Precision**: Measures signal-to-noise ratio in retrieved context
- **Context Recall**: Measures if all relevant information was retrieved

**Documentation**: [RAGAS Documentation](https://docs.ragas.io/)

**Integration**:
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
```

### DeepEval

**Version**: deepeval>=3.8.1

**Key Features**:
- Alternative LLM-as-Judge framework
- More customizable evaluation criteria
- Support for multi-turn conversations
- Integration with CI/CD pipelines

**Documentation**: [DeepEval Documentation](https://docs.confident-ai.com/)

**Integration**:
```python
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

metric = AnswerRelevancyMetric(threshold=0.8)
metric.measure(test_case, actual_output)
```

---

## Academic References

### RAG Evaluation Papers

1. **RAGAS: Framework for RAG Evaluation**
   - Authors: RAGAS Team
   - Link: [RAGAS Paper](https://arxiv.org/abs/2309.15217)
   - Key Insights: Metric definitions, evaluation methodology

2. **Evaluating RAG Systems**
   - Authors: Various
   - Survey of RAG evaluation approaches
   - Comparison: Human evaluation vs LLM-as-Judge

3. **Hallucination Detection in RAG**
   - Techniques for detecting factual inconsistencies
   - Faithfulness scoring methods
   - Context relevance measurement

### Persona-Based Testing

1. **User Persona Simulation for QA**
   - Benefits: Realistic usage patterns
   - Challenges: Accurate persona modeling
   - Best Practices: Multi-persona coverage

2. **Multi-Turn Conversation Evaluation**
   - Context maintenance metrics
   - Conversation flow assessment
   - Turn-by-turn evaluation

---

## Tools and Libraries

### Core Dependencies

```toml
[project.dependencies]
ragas = ">=0.4.3"           # RAGAS evaluation framework
deepeval = ">=3.8.1"        # Alternative evaluation framework
openai = ">=2.11.0"         # GPT-4o for LLM-as-Judge
llama-index = ">=0.14.10"   # RAG system integration
chromadb = ">=1.4.0"        # Vector store
```

### CLI Integration

```bash
# RAG query execution
regulation ask "{query}" --format json

# Evaluation run
python -m rag_quality_local --full

# Targeted evaluation
python -m rag_quality_local --persona professor --category edge_cases
```

---

## Best Practices

### LLM-as-Judge Guidelines

1. **Model Selection**
   - Primary: GPT-4o (best quality)
   - Alternative: GPT-4o-mini (10x cheaper, slightly lower quality)
   - Fallback: Claude Sonnet (if OpenAI unavailable)

2. **Prompt Engineering**
   - Clear scoring criteria (0.0-1.0 scale)
   - Examples for each score level
   - Structured JSON output format
   - Context-specific instructions

3. **Consistency**
   - Temperature = 0.0 for deterministic evaluation
   - Consistent prompt structure
   - Validated JSON parsing

### Evaluation Design

1. **Query Coverage**
   - Include all persona types
   - Cover all difficulty levels
   - Include edge cases
   - Test multi-turn conversations

2. **Expected Answers**
   - Define required information points
   - Specify regulation references
   - Include exceptions
   - Set persona-appropriate style

3. **Pass/Fail Criteria**
   - Overall threshold: 80%
   - Individual metric thresholds
   - Automatic failure conditions (hallucination)
   - Edge case handling

### Result Analysis

1. **Failure Pattern Detection**
   - Group failures by type
   - Identify affected personas
   - Track failure frequency
   - Prioritize improvements

2. **Trend Analysis**
   - Compare to baseline
   - Track improvement over time
   - Detect regressions
   - Visualize metrics

3. **Actionable Insights**
   - Generate improvement SPECs
   - Provide specific recommendations
   - Prioritize by impact
   - Include implementation guidance

---

## Community Resources

### RAG Evaluation Community

1. **RAGAS Discord**: Community discussion, troubleshooting
2. **LangChain RAG Discussion**: Best practices, examples
3. **Papers with Code**: Latest RAG evaluation research

### University Regulation Management

1. **Project GitHub**: Repository with code and documentation
2. **Internal Wiki**: Project-specific knowledge base
3. **Evaluation Dashboard**: Gradio web interface for quality monitoring

---

## Quick Reference Card

### Evaluation Pipeline

```
1. Spawn Sub-Agents → 6 personas in parallel
2. Generate Queries → 150+ scenarios
3. Execute RAG CLI → regulation ask "{query}"
4. LLM-as-Judge → GPT-4o evaluation
5. Calculate Metrics → Precision, Recall, F1, Context
6. Generate Reports → JSON + Markdown + SPEC
```

### Metric Thresholds

| Metric | Target | Description |
|--------|--------|-------------|
| Overall | 0.80 | Weighted average of all metrics |
| Accuracy | 0.85 | No hallucinations |
| Completeness | 0.75 | All key info present |
| Citations | 0.70 | Accurate regulation refs |
| Context Relevance | 0.75 | Relevant sources |

### Persona Performance

| Persona | Avg Score | Pass Rate |
|---------|-----------|-----------|
| Professor | 0.85 | 85% |
| Graduate | 0.83 | 82% |
| Staff | 0.81 | 80% |
| Undergraduate | 0.79 | 75% |
| Parent | 0.78 | 73% |
| International | 0.76 | 70% |

### Category Performance

| Category | Avg Score | Pass Rate |
|----------|-----------|-----------|
| Adversarial | 0.91 | 90% |
| Simple | 0.89 | 90% |
| Domain-Specific | 0.86 | 88% |
| Complex | 0.82 | 80% |
| Multi-turn | 0.78 | 75% |
| Edge Cases | 0.71 | 70% |

---

## Version History

### v1.0.0 (2025-01-07)
- Initial release
- 6 persona sub-agents
- 150+ test scenarios
- LLM-as-Judge evaluation
- Automated SPEC generation
- Comprehensive metrics

### Future Enhancements
- [ ] Additional personas (alumni, part-time student)
- [ ] More edge case scenarios
- [ ] Visual trend dashboard
- [ ] CI/CD integration
- [ ] Multi-language support

---

## Contact and Support

### Project Maintenance

**Primary Maintainer**: RAG Team
**Issue Tracker**: Project GitHub Issues
**Discussion**: Team Slack #rag-quality

### Skill Updates

**Skill File**: `.claude/skills/rag-quality-local/SKILL.md`
**Modules**: `.claude/skills/rag-quality-local/modules/`
**Update Frequency**: As needed based on RAG system changes

### Feedback and Contributions

Feedback and contributions welcome! Please:
1. Report bugs via GitHub Issues
2. Suggest improvements via team discussion
3. Submit pull requests for enhancements
4. Share evaluation results for benchmarking

---

## License

This skill follows the project's MIT License.

See LICENSE file in project root for details.
