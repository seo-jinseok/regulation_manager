======================================================================
EVALUATION METRICS VERIFICATION REPORT
Generated: 2026-02-20T09:45:41.029041
======================================================================

## RAGAS Environment
- chromadb: OK
  - Version: 1.5.0
- ragas: OK
  - Version: 0.4.3
- ragas_metrics: OK
- llm_configured: MISSING

### Warnings:
- Using deprecated ragas.metrics import. Update to ragas.metrics.collections
- OPENAI_API_KEY not set

## Citation Analysis
- Total responses: 150
- With citations: 1
- Citation rate: 0.7%

## Score Analysis
- Is uniform: True
- Likely default: True
- Warning: CRITICAL: All 5 scores are exactly 0.50. This is the default value when RAGAS cannot calculate metrics. Likely causes: Missing LLM configuration, chromadb issues, or metric calculation failure.

## Small Scale Test Results
- RAGAS available: True
- Test count: 3

### Errors:
- RAGAS evaluation failed: cannot import name 'AnswerRelevance' from 'ragas.metrics._answer_relevance' (/Users/truestone/Dropbox/repo/University/regulation_manager/.venv/lib/python3.11/site-packages/ragas/metrics/_answer_relevance.py)

## Recommendations

2. Configure LLM for RAGAS:
   export OPENAI_API_KEY=your_key

3. Fix Default Value Issue:
   - RAGAS is returning default 0.50 values
   - Check OPENAI_API_KEY is set correctly
   - Verify LLM provider configuration
   - Check RAGAS metrics initialization

4. Improve Citation Rate:
   - Current citation rate is low
   - Ensure answers include regulation references

======================================================================
END OF REPORT
======================================================================