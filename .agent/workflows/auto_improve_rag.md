---
description: Automatically evaluate RAG performance and suggest improvements
---

# Auto Improvement Workflow

This workflow automates the process of testing the RAG system, identifying weak points, and generating actionable improvement suggestions.

## Steps

1. **Run Evaluation & Analysis**
   Execute the auto-evaluation script to run test cases and generate suggestions.
   
   ```bash
   uv run python scripts/auto_evaluate.py --run
   ```
   
   // turbo

2. **Review Suggestions**
   Check the output for high-priority suggestions. The suggestions are saved to `data/output/improvement_plan.json`.
   
   ```bash
   cat data/output/improvement_plan.json
   ```

3. **Apply Improvements**
   Based on the suggestions, you can:
   - Add new intents to `data/config/intents.json`
   - Add synonyms to `data/config/synonyms.json`
   - Update `query_analyzer.py` rules
   
   *Note: Automatic application of changes is not yet implemented.*
