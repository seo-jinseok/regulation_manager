# Research: rag-quality-local Skill Production Readiness

## Executive Summary

rag-quality-local 스킬은 RAG 시스템 품질 평가를 위한 포괄적인 기능을 제공하지만, **"어떻게 사용하는지"에 대한 실행 가능한 인터페이스가 부족**하여 프로덕션 완성을 위해서는 개선이 필요합니다.

## Current State Analysis

### 1. Skill Structure (Current)

```
.claude/skills/rag-quality-local/
├── SKILL.md              # ~365 lines - Main coordinator
├── modules/
│   ├── personas.md       # ~614 lines - 6 persona definitions
│   ├── scenarios.md      # ~690 lines - Test scenarios
│   ├── evaluation.md     # ~395 lines - LLM-as-Judge prompts
│   └── metrics.md        # ~544 lines - Metric calculations
└── reference.md          # ~333 lines - External resources

Total: ~2,950 lines of documentation
```

### 2. Implementation Status

**Fully Implemented (src/rag/domain/evaluation/):**
| Component | File | Status |
|-----------|------|--------|
| LLM-as-Judge | llm_judge.py, custom_judge.py | ✅ Complete |
| Personas | personas.py, persona_definition.py | ✅ Complete |
| Metrics | quality_analyzer.py | ✅ Complete |
| Failure Classification | failure_classifier.py | ✅ Complete |
| SPEC Generation | spec_generator.py | ✅ Complete |
| Recommendations | recommendation_engine.py | ✅ Complete |
| Parallel Evaluation | parallel_evaluator.py | ✅ Complete |
| Batch Execution | batch_executor.py | ✅ Complete |
| Query Generation | regulation_query_generator.py | ✅ Complete |

**Sub-agents (7 total):**
- rag-quality-evaluator.md
- rag-student-undergraduate.md
- rag-student-graduate.md
- rag-professor.md
- rag-staff-admin.md
- rag-parent.md
- rag-international-student.md

### 3. Related SPECs (16 documents)
- SPEC-RAG-Q-001 through SPEC-RAG-Q-011
- SPEC-RAG-QUALITY-001 through SPEC-RAG-QUALITY-009
- SPEC-RAG-EVAL-001
- SPEC-RAG-MONITOR-001

## Gap Analysis

### Critical Gaps (P0)

#### Gap 1: No Executable Entry Point
**Problem:** Skill describes "what" but not "how to invoke"
- Current: "Use the rag-quality-local skill to run evaluation"
- Missing: Actual command to execute the skill
- Impact: Users cannot run the skill directly

**Evidence:**
```markdown
# Current SKILL.md Quick Start
**Option 1: Full Evaluation**
```
Use the rag-quality-local skill to run a comprehensive RAG quality evaluation:
1. Spawn all 6 persona sub-agents in parallel...
```
```

**Improvement Needed:**
```bash
# Expected command
/rag-quality full --output report.md

# Or CLI
uv run python run_rag_quality_eval.py --full
```

#### Gap 2: Result Verification Path Unclear
**Problem:** Users don't know where to check evaluation results
- Outputs mentioned: JSON, Markdown, SPEC template
- Locations mentioned: data/evaluations/
- Missing: Clear instructions on how to view/interpret results

**Improvement Needed:**
- Dashboard link
- Result summary command
- File location with example

#### Gap 3: No Follow-up Action Guidance
**Problem:** "Evaluation complete" then what?
- Current: Generates SPEC for failures
- Missing: How to execute the improvement SPEC
- Missing: Connection to /moai run workflow

### Important Gaps (P1)

#### Gap 4: MoAI Workflow Integration Missing
- No triggers for /moai plan/run/sync
- No quality gate enforcement
- No automatic SPEC-to-implementation pipeline

#### Gap 5: Documentation-Code Synchronization
- Skill documentation (~2950 lines) separate from code
- API changes may not reflect in documentation
- No version synchronization mechanism

### Enhancement Gaps (P2)

#### Gap 6: CI/CD Integration
- No GitHub Actions workflow example
- No pre-commit hook integration
- No scheduled evaluation guidance

#### Gap 7: Alerting System
- No notification when quality drops
- No threshold breach alerts
- No trend anomaly detection

## MoAI Standards Compliance

### Compliant Items
- ✅ Required YAML frontmatter fields (name, description, version, category, status)
- ✅ metadata with tags, related-skills
- ✅ progressive_disclosure configuration
- ✅ triggers configuration (keywords, agents, phases, languages)
- ✅ allowed-tools definition

### Non-Compliant Items
- ⚠️ user-invocable not specified (defaults to true)
- ⚠️ No clear slash command definition
- ⚠️ description approaches 1024 char limit

### Recommendations
1. Add `user-invocable: true` explicitly
2. Create `.claude/commands/rag-quality.md` for slash command
3. Trim description or use more concise language

## User Experience Issues

### Issue 1: High Entry Barrier
- Total documentation: ~2950 lines
- Quick Start exists but not actionable
- Mixed developer/user content

### Issue 2: Abstract Instructions
- "Spawn sub-agents" - requires understanding Claude Code internals
- "Execute queries via CLI" - no concrete command
- "LLM-as-Judge evaluation" - jargon without explanation

### Issue 3: Result Confusion
- Multiple output formats (JSON, MD, SPEC)
- No unified view
- Dashboard mentioned but not linked

## Improvement Plan Summary

### P0: Essential (Production Minimum)
1. Add executable slash command `/rag-quality`
2. Improve Quick Start with actual commands
3. Add result verification section

### P1: Important (Usability)
4. Add MoAI workflow integration
5. Create dashboard usage guide
6. Add quality gate configuration

### P2: Recommended (Polish)
7. Add CI/CD integration examples
8. Add scheduled evaluation guide
9. Add alerting configuration

## Success Criteria (Definition of Done)

A user should be able to:
1. ✅ Run full evaluation with single command
2. ✅ View results in dashboard or file
3. ✅ Get improvement suggestions automatically
4. ✅ Execute improvement SPEC with one click
5. ✅ Integrate into CI/CD pipeline

## Reference Files

### Primary Analysis Sources
- `.claude/skills/rag-quality-local/SKILL.md`
- `.claude/skills/rag-quality-local/modules/*.md`
- `.claude/agents/rag-quality-evaluator.md`

### Implementation References
- `src/rag/domain/evaluation/__init__.py`
- `src/rag/domain/evaluation/llm_judge.py`
- `src/rag/domain/evaluation/parallel_evaluator.py`
- `run_rag_quality_eval.py`

### Configuration References
- `.moai/config/sections/quality.yaml`
- `.claude/rules/moai/development/skill-authoring.md`

---

**Research Date:** 2026-02-24
**Researcher:** MoAI Strategic Orchestrator
**Method:** Sequential Thinking MCP (8 thoughts)
