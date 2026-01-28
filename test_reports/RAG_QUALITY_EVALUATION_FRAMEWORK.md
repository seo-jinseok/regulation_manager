# RAG Quality Evaluation Framework Report

**Date:** 2026-01-28
**Evaluator:** RAG Quality Assurance Specialist
**Project:** University Regulation Management System

---

## Executive Summary

This document provides a comprehensive analysis of the RAG (Retrieval-Augmented Generation) quality evaluation framework for the University Regulation Management System. The framework has been designed to rigorously test system quality across diverse user personas, query styles, and conversation patterns.

### Key Findings

| Metric | Value |
|--------|-------|
| **Total Test Scenarios** | 51 (34 single-turn + 17 multi-turn turns) |
| **User Personas** | 6 types (Freshman, Graduate, Professor, Staff, Parent, International) |
| **Query Styles** | 7 variations (Precise, Ambiguous, Colloquial, Incorrect Terminology, Multi-part, Context-dependent, Typo/Grammar errors) |
| **Quality Dimensions** | 6 (Accuracy, Completeness, Relevance, Source Citation, Practicality, Actionability) |
| **Maximum Score** | 5.0 points |
| **Passing Threshold** | 4.0 points (80%) |

---

## Phase 1: Diverse User Persona Simulation

### Test Coverage by Persona

#### 1. Freshman Student (11 queries)
**Characteristics:**
- Unfamiliar with university terminology
- Uses casual language
- Needs step-by-step guidance
- Focus on: Registration, deadlines, basic procedures

**Query Examples:**
- "수강 신청 언제까지야?" (Colloquial)
- "졸업하려면 학점 몇 점 필요해?" (Precise)
- "휴학하고 싶은데 어떻게 해?" (Ambiguous)

**Success Criteria:**
- System should use simple, clear language
- Provide specific deadlines and procedures
- Offer clarification when terminology is unclear

#### 2. Graduate Student (7 queries)
**Characteristics:**
- Research-focused questions
- Familiar with academic procedures
- Needs specific policy details
- Focus on: Research funding, thesis guidelines, academic requirements

**Query Examples:**
- "박사과정 연구장려금 지급 기준과 신청 서류가 궁금합니다." (Precise)
- "연차 학회 참가비 지원 가능한가요?" (Precise)

**Success Criteria:**
- Provide detailed regulation references
- Include specific eligibility criteria
- Reference relevant academic articles

#### 3. Professor (6 queries)
**Characteristics:**
- Expert knowledge of university structure
- Needs administrative procedure details
- Focus on: Evaluation policies, research expenditure, hiring

**Query Examples:**
- "교원 인사 평가 정책 중 연구 성과 평가 기준" (Precise, Expert)
- "연구비 집행 시 유의해야 할 규정 사항" (Precise, Expert)

**Success Criteria:**
- Reference specific faculty regulations
- Provide detailed procedural requirements
- Include relevant compliance information

#### 4. Staff Member (6 queries)
**Characteristics:**
- Internal process focus
- Needs workflow and approval details
- Focus on: Leave policies, procurement, certifications

**Query Examples:**
- "직원 복무 규정 중 연차 사용에 관한 규정" (Precise)
- "구매 입찰 진행 절차와 필요 서류" (Precise)

**Success Criteria:**
- Reference staff regulations accurately
- Provide step-by-step procedures
- Include required forms/deadlines

#### 5. Parent (4 queries)
**Characteristics:**
- No university knowledge
- Focus on financial and practical matters
- Uses simple language

**Query Examples:**
- "학생 복지 카드 사용 가능한 곳과 할인 혜택" (Colloquial)
- "학생이 휴학하면 등록금 환불되나요?" (Colloquial)

**Success Criteria:**
- Avoid jargon
- Provide clear contact information
- Include specific dollar amounts/deadlines

---

## Phase 2: Ambiguous and Poorly-Phrased Query Testing

### Query Style Analysis

#### Single-Word Queries (4 queries)
**Examples:** "졸업", "등록", "장학", "휴가"

**Expected Behavior:**
- System should identify ambiguity
- Ask clarifying questions
- NOT guess at user intent
- Provide multiple interpretation options

**Test Cases:**
1. "졸업" → Should clarify: Requirements? Application? Ceremony?
2. "등록" → Should clarify: Course registration? Fee payment?
3. "장학" → Should clarify: Types? Eligibility? Application?
4. "휴가" → Should clarify: Type of leave? Duration?

#### Multi-Part Queries (3 queries)
**Examples:**
- "수강 신청 기간과 정정 기간, 그리고 취소 기간을 알려주세요."
- "연구장려금 신청 자격과 절차, 그리고 제출 서류가 무엇인가요?"

**Expected Behavior:**
- Address all sub-questions
- Provide structured response
- Use bullet points or numbered lists
- Reference each sub-question separately

#### Incorrect Terminology (3 queries)
**Examples:**
- "학기 말 시험 일정 알려줘" (instead of "기말고사")
- "교수님들 급여 체계가 어떻게 되나요?" (too casual)
- "학교 도서관 대출 연장 방법" (instead of "도서대출")

**Expected Behavior:**
- Detect actual intent despite wrong terms
- Use correct terminology in response
- Gently correct user terminology
- Provide accurate information

#### Typo/Grammar Errors (3 queries)
**Examples:**
- "성적 이의 신청하는법 알려줘" (missing spaces)
- "졸업 논문 제출 마감이 언제인가요??" (double question mark)
- "연구비 집행시 유의사항과 영수증 제출방법" (missing spaces)

**Expected Behavior:**
- Handle typos gracefully
- Ignore punctuation variations
- Focus on content meaning
- Provide professional response

---

## Phase 3: Multi-Turn Conversation Testing

### Conversation Scenarios (5 scenarios, 17 total turns)

#### Scenario 1: Freshman Registration Confusion (4 turns)
**Persona:** Freshman Student
**Challenge:** Confused about registration process

| Turn | Query | Expected Follow-Up | Context Preservation |
|------|-------|-------------------|---------------------|
| 1 | "수강 신청 언제 해?" | Clarification | N/A |
| 2 | "아, 그리고 정정 기간은?" | Deepen | Must remember registration context |
| 3 | "신청한 거 취소도 가능해?" | Deepen | Must link to original query |
| 4 | "연장제 수강 신청도 같은 기간인가요?" | Refine | Must track extended education context |

**Success Metrics:**
- Context preservation rate ≥ 70%
- All turns reference appropriate regulations
- No contradictory information across turns

#### Scenario 2: Graduate Research Funding (4 turns)
**Persona:** Graduate Student
**Challenge:** Exploring funding options

| Turn | Query | Expected Follow-Up |
|------|-------|-------------------|
| 1 | "연구장려금 신청할 수 있어?" | Clarification |
| 2 | "신청 자격이 어떻게 돼?" | Deepen |
| 3 | "어디에 제출하면 돼?" | Deepen |
| 4 | "학회 참가비 지원도 받을 수 있어?" | Shift (to related topic) |

**Success Metrics:**
- Smooth topic transition handling
- Context maintained across related queries
- Proper context separation on topic shift

#### Scenario 3: Professor Evaluation Process (3 turns)
**Persona:** Professor
**Challenge:** Understanding evaluation policies

**Success Metrics:**
- Detailed regulatory references
- Procedural clarity
- Appeal process information included

#### Scenario 4: Staff Leave Inquiry (3 turns)
**Persona:** Staff Member
**Challenge:** Leave policy navigation

**Success Metrics:**
- Accurate staff regulation references
- All leave types covered
- Clear procedural steps

#### Scenario 5: Parent Tuition Refund (3 turns)
**Persona:** Parent
**Challenge:** Understanding refund policy

**Success Metrics:**
- Simple language
- Specific amounts/deadlines
- Contact information included

---

## Phase 4: Quality Metrics Evaluation

### Six-Dimension Scoring Framework

#### 1. Accuracy (0.0 - 1.0)
**Definition:** Correctness of factual information based on regulations

**Scoring:**
| Score | Criteria |
|-------|----------|
| 1.0 | All facts correct, properly cited |
| 0.8 | Minor inaccuracies, not material |
| 0.5 | Some incorrect information |
| 0.0 | Major factual errors or hallucinations |

**Auto-Fail Conditions:**
- Hallucinated phone numbers (e.g., "02-XXXX-XXXX")
- References to other universities
- Generic disclaimers ("대학마다 다릅니다")
- Numbers without regulatory basis

#### 2. Completeness (0.0 - 1.0)
**Definition:** Coverage of all aspects of the question

**Scoring:**
| Score | Criteria |
|-------|----------|
| 1.0 | All question aspects addressed |
| 0.8 | Minor aspects missing |
| 0.5 | Significant aspects missing |
| 0.0 | Fails to address core question |

**Evaluation Factors:**
- Multi-part query handling
- Prerequisites mentioned
- Exceptions noted
- Related information included

#### 3. Relevance (0.0 - 1.0)
**Definition:** Alignment with user intent and question focus

**Scoring:**
| Score | Criteria |
|-------|----------|
| 1.0 | Directly addresses intent |
| 0.8 | Mostly relevant, minor tangents |
| 0.5 | Partially relevant |
| 0.0 | Irrelevant to question |

**Evaluation Factors:**
- Intent recognition accuracy
- Audience-appropriate response
- Query type consideration

#### 4. Source Citation (0.0 - 1.0)
**Definition:** Proper citation of regulation names and articles

**Scoring:**
| Score | Criteria |
|-------|----------|
| 1.0 | Full citation (regulation + article) for all claims |
| 0.8 | Citations for most claims |
| 0.5 | Partial citations |
| 0.0 | No citations or incorrect citations |

**Required Format:**
- "규정명 제N조" (e.g., "학칙 제15조", "직원복무규정 제26조")
- Must include regulation name AND article number
- Multiple regulations must each be cited

#### 5. Practicality (0.0 - 0.5)
**Definition:** Inclusion of actionable details (deadlines, documents, departments)

**Scoring:**
| Score | Criteria |
|-------|----------|
| 0.5 | All practical info included |
| 0.3 | Most practical info included |
| 0.1 | Minimal practical info |
| 0.0 | No practical info |

**Practical Elements:**
- Specific deadlines (dates/timeframes)
- Required documents
- Department/office locations
- Contact methods (generic, not hallucinated)
- Forms or applications

#### 6. Actionability (0.0 - 0.5)
**Definition:** User's ability to take immediate action based on response

**Scoring:**
| Score | Criteria |
|-------|----------|
| 0.5 | Clear next steps, user can act immediately |
| 0.3 | Some steps clear, minor clarifications needed |
| 0.1 | Vague guidance, research required |
| 0.0 | No actionable guidance |

**Actionability Elements:**
- Step-by-step procedures
- "To do X, you must Y" format
- Clear decision points
- Alternative options presented

### Total Score Calculation

```
Total = Accuracy + Completeness + Relevance + Source Citation + Practicality + Actionability
Maximum = 1.0 + 1.0 + 1.0 + 1.0 + 0.5 + 0.5 = 5.0
Passing = 4.0 (80%)
```

---

## Phase 5: Failure Analysis (5-Why Method)

### Root Cause Analysis Framework

For each test failure, perform 5-Why analysis:

**Example: Failure on "수강 신청 언제까지야?"**

| Why Level | Question | Answer |
|-----------|----------|--------|
| 1 | Why did the system fail? | Incorrect deadline provided |
| 2 | Why incorrect deadline? | Retrieved outdated regulation |
| 3 | Why outdated? | Database not synced with latest regulations |
| 4 | Why not synced? | Incremental sync failed silently |
| 5 | Why failed silently? | No error handling for sync failures |

**Root Cause:** Lack of error handling in incremental sync process

**Suggested Fix:**
- **Component to Patch:** `sync_usecase.py`
- **Code Change Required:** Yes
- **Fix:** Add error logging and fallback mechanisms for sync failures

---

## Comprehensive Test Results Summary

### Single-Turn Query Breakdown

| Persona | Total Queries | Query Styles |
|---------|---------------|--------------|
| Freshman Student | 11 | Precise, Colloquial, Ambiguous |
| Graduate Student | 7 | Precise, Multi-part |
| Professor | 6 | Precise, Multi-part |
| Staff Member | 6 | Precise, Incorrect Terminology |
| Parent | 4 | Colloquial, Precise |
| **Ambiguous Single-Word** | 4 | All Ambiguous |
| **Multi-Part** | 3 | All Multi-part |
| **Incorrect Terminology** | 3 | All Incorrect Terminology |
| **Typo/Grammar Errors** | 3 | All Typo/Grammar Error |

**Total Single-Turn Queries: 34**

### Multi-Turn Scenario Breakdown

| Scenario ID | Persona | Turns | Focus |
|-------------|---------|-------|-------|
| freshman_registration_confusion | Freshman | 4 | Registration process |
| graduate_research_funding | Graduate | 4 | Research funding |
| professor_evaluation_process | Professor | 3 | Evaluation policies |
| staff_leave_inquiry | Staff | 3 | Leave policies |
| parent_tuition_refund | Parent | 3 | Refund policy |

**Total Scenarios: 5**
**Total Conversation Turns: 17**

---

## Quality Dimensions Expected Performance

### Persona-Based Performance Expectations

| Persona | Expected Accuracy | Expected Source Citation | Expected Practicality |
|---------|------------------|------------------------|----------------------|
| Freshman | 0.8+ | 0.7+ | 0.4+ |
| Graduate | 0.9+ | 0.9+ | 0.5 |
| Professor | 0.95+ | 0.95+ | 0.5 |
| Staff | 0.9+ | 0.9+ | 0.5 |
| Parent | 0.7+ | 0.6+ | 0.4 |

### Query Style Performance Expectations

| Query Style | Expected Accuracy | Expected Relevance |
|-------------|------------------|-------------------|
| Precise | 0.95+ | 0.95+ |
| Colloquial | 0.7+ | 0.8+ |
| Ambiguous | 0.5+ (after clarification) | 0.6+ |
| Incorrect Terminology | 0.75+ | 0.7+ |
| Multi-part | 0.85+ | 0.9+ |
| Typo/Grammar Error | 0.8+ | 0.85+ |

---

## Recommendations

### Immediate Actions (High Priority)

1. **Clarification Response System**
   - Implement automatic clarification requests for single-word queries
   - Add multi-option clarification UI
   - Target: Reduce ambiguity failures by 80%

2. **Source Citation Enforcement**
   - Add post-processing validation for regulation citations
   - Require "규정명 제N조" format
   - Flag responses without proper citations for regeneration

3. **Hallucination Prevention**
   - Add phone number detection and blocking
   - Add other-university reference detection
   - Add generic disclaimer detection
   - Target: 0% hallucination rate

### Medium-Term Improvements

4. **Multi-Turn Context Management**
   - Implement conversation context window
   - Add context preservation tracking
   - Implement topic transition handling
   - Target: 70%+ context preservation rate

5. **Query Expansion Enhancement**
   - Add colloquial-to-formal term mapping
   - Add typo correction
   - Add intent recognition for ambiguous queries
   - Target: 90%+ intent recognition accuracy

6. **Practical Information Extraction**
   - Implement deadline extraction from regulations
   - Implement required document extraction
   - Implement department/office extraction
   - Target: 0.4+ practicality score average

### Long-Term Enhancements

7. **Automated Fact Checking**
   - Implement fact-checking against regulation database
   - Add claim verification pipeline
   - Target: 95%+ factual accuracy

8. **Personalization by Persona**
   - Implement language level adjustment
   - Implement detail level customization
   - Implement persona-specific response templates
   - Target: 0.9+ user satisfaction by persona

---

## Implementation Roadmap

### Phase 1: Critical Fixes (Week 1-2)
- [ ] Implement clarification request system
- [ ] Add source citation validation
- [ ] Implement hallucination detection

### Phase 2: Quality Improvements (Week 3-4)
- [ ] Enhance query expansion
- [ ] Implement multi-turn context management
- [ ] Add practical information extraction

### Phase 3: Advanced Features (Week 5-6)
- [ ] Implement automated fact checking
- [ ] Add persona-based personalization
- [ ] Implement continuous evaluation pipeline

---

## Conclusion

The RAG Quality Evaluation Framework provides a comprehensive, systematic approach to testing the University Regulation Management System across diverse user personas, query styles, and conversation patterns. The 51 test scenarios (34 single-turn + 17 multi-turn turns) cover the full spectrum of expected user interactions.

The six-dimensional quality scoring framework ensures rigorous evaluation of:
1. **Accuracy** - Factual correctness
2. **Completeness** - Full question coverage
3. **Relevance** - Intent alignment
4. **Source Citation** - Proper attribution
5. **Practicality** - Actionable details
6. **Actionability** - Clear next steps

By implementing the recommended improvements, the system can achieve:
- 80%+ pass rate on single-turn queries
- 70%+ context preservation on multi-turn conversations
- 95%+ factual accuracy
- 0% hallucination rate

This framework provides the foundation for continuous quality improvement and ensures the RAG system meets the diverse needs of all university stakeholders.

---

**Report Generated:** 2026-01-28
**Framework Version:** 1.0
**Status:** Framework Complete, Ready for Execution

<moai>DONE</moai>
