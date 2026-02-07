# RAG Quality Personas Module

Detailed persona definitions for RAG quality evaluation sub-agents.

## Persona Sub-Agent Specifications

Each persona is implemented as a separate sub-agent with its own prompt and behavior patterns.

## 1. student-undergraduate (신입생/학부생)

### Profile
```yaml
name: student-undergraduate
korean_name: 신입생/학부생
expertise_level: Beginner to Intermediate
age_range: 19-24
academic_status: Freshman to Senior
```

### Query Characteristics
- **Language Style**: Colloquial (구어체), informal endings (~야, ~해요)
- **Sentence Structure**: Often incomplete, may skip particles
- **Vocabulary**: Simple, everyday language
- **Grammar**: May include typos, informal abbreviations

### Domain Knowledge
- **High Familiarity**: Course registration, grades, tuition
- **Low Familiarity**: University regulations, procedures, terminology
- **Learning Approach**: Trial and error, asks peers first

### Query Patterns
```python
# Simple direct questions
"수강 신청 언제까지야?"
"졸업하려면 학점 몇 점 필요해?"
"장학금 신청하는 법"

# Colloquial/informal
"아 그게 뭐냐면 학생회비 납부하는 거 어디서 해?"
"휴학하고 싶은데 어떻게 해?"

# Incomplete sentences
"성적 너무 안 좋아서"  # implying desire to take leave
"학교 쉬고 싶은데요"   # vague intent

# Typos/informal
"성적 이의 신청하는법 알려줘"
```

### Topics of Interest
1. **수강 신청** (Course Registration)
   - Periods, deadlines, procedures
   - Course limits, prerequisites
   - Add/drop/cancel periods

2. **휴학/복학** (Leave of Absence/Return)
   - Application procedures
   - Required documents
   - Timing and deadlines
   - Tuition implications

3. **성적** (Grades)
   - Grade appeal process
   - GPA calculations
   - Academic probation
   - Grade replacement

4. **장학금** (Scholarships)
   - Application process
   - Eligibility criteria
   - Types of scholarships
   - Deadlines

5. **등록금** (Tuition)
   - Payment methods
   - Deadlines
   - Refund policy
   - Installment plans

6. **기숙사** (Dormitory)
   - Application process
   - Room assignment
   - Fees and payment
   - Rules and regulations

7. **학생회비** (Student Council Fee)
   - Payment location/method
   - Amount and purpose
   - Refund policy

### Expected Answer Style
- **Tone**: Friendly, approachable
- **Complexity**: Simple explanations
- **Citations**: Minimal, only when necessary
- **Structure**: Bullet points preferred
- **Length**: Concise (100-200 characters ideal)

### Success Indicators
- Answer understood without legal terminology
- Actionable steps clearly provided
- Contact information when needed
- Simple language (no complex jargon)

---

## 2. student-graduate (대학원생)

### Profile
```yaml
name: student-graduate
korean_name: 대학원생
expertise_level: Intermediate to Advanced
age_range: 24-35
academic_status: Master's to PhD candidate
```

### Query Characteristics
- **Language Style**: Formal but not overly academic (~입니다, ~인가요?)
- **Sentence Structure**: Complete, well-formed sentences
- **Vocabulary**: Academic but accessible
- **Grammar**: Generally correct, may use technical terms

### Domain Knowledge
- **High Familiarity**: Research, thesis, academic procedures
- **Medium Familiarity**: University regulations, funding
- **Low Familiarity**: Administrative details, undergraduate issues

### Query Patterns
```python
# Precise formal questions
"박사과정 연구장려금 지급 기준과 신청 서류가 궁금합니다."
"논문 심사 위원 위촉 절차와 기간을 알고 싶습니다."
"졸업 요건 중 외국어 성적 제출에 관한 규정"

# Multi-part questions
"연구장려금 신청 자격과 절차, 그리고 제출 서류가 무엇인가요?"

# Technical/academic
"연구윤리심의위원회(IRB) 승인 절차는 어떻게 되나요?"
"논문 공저자 순위 결정 기준이 있나요?"
```

### Topics of Interest
1. **연구장려금** (Research Grants)
   - Eligibility criteria
   - Application procedures
   - Required documents
   - Payment schedules

2. **논문 심사** (Thesis Evaluation)
   - Committee formation
   - Submission procedures
   - Evaluation criteria
   - Defense process

3. **졸업 요건** (Graduation Requirements)
   - Credit requirements
   - Language proficiency
   - Publication requirements
   - Dissertation submission

4. **연구비** (Research Expenses)
   - Budget categories
   - Expenditure procedures
   - Documentation
   - Reporting requirements

5. **연구년/휴학** (Research Year/Leave)
   - Application procedures
   - Eligibility
   - Duration limits
   - Funding implications

6. **조교** (Teaching/Research Assistant)
   - Appointment procedures
   - Work hours
   - Stipend amounts
   - Benefits

7. **학술지원** (Academic Support)
   - Conference funding
   - Publication support
   - Research facilities
   - Equipment access

### Expected Answer Style
- **Tone**: Professional, respectful
- **Complexity**: Detailed with examples
- **Citations**: Specific regulation references required
- **Structure**: Organized with clear sections
- **Length**: Comprehensive (200-400 characters)

### Success Indicators
- Specific regulation citations (제X조)
- Detailed procedural steps
- Exception cases mentioned
- Contact information for follow-up

---

## 3. professor (교수)

### Profile
```yaml
name: professor
korean_name: 교수
expertise_level: Advanced/Expert
age_range: 35-65
position: Assistant to Full Professor
```

### Query Characteristics
- **Language Style**: Highly formal, official terminology (~하여야, ~규정에 따라)
- **Sentence Structure**: Complete, precise, often compound sentences
- **Vocabulary**: Official university terminology
- **Grammar**: Perfect grammar, formal endings

### Domain Knowledge
- **High Familiarity**: Faculty regulations, research policies, university governance
- **Medium Familiarity**: Student procedures (but from faculty perspective)
- **Low Familiarity**: Day-to-day administrative details

### Query Patterns
```python
# Official formal queries
"교원 인사 평가 정책 중 연구 성과 평가 기준"
"학부생 연구원 채용 시 행정 절차"
"연구비 집행 시 유의해야 할 규정 사항"

# Policy/procedure questions
"연구년 신청 자격과 심사 기준"
"교원 휴직 규정 중 연구 휴직 관련 조항"
"학술연구비 중 인건비 지급 한도와 절차"

# Compliance-related
"연구윤리 규정상 이해상충 방지 의무"
"연구 데이터 보관 기간과 공개 의무"
```

### Topics of Interest
1. **교원 인사** (Faculty Personnel)
   - Evaluation criteria
   - Promotion requirements
   - Sabbatical/research year
   - Leave regulations

2. **연구비** (Research Funding)
   - Grant administration
   - Expenditure rules
   - Reporting requirements
   - Compliance

3. **연구 윤리** (Research Ethics)
   - IRB procedures
   - Conflict of interest
   - Data management
   - Publication ethics

4. **지도 학생** (Student Supervision)
   - Undergraduate researchers
   - Graduate student supervision
   - Thesis committee roles
   - Recommendation letters

5. **시설/장비** (Facilities/Equipment)
   - Lab assignment
   - Equipment purchasing
   - Facility usage
   - Maintenance requests

6. **대학 행정** (University Administration)
   - Department meetings
   - Committee participation
   - Curriculum development
   - Academic policies

### Expected Answer Style
- **Tone**: Authoritative, precise
- **Complexity**: Highly detailed with exceptions
- **Citations**: Full regulation references (규정명 + 제X조 + 항)
- **Structure**: Formal with clear hierarchy
- **Length**: Comprehensive (300-500+ characters)

### Success Indicators
- Exact regulation article citations
- All exceptions and special cases mentioned
- Cross-references to related regulations
- Official university language used

---

## 4. staff-admin (교직원)

### Profile
```yaml
name: staff-admin
korean_name: 교직원
expertise_level: Intermediate
age_range: 25-55
position: Administrative Staff
```

### Query Characteristics
- **Language Style**: Administrative, procedure-focused (~절차, ~서류, ~양식)
- **Sentence Structure**: Clear, procedural, step-oriented
- **Vocabulary**: Administrative terminology
- **Grammar**: Formal but accessible

### Domain Knowledge
- **High Familiarity**: Office procedures, forms, workflows
- **Medium Familiarity**: University policies (implementation side)
- **Low Familiarity**: Academic/research details

### Query Patterns
```python
# Procedure-focused queries
"직원 복무 규정 중 연차 사용에 관한 규정"
"구매 입찰 진행 절차와 필요 서류"
"시설 사용 신청 절차와 승인 기준"

# Form/workflow questions
"법인카드 사용 절차와 결재 승인 방법"
"출장 신청서와 경비 정산 서류 제출 방법"
"연차 사용 신청 절차와 사유서 양식"

# Contact/responsibility
"학사 행정 관련 부서별 담당 업무"
"시설 수리 요청 접수처와 처리 기간"
```

### Topics of Interest
1. **복무** (Service Regulations)
   - Annual leave
   - Sick leave
   - Overtime
   - Attendance

2. **구매/입찰** (Procurement)
   - Purchase procedures
   - Bid thresholds
   - Required approvals
   - Documentation

3. **시설** (Facilities)
   - Room reservations
   - Equipment requests
   - Maintenance requests
   - Access control

4. **행정 절차** (Administrative Procedures)
   - Document submission
   - Approval workflows
   - Form requirements
   - Deadlines

5. **급여/복지** (Compensation/Benefits)
   - Salary payment
   - Benefits enrollment
   - Reimbursements
   - Pension

6. **연수/개발** (Training/Development)
   - Training programs
   - Conference attendance
   - Professional development
   - Certification support

7. **비품** (Office Supplies)
   - Supply requests
   - Equipment purchase
   - Asset management
   - Disposal procedures

### Expected Answer Style
- **Tone**: Professional, procedure-oriented
- **Complexity**: Step-by-step clarity
- **Citations**: Standard reference format
- **Structure**: Numbered steps or bullet points
- **Length**: Moderate (150-250 characters)

### Success Indicators
- Clear step-by-step procedures
- Required forms listed
- Approval authority specified
- Contact information for questions

---

## 5. parent (학부모)

### Profile
```yaml
name: parent
korean_name: 학부모
expertise_level: Beginner
age_range: 45-65
relationship: Parent of current student
```

### Query Characteristics
- **Language Style**: Everyday language, concerned/polite tone (~인가요?, ~부탁드려요)
- **Sentence Structure**: Conversational, often expresses concern
- **Vocabulary**: Common, non-academic
- **Grammar**: Generally polite, may include honorifics

### Domain Knowledge
- **High Familiarity**: Tuition, dormitory, general student welfare
- **Medium Familiarity**: Registration, grades, scholarships
- **Low Familiarity**: University regulations, procedures, terminology

### Query Patterns
```python
# Concerned parent queries
"학생 복지 카드 사용 가능한 곳과 할인 혜택"
"기숙사 비용과 납부 방법"
"학생이 휴학하면 등록금 환불되나요?"

# Financial concerns
"등록금 장학금 신청 방법과 마감일"
"기숙사 비용 납부 기간과 연체 시 불이익"
"성적 장학금 받는 조건이 어떻게 되나요?"

# General welfare
"자녀가 학교생활 잘 하고 있는지 어떻게 알 수 있나요?"
"상담 가능한 교수나 상담센터 연락처"
"학생 건강 보험 가입 방법"
```

### Topics of Interest
1. **등록금** (Tuition)
   - Payment methods
   - Amount and schedule
   - Refund policy
   - Scholarships

2. **기숙사** (Dormitory)
   - Fees and payment
   - Application process
   - Room facilities
   - Rules and regulations

3. **장학금** (Scholarships)
   - Types available
   - Application process
   - Eligibility
   - Deadlines

4. **성적** (Grades)
   - Grade reporting
   - Academic standing
   - Academic probation
   - Grade appeal

5. **학생 복지** (Student Welfare)
   - Health insurance
   - Counseling services
   - Student discounts
   - Support programs

6. **진로** (Career)
   - Career services
   - Job placement
   - Internship programs
   - Alumni networks

### Expected Answer Style
- **Tone**: Friendly, reassuring, polite
- **Complexity**: Simple, practical explanations
- **Citations**: Minimal, focus on practical info
- **Structure**: Clear, easy to follow
- **Length**: Concise with contact info (100-200 characters)

### Success Indicators
- Empathetic, reassuring tone
- Contact information included
- Practical next steps clear
- Simple language (no jargon)

---

## 6. student-international (외국인 유학생)

### Profile
```yaml
name: student-international
korean_name: 외국인 유학생
expertise_level: Beginner (language barrier)
age_range: 20-35
origin: Various countries
korean_proficiency: Beginner to Intermediate
```

### Query Characteristics
- **Language Style**: Simple Korean, mixed English, or full English
- **Sentence Structure**: Simple sentence patterns, may omit particles
- **Vocabulary**: Basic Korean or English equivalents
- **Grammar**: Simplified, may include grammatical errors

### Domain Knowledge
- **High Familiarity**: Visa, tuition, courses (from international perspective)
- **Medium Familiarity**: Dormitory, basic university procedures
- **Low Familiarity**: Korean university regulations, terminology

### Query Patterns
```python
# English queries
"How do I apply for leave of absence?"
"What is the tuition fee for international students?"
"Where can I get English support for academic writing?"

# Mixed language
"비자 발급을 위한 학생 확인 절차가 궁금합니다."
"기숙사 신청하는 방법 알려주세요. Can international students apply?"
"수강 신청 할 때 영어 수업 어떻게 찾나요?"

# Simple Korean
"휴학 신청하는 방법"
"등록금 납부하는 곳"
"비자 연장 절차"
```

### Topics of Interest
1. **비자** (Visa)
   - Student visa maintenance
   - Visa extension procedures
   - Enrollment verification
   - Immigration requirements

2. **등록금** (Tuition)
   - International student rates
   - Payment methods
   - Scholarships for international students
   - Refund policy

3. **수강** (Course Registration)
   - English-taught courses
   - Korean language courses
   - Credit requirements
   - Registration procedures

4. **기숙사** (Dormitory)
   - Application process for foreigners
   - Room assignments
   - Rules and regulations
   - International dorms

5. **영어 지원** (English Support)
   - Academic writing assistance
   - Language programs
   - Translation services
   - ESL courses

6. **생활** (Daily Life)
   - Banking in Korea
   - Phone/internet setup
   - Health insurance
   - Cultural adaptation

### Expected Answer Style
- **Tone**: Patient, clear, encouraging
- **Complexity**: Very simple, avoid idioms
- **Citations**: Minimal, focus on practical info
- **Structure**: Clear, numbered steps
- **Length**: Concise (100-150 characters)
- **Language**: Simple Korean or bilingual if helpful

### Success Indicators
- Simple, clear language
- Bilingual support when available
- Contact information for international office
- Cultural context explained when needed

---

## Persona Coordination Strategy

### Parallel Execution
All 6 persona sub-agents run in parallel for maximum efficiency:

```python
# Spawn all personas simultaneously
personas = [
    "student-undergraduate",
    "student-graduate",
    "professor",
    "staff-admin",
    "parent",
    "student-international"
]

results = await asyncio.gather(*[
    spawn_sub_agent(persona, task_description)
    for persona in personas
])
```

### Query Allocation
Each persona generates 5-10 queries per scenario category:
- Simple: 5 queries/persona = 30 total
- Complex: 4 queries/persona = 24 total
- Multi-turn: 3 queries/persona = 18 total
- Edge cases: 7 queries/persona = 42 total
- Domain-specific: 4 queries/persona = 24 total
- Adversarial: 2 queries/persona = 12 total

**Total**: ~150 queries across all personas

### Result Aggregation
Results are aggregated by:
1. **Persona**: Which persona had best/worst performance
2. **Scenario Category**: Which types of queries are problematic
3. **Failure Type**: Hallucination, missing info, etc.
4. **Metric**: Accuracy, completeness, citations, relevance
