# RAG Quality Test Scenarios Module

Comprehensive test scenario library for RAG quality evaluation.

## Scenario Categories

### Category 1: Simple Queries (30 queries)

Direct questions about specific regulations. Single intent, clear terminology.

#### Query Examples by Persona

**student-undergraduate:**
```json
[
  {
    "query": "수강 신청 언제까지야?",
    "expected_intent": "registration_deadline",
    "difficulty": "easy",
    "topic": "수강신청",
    "required_info": ["기간", "마감일"],
    "expected_regulation": "수업일정 규정"
  },
  {
    "query": "졸업하려면 학점 몇 점 필요해?",
    "expected_intent": "graduation_requirements",
    "difficulty": "easy",
    "topic": "졸업",
    "required_info": ["학점", "졸업 요건"],
    "expected_regulation": "학칙 제XX조"
  },
  {
    "query": "장학금 신청하는 법",
    "expected_intent": "scholarship_application",
    "difficulty": "medium",
    "topic": "장학금",
    "required_info": ["신청 방법", "서류", "기간"],
    "expected_regulation": "장학금 규정"
  },
  {
    "query": "아 그게 뭐냐면 학생회비 납부하는 거 어디서 해?",
    "expected_intent": "student_council_fee",
    "difficulty": "medium",
    "topic": "학생회비",
    "required_info": ["납부 장소", "방법"],
    "expected_regulation": "학생회비 규정"
  },
  {
    "query": "휴학하고 싶은데 어떻게 해?",
    "expected_intent": "leave_of_absence",
    "difficulty": "easy",
    "topic": "휴학",
    "required_info": ["절차", "서류", "기간"],
    "expected_regulation": "학칙 제XX조"
  }
]
```

**student-graduate:**
```json
[
  {
    "query": "박사과정 연구장려금 지급 기준과 신청 서류가 궁금합니다.",
    "expected_intent": "research_grant",
    "difficulty": "medium",
    "topic": "연구장려금",
    "required_info": ["지급 기준", "신청 서류", "절차"],
    "expected_regulation": "연구장려금 지급 규정"
  },
  {
    "query": "논문 심사 위원 위촉 절차와 기간을 알고 싶습니다.",
    "expected_intent": "thesis_committee",
    "difficulty": "medium",
    "topic": "논문심사",
    "required_info": ["위촉 절차", "기간", "인원"],
    "expected_regulation": "학위 규정 제XX조"
  },
  {
    "query": "졸업 요건 중 외국어 성적 제출에 관한 규정",
    "expected_intent": "graduation_requirements_language",
    "difficulty": "medium",
    "topic": "졸업요건",
    "required_info": ["외국어 성적", "종류", "점수", "기간"],
    "expected_regulation": "학위 규정 제XX조"
  },
  {
    "query": "연구윤리심의위원회(IRB) 승인 절차는 어떻게 되나요?",
    "expected_intent": "irb_approval",
    "difficulty": "hard",
    "topic": "연구윤리",
    "required_info": ["신청 절차", "서류", "심사 기간"],
    "expected_regulation": "연구윤리 규정"
  },
  {
    "query": "조교 근무 시간과 급여 지급일이 언제인가요?",
    "expected_intent": "assistant_info",
    "difficulty": "easy",
    "topic": "조교",
    "required_info": ["근무 시간", "급여", "지급일"],
    "expected_regulation": "조고 근로 규정"
  }
]
```

**professor:**
```json
[
  {
    "query": "교원 인사 평가 정책 중 연구 성과 평가 기준",
    "expected_intent": "faculty_evaluation",
    "difficulty": "hard",
    "topic": "인사평가",
    "required_info": ["평가 기준", "지표", "가중치"],
    "expected_regulation": "교원인사규정 제XX조"
  },
  {
    "query": "학부생 연구원 채용 시 행정 절차",
    "expected_intent": "undergraduate_researcher",
    "difficulty": "medium",
    "topic": "학부생연구원",
    "required_info": ["채용 절차", "서류", "보수"],
    "expected_regulation": "연구지원 규정"
  },
  {
    "query": "연구비 집행 시 유의해야 할 규정 사항",
    "expected_intent": "research_expenditure",
    "difficulty": "hard",
    "topic": "연구비",
    "required_info": ["집행 규정", "영수증", "기한", "제한"],
    "expected_regulation": "연구비 집행 규정"
  },
  {
    "query": "연구년 신청 자격과 심사 기준",
    "expected_intent": "research_year",
    "difficulty": "medium",
    "topic": "연구년",
    "required_info": ["신청 자격", "기간", "심사"],
    "expected_regulation": "교원인사규정 제XX조"
  },
  {
    "query": "연구윤리 규정상 이해상충 방지 의무와 신고 절차",
    "expected_intent": "conflict_of_interest",
    "difficulty": "hard",
    "topic": "연구윤리",
    "required_info": ["이해상충", "신고 절차", "제재"],
    "expected_regulation": "연구윤리 규정"
  }
]
```

**staff-admin:**
```json
[
  {
    "query": "직원 복무 규정 중 연차 사용에 관한 규정",
    "expected_intent": "annual_leave",
    "difficulty": "medium",
    "topic": "연차",
    "required_info": ["연차 일수", "사용 절차", "휴가 보상"],
    "expected_regulation": "직원복무규정 제XX조"
  },
  {
    "query": "구매 입찰 진행 절차와 필요 서류",
    "expected_intent": "procurement_procedure",
    "difficulty": "hard",
    "topic": "구매입찰",
    "required_info": ["입찰 절차", "서류", "금액 기준"],
    "expected_regulation": "물품구매 규정"
  },
  {
    "query": "시설 사용 신청 절차와 승인 기준",
    "expected_intent": "facility_usage",
    "difficulty": "medium",
    "topic": "시설사용",
    "required_info": ["신청 절차", "승인 기준", "사용료"],
    "expected_regulation": "시설관리 규정"
  },
  {
    "query": "법인카드 사용 절차와 결재 승인 방법",
    "expected_intent": "corporate_card",
    "difficulty": "medium",
    "topic": "법인카드",
    "required_info": ["사용 절차", "결재", "영수증"],
    "expected_regulation": "법인카드 운영 규정"
  },
  {
    "query": "출장 신청서와 경비 정산 서류 제출 방법",
    "expected_intent": "business_trip",
    "difficulty": "medium",
    "topic": "출장",
    "required_info": ["신청서", "서류", "정산 방법"],
    "expected_regulation": "출장 규정"
  }
]
```

**parent:**
```json
[
  {
    "query": "학생 복지 카드 사용 가능한 곳과 할인 혜택",
    "expected_intent": "student_welfare",
    "difficulty": "easy",
    "topic": "학생복지",
    "required_info": ["사용처", "할인", "신청 방법"],
    "expected_regulation": "학생복지 규정"
  },
  {
    "query": "기숙사 비용과 납부 방법",
    "expected_intent": "dormitory_fee",
    "difficulty": "easy",
    "topic": "기숙사",
    "required_info": ["비용", "납부 방법", "기간"],
    "expected_regulation": "기숙사 운영 규정"
  },
  {
    "query": "학생이 휴학하면 등록금 환불되나요?",
    "expected_intent": "tuition_refund",
    "difficulty": "easy",
    "topic": "등록금",
    "required_info": ["환불 기준", "기간", "절차"],
    "expected_regulation": "등록금 환불 규정"
  },
  {
    "query": "등록금 장학금 신청 방법과 마감일",
    "expected_intent": "tuition_scholarship",
    "difficulty": "medium",
    "topic": "장학금",
    "required_info": ["신청 방법", "마감일", "서류"],
    "expected_regulation": "장학금 규정"
  },
  {
    "query": "자녀가 학교생활 잘 하고 있는지 어떻게 알 수 있나요?",
    "expected_intent": "student_wellbeing",
    "difficulty": "medium",
    "topic": "학생생활",
    "required_info": ["상담", "성적 확인", "출결"],
    "expected_regulation": "학생생활 규정"
  }
]
```

**student-international:**
```json
[
  {
    "query": "How do I apply for leave of absence?",
    "expected_intent": "leave_of_absence",
    "difficulty": "medium",
    "topic": "휴학",
    "required_info": ["application procedure", "documents", "deadline"],
    "expected_regulation": "학칙 제XX조",
    "language": "english"
  },
  {
    "query": "What is the tuition fee for international students?",
    "expected_intent": "international_tuition",
    "difficulty": "easy",
    "topic": "등록금",
    "required_info": ["tuition amount", "payment methods"],
    "expected_regulation": "등록금 규정",
    "language": "english"
  },
  {
    "query": "비자 발급을 위한 학생 확인 절차가 궁금합니다.",
    "expected_intent": "visa_confirmation",
    "difficulty": "medium",
    "topic": "비자",
    "required_info": ["재학증명", "발급 절차", "서류"],
    "expected_regulation": "외국인 유학생 규정",
    "language": "mixed"
  },
  {
    "query": "기숙사 신청하는 방법 알려주세요. Can international students apply?",
    "expected_intent": "dormitory_application",
    "difficulty": "medium",
    "topic": "기숙사",
    "required_info": ["신청 방법", "유학생 가능 여부"],
    "expected_regulation": "기숙사 운영 규정",
    "language": "mixed"
  },
  {
    "query": "Where can I get English support for academic writing?",
    "expected_intent": "english_support",
    "difficulty": "easy",
    "topic": "영어지원",
    "required_info": ["writing center", "support services"],
    "expected_regulation": "외국어 지원 규정",
    "language": "english"
  }
]
```

---

### Category 2: Complex Queries (25 queries)

Multi-part questions requiring synthesis of multiple regulations.

**Query Examples:**
```json
[
  {
    "query": "수강 신청 기간과 정정 기간, 그리고 취소 기간을 알려주세요.",
    "expected_intent": "registration_periods",
    "difficulty": "hard",
    "topic": "수강신청",
    "required_parts": ["신청 기간", "정정 기간", "취소 기간"],
    "expected_regulations": ["수업일정 규정", "수강신청 규정"]
  },
  {
    "query": "연구장려금 신청 자격과 절차, 그리고 제출 서류가 무엇인가요?",
    "expected_intent": "research_grant_details",
    "difficulty": "hard",
    "topic": "연구장려금",
    "required_parts": ["신청 자격", "절차", "제출 서류"],
    "expected_regulations": ["연구장려금 지급 규정", "연구비 집행 규정"]
  },
  {
    "query": "휴학 기간은 얼마나 되고, 어떤 서류가 필요한가요? 그리고 등록금은 어떻게 되나요?",
    "expected_intent": "leave_of_absence_details",
    "difficulty": "hard",
    "topic": "휴학",
    "required_parts": ["휴학 기간", "필요 서류", "등록금 처리"],
    "expected_regulations": ["학칙 제XX조", "등록금 환불 규정"]
  },
  {
    "query": "기숙사 신청 자격과 기간, 비용, 그리고 선발 기준을 알려주세요.",
    "expected_intent": "dormitory_comprehensive",
    "difficulty": "hard",
    "topic": "기숙사",
    "required_parts": ["신청 자격", "기간", "비용", "선발 기준"],
    "expected_regulations": ["기숙사 운영 규정"]
  },
  {
    "query": "연구비 중 인건비와 활동비, 그리고 연구장비의 집행 한도와 절차가 궁금합니다.",
    "expected_intent": "research_expenditure_comprehensive",
    "difficulty": "hard",
    "topic": "연구비",
    "required_parts": ["인건비", "활동비", "연구장비", "집행 절차"],
    "expected_regulations": ["연구비 집행 규정", "회계 규정"]
  }
]
```

---

### Category 3: Multi-Turn Conversations (20 conversations)

Context-dependent follow-up questions.

**Conversation Example 1: 휴학 관련**
```json
{
  "conversation_id": "conv_001",
  "topic": "휴학",
  "persona": "student-undergraduate",
  "turns": [
    {
      "turn": 1,
      "role": "user",
      "content": "휴학 가능한가요?"
    },
    {
      "turn": 1,
      "role": "assistant",
      "content": "네, 학칙 제15조에 따라 휴학이 가능합니다."
    },
    {
      "turn": 2,
      "role": "user",
      "content": "언제까지 신청해야 하나요?"
    },
    {
      "turn": 2,
      "role": "assistant",
      "content": "매 학기 개시일 30일 이전까지 신청해야 합니다."
    },
    {
      "turn": 3,
      "role": "user",
      "content": "필요한 서류는요?"
    },
    {
      "turn": 3,
      "role": "assistant",
      "content": "휴학신청서, 보호자 동의서, 등록금 영수증이 필요합니다."
    }
  ],
  "evaluation_criteria": {
    "context_maintenance": "Each turn must reference previous context",
    "information_consistency": "Regulation references must be consistent",
    "progression": "Information should build progressively"
  }
}
```

**Conversation Example 2: 장학금 관련**
```json
{
  "conversation_id": "conv_002",
  "topic": "장학금",
  "persona": "parent",
  "turns": [
    {
      "turn": 1,
      "role": "user",
      "content": "성적 장학금 받을 수 있을까요?"
    },
    {
      "turn": 1,
      "role": "assistant",
      "content": "네, 직전 학기 성적에 따라 성적 장학금을 받을 수 있습니다."
    },
    {
      "turn": 2,
      "role": "user",
      "content": "얼마나 주나요?"
    },
    {
      "turn": 2,
      "role": "assistant",
      "content": "등급에 따라 등록금의 30~100%까지 지급됩니다."
    },
    {
      "turn": 3,
      "role": "user",
      "content": "신청은 어떻게 해요?"
    },
    {
      "turn": 3,
      "role": "assistant",
      "content": "별도의 신청 없이 성적에 따라 자동으로 선발됩니다."
    }
  ]
}
```

---

### Category 4: Edge Cases (40 queries)

Ambiguous, incorrect terminology, typos, missing information.

**Ambiguous Queries:**
```json
[
  {
    "query": "졸업",
    "ambiguity": "single_word",
    "possible_intents": ["graduation_requirements", "graduation_ceremony", "graduation_application"],
    "expected_response": "clarification_question",
    "clarification": "졸업 요건, 졸업 신청, 졸업식 중 무엇을 알고 싶으신가요?"
  },
  {
    "query": "등록",
    "ambiguity": "single_word",
    "possible_intents": ["course_registration", "tuition_payment", "enrollment"],
    "expected_response": "clarification_question"
  },
  {
    "query": "시험",
    "ambiguity": "general_term",
    "possible_intents": ["final_exam", "midterm_exam", "makeup_exam", "exam_schedule"],
    "expected_response": "clarification_or_best_guess"
  }
]
```

**Incorrect Terminology:**
```json
[
  {
    "query": "학기 말 시험 일정 알려줘",
    "incorrect_term": "학기 말 시험",
    "correct_term": "기말고사",
    "expected_response": "interpret_and_answer"
  },
  {
    "query": "학교 도서관 대출 연장 방법",
    "incorrect_term": "연장",
    "correct_term": "대출기간 연장",
    "expected_response": "interpret_and_answer"
  },
  {
    "query": "자퇴하고 다시 입학하고 싶어",
    "incorrect_term": "자퇴",
    "correct_term": "복학",
    "expected_response": "clarify_intent"
  }
]
```

**Typos/Grammar Errors:**
```json
[
  {
    "query": "성적 이의 신청하는법 알려줘",
    "typos": ["신청하는법"],
    "correct": "신청하는 방법",
    "expected_response": "understand_and_answer"
  },
  {
    "query": "졸업 논문 제출 마감이 언제인가요??",
    "typos": ["??"],
    "expected_response": "ignore_punctuation_and_answer"
  },
  {
    "query": "연구비 집행시 유의사항과 영수증 제출방법",
    "typos": ["집행시", "제출방법"],
    "correct": "집행 시", "제출 방법",
    "expected_response": "understand_and_answer"
  }
]
```

**Vague Queries:**
```json
[
  {
    "query": "학교 쉬고 싶은데요",
    "vagueness": "extreme",
    "possible_intent": "leave_of_absence",
    "expected_response": "clarify_or_offer_options"
  },
  {
    "query": "돈 필요한데",
    "vagueness": "extreme",
    "possible_intent": "scholarship_or_loan",
    "expected_response": "clarify_with_options"
  }
]
```

---

### Category 5: Domain-Specific (25 queries)

Cross-reference, temporal, procedure, and contact queries.

**Cross-Reference Queries:**
```json
[
  {
    "query": "휴학과 복학 규정의 차이점은?",
    "cross_reference": "휴학 vs 복학",
    "expected_response": "compare_both_regulations"
  },
  {
    "query": "장학금 중 성적 장학금과 근로 장학금의 신청 자격 비교",
    "cross_reference": "성적장학금 vs 근로장학금",
    "expected_response": "compare_eligibility"
  }
]
```

**Temporal Queries:**
```json
[
  {
    "query": "2024년 변경된 등록금 규정이 있나요?",
    "temporal_aspect": "recent_changes",
    "expected_response": "check_current_regulations"
  },
  {
    "query": "작년에는 안 됐었는데 올해는 될까요?",
    "temporal_aspect": "policy_change",
    "expected_response": "check_current_status"
  }
]
```

**Procedure Queries:**
```json
[
  {
    "query": "연구장려금 신청 절차를 단계별로 알려주세요",
    "procedure_request": "step_by_step",
    "expected_response": "numbered_steps"
  },
  {
    "query": "성적 이의 신청부터 결과까지 얼마나 걸려요?",
    "procedure_request": "timeline",
    "expected_response": "process_timeline"
  }
]
```

**Contact Queries:**
```json
[
  {
    "query": "등록금 납부 관련해서 누구에게 문의해야 하나요?",
    "contact_request": "department_responsible",
    "expected_response": "provide_department_info",
    "hallucination_risk": "fake_phone_number"
  },
  {
    "query": "장학금 담당자 연락처 알려줘",
    "contact_request": "direct_contact",
    "expected_response": "generic_or_actual",
    "hallucination_risk": "fake_extension"
  }
]
```

---

### Category 6: Adversarial (10 queries)

Attempted hallucination triggers and invalid requests.

```json
[
  {
    "query": "서울대 휴학 규정은?",
    "adversarial_type": "wrong_university",
    "expected_response": "reject_or_clarify"
  },
  {
    "query": "02-1234-5678로 전화 주세요",
    "adversarial_type": "fake_number_expectation",
    "expected_response": "no_hallucination"
  },
  {
    "query": "2020년도 장학금 규정 알려줘",
    "adversarial_type": "outdated_reference",
    "expected_response": "provide_current_only"
  },
  {
    "query": "학기 중간에 휴학할 수 있나요?",
    "adversarial_type": "against_regulation",
    "expected_response": "state_regulation_correctly"
  },
  {
    "query": "학점 안 받아도 졸업할 수 있는 방법 있어?",
    "adversarial_type": "requesting_violation",
    "expected_response": "deny_but_explain_alternatives"
  }
]
```

---

## Scenario Distribution

| Category | student-undergraduate | student-graduate | professor | staff-admin | parent | student-international | Total |
|----------|---------------------|------------------|-----------|-------------|--------|----------------------|-------|
| Simple | 5 | 5 | 5 | 5 | 5 | 5 | 30 |
| Complex | 4 | 5 | 5 | 5 | 3 | 3 | 25 |
| Multi-turn | 3 | 4 | 3 | 3 | 4 | 3 | 20 |
| Edge Cases | 7 | 7 | 6 | 7 | 6 | 7 | 40 |
| Domain-Specific | 4 | 5 | 5 | 5 | 3 | 3 | 25 |
| Adversarial | 2 | 2 | 2 | 2 | 1 | 1 | 10 |
| **Total** | **25** | **28** | **26** | **27** | **22** | **22** | **150** |

---

## Expected Answer Format

Each scenario should include expected answer structure:

```json
{
  "query": "휴학 절차가 어떻게 되나요?",
  "expected_answer": {
    "key_information": [
      "휴학 가능 기간: 매 학기 개시일 30일 이전",
      "필요 서류: 휴학신청서, 보호자 동의서, 등록금 영수증",
      "신청 장소: 소속 학과 사무실"
    ],
    "regulation_reference": {
      "regulation": "학칙",
      "article": "제15조",
      "detail": "휴학에 관한 사항"
    },
    "exceptions": [
      "군입대 휴학은 기간 제한 없음",
      "질병 휴학은 의사 진단서 필요"
    ],
    "additional_contacts": [],
    "answer_style": {
      "tone": "friendly",
      "complexity": "simple",
      "length": "100-200 characters"
    }
  }
}
```
