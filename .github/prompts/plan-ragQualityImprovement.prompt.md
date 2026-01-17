# Plan: RAG 시스템 품질 개선

**TL;DR**: 테스트에서 발견된 44.4% 동적 쿼리 성공률을 80%로 올리기 위해, (1) 즉시 조치로 intents.json 강화 및 동의어 확장, (2) 근본 해결책으로 복합 쿼리 분해 로직 및 삭제 조항 경고 시스템을 구현합니다.

---

## 🔧 즉시 조치 (1-2시간)

### Step 1: intents.json 부당대우→인권센터 매핑 강화
- [intents.json](data/config/intents.json#L1508) `student_rights` 인텐트에 `keywords`에 "인권센터", "고충처리위원회" 추가
- `regulation_hint`에 "인권센터규정, 고충처리규정 우선 참조" 추가

### Step 2: synonyms.json 동의어 확장
- [synonyms.json](data/config/synonyms.json)에 다음 매핑 추가:
  - "부당대우" → ["갑질", "인권침해", "고충", "부당행위"]
  - "학자금" → ["등록금", "장학금", "학비", "수업료"]

### Step 3: 장학금+휴학 복합 인텐트 가중치 상향
- [intents.json](data/config/intents.json#L1888) `scholarship_leave` 인텐트의 `weight`를 1.5 → 2.0으로 상향
- `keywords`에 "장학금 중단", "장학금 유지 조건" 추가

---

## 🏗️ 근본 해결책 (4-8시간)

### Step 4: 복합 쿼리 분해 로직 구현
- [query_analyzer.py](src/rag/infrastructure/query_analyzer.py)에 `decompose_query()` 메서드 추가:
  - 패턴 기반 분해: "A하면서 B" → ["A 조건", "B 절차"]
  - 접속사 감지: "그리고", "또한", "동시에" 등
  - 복합 인텐트 매칭 시 자동 서브쿼리 생성

### Step 5: SearchUseCase에 복합 검색 병합 로직 추가
- [search_usecase.py](src/rag/application/search_usecase.py)에 `_search_composite()` 메서드 추가:
  - 분해된 서브쿼리 각각 검색
  - RRF(Reciprocal Rank Fusion)로 결과 병합
  - 중복 제거 및 다양성 보장

### Step 6: 삭제 조항 경고 시스템
- [query_handler.py](src/rag/interface/query_handler.py)에 결과 후처리 로직 추가:
  - 검색 결과 텍스트에 "삭제", "폐지" 포함 시 경고 메시지 추가
  - "이 조항은 [날짜]에 삭제되었습니다" 형식으로 사용자에게 안내

---

## Further Considerations

1. **LLM 기반 쿼리 분해 vs 패턴 기반?** : A, B 결과를 모두 활용
   - Option A: 패턴 기반 (빠름, 예측 가능) ← 권장
   - Option B: LLM 기반 (유연함, 느림)
   - Option C: 하이브리드 (패턴 실패 시 LLM)

2. **복합 검색 결과 병합 전략?** : 
   - Option A: RRF (Reciprocal Rank Fusion) ← 권장
   - Option B: Score Averaging
   - Option C: Max Score Selection

3. **테스트 우선순위**: 즉시 조치 후 동일 테스트 쿼리로 재검증 필요. 특히 "장학금+휴학", "부당대우 신고" 쿼리 재테스트 권장
