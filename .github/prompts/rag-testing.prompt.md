# RAG 시스템 품질 테스트 및 개선

AI 에이전트가 자율적으로 RAG 시스템의 품질을 테스트하고 개선하는 워크플로우입니다.

## 성공 기준

| 메트릭 | 목표 |
|--------|------|
| 테스트 통과율 | ≥ 85% |
| 인텐트 정확도 | ≥ 80% |
| 키워드 커버리지 평균 | ≥ 70% |
| 회귀 발생 | 0건 |

## 워크플로우 단계

### Phase 0: 사전 검증

1. 시스템 상태 확인:
```bash
uv run regulation status
```

2. 데이터가 없으면 동기화:
```bash
uv run regulation sync data/output/규정집.json
```

3. 이전 결과 확인:
```bash
cat data/output/improvement_plan.json 2>/dev/null || echo "이전 결과 없음"
```

### Phase 1: 평가 실행

1. 자동 평가 실행:
```bash
uv run python scripts/auto_evaluate.py --run
```

2. 결과 확인:
   - 통과율: 전체 테스트 케이스 중 통과 비율
   - 인텐트 정확도: 의도 인식 정확도
   - 실패 케이스 수: 개선 필요한 쿼리 수

**목표 달성 시 → Phase 6으로 이동**

### Phase 2: 실패 분석

1. 개선 제안 확인:
```bash
cat data/output/improvement_plan.json | python -m json.tool
```

2. 제안 유형별 처리:

| 유형 | 처리 방법 |
|------|----------|
| `intent` | Phase 3.1에서 `intents.json` 패치 |
| `synonym` | Phase 3.1에서 `synonyms.json` 패치 |
| `code_pattern` | Phase 3.2에서 `query_analyzer.py` 수정 |
| `code_weight` | Phase 3.2에서 가중치 조정 |
| `code_audience` | Phase 3.2에서 대상 감지 로직 개선 |
| `architecture` | Phase 6에서 수동 검토 보고 |

### Phase 3: 개선 적용

#### 3.1 데이터 패치 (intent/synonym)

**인텐트 추가** - `data/config/intents.json`:
```json
{
  "intent_id": "overseas_conference",
  "triggers": ["해외학회", "해외 학회", "국제학회"]
}
```

**동의어 추가** - `data/config/synonyms.json`에 새 항목 추가

#### 3.2 코드 개선 (code_*)

| 제안 유형 | 수정 파일 | 수정 대상 |
|-----------|----------|----------|
| `code_pattern` | `src/rag/infrastructure/query_analyzer.py` | `INTENT_PATTERNS` |
| `code_weight` | `src/rag/infrastructure/query_analyzer.py` | `WEIGHT_PRESETS` |
| `code_audience` | `src/rag/infrastructure/query_analyzer.py` | `*_KEYWORDS` 상수 |

**수정 원칙**: 기존 항목 삭제 금지, 새 항목만 추가

#### 3.3 패치 검증
```bash
uv run pytest tests/rag/unit/infrastructure/test_query_analyzer.py -v --tb=short
```

### Phase 4: 재평가

1. 단위 테스트 확인:
```bash
uv run pytest tests/rag/ -v --tb=short
```

2. 평가 재실행:
```bash
uv run python scripts/auto_evaluate.py --run
```

3. 결과 비교:
   - 통과율 증가 → 개선 성공
   - 통과율 동일 → 다른 접근 필요
   - 통과율 감소 → 회귀 발생, 수정 되돌리기

### Phase 5: 반복 판단

**종료 조건** (하나라도 해당 시 Phase 6으로):
1. 목표 달성: 통과율 ≥ 85%
2. 개선 한계: 2회 연속 동일 결과
3. 최대 반복: 3회 사이클 완료
4. 구조적 문제: `architecture` 유형만 남음

종료 조건 미해당 시 **Phase 2로 반복**

### Phase 6: 완료 보고

세션 요약 생성:
1. **시작 상태**: 초기 통과율, 실패 케이스 수
2. **적용된 개선**: 인텐트/동의어 추가, 코드 수정 내역
3. **최종 상태**: 최종 통과율, 남은 실패 케이스
4. **남은 문제**: 해결되지 않은 제안
5. **다음 단계 권장사항**

## 트러블슈팅

```bash
# LLM 연결 확인
curl http://localhost:11434/api/tags 2>/dev/null || echo "Ollama 미실행"

# 특정 테스트 디버깅
uv run pytest tests/rag/unit/infrastructure/test_query_analyzer.py::test_specific -v -s

# 변경사항 확인/되돌리기
git diff data/config/
git checkout -- data/config/intents.json
```

## 체크리스트

- [ ] Phase 0: 시스템 상태 확인
- [ ] Phase 1: 초기 평가 완료
- [ ] Phase 2: 실패 분석 완료
- [ ] Phase 3: 개선 적용 완료
- [ ] Phase 4: 재평가 완료
- [ ] Phase 5: 종료 조건 확인
- [ ] Phase 6: 보고서 생성
