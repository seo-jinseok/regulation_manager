# Graph RAG 도입 평가 결과

> 평가일: 2026-01-16

## 결론 요약

**Graph RAG는 현재 시점에서 도입하지 않습니다.** ROI가 낮고, 95% 이상의 쿼리는 현재 시스템으로 충분히 처리됩니다.

---

## 현재 시스템 vs Graph RAG

### 이미 보유한 기능
| 기능 | 설명 | 상태 |
|------|------|------|
| Hybrid Search | BM25 + Dense 검색 | ✅ |
| HyDE | 모호한 쿼리에 가상 문서 생성 | ✅ |
| Corrective RAG | 결과 품질 자동 보정 | ✅ |
| Self-RAG | 검색 필요성 판단 | ✅ |
| Fact Check | 할루시네이션 방지 | ✅ |
| 계층 구조 | `parent_path`: 규정 > 장 > 조 | ✅ |

### Graph RAG가 해결하는 문제 (< 5% 쿼리)
- 다중 홉 추론: "휴직 후 복직 시 호봉은?"
- 규정 간 참조: "A규정 제5조가 인용하는 B규정은?"
- 전체 구조 파악: "인사 관련 규정 전체 체계?"

### 비용 대비 효과
```
도입 비용:
├── 개발: 2-4주
├── 인프라: Neo4j 또는 메모리 오버헤드
├── 유지보수: 규정 변경 시 그래프 동기화
└── 응답 지연: 그래프 탐색 추가 시간

효과:
└── 5% 미만의 복잡한 질문 개선
```

---

## 대안: 점진적 Ontology 도입

### Phase 1: 경량 Ontology (✅ 완료)

**파일**: `data/config/ontology.json`

```json
{
  "entities": {
    "교원": { "subTypes": ["교수", "부교수", ...], "relatedRegulations": [...] },
    "학생": { "subTypes": ["학부생", "대학원생", ...], "relatedRegulations": [...] }
  },
  "relations": {
    "교원인사규정": { "참조": ["정관"], "세부규정": ["교원휴직규정", ...] }
  },
  "queryExpansionRules": {
    "휴직": { "교원이면": ["교원휴직규정"], "직원이면": ["취업규칙"] }
  }
}
```

**활용 방안**:
1. `QueryAnalyzer.expand_query()`에서 ontology 기반 규정 힌트 추가
2. `intents.json`과 연계하여 더 정교한 쿼리 확장
3. 검색 결과 필터링 시 관련 규정 우선순위 조정

### Phase 2: 규정 간 참조 그래프 (중기 - 필요 시)

규정 JSON에서 "참조" 관계 추출:
```
[교원인사규정] --참조--> [정관 제X조]
[교원인사규정 제36조] --세부규정--> [교원휴직규정]
```

**트리거 조건**:
- 사용자로부터 "A규정이 참조하는 B규정" 유형 질문 빈도 증가
- 피드백 분석에서 규정 간 연결 관련 불만 다수 발생

### Phase 3: 전체 Graph RAG (장기 - 필요성 검증 후)

- Neo4j 또는 NetworkX 기반
- Microsoft GraphRAG 패턴 적용

**트리거 조건**:
- Phase 2 적용 후에도 복잡한 다중 홉 쿼리 해결 안 됨
- 대학 규정 체계가 대폭 복잡해짐

---

## QueryAnalyzer 통합 방안

### ontology.json 로드
```python
# query_analyzer.py에 추가
def _load_ontology(self) -> Dict:
    ontology_path = Path(__file__).parent.parent.parent.parent / "data/config/ontology.json"
    if ontology_path.exists():
        with open(ontology_path, encoding="utf-8") as f:
            return json.load(f)
    return {}
```

### 규정 힌트 확장
```python
def get_regulation_hints(self, query: str, audience: Audience) -> List[str]:
    """ontology 기반으로 관련 규정 힌트 반환"""
    hints = []
    rules = self._ontology.get("queryExpansionRules", {})
    
    for keyword, rule in rules.items():
        if keyword in query:
            if audience == Audience.FACULTY and "교원이면" in rule:
                hints.extend(rule["교원이면"])
            elif audience == Audience.STUDENT and "학생이면" in rule:
                hints.extend(rule["학생이면"])
            else:
                hints.extend(rule.get("기본", []))
    
    return list(set(hints))
```

### 통합 우선순위
1. **즉시**: ontology.json은 참조 문서로만 활용
2. **단기**: intents.json의 `regulation_hint` 필드 확장
3. **중기**: QueryAnalyzer에 ontology 로딩 및 활용 로직 추가

---

## 사용자 만족도 향상 다른 방법

| 방법 | 효과 | 난이도 |
|------|------|--------|
| UI/UX 개선 (후속 질문 제안) | ⭐⭐⭐⭐ | 낮음 |
| 답변 포맷팅 개선 | ⭐⭐⭐ | 낮음 |
| FAQ 자동 생성 | ⭐⭐⭐⭐ | 중간 |
| 피드백 수집/분석 | ⭐⭐⭐⭐⭐ | 낮음 |
| Graph RAG | ⭐⭐ | 높음 |

---

## 최종 권장사항

| 시점 | 행동 |
|------|------|
| **현재** | ontology.json 참조 문서로 유지, Graph RAG 도입 보류 |
| **3개월 후** | 피드백 분석하여 규정 간 참조 질문 빈도 확인 |
| **필요 시** | Phase 2 (규정 간 참조 그래프) 검토 |

---

## 관련 파일

- `data/config/ontology.json` - 경량 Ontology 정의
- `data/config/intents.json` - 자연어 의도 인식 규칙
- `src/rag/infrastructure/query_analyzer.py` - 쿼리 분석기
