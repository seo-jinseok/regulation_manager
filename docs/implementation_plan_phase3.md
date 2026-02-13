# 프로젝트 종합 분석 및 개선 제안

## 프로젝트 최종 목표 요약

**대학 규정 관리 시스템 (Regulation Manager)**은 다음을 목표로 합니다:

1. **HWPX → JSON 변환**: 대학 규정집(HWPX)을 계층 구조 보존하며 JSON으로 변환
2. **Hybrid RAG 검색**: BM25 + Dense + Reranker 기반 고품질 검색
3. **LLM 기반 Q&A**: 자연어 질문에 대한 정확한 답변 생성
4. **다양한 인터페이스**: CLI, Web UI (Gradio), MCP Server

---

## 현재 구현 상태 분석

### 📊 프로젝트 규모

| 항목 | 수치 |
|------|------|
| **총 소스 코드 라인** | ~16,753줄 (src/rag/) |
| **테스트 라인** | ~4,479줄 (tests/rag/unit/) |
| **테스트 케이스** | 288개 |
| **테스트 통과율** | 97.6% (281/288) |
| **평가 데이터셋** | 18개 케이스 |

### ✅ 잘 구현된 부분

1. **Clean Architecture 준수**
   - Domain/Application/Infrastructure/Interface 레이어 분리
   - 의존성 역전 원칙 적용 (IVectorStore, ILLMClient 인터페이스)

2. **고급 RAG 기법 구현**
   - Hybrid Search (BM25 + Dense + RRF Fusion)
   - BGE Reranker
   - Corrective RAG
   - Intent 확장 및 동의어 확장

3. **다양한 인터페이스 지원**
   - CLI: 대화형 모드, 스트리밍 출력
   - Web UI: Modern Glassmorphism 디자인
   - MCP Server: AI 에이전트 연동

4. **통합된 QueryHandler**
   - CLI/Web/MCP 공용 쿼리 처리 로직

---

## 개선 필요 사항

### 🔴 1. 테스트 실패 수정 (우선순위: 높음)

현재 7개 테스트가 실패 중입니다:

| 테스트 | 문제 |
|--------|------|
| `test_detects_regulation_name` | "장학금규정" → `REGULATION_NAME` 기대, 다른 타입 반환 |
| `test_detects_academic_keywords` | "휴학" → `REGULATION_NAME` 기대, 다른 타입 반환 |
| `test_get_weights_regulation` | "인사규정" 가중치 (0.5, 0.5) 기대, 다른 값 반환 |
| `test_search_rule_code_filters_by_rule_code` | 규정 코드 검색 필터 문제 |
| `test_search_with_injected_hybrid_searcher` | Hybrid Searcher 주입 테스트 |
| `test_query_handler_initialization_*` | QueryHandler 초기화 통합 테스트 |

**권장 조치:**
```bash
# 실패 테스트 상세 확인
uv run pytest tests/rag/unit/infrastructure/test_query_analyzer.py -v --tb=short
```

---

### 🟡 2. 코드 품질 및 리팩토링 (우선순위: 중간)

#### 2.1 대형 파일 분리

| 파일 | 라인 수 | 권장 |
|------|---------|------|
| [gradio_app.py](file:///Users/truestone/Dropbox/repo/University/regulation_manager/src/rag/interface/gradio_app.py) | 1,750줄 | UI 컴포넌트, 이벤트 핸들러, CSS 분리 |
| [search_usecase.py](file:///Users/truestone/Dropbox/repo/University/regulation_manager/src/rag/application/search_usecase.py) | 1,278줄 | 검색 전략별 클래스 분리 |
| [query_handler.py](file:///Users/truestone/Dropbox/repo/University/regulation_manager/src/rag/interface/query_handler.py) | 1,237줄 | 쿼리 타입별 핸들러 분리 |
| [query_analyzer.py](file:///Users/truestone/Dropbox/repo/University/regulation_manager/src/rag/infrastructure/query_analyzer.py) | 949줄 | 인텐트/동의어 로직 별도 모듈화 |
| [cli.py](file:///Users/truestone/Dropbox/repo/University/regulation_manager/src/rag/interface/cli.py) | 1,309줄 | 출력 포매팅 로직 분리 |

**권장 구조:**
```
src/rag/interface/
├── gradio/
│   ├── __init__.py
│   ├── app.py              # 메인 앱 조립
│   ├── components.py       # UI 컴포넌트
│   ├── handlers.py         # 이벤트 핸들러
│   └── styles.css          # CSS 분리
├── query_handler/
│   ├── __init__.py
│   ├── base.py             # QueryHandler 기본
│   ├── overview.py         # 규정 개요 핸들러
│   ├── article.py          # 조항 핸들러
│   └── search.py           # 검색 핸들러
```

#### 2.2 TODO 미완성 항목

```python
# tool_executor.py:190
# TODO: Implement direct article lookup
```

---

### 🟡 3. RAG 검색 품질 개선 (우선순위: 중간)

#### 3.1 평가 데이터셋 확장

현재 18개 케이스 → **50개+ 확대 권장**

| 카테고리 | 현재 | 권장 |
|----------|------|------|
| 학생 고충 | 2 | 5+ |
| 학적 | 1 | 5+ |
| 인사 | 3 | 8+ |
| 장학 | 1 | 3+ |
| 윤리 | 2 | 3+ |
| 조문/규정 검색 | 3 | 5+ |
| Edge Cases | 6 | 10+ |

#### 3.2 인텐트 커버리지 확장

[intents.json](file:///Users/truestone/Dropbox/repo/University/regulation_manager/data/config/intents.json) 현재 51개 규칙

**누락된 인텐트 예시:**
- 학위 논문 관련 ("논문 제출 언제?", "논문 심사")
- 등록금 관련 ("등록금 환불", "분할 납부")
- 시험/성적 관련 ("성적 이의신청", "재시험")
- 국제 학생 관련 ("비자 연장", "영문 증명서")

#### 3.3 Self-RAG 활성화 검토

현재 비활성화 상태 - LLM 호출 비용 대비 품질 개선 효과 측정 필요

---

### 🟢 4. 문서화 개선 (우선순위: 낮음)

#### 4.1 API 문서 자동 생성

```bash
# pdoc 또는 Sphinx 도입
uv add --dev pdoc3
uv run pdoc --html src/rag -o docs/api
```

#### 4.2 빈 docs/ 폴더 활용

```
docs/
├── api/                # 자동 생성 API 문서
├── architecture.md     # 아키텍처 상세
├── deployment.md       # 배포 가이드
└── contributing.md     # 기여 가이드
```

---

### 🟢 5. 성능 최적화 (우선순위: 낮음)

#### 5.1 임베딩 캐싱

```python
# 현재: 매 검색마다 임베딩 생성
# 개선: Redis/파일 기반 임베딩 캐시
```

#### 5.2 청크 크기 최적화

현재 규정 조항 단위 청킹 → **토큰 기반 동적 청킹** 검토

#### 5.3 배치 처리

대량 동기화 시 배치 임베딩 처리 개선

---

### 🟢 6. 기능 확장 제안 (우선순위: 낮음)

| 기능 | 설명 |
|------|------|
| **규정 변경 추적** | 버전 관리 및 변경 이력 시각화 |
| **알림 시스템** | 관심 규정 변경 시 알림 |
| **내보내기** | PDF/Word 형식 규정 내보내기 |
| **다국어 지원** | 영문 규정 번역 지원 |
| **피드백 시스템** | 사용자 피드백으로 검색 품질 개선 |

---

## 실행 계획 (권장 순서)

### Phase 1: 즉시 수정 (1-2일)

1. ✅ 7개 실패 테스트 수정
2. ✅ `tool_executor.py` TODO 완료

### Phase 2: 코드 품질 (1주)

1. 대형 파일 리팩토링 (gradio_app.py 분리)
2. CSS 외부 파일로 분리
3. 타입 힌트 일관성 개선

### Phase 3: RAG 품질 (2주)

1. 평가 데이터셋 50개로 확장
2. 자동화된 RAG 평가 파이프라인 강화
3. 인텐트/동의어 사전 확장

### Phase 4: 유지보수 (지속)

1. API 문서 자동 생성
2. 성능 모니터링 대시보드
3. 통합 테스트 커버리지 확대

---

## 요약

| 영역 | 현재 상태 | 개선 우선순위 |
|------|----------|---------------|
| **아키텍처** | ✅ Clean Architecture 준수 | 유지 |
| **테스트** | 🔴 7개 실패 (97.6%) | 🔴 높음 |
| **코드 품질** | 🟡 대형 파일 존재 | 🟡 중간 |
| **RAG 품질** | 🟡 18개 평가 케이스 | 🟡 중간 |
| **문서화** | 🟢 기본 문서 존재 | 🟢 낮음 |
| **성능** | 🟢 기본 최적화 | 🟢 낮음 |

> [!TIP]
> 가장 즉각적인 효과를 위해 **Phase 1 (테스트 수정)**부터 시작하는 것을 권장합니다.
