# Plan: RAG 테스트 프롬프트 v2.1 업데이트

최근 추가된 기능(Fact Check, synonym CLI, 보안 검증 등)을 반영하고, outdated된 내용을 수정하여 테스트 프롬프트를 현행화합니다.

## 변경 요약

| 구분 | 항목 수 |
|------|--------|
| 프롬프트와 일치 | 3개 (BM25 모드, Corrective RAG 임계값, Self-RAG/HyDE 기본) |
| 코드에만 있는 새 기능 | 8개 |
| 프롬프트 업데이트 필요 항목 | 8개 |

---

## Steps

### 1. 버전 및 변경사항 업데이트
- **위치**: 프롬프트 헤더 (Line 1-10)
- **변경**: v2.0 → v2.1
- **추가 내용**:
  ```markdown
  **v2.1 주요 변경사항:**
  - 🆕 Fact Check 시스템 검증 추가
  - 🆕 synonym CLI 명령어 문서화
  - 🆕 입력 검증 및 보안 테스트 섹션
  - 🆕 Query Decomposition 컴포넌트 추가
  - CLI 옵션 정확성 업데이트 (--tool-calling 제거)
  ```

### 2. 고급 RAG 컴포넌트 개요 테이블 업데이트
- **위치**: "고급 RAG 컴포넌트 개요" 테이블 (Line 31-45)
- **추가할 컴포넌트**:

| 컴포넌트 | 역할 | 활성화 조건 | 검증 포인트 |
|----------|------|-------------|-------------|
| **Fact Check** | 답변 생성 후 팩트체크 및 재생성 | `ENABLE_FACT_CHECK=true` (기본 OFF) | 오류 발견 시 재생성 여부 |
| **Query Decomposition** | 복합 질문 → 하위 질문 분해 | 내부 로직 | 복합 쿼리 처리 정확도 |
| **Dynamic RRF** | RRF k 값 동적 조정 | 코드 내 설정 | BM25/Dense 균형 |

- **수정할 항목**:
  - `Tool Calling` 행에서 `--tool-calling` 옵션 → `serve --mcp` 또는 내부 로직으로 수정

### 3. Phase 0.4 고급 RAG 설정 확인 업데이트
- **위치**: Phase 0.4 (Line 75-90)
- **추가할 환경변수**:
  ```bash
  # 현재 RAG 설정 확인 (확장)
  cat .env | grep -E "(ENABLE_SELF_RAG|ENABLE_HYDE|BM25_TOKENIZE_MODE|HYDE_CACHE|ENABLE_FACT_CHECK|FACT_CHECK_MAX_RETRIES)"

  # 기본 설정값 (미설정 시):
  # ENABLE_SELF_RAG=true
  # ENABLE_HYDE=true  
  # BM25_TOKENIZE_MODE=konlpy
  # HYDE_CACHE_ENABLED=true
  # ENABLE_FACT_CHECK=false  # 🆕
  # FACT_CHECK_MAX_RETRIES=2  # 🆕
  ```

### 4. Phase 1.5에 Fact Check 검증 섹션 신설
- **위치**: Phase 1.5.7 이후 (Line 270 부근)
- **신규 섹션**:
  ```markdown
  ### 1.5.8 Fact Check 검증 (선택적)

  Fact Check는 LLM 답변 생성 후 팩트체크를 수행하고 오류 발견 시 재생성합니다.

  > ⚠️ 기본 비활성화(`ENABLE_FACT_CHECK=false`)이므로 활성화 후 테스트합니다.

  **테스트 케이스:**

  | 시나리오 | 기대 동작 |
  |----------|----------|
  | 정확한 답변 | 팩트체크 통과, 재생성 없음 |
  | 부정확한 답변 | 팩트체크 실패, 최대 N회 재생성 |

  ```bash
  # Fact Check 활성화 테스트
  ENABLE_FACT_CHECK=true uv run regulation search "휴학 신청 기간" -a -n 5
  # 로그에서 "Fact check passed" 또는 "Fact check failed, regenerating" 확인
  ```

  **검증 기록:**
  ```
  [Fact Check 검증]
  - 활성화 상태: ENABLE_FACT_CHECK=true
  - 정확한 답변: ✅ 팩트체크 통과 / ❌ 불필요한 재생성
  - 부정확한 답변 시뮬레이션: ✅ 재생성 트리거 / ❌ 미트리거
  ```
  ```

### 5. 보안 검증 섹션 신설
- **위치**: Phase 1.5.8 이후 (새로운 섹션)
- **신규 섹션**:
  ```markdown
  ### 1.5.9 입력 검증 및 보안 테스트

  QueryHandler의 입력 검증이 올바르게 작동하는지 확인합니다.

  **검증 항목:**
  - 최대 쿼리 길이: 500자
  - 금지 패턴 차단: XSS, SQL Injection, Template Injection

  **테스트 케이스:**
  ```bash
  # 길이 제한 테스트 (500자 초과)
  uv run regulation search "$(python -c 'print("a"*501)')" -n 1
  # 기대: 오류 메시지 또는 잘린 쿼리

  # XSS 패턴 차단 테스트
  uv run regulation search "<script>alert(1)</script>" -n 1
  # 기대: 차단 또는 무시

  # SQL Injection 패턴 차단 테스트
  uv run regulation search "'; DROP TABLE regulations; --" -n 1
  # 기대: 차단 또는 무시
  ```

  **검증 기록:**
  ```
  [입력 검증 테스트]
  - 500자 초과 쿼리: ✅ 차단/잘림 / ❌ 그대로 처리
  - XSS 패턴: ✅ 차단 / ❌ 처리됨
  - SQL Injection 패턴: ✅ 차단 / ❌ 처리됨
  ```
  ```

### 6. synonym CLI 명령어 추가
- **위치**: 트러블슈팅 섹션 (Line 810-840)
- **추가할 내용**:
  ```bash
  # 동의어 관리 (synonym CLI)
  uv run regulation synonym suggest "휴학"    # LLM 기반 동의어 제안
  uv run regulation synonym add 휴학 학업중단  # 동의어 추가
  uv run regulation synonym remove 휴학       # 동의어 삭제
  uv run regulation synonym list              # 전체 동의어 목록
  ```

### 7. Phase 5.3 동의어 패치 방법 업데이트
- **위치**: Phase 5.3 (Line 620 부근)
- **변경**: 수동 JSON 편집 대신 CLI 사용 권장
  ```markdown
  **동의어 추가** - CLI 사용 권장:
  ```bash
  # 방법 1: CLI (권장)
  uv run regulation synonym add 휴학 학업중단

  # 방법 2: 수동 편집 (필요시)
  # data/config/synonyms.json에 직접 추가
  ```
  ```

### 8. CLI 옵션 정확성 검토
- **위치**: 고급 RAG 컴포넌트 개요 테이블의 Tool Calling 행
- **변경**: `--tool-calling` 옵션 제거 또는 수정
  - 현재 CLI에 `--tool-calling` 옵션이 없음
  - `serve --mcp` 명령으로 MCP 서버 모드에서만 Tool Calling 사용

### 9. 실제 CLI 옵션 문서화
- **위치**: Phase 4.6 CLI에서 멀티턴 테스트 (Line 540 부근)
- **변경**: 실제 옵션으로 업데이트
  ```bash
  # search 명령 실제 옵션
  uv run regulation search "query" [OPTIONS]
    -n, --top-k INT      결과 개수 (기본: 10)
    -a, --answer         LLM 답변 생성
    -q, --quick          빠른 검색 (리랭킹 생략)
    --no-rerank          리랭킹 비활성화
    --debug              디버그 모드
    --interactive        대화형 모드
    --feedback           피드백 수집 모드
  ```

---

## Further Considerations

### 1. Fact Check 테스트 깊이 : B
- **현재**: Fact Check은 기본 비활성화(`false`)
- **옵션 A**: 선택적 테스트 섹션으로 분리 (현재 계획)
- **옵션 B**: 필수 테스트로 포함 (활성화 권장)
- **결정 필요**: 테스트 시 활성화 권장 여부

### 2. 버전 넘버링 : A
- **변경 범위**: 8개 신기능 추가, CLI 옵션 수정
- **옵션 A**: v2.1 마이너 업데이트 (현재 계획)
- **옵션 B**: v3.0 메이저 업데이트
- **권장**: v2.1 (기존 워크플로우 호환 유지)

### 3. synonym CLI 자동화 통합 : 제안 ok
- **현재**: Phase 5에서 수동 JSON 편집
- **제안**: CLI 사용 권장으로 변경
- **장점**: 오타 방지, 포맷 검증 자동화

### 4. Query Decomposition 테스트 : 제안 ok
- **현재**: 프롬프트에 미포함
- **제안**: Phase 1.5에 테스트 케이스 추가
- **테스트 쿼리**: "휴학하면 장학금은? 복학은? 등록금은?"

### 5. Dynamic RRF 테스트 : 제안 ok
- **현재**: 코드에 존재하나 프롬프트에 미포함
- **제안**: Hybrid Search 검증 섹션에 통합
- **테스트 방법**: 디버그 로그에서 RRF k 값 확인

---

## 변경 파일 목록

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `.github/prompts/rag-testing.prompt.md` | 수정 | 전체 업데이트 |

## 예상 작업 시간

- 컴포넌트 테이블 업데이트: 5분
- Phase 0.4 환경변수 추가: 2분
- Fact Check 섹션 신설: 10분
- 보안 검증 섹션 신설: 10분
- synonym CLI 추가: 5분
- CLI 옵션 수정: 5분
- 검토 및 테스트: 10분

**총 예상 시간: 45분**

---

## 승인 후 작업 순서

1. [ ] 버전 및 변경사항 헤더 업데이트
2. [ ] 컴포넌트 개요 테이블 업데이트
3. [ ] Phase 0.4 환경변수 추가
4. [ ] Phase 1.5.8 Fact Check 섹션 추가
5. [ ] Phase 1.5.9 보안 검증 섹션 추가
6. [ ] 트러블슈팅에 synonym CLI 추가
7. [ ] Phase 5.3 동의어 패치 방법 업데이트
8. [ ] CLI 옵션 정확성 수정
9. [ ] 최종 검토 및 테스트
