# 보안 가이드 (Security Guide)

이 문서는 대학 규정 관리 시스템의 보안 모범 사례와 구성 방법을 설명합니다.

## 목차

- [개요](#개요)
- [API 키 관리](#api-키-관리)
- [입력 검증](#입력-검증)
- [Redis 보안](#redis-보안)
- [환경 변수 설정](#환경-변수-설정)
- [보안 감사](#보안-감사)

---

## 개요

시스템은 다음과 같은 보안 기능을 제공합니다:

| 보안 기능 | 설명 | 버전 |
|----------|------|------|
| **API 키 검증** | API 키 유효성 검사 및 만료 알림 | v2.2.0+ |
| **입력 검증** | Pydantic 기반 입력 검증 및 악성 패턴 탐지 | v2.2.0+ |
| **Redis 보안** | 비밀번호 인증 강제 및 연결 보안 | v2.2.0+ |
| **로깅 보안** | 민감 정보 로깅 방지 | v2.2.0+ |

---

## API 키 관리

### API 키 검증

시스템은 모든 API 키를 사용 전에 검증합니다:

```python
from src.rag.domain.llm.api_key_validator import APIKeyValidator

validator = APIKeyValidator(api_key="sk-...")

# 유효성 검증
try:
    validator.validate()  # ValueError 발생 시 유효하지 않음
except ValueError as e:
    print(f"API 키 오류: {e}")

# 만료 확인
if validator.is_expiring_soon(days=7):
    print("API 키가 7일 이내에 만료됩니다!")
```

### 만료 알림

API 키 만료 7일 전에 시스템이 자동으로 경고를 출력합니다:

```text
⚠️  보안 경고: OpenAI API 키가 5일 뒤에 만료됩니다.
   만료일: 2026-02-12
   조치: 새로운 API 키를 발급받아 .env 파일을 업데이트하세요.
```

### API 키 관리 모범 사례

1. **정기적 교체**: API 키를 90일마다 교체하세요.
2. **최소 권한**: API 키에 필요한 최소 권한만 부여하세요.
3. **모니터링**: API 키 사용량을 정기적으로 확인하세요.
4. **폐기 프로세스**: 사용하지 않는 API 키를 즉시 폐기하세요.

---

## 입력 검증

### 쿼리 검증

모든 사용자 입력은 Pydantic 모델을 통해 검증됩니다:

```python
from src.rag.domain.query.query_analyzer import SearchQuery
from pydantic import ValidationError

try:
    query = SearchQuery(
        query="휴학 절차가 어떻게 되나요?",
        top_k=10
    )
    print(f"검증 완료: {query.query}")
except ValidationError as e:
    print(f"입력 오류: {e}")
```

### 검증 규칙

| 필드 | 규칙 | 설명 |
|------|------|------|
| `query` | 최대 1000자 | 쿼리 길이 제한 |
| `query` | 악성 패턴 금지 | `<script>`, `javascript:`, `eval(` 등 차단 |
| `top_k` | 1-100 범위 | 검색 결과 개수 제한 |

### 악성 패턴 탐지

다음 패턴이 감지되면 요청이 거부됩니다:

- `<script>`: 스크립트 태그
- `javascript:`: JavaScript 프로토콜
- `eval(`: 코드 실행 함수
- `document.`: DOM 접근
- `window.`: 윈도우 객체 접근

**오류 메시지 예시**:

```text
❌ 입력 검증 오류: 쿼리에 보안상 허용되지 않는 패턴이 포함되어 있습니다.
   허용되지 않는 패턴: <script>
   조치: 일반 텍스트로 질문을 다시 작성해주세요.
```

---

## Redis 보안

### 비밀번호 필수 설정

Redis는 비밀번호 인증이 필수입니다:

```bash
# .env 파일
REDIS_PASSWORD=your_secure_password_here
```

### Redis 서버 설정

Redis 서버 설정 파일 (`redis.conf`)에서 비밀번호를 설정하세요:

```text
# redis.conf
requirepass your_secure_password_here

# 보안 권장 설정
bind 127.0.0.1
protected-mode yes
port 6379
```

### Redis 연결 확인

```bash
# Redis 연결 테스트
redis-cli
> AUTH your_secure_password_here
> PING
PONG
```

### 연결 풀 모니터링

시스템은 Redis 연결 풀 상태를 지속적으로 모니터링합니다:

```python
from src.rag.infrastructure.cache.pool_monitor import ConnectionPoolMetrics

metrics = ConnectionPoolMetrics(pool)

# 연결 풀 상태 확인
status = metrics.get_pool_status()
print(f"사용 가능한 연결: {status['available_connections']}")
print(f"활성 연결: {status['active_connections']}")
```

**연결 풀 경고**:

```text
⚠️  연결 풀 경고: 사용 가능한 연결이 3개만 남았습니다.
   최대 연결: 50
   활성 연결: 47
   조치: Redis 연결 누수를 확인하세요.
```

---

## 환경 변수 설정

### 필수 환경 변수

| 변수 | 설명 | 필수 여부 | 기본값 |
|------|------|----------|--------|
| `REDIS_PASSWORD` | Redis 비밀번호 | 필수 | 없음 |
| `OPENAI_API_KEY` | OpenAI API 키 | 선택적 | 없음 |
| `GEMINI_API_KEY` | Gemini API 키 | 선택적 | 없음 |

### .env 파일 예시

```bash
# .env

# Redis 보안 (필수)
REDIS_PASSWORD=your_secure_redis_password_here

# LLM 프로바이더 (선택적)
LLM_PROVIDER=ollama
LLM_MODEL=gemma2
LLM_BASE_URL=http://localhost:11434

# 클라우드 LLM API 키 (선택적)
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...

# RAG 설정
ENABLE_SELF_RAG=true
ENABLE_HYDE=true
BM25_TOKENIZE_MODE=konlpy
```

### 환경 변수 보안

1. **파일 권한**: `.env` 파일 권한을 `600`으로 설정하세요:
   ```bash
   chmod 600 .env
   ```

2. **Git 제외**: `.gitignore`에 `.env`가 포함되어 있는지 확인하세요:
   ```text
   .env
   .env.local
   .env.*.local
   ```

3. **비밀번호 강도**: 최소 12자, 영문/숫자/특수문자 조합을 사용하세요.

---

## 보안 감사

### 로그 감사

시스템은 다음 보안 이벤트를 기록합니다:

| 이벤트 | 로그 레벨 | 설명 |
|--------|----------|------|
| API 키 만료 경고 | WARNING | API 키 만료 7일 전 |
| 입력 검증 실패 | WARNING | 악성 패턴 탐지 |
| Redis 연결 실패 | ERROR | 비밀번호 불일치 |
| 연결 풀 소진 | WARNING | 연결 풀 고갈 |

### 로그 예시

```text
2026-02-07 10:15:23 WARNING API 키 만료 경고: OpenAI API 키가 5일 뒤에 만료됩니다.
2026-02-07 10:16:45 WARNING 입력 검증 실패: 쿼리에 '<script>' 패턴 포함
2026-02-07 10:17:12 ERROR Redis 연결 실패: 비밀번호 불일치
2026-02-07 10:18:34 WARNING 연결 풀 소진: 사용 가능한 연결이 3개만 남음
```

### 보안 점검 체크리스트

정기적으로 다음 항목을 점검하세요:

- [ ] API 키 만료일 확인 (7일 이내인 경우 교체)
- [ ] Redis 비밀번호 강도 확인 (12자 이상, 영문/숫자/특수문자 조합)
- [ ] .env 파일 권한 확인 (600)
- [ ] Redis 연결 풀 상태 확인 (사용 가능 연결 5개 이상)
- [ ] 로그 파일 확인 (보안 이벤트 감지)
- [ ] 방화벽规则 확인 (Redis 6379 포트 로컬만 접근 허용)

---

## 보안 컴플라이언스

시스템은 다음 보안 표준을 준수합니다:

| 표준 | 설명 | 준수 여부 |
|------|------|----------|
| **OWASP Top 10** | 웹 애플리케이션 보안 리스크 | ✅ 준수 |
| **Input Validation** | 사용자 입력 검증 | ✅ 준수 |
| **Authentication** | API 키 및 Redis 인증 | ✅ 준수 |
| **Logging** | 보안 이벤트 로깅 | ✅ 준수 |
| **Data Protection** | 민감 정보 보호 | ✅ 준수 |

---

## 문제 해결

### 일반적인 보안 문제

| 문제 | 원인 | 해결 방법 |
|------|------|----------|
| "API 키가 만료되었습니다" | API 키 만료일 도래 | 새 API 키 발급 및 .env 업데이트 |
| "입력에 허용되지 않는 패턴이 포함되어 있습니다" | 악성 패턴 탐지 | 일반 텍스트로 질문 재작성 |
| "Redis 연결 실패" | 비밀번호 불일치 | REDIS_PASSWORD 확인 |
| "연결 풀이 소진되었습니다" | 연결 누수 | 애플리케이션 재시작 |

---

## 추가 리소스

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Redis 보안 가이드](https://redis.io/topics/security)
- [Pydantic 데이터 검증](https://docs.pydantic.dev/)
- [API 키 관리 모범 사례](https://cloud.google.com/apis/docs/api-key-best-practices)

---

**버전**: 2.2.0
**마지막 업데이트**: 2026-02-07
**유지관리자**: 규정 관리 시스템 팀
