# 테스트 인프라 가이드 (Testing Infrastructure Guide)

이 문서는 대학 규정 관리 시스템의 테스트 인프라 구성과 사용 방법을 설명합니다.

## 목차

- [개요](#개요)
- [pytest 설정](#pytest-설정)
- [단위 테스트](#단위-테스트)
- [통합 테스트](#통합-테스트)
- [성능 벤치마크](#성능-벤치마크)
- [테스트 실행](#테스트-실행)
- [커버리지](#커버리지)

---

## 개요

시스템은 다음과 같은 테스트 인프라를 제공합니다:

| 테스트 유형 | 수량 | 설명 | 목표 커버리지 |
|----------|------|------|-------------|
| **단위 테스트** | 67개 | 개별 컴포넌트 테스트 | 85%+ |
| **통합 테스트** | 25개 | RAG 파이프라인 테스트 | 80%+ |
| **벤치마크** | 15개 | 성능 측정 | N/A |
| **전체** | 107개 | 종합 테스트 스위트 | 87.3% |

---

## pytest 설정

### pytest.ini 구성

```ini
# pytest.ini
[pytest]
# 테스트 경로
testpaths = tests

# 테스트 파일 패턴
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# asyncio 설정
asyncio_mode = auto

# 추가 옵션
addopts =
    -v
    --strict-markers
    --tb=short
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=85

# 마커 정의
markers =
    unit: 단위 테스트
    integration: 통합 테스트
    benchmark: 성능 벤치마크
    slow: 느린 테스트
```

### 테스트 의존성

```toml
# pyproject.toml
[tool.poetry.dev-dependencies]
pytest = "^8.0.0"
pytest-asyncio = "^0.23.0"
pytest-cov = "^5.0.0"
pytest-benchmark = "^4.0.0"
pytest-timeout = "^2.2.0"
pytest-xdist = "^3.5.0"  # 병렬 테스트 실행
```

---

## 단위 테스트

### Kiwi 토크나이저 테스트

```python
# tests/unit/test_kiwi_tokenizer.py
import pytest
from src.rag.infrastructure.nlp.kiwi_tokenizer import KiwiTokenizer

class TestKiwiTokenizer:
    """Kiwi 토크나이저 단위 테스트"""

    def test_singleton_pattern(self):
        """싱글톤 패턴 테스트"""
        tokenizer1 = KiwiTokenizer.get_instance()
        tokenizer2 = KiwiTokenizer.get_instance()
        assert tokenizer1 is tokenizer2

    def test_tokenize_korean_text(self):
        """한국어 텍스트 토큰화 테스트"""
        tokenizer = KiwiTokenizer.get_instance()
        tokens = tokenizer.tokenize("교원연구년신청절차")
        assert "교원" in tokens
        assert "연구년" in tokens
        assert "신청" in tokens
        assert "절차" in tokens

    @pytest.mark.slow
    def test_lazy_loading(self):
        """지연 로딩 테스트"""
        # 첫 번째 호출 시에만 초기화
        assert KiwiTokenizer._instance is None
        tokenizer = KiwiTokenizer.get_instance()
        assert KiwiTokenizer._instance is not None
```

### BM25 캐시 테스트

```python
# tests/unit/test_bm25_cache.py
import pytest
import tempfile
import os
from src.rag.infrastructure.cache.bm25_cache import BM25Cache

class TestBM25Cache:
    """BM25 캐시 단위 테스트"""

    def test_save_and_load_with_msgpack(self):
        """msgpack 직렬화 테스트"""
        cache = BM25Cache()
        index = cache.create_sample_index()

        # msgpack으로 저장
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        try:
            cache.save_with_msgpack(index, temp_path)
            assert os.path.exists(temp_path)

            # msgpack에서 로드
            loaded_index = cache.load_with_msgpack(temp_path)
            assert loaded_index is not None
        finally:
            os.unlink(temp_path)

    def test_msgpack_vs_pickle_size(self):
        """msgpack이 pickle보다 파일 크기가 작은지 확인"""
        import pickle
        import msgpack

        cache = BM25Cache()
        index = cache.create_sample_index()

        with tempfile.NamedTemporaryFile(delete=False) as f1, \
             tempfile.NamedTemporaryFile(delete=False) as f2:
            pickle_path = f1.name
            msgpack_path = f2.name

        try:
            # pickle 저장
            with open(pickle_path, 'wb') as f:
                pickle.dump(index.to_dict(), f)

            # msgpack 저장
            with open(msgpack_path, 'wb') as f:
                msgpack.dump(index.to_dict(), f)

            # 파일 크기 비교
            pickle_size = os.path.getsize(pickle_path)
            msgpack_size = os.path.getsize(msgpack_path)
            assert msgpack_size < pickle_size
        finally:
            os.unlink(pickle_path)
            os.unlink(msgpack_path)
```

### HyDE 캐시 테스트

```python
# tests/unit/test_hyde_cache.py
import pytest
from src.rag.infrastructure.cache.hyde_cache import HyDECache

class TestHyDECache:
    """HyDE 캐시 단위 테스트"""

    def test_lru_cache_maxsize(self):
        """LRU 캐시 최대 크기 테스트"""
        cache = HyDECache(maxsize=10)

        # 11개 항목 추가 (maxsize=10 초과)
        for i in range(11):
            cache.get(f"query_{i}")

        # 첫 번째 항목이 제거되었는지 확인
        assert cache.get_cache_info().currsize == 10

    def test_zlib_compression(self):
        """zlib 압축 테스트"""
        import zlib
        original = "가상의 규정 문서입니다. " * 100

        compressed = zlib.compress(original.encode())
        decompressed = zlib.decompress(compressed).decode()

        assert decompressed == original
        assert len(compressed) < len(original.encode())

    @pytest.mark.benchmark
    def test_cache_hit_rate(self, benchmark):
        """캐시 적중률 벤치마크"""
        cache = HyDECache(maxsize=1000)

        # 캐시 웜업
        for i in range(100):
            cache.get(f"query_{i}")

        # 캐시 적중 테스트
        def cache_hit_test():
            return cache.get("query_50")  # 캐시 적중

        result = benchmark(cache_hit_test)
        assert result is not None
```

### API 키 검증 테스트

```python
# tests/unit/test_api_key_validator.py
import pytest
from datetime import datetime, timedelta
from src.rag.domain.llm.api_key_validator import APIKeyValidator

class TestAPIKeyValidator:
    """API 키 검증 단위 테스트"""

    def test_validate_valid_key(self):
        """유효한 API 키 검증 테스트"""
        validator = APIKeyValidator(api_key="sk-valid-key")
        assert validator.validate() is True

    def test_validate_empty_key(self):
        """빈 API 키 검증 테스트"""
        validator = APIKeyValidator(api_key="")
        with pytest.raises(ValueError, match="API key is missing"):
            validator.validate()

    def test_is_expired_true(self):
        """만료된 API 키 테스트"""
        expiry_date = datetime.now() - timedelta(days=1)
        validator = APIKeyValidator(api_key="sk-expired-key")
        validator.expiry_date = expiry_date

        assert validator.is_expired() is True
        with pytest.raises(ValueError, match="API key has expired"):
            validator.validate()

    def test_is_expiring_soon_true(self):
        """7일 이내 만료 API 키 테스트"""
        expiry_date = datetime.now() + timedelta(days=5)
        validator = APIKeyValidator(api_key="sk-expiring-soon-key")
        validator.expiry_date = expiry_date

        assert validator.is_expiring_soon(days=7) is True
```

---

## 통합 테스트

### RAG 파이프라인 통합 테스트

```python
# tests/integration/test_rag_pipeline.py
import pytest
from src.rag.application.search_service import SearchService

class TestRAGPipeline:
    """RAG 파이프라인 통합 테스트"""

    @pytest.mark.asyncio
    async def test_end_to_end_search(self):
        """종단 간 검색 테스트"""
        service = SearchService()

        query = "휴학 신청 절차가 어떻게 되나요?"
        results = await service.search(query, top_k=5)

        assert len(results) > 0
        assert all(r.citation for r in results)
        assert all(r.confidence > 0.5 for r in results)

    @pytest.mark.asyncio
    async def test_cache_integration(self):
        """캐시 통합 테스트"""
        service = SearchService()

        query = "교원 연구년 자격"

        # 첫 번째 검색 (캐시 미스)
        results1 = await service.search(query, top_k=5)

        # 두 번째 검색 (캐시 적중)
        results2 = await service.search(query, top_k=5)

        assert len(results1) == len(results2)
        assert [r.id for r in results1] == [r.id for r in results2]

    @pytest.mark.asyncio
    async def test_bm25_dense_fusion(self):
        """BM25 + Dense 검색 융합 테스트"""
        service = SearchService()

        query = "제15조"
        results = await service.search(query, top_k=10)

        # BM25와 Dense 결과가 융합되었는지 확인
        assert len(results) > 0
        assert any(r.score > 0.8 for r in results)
```

---

## 성능 벤치마크

### 검색 성능 벤치마크

```python
# tests/benchmarks/test_performance.py
import pytest

class TestPerformance:
    """성능 벤치마크 테스트"""

    @pytest.mark.benchmark
    def test_bm25_retrieval_latency(self, benchmark):
        """BM25 검색 지연 시간 측정"""
        from src.rag.infrastructure.search.bm25_retriever import BM25Retriever

        retriever = BM25Retriever()
        query = "휴학 규정"

        result = benchmark(retriever.retrieve, query, top_k=10)
        assert len(result) > 0

    @pytest.mark.benchmark
    def test_dense_retrieval_latency(self, benchmark):
        """Dense 검색 지연 시간 측정"""
        from src.rag.infrastructure.search.dense_retriever import DenseRetriever

        retriever = DenseRetriever()
        query = "휴학 규정"

        result = benchmark(retriever.retrieve, query, top_k=10)
        assert len(result) > 0

    @pytest.mark.benchmark
    def test_hyde_generation_latency(self, benchmark):
        """HyDE 생성 지연 시간 측정"""
        from src.rag.infrastructure.hyde.hyde_generator import HyDEGenerator

        generator = HyDEGenerator()
        query = "학교에 가기 싫어"

        result = benchmark(generator.generate, query)
        assert len(result) > 0

    @pytest.mark.benchmark
    def test_full_rag_pipeline_latency(self, benchmark):
        """전체 RAG 파이프라인 지연 시간 측정"""
        from src.rag.application.search_service import SearchService

        service = SearchService()
        query = "휴학 신청 절차가 어떻게 되나요?"

        async def run_search():
            return await service.search(query, top_k=5)

        result = benchmark(asyncio.run, run_search())
        assert len(result) > 0
```

---

## 테스트 실행

### 전체 테스트 실행

```bash
# 전체 테스트 실행
pytest

# 상세 출력 모드
pytest -v

# 특정 마커만 실행
pytest -m unit        # 단위 테스트만
pytest -m integration # 통합 테스트만
pytest -m benchmark   # 벤치마크만
```

### 병렬 테스트 실행

```bash
# pytest-xdist로 병렬 실행 (2개 worker)
pytest -n 2

# worker 수 지정
pytest -n 4
```

### 테스트 필터링

```bash
# 특정 파일만 테스트
pytest tests/unit/test_bm25_cache.py

# 특정 클래스만 테스트
pytest tests/unit/test_bm25_cache.py::TestBM25Cache

# 특정 테스트만 실행
pytest tests/unit/test_bm25_cache.py::TestBM25Cache::test_save_and_load_with_msgpack
```

---

## 커버리지

### 커버리지 리포트 생성

```bash
# 터미널에 커버리지 출력
pytest --cov=src --cov-report=term-missing

# HTML 리포트 생성
pytest --cov=src --cov-report=html

# 특정 모듈만 커버리지 확인
pytest --cov=src.rag.infrastructure.cache --cov-report=term-missing
```

### 커버리지 목표

| 모듈 | 현재 커버리지 | 목표 | 상태 |
|------|--------------|------|------|
| `src.rag.infrastructure.nlp` | 89.2% | 85% | ✅ |
| `src.rag.infrastructure.cache` | 91.5% | 85% | ✅ |
| `src.rag.domain.llm` | 87.3% | 85% | ✅ |
| `src.rag.domain.query` | 86.7% | 85% | ✅ |
| `src.rag.application` | 84.1% | 85% | ⚠️ |
| **전체** | **87.3%** | **85%** | ✅ |

### 커버리지 바젯 설정

```ini
# pytest.ini
addopts =
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=85  # 85% 미만 시 테스트 실패
```

---

## CI/CD 통합

### GitHub Actions 워크플로우

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-asyncio pytest-cov pytest-benchmark

      - name: Run tests
        run: pytest --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

---

## 테스트 작성 가이드

### 테스트 네이밍 컨벤션

```python
# 올바른 테스트 이름
def test_save_bm25_index_with_msgpack():
    """기능_테스트"""

def test_api_key_validation_raises_error_when_empty():
    """기능_예상_결과_when_조건"""

# 잘못된 테스트 이름
def test1():  # 너무 일반적
def test_bm25():  # 무엇을 테스트하는지 불명확
```

### AAA 패턴 (Arrange-Act-Assert)

```python
def test_api_key_validator_rejects_expired_key(self):
    # Arrange (준비)
    expiry_date = datetime.now() - timedelta(days=1)
    validator = APIKeyValidator(api_key="sk-expired-key")
    validator.expiry_date = expiry_date

    # Act (실행)
    with pytest.raises(ValueError) as exc_info:
        validator.validate()

    # Assert (단언)
    assert "expired" in str(exc_info.value).lower()
```

### 테스트 픽스처 활용

```python
# conftest.py
import pytest
from src.rag.infrastructure.cache.bm25_cache import BM25Cache

@pytest.fixture
def bm25_cache():
    """BM25 캐시 인스턴스"""
    return BM25Cache()

@pytest.fixture
def sample_query():
    """샘플 쿼리"""
    return "휴학 규정"

# 테스트에서 픽스처 사용
def test_bm25_cache(bm25_cache, sample_query):
    results = bm25_cache.retrieve(sample_query)
    assert len(results) > 0
```

---

## 문제 해결

### 일반적인 테스트 문제

| 문제 | 원인 | 해결 방법 |
|------|------|----------|
| "ImportError" | 의존성 미설치 | `uv sync` 실행 |
| "asyncio_test timeout" | 비동기 테스트 타임아웃 | `pytest.ini`에 `asyncio_mode=auto` 추가 |
| "Coverage below 85%" | 커버리지 부족 | 테스트 케이스 추가 |
| "ModuleNotFoundError" | Python 경로 문제 | `PYTHONPATH` 설정 또는 `pytest.ini` 확인 |

---

## 추가 리소스

- [pytest 문서](https://docs.pytest.org/)
- [pytest-asyncio 문서](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov 문서](https://pytest-cov.readthedocs.io/)
- [pytest-benchmark 문서](https://pytest-benchmark.readthedocs.io/)

---

**버전**: 2.2.0
**마지막 업데이트**: 2026-02-07
**유지관리자**: 규정 관리 시스템 팀
