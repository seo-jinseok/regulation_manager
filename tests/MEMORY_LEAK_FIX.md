# 메모리 누수 해결 가이드

## 문제 진단

### 원인 분석

1. **ChromaDB PersistentClient 누수**
   - `ChromaVectorStore` 클래스에 리소스 정리 메서드 부재
   - 각 테스트에서 생성된 클라이언트가 메모리에 유지됨

2. **pytest-cov 데이터 축적**
   - 2111개 테스트 실행 시 coverage 데이터가 계속 메모리에 축적
   - 데이터 파일 위치가 지정되지 않아 기본 위치 사용

3. **Mock 객체 누적**
   - `unittest.mock` 객체들이 순환 참조 생성
   - `gc.collect()`만으로는 완전히 수거되지 않음

## 해결 방안

### 1. Coverage 설정 최적화 (`.coveragerc`)

```ini
[run]
data_file = .cache/coverage  # 임시 디렉토리 사용

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    if TYPE_CHECKING:
```

### 2. ChromaVectorStore 리소스 정리

`src/rag/infrastructure/chroma_store.py`에 추가:

```python
def close(self):
    """Release ChromaDB client resources."""
    try:
        if hasattr(self, "_client") and self._client is not None:
            self._collection = None
            self._client = None
    except Exception:
        pass

def __enter__(self):
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()
    return False

def __del__(self):
    self.close()
```

### 3. pytest-xdist 병렬 실행

**설치**:
```bash
uv pip install pytest-xdist pytest-timeout
```

**사용법**:

```bash
# 4개 worker로 병렬 실행 (권장)
pytest -n 4 --cov=src tests/

# CPU 코어 수에 맞춰 자동 설정
pytest -n auto --cov=src tests/

# 특정 테스트만 실행
pytest -n 2 tests/test_converter_coverage.py -x
```

### 4. 개선된 conftest.py

`tests/conftest.py`의 새로운 fixture들:

```python
# Vector store fixture (자동 정리)
@pytest.fixture
def vector_store():
    """Create a test vector store with automatic cleanup."""
    store = ChromaVectorStore(...)
    yield store
    store.close()  # 자동 정리

# Mock cleanup fixture
@pytest.fixture
def clean_mock():
    """Mock with automatic cleanup."""
    # 사용 예시
    mock.patch(...)  # 자동으로 정리됨
```

## 테스트 실행 방법

### 방법 1: 스크립트 사용 (권장)

```bash
./scripts/run_tests_with_memory_limit.sh 4
```

### 방법 2: 직접 pytest 실행

```bash
# 전체 테스트 (4개 worker)
pytest -n 4 --cov=src tests/

# 빠른 테스트 (coverage 없이)
pytest -n 4 tests/ -x

# 특정 모듈만
pytest -n 2 tests/test_converter_coverage.py -v
```

## 메모리 사용량 모니터링

### macOS

```bash
# 터미널 1: 테스트 실행
pytest -n 4 --cov=src tests/

# 터미널 2: 메모리 모니터링
while true; do
    clear
    ps aux | grep pytest | head -10
    echo "---"
    ps aux | grep pytest | awk '{sum+=$6} END {print "Total MB:", sum/1024}'
    sleep 2
done
```

### Activity Monitor 사용

1. Activity Monitor 열기
2. CPU 탭에서 Python 프로세스 확인
3. 메모리 탭에서 실시간 메모리 사용량 확인

## 예상 메모리 개선 효과

### 변경 전
- 단일 프로세스: 31GB (누적)
- 테스트 실패 후 메모리 유지

### 변경 후
- Worker당 약 2-4GB (격리됨)
- 테스트 완료 후 자동 해제
- 전체 사용량: workers × 4GB = 8-16GB (4 workers 기준)

## 추가 최적화 팁

### 1. 테스트 분할 실행

```bash
# Converter 테스트만
pytest -n 2 tests/test_converter*.py -x

# RAG 테스트만
pytest -n 2 tests/rag/ -x

# 단위 테스트만
pytest -n 4 tests/rag/unit/ -x
```

### 2. Coverage 레벨 조정

```bash
# 빠른 테스트 (coverage 없음)
pytest -n 4 tests/ -x --no-cov

# 최소 coverage만
pytest -n 4 --cov=src --cov-report=term tests/
```

### 3. 테스트 타임아웃 설정

```bash
# 개별 테스트 30초 타임아웃
pytest -n 4 --timeout=30 tests/

# 전체 10분 타임아웃
pytest -n 4 --timeout=600 tests/
```

## 문제 해결

### pytest-xdist 관련 이슈

**문제**: "import error" 또는 "module not found"
**해결**:
```bash
uv pip install pytest-xdist pytest-timeout
```

**문제**: "worker crashed"
**해결**: Worker 수 줄이기
```bash
pytest -n 2 --cov=src tests/  # 4에서 2로 줄임
```

### ChromaDB 관련 이슈

**문제**: "database is locked"
**해결**: `vector_store` fixture 사용 (자동으로 임시 디렉토리 사용)

**문제**: 테스트 간 데이터 겹침
**해결**: 각 테스트에서 고유 collection 사용

## 참고 자료

- [pytest-xdist 문서](https://pytest-xdist.readthedocs.io/)
- [Coverage.py 설정](https://coverage.readthedocs.io/)
- [Python Memory Profiling](https://docs.python.org/3/library/tracemalloc.html)
