# 명령어 별침 설정 방안

## 현재 상황

- 명령어: `uv run src.main`
- 문제: 너무 김, 혼동 (src.main, regulation, regulation.py 등)
- 사용자가 더 간단한 명령어 원함

## 제안된 방안

### 방안 A: pyproject.toml 별칭 (권장)

```toml
[project.scripts]
run_regulation = "python -m src.main"
```

- 장점: 기존 코드 수정 없음, 설정만 추가
- 단점: pyproject.toml 수정 필요

### 방안 B: 현재 방식 유지

```bash
# 현재 그대로 사용
uv run src.main file.hwpx
```

- 장점: pyproject.toml 수정 불필요
- 단점: 여전히 `src.main` 입력해야 함

## 권장사항

**방안 A (별칭 추가)**가 더 간단하고 명확합니다.

그럼 `uv run reg` 명령어를 추가하면:
- `uv run reg` → `python -m src.main`과 동일
- 사용자가 더 간단하게 `uv run reg` 사용 가능
