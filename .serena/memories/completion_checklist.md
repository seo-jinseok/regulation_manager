# Completion Checklist

When a task is completed, ensure:
1. **Tests Pass**: `uv run pytest`
2. **Linting**: `ruff check .`
3. **Formatting**: `ruff format .`
4. **Verification**: Manual check with `regulation` CLI if applicable.
5. **No Secret Leaks**: Check for API keys in code.
