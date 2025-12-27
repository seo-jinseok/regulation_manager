# GitHub Copilot Instructions

This file provides project-specific context to GitHub Copilot.

## Project Context

See [AGENTS.md](../AGENTS.md) for comprehensive project context including:
- Project structure and architecture
- TDD and Clean Architecture principles
- Coding rules and conventions
- Command references

## Quick Reference

### Development Principles
- **TDD**: Write tests first (RED → GREEN → REFACTOR)
- **Clean Architecture**: Domain → Application → Infrastructure/Interface
- **Package Manager**: Use `uv` (not pip or conda)

### Key Commands
```bash
uv run pytest                    # Run tests
uv run regulation-rag search     # Search regulations
uv run regulation-rag ask        # Ask questions with LLM
```

### Coding Style
- Python 3.11+
- `snake_case` for functions/variables
- `CamelCase` for classes
- Type hints required
- Google-style docstrings

### Forbidden
- Do not modify `domain/` layer to depend on external libraries
- Do not add features without tests
- Do not manually edit `sync_state.json` or `data/chroma_db/`
