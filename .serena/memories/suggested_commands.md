# Suggested Commands

## Environment
- `uv sync`: Install dependencies
- `uv run regulation --help`: Show CLI help

## Conversion & Sync
- `uv run regulation convert <input.hwp>`: Convert HWP to JSON
- `uv run regulation sync <output.json>`: Sync JSON to Vector DB
- `uv run regulation status`: Check DB status
- `uv run regulation reset --confirm`: Reset database

## Search & Q&A
- `uv run regulation search "<query>"`: Search regulations
- `uv run regulation ask "<question>"`: Ask question to LLM

## Web & MCP
- `uv run regulation serve --web`: Start Gradio UI
- `uv run regulation serve --mcp`: Start MCP server

## Development
- `uv run pytest`: Run all tests
- `ruff check .`: Linting
- `ruff format .`: Formatting
