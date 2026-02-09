#!/bin/bash
# Wrapper script to run RAG evaluation
# Uses existing .venv if available, otherwise uses uv
# Loads .env file for API keys
# Removes proxy env vars for local LLM (lmstudio)

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "========================================"
echo "RAG Quality Evaluation"
echo "========================================"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Remove proxy environment variables for local LLM (lmstudio doesn't need them)
unset ALL_PROXY
unset all_proxy
unset HTTP_PROXY
unset http_proxy
unset HTTPS_PROXY
unset https_proxy
unset NO_PROXY
unset no_proxy

# Load .env file if exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading environment variables from .env..."
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
    echo "Environment variables loaded."
    echo ""
fi

# Function to run with existing venv
run_with_venv() {
    if [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
        echo "Using existing .venv environment..."
        exec "$PROJECT_ROOT/.venv/bin/python" "$@"
    fi
    return 1
}

# Function to run with uv
run_with_uv() {
    if command -v uv &> /dev/null; then
        echo "Using uv environment..."
        # Set project-local cache to avoid system cache issues
        export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
        mkdir -p "$UV_CACHE_DIR"
        exec uv run python "$@"
    fi
    return 1
}

# Try existing venv first, then uv
if ! run_with_venv "scripts/run_parallel_evaluation.py" "$@"; then
    echo "No existing .venv found, trying uv..."
    run_with_uv "scripts/run_parallel_evaluation.py" "$@"
fi
