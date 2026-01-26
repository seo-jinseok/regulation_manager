#!/usr/bin/env python3
"""
Test LM Studio Tool Calling API.

Tests whether the current LM Studio server supports function calling.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_lmstudio_tool_calling():
    """Test LM Studio's tool calling capability."""
    import requests

    base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234")
    model = os.getenv("LLM_MODEL", "eeve-korean-instruct-7b-v2.0-preview-mlx")

    print(f"Testing LM Studio at: {base_url}")
    print(f"Model: {model}")

    # Define a simple tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_regulations",
                "description": "Search for university regulations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    # Test request
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "교원 연구년 신청 자격을 검색해줘"}
        ],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0
    }

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )

        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return False

        data = response.json()
        message = data.get("choices", [{}])[0].get("message", {})

        print("\nResponse:")
        print(f"  Content: {message.get('content', '')[:200]}")

        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            print(f"  ✅ Tool calls detected: {len(tool_calls)}")
            for tc in tool_calls:
                func = tc.get("function", {})
                print(f"    - {func.get('name')}: {func.get('arguments')}")
            return True
        else:
            print("  ⚠️ No tool calls in response")
            print("  This may mean the model doesn't support tool calling")
            print("  or the prompt wasn't interpreted as needing a tool.")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    # Load .env
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

    print("=" * 60)
    print("LM Studio Tool Calling Test")
    print("=" * 60)

    success = test_lmstudio_tool_calling()

    print("\n" + "=" * 60)
    if success:
        print("✅ LM Studio supports tool calling!")
        print("You can use the current model for FunctionGemma-style queries.")
    else:
        print("⚠️ Tool calling may not be fully supported")
        print("Consider:")
        print("  1. Loading FunctionGemma in LM Studio")
        print("  2. Installing Ollama locally and using 'ollama pull functiongemma'")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
