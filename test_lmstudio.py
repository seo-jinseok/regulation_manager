#!/usr/bin/env python3
import os
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['ALL_PROXY'] = ''

from src.rag.infrastructure.llm_adapter import LLMClientAdapter

client = LLMClientAdapter(
    provider='lmstudio',
    model='exaone-4.0-32b-mlx',
    base_url='http://game-mac-studio:1234'
)

result = client.complete('Hello test')
result_preview = result[:100] if result else "Empty"
print(f"Result: {result_preview}...")
