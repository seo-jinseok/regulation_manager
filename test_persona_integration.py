#!/usr/bin/env python3
"""Quick test to verify persona prompt integration."""

import logging
logging.basicConfig(level=logging.DEBUG)

from src.rag.domain.personas.persona_generator import PersonaAwareGenerator
from src.rag.application.search_usecase import SearchUseCase, REGULATION_QA_PROMPT
from src.rag.infrastructure.chroma_store import ChromaVectorStore
from src.rag.infrastructure.llm_adapter import LLMClientAdapter

# Initialize components
store = ChromaVectorStore(persist_directory="data/chroma_db")
llm_client = LLMClientAdapter(provider="openai", model="gpt-4o")
search_usecase = SearchUseCase(store=store, llm_client=llm_client, use_reranker=True)
persona_generator = PersonaAwareGenerator()

# Test query
query = "휴학 방법 알려줘"
persona = "student-undergraduate"

# Generate persona prompt
persona_prompt = persona_generator.enhance_prompt(
    base_prompt=REGULATION_QA_PROMPT,
    persona=persona,
    query=query
)

print(f"\n=== Persona: {persona} ===")
print(f"Base prompt length: {len(REGULATION_QA_PROMPT)}")
print(f"Enhanced prompt length: {len(persona_prompt)}")
print(f"Added: {len(persona_prompt) - len(REGULATION_QA_PROMPT)} characters")

# Show preview of persona enhancement
enhancement_start = len(REGULATION_QA_PROMPT)
persona_enhancement = persona_prompt[enhancement_start:enhancement_start+300]
print(f"\nPersona enhancement preview:\n{persona_enhancement}...")

# Test with custom prompt
print("\n=== Testing SearchUseCase with custom prompt ===")
try:
    result = search_usecase.ask(
        question=query,
        top_k=5,
        include_abolished=False,
        custom_prompt=persona_prompt
    )
    print(f"Response generated: {len(result.text)} characters")
    print(f"Response preview: {result.text[:200]}...")
except Exception as e:
    print(f"Error: {e}")

print("\n=== Test complete ===")
