#!/usr/bin/env python3
"""
Quick verification script for Phase 2 PersonaAwareGenerator implementation.

Tests persona prompt generation and integration without running full evaluation.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_persona_generator():
    """Test PersonaAwareGenerator functionality."""
    from src.rag.domain.personas import PersonaAwareGenerator

    print("=" * 60)
    print("Phase 2 Verification: PersonaAwareGenerator")
    print("=" * 60)

    generator = PersonaAwareGenerator()

    # Test 1: Persona ID mapping
    print("\n[Test 1] Persona ID Mapping:")
    mappings = [
        ("student-undergraduate", "freshman"),
        ("professor", "professor"),
        ("parent", "parent"),
    ]
    for evaluator_id, internal_name in mappings:
        result = generator.get_persona_name(evaluator_id)
        status = "✓" if result == internal_name else "✗"
        print(f"  {status} {evaluator_id} → {result} (expected: {internal_name})")

    # Test 2: Prompt enhancement
    print("\n[Test 2] Prompt Enhancement:")
    base_prompt = "당신은 규정 전문가입니다."
    personas = ["professor", "parent", "student-international"]

    for persona in personas:
        enhanced = generator.enhance_prompt(base_prompt, persona)
        enhancement_size = len(enhanced) - len(base_prompt)
        print(f"  ✓ {persona}: +{enhancement_size} characters")

    # Test 3: Integration with SearchUseCase
    print("\n[Test 3] SearchUseCase Integration:")
    try:
        from src.rag.application.search_usecase import SearchUseCase
        import inspect

        # Check if ask() method has custom_prompt parameter
        sig = inspect.signature(SearchUseCase.ask)
        has_custom_prompt = "custom_prompt" in sig.parameters
        status = "✓" if has_custom_prompt else "✗"
        print(f"  {status} SearchUseCase.ask() has custom_prompt parameter")

    except Exception as e:
        print(f"  ✗ Error checking SearchUseCase: {e}")

    # Test 4: ParallelPersonaEvaluator integration
    print("\n[Test 4] ParallelPersonaEvaluator Integration:")
    try:
        from src.rag.domain.evaluation.parallel_evaluator import ParallelPersonaEvaluator
        import inspect

        # Check if __init__ method initializes persona_generator
        init_source = inspect.getsource(ParallelPersonaEvaluator.__init__)
        has_persona_gen_init = 'self.persona_generator' in init_source
        status = "✓" if has_persona_gen_init else "✗"
        print(f"  {status} ParallelPersonaEvaluator.__init__() has persona_generator")

        # Check if _evaluate_single_query uses persona_generator
        try:
            eval_source = inspect.getsource(ParallelPersonaEvaluator._evaluate_single_query)
            uses_persona_gen = 'self.persona_generator' in eval_source
            status = "✓" if uses_persona_gen else "✗"
            print(f"  {status} _evaluate_single_query() uses persona_generator")
        except:
            print(f"  - Could not check _evaluate_single_query() source")

    except Exception as e:
        print(f"  ✗ Error checking ParallelPersonaEvaluator: {e}")

    # Test 5: Enhanced prompt content
    print("\n[Test 5] Enhanced Prompt Content:")

    from src.rag.application.search_usecase import REGULATION_QA_PROMPT

    # Check for new sections in base prompt
    has_completeness = "Completeness Requirements" in REGULATION_QA_PROMPT
    has_citation_quality = "Citation Quality Requirements" in REGULATION_QA_PROMPT

    print(f"  {'✓' if has_completeness else '✗'} Base prompt has Completeness Requirements")
    print(f"  {'✓' if has_citation_quality else '✗'} Base prompt has Citation Quality Requirements")
    print(f"  Base prompt length: {len(REGULATION_QA_PROMPT)} characters")

    # Test 6: Persona-specific instructions
    print("\n[Test 6] Persona-Specific Instructions:")

    test_cases = [
        ("professor", "교수님"),
        ("parent", "학부모"),
        ("student-international", "외국인유학생"),
    ]

    for persona_id, keyword in test_cases:
        enhanced = generator.enhance_prompt(base_prompt, persona_id)
        has_instruction = keyword in enhanced
        status = "✓" if has_instruction else "✗"
        print(f"  {status} {persona_id} prompt contains '{keyword}'")

    # Summary
    print("\n" + "=" * 60)
    print("Phase 2 Verification Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Run full evaluation: python3 scripts/run_parallel_evaluation_simple.py")
    print("2. Check metrics improvement in target personas")
    print("3. Verify overall completeness ≥ 0.750")


if __name__ == "__main__":
    test_persona_generator()
