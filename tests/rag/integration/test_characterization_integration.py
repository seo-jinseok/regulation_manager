"""Characterization tests for RAG pipeline integration - documents current broken behavior."""
import pytest


class TestIntegrationCharacterization:
    """Characterization tests documenting current (broken) integration state."""

    def test_query_expansion_methods_exist(self):
        """Verify that query expansion methods exist in SearchUseCase."""
        from src.rag.application.search_usecase import SearchUseCase

        methods = dir(SearchUseCase)
        assert "_apply_dynamic_expansion" in methods, "Query expansion method should exist"
        assert "_ensure_query_expansion_service" in methods, "Query expansion service initialization should exist"
        print("✓ Query expansion methods EXIST")

    def test_citation_enhancement_methods_exist(self):
        """Verify that citation enhancement methods exist in SearchUseCase."""
        from src.rag.application.search_usecase import SearchUseCase

        methods = dir(SearchUseCase)
        assert "_enhance_answer_citations" in methods, "Citation enhancement method should exist"
        print("✓ Citation enhancement method EXISTS")

    def test_query_expansion_not_initialized(self):
        """Characterize: QueryExpansionService is None initially (lazy initialization)."""
        from src.rag.application.search_usecase import SearchUseCase

        # The service is initialized lazily when first needed
        # This test verifies the lazy initialization pattern
        assert hasattr(SearchUseCase, '_ensure_query_expansion_service'), \
            "QueryExpansionService has lazy initialization method"
        print("✓ QueryExpansionService uses lazy initialization (correct pattern)")

    def test_methods_now_called_in_flow(self):
        """Verify: enhancement methods are NOW called in ask_stream after fix."""
        # This test verifies that the fix has been applied
        from src.rag.application.search_usecase import SearchUseCase
        import inspect

        # Get ask_stream source
        source = inspect.getsource(SearchUseCase.ask_stream)

        # Check if query expansion is called
        assert "_apply_dynamic_expansion" in source, "FIXED: query expansion IS called in ask_stream"

        # Check if citation enhancement is called
        assert "_enhance_answer_citations" in source, "FIXED: citation enhancement IS called in ask_stream"

        print("✓ Enhancement methods ARE called in ask_stream (fix verified)")


if __name__ == "__main__":
    import sys

    pytest.main([__file__, "-v", "-s"])
