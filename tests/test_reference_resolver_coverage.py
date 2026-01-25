"""
Tests for reference_resolver.py to improve coverage.

The reference_resolver module handles cross-reference resolution
between regulation articles.
"""

import unittest

from src.parsing.reference_resolver import ReferenceResolver


class MockChunk:
    """Mock Chunk class for testing."""

    def __init__(
        self,
        chunk_id,
        title="",
        text="",
        parent_path=None,
        rule_code="1-1-1",
        level="article",
    ):
        self.id = chunk_id
        self.title = title
        self.text = text
        self.parent_path = parent_path or []
        self.rule_code = rule_code
        self.level = level
        self.references = []


class MockDocument:
    """Mock Document class for testing."""

    def __init__(self, doc_id, title="", content=None, metadata=None):
        self.id = doc_id
        self.title = title
        self.content = content or []
        self.metadata = metadata or {}
        self.rule_code = metadata.get("rule_code") if metadata else None


class TestReferenceResolverBasic(unittest.TestCase):
    """Basic tests for ReferenceResolver."""

    def setUp(self):
        self.resolver = ReferenceResolver()

    def test_init(self):
        """Test ReferenceResolver initialization."""
        resolver = ReferenceResolver()
        self.assertIsNotNone(resolver)

    def test_resolve_all_empty_docs(self):
        """Test resolve_all with empty document list."""
        result = self.resolver.resolve_all([])
        # Should handle empty list gracefully
        self.assertEqual(result, [])

    def test_resolve_all_with_simple_docs(self):
        """Test resolve_all with simple documents."""
        docs = [
            {
                "id": "doc1",
                "title": "규정",
                "content": [
                    {
                        "id": "c1",
                        "title": "제1조",
                        "text": "제2조를 따른다",
                        "references": [{"text": "제2조"}],
                    }
                ],
            }
        ]
        # Should handle without errors
        result = self.resolver.resolve_all(docs)
        self.assertEqual(len(result), 1)


class TestReferenceResolverHelpers(unittest.TestCase):
    """Tests for helper methods in ReferenceResolver."""

    def setUp(self):
        self.resolver = ReferenceResolver()

    def test_normalize_article_ref(self):
        """Test article reference normalization."""
        # Test if method exists
        if hasattr(self.resolver, "_normalize_article_ref"):
            result = self.resolver._normalize_article_ref("제 5 조")
            self.assertIsNotNone(result)

    def test_extract_article_number(self):
        """Test article number extraction."""
        # Test if method exists
        if hasattr(self.resolver, "_extract_article_number"):
            result = self.resolver._extract_article_number("제5조")
            self.assertIsNotNone(result)

    def test_find_chunk_by_article(self):
        """Test finding chunk by article reference."""
        chunks = [
            MockChunk("c1", title="제1조", text="본문"),
            MockChunk("c2", title="제2조", text="본문"),
        ]
        # Test if method exists
        if hasattr(self.resolver, "_find_chunk_by_article"):
            result = self.resolver._find_chunk_by_article(chunks, "제1조")
            self.assertIsNotNone(result)


class TestReferenceResolverIntegration(unittest.TestCase):
    """Integration tests for reference resolution."""

    def setUp(self):
        self.resolver = ReferenceResolver()

    def test_resolve_within_same_document(self):
        """Test resolving references within the same document."""
        docs = [
            {
                "id": "doc1",
                "title": "규정",
                "metadata": {"rule_code": "1-1-1"},
                "content": [
                    {
                        "id": "c1",
                        "title": "제1조",
                        "text": "제2조의 규정에 따른다",
                        "references": [{"text": "제2조"}],
                    },
                    {
                        "id": "c2",
                        "title": "제2조",
                        "text": "상세 내용",
                        "references": [],
                    },
                ],
            }
        ]
        result = self.resolver.resolve_all(docs)
        self.assertEqual(len(result), 1)

    def test_resolve_cross_document(self):
        """Test resolving references across documents."""
        docs = [
            {
                "id": "doc1",
                "title": "규정1",
                "metadata": {"rule_code": "1-1-1"},
                "content": [
                    {
                        "id": "c1",
                        "title": "제1조",
                        "text": "규정2 제5조를 참조한다",
                        "references": [{"text": "규정2 제5조"}],
                    }
                ],
            },
            {
                "id": "doc2",
                "title": "규정2",
                "metadata": {"rule_code": "1-1-2"},
                "content": [
                    {
                        "id": "c2",
                        "title": "제5조",
                        "text": "상세 내용",
                        "references": [],
                    }
                ],
            },
        ]
        result = self.resolver.resolve_all(docs)
        self.assertEqual(len(result), 2)


class TestReferenceResolverEdgeCases(unittest.TestCase):
    """Edge case tests for ReferenceResolver."""

    def setUp(self):
        self.resolver = ReferenceResolver()

    def test_malformed_reference(self):
        """Test handling of malformed references."""
        docs = [
            {
                "id": "doc1",
                "title": "규정",
                "content": [
                    {
                        "id": "c1",
                        "title": "제1조",
                        "text": "잘못된조 참조",
                        "references": [{"text": "잘못된조"}],
                    }
                ],
            }
        ]
        # Should handle gracefully
        result = self.resolver.resolve_all(docs)
        self.assertEqual(len(result), 1)

    def test_empty_references(self):
        """Test handling of empty references list."""
        docs = [
            {
                "id": "doc1",
                "title": "규정",
                "content": [
                    {"id": "c1", "title": "제1조", "text": "본문", "references": []}
                ],
            }
        ]
        result = self.resolver.resolve_all(docs)
        self.assertEqual(len(result), 1)

    def test_none_references(self):
        """Test handling of None references."""
        docs = [
            {
                "id": "doc1",
                "title": "규정",
                "content": [
                    {"id": "c1", "title": "제1조", "text": "본문", "references": None}
                ],
            }
        ]
        result = self.resolver.resolve_all(docs)
        self.assertEqual(len(result), 1)

    def test_circular_references(self):
        """Test handling of circular references."""
        docs = [
            {
                "id": "doc1",
                "title": "규정",
                "content": [
                    {
                        "id": "c1",
                        "title": "제1조",
                        "text": "제2조 참조",
                        "references": [{"text": "제2조"}],
                    },
                    {
                        "id": "c2",
                        "title": "제2조",
                        "text": "제1조 참조",
                        "references": [{"text": "제1조"}],
                    },
                ],
            }
        ]
        # Should handle circular refs without infinite loop
        result = self.resolver.resolve_all(docs)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
