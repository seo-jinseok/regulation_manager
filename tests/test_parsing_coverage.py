# -*- coding: utf-8 -*-
"""
Tests for parsing modules to improve coverage.

Focuses on reference_resolver and other parsing components.
"""

import unittest

from src.parsing.reference_resolver import ReferenceResolver


class TestReferenceResolverBasic(unittest.TestCase):
    """Basic tests for ReferenceResolver."""

    def setUp(self):
        self.resolver = ReferenceResolver()

    def test_init(self):
        """Test ReferenceResolver initialization."""
        resolver = ReferenceResolver()
        self.assertIsNotNone(resolver)

    def test_resolve_all_empty_list(self):
        """Test resolve_all with empty document list."""
        result = self.resolver.resolve_all([])
        # Method returns the input list (modified in place)
        self.assertEqual(result, [])

    def test_resolve_all_simple_docs(self):
        """Test resolve_all with simple documents structure."""
        docs = [
            {
                "id": "doc1",
                "title": "regulation",
                "content": [
                    {
                        "id": "c1",
                        "title": "Article 1",
                        "text": "See Article 2",
                        "references": [{"text": "Article 2"}],
                    }
                ],
            }
        ]
        result = self.resolver.resolve_all(docs)
        # Method returns the processed documents
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)


class TestReferenceResolverWithReferences(unittest.TestCase):
    """Tests for reference resolution functionality."""

    def setUp(self):
        self.resolver = ReferenceResolver()

    def test_resolve_with_empty_references(self):
        """Test handling of documents with no references."""
        docs = [
            {
                "id": "doc1",
                "title": "regulation",
                "content": [
                    {
                        "id": "c1",
                        "title": "Article 1",
                        "text": "Content",
                        "references": [],
                    }
                ],
            }
        ]
        result = self.resolver.resolve_all(docs)
        # Method returns the processed documents
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

    def test_resolve_with_none_references(self):
        """Test handling of documents with None references."""
        docs = [
            {
                "id": "doc1",
                "title": "regulation",
                "content": [
                    {
                        "id": "c1",
                        "title": "Article 1",
                        "text": "Content",
                        "references": None,
                    }
                ],
            }
        ]
        result = self.resolver.resolve_all(docs)
        # Method should handle gracefully and return the documents
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

    def test_resolve_within_document(self):
        """Test resolving references within the same document."""
        docs = [
            {
                "id": "doc1",
                "title": "regulation",
                "metadata": {"rule_code": "1-1-1"},
                "content": [
                    {
                        "id": "c1",
                        "title": "Article 1",
                        "text": "See Article 2",
                        "references": [{"text": "Article 2"}],
                    },
                    {
                        "id": "c2",
                        "title": "Article 2",
                        "text": "Details",
                        "references": [],
                    },
                ],
            }
        ]
        result = self.resolver.resolve_all(docs)
        # Method returns the processed documents
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]["content"]), 2)


class TestReferenceResolverEdgeCases(unittest.TestCase):
    """Edge case tests for ReferenceResolver."""

    def setUp(self):
        self.resolver = ReferenceResolver()

    def test_malformed_reference(self):
        """Test handling of malformed references."""
        docs = [
            {
                "id": "doc1",
                "title": "regulation",
                "content": [
                    {
                        "id": "c1",
                        "title": "Article 1",
                        "text": "See invalid_reference",
                        "references": [{"text": "invalid_reference"}],
                    }
                ],
            }
        ]
        result = self.resolver.resolve_all(docs)
        # Should handle gracefully without crashing
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

    def test_circular_references(self):
        """Test handling of circular references."""
        docs = [
            {
                "id": "doc1",
                "title": "regulation",
                "content": [
                    {
                        "id": "c1",
                        "title": "Article 1",
                        "text": "See Article 2",
                        "references": [{"text": "Article 2"}],
                    },
                    {
                        "id": "c2",
                        "title": "Article 2",
                        "text": "See Article 1",
                        "references": [{"text": "Article 1"}],
                    },
                ],
            }
        ]
        result = self.resolver.resolve_all(docs)
        # Should handle circular refs without infinite loop
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]["content"]), 2)


if __name__ == "__main__":
    unittest.main()
