# Product: Regulation Manager

## Overview

**Regulation Manager** is an AI-powered search system for university regulations. It converts HWP (Korean word processor) regulation files into structured JSON and provides natural language question answering using RAG (Retrieval-Augmented Generation).

## Problem Statement

University regulations contain hundreds of regulations with thousands of articles. Traditional search methods require users to:
- Manually search through PDF or HWP files
- Know specific article numbers or keywords
- Spend significant time finding relevant information

## Solution

Regulation Manager provides:
1. **HWP to JSON Conversion**: Structured data with hierarchical regulation format
2. **Hybrid Search**: Combines keyword matching (BM25) with semantic search (vector embeddings)
3. **AI-Powered Answers**: LLM generates accurate answers with proper citations
4. **Multiple Interfaces**: CLI, Web UI, and MCP server for AI agent integration

## Target Users

| User Type | Use Case |
|-----------|----------|
| **Students** | "How do I apply for leave of absence?" |
| **Professors** | "What are the eligibility requirements for sabbatical?" |
| **Staff** | Quick regulation lookup for administrative tasks |
| **Developers** | Integration with AI agents (Claude, Cursor) via MCP |

## Key Features

### Core Features
- **Regulation Conversion**: HWP → JSON with preserved hierarchy (편/장/절/조/항/호/목)
- **Vector Database**: ChromaDB with BGE-M3 embeddings (1024 dimensions)
- **Hybrid Search**: BM25 + Dense search with RRF fusion
- **Reranking**: BGE Reranker v2-m3 for precision
- **Multi-turn Conversation**: Context-aware dialogue support

### Advanced RAG Techniques
- **Agentic RAG**: LLM tool selection and execution
- **Corrective RAG**: Automatic query expansion and re-retrieval
- **Self-RAG**: LLM evaluates retrieval necessity and result quality
- **HyDE**: Hypothetical document embeddings for ambiguous queries
- **KoNLPy BM25**: Korean morpheme analysis for keyword search

### Additional Capabilities
- **Quality Evaluation**: LLM-as-Judge evaluation system with 6 persona simulation
- **A/B Testing**: Statistical framework for component comparison
- **Circuit Breaker**: LLM connection failure detection and recovery
- **Ambiguity Classification**: Automatic detection and clarification of ambiguous queries
- **Emotional Query Support**: Empathetic responses for distressed users
- **Citation Enhancement**: Accurate article number extraction and validation

## Version History

| Version | Date | Key Improvements |
|---------|------|------------------|
| v1.0 | Initial | Basic RAG system |
| v2.0 | 2025-01 | +33.8% accuracy, +70.8% speed |
| v2.1 | 2026-01 | SPEC-RAG-001: 7 new components, 467 tests |
| v2.2 | 2026-02 | SPEC-RAG-002: Quality improvements, 87.3% coverage |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 87% |
| NDCG@10 | 0.82 |
| MRR | 0.89 |
| Avg Response Time | 280ms |
| Cache Hit Rate | 78% |
| Test Coverage | 87.3% |

## Project Goals

1. **Accuracy**: Provide precise regulation-based answers
2. **Usability**: Multiple interfaces for different user types
3. **Extensibility**: Clean Architecture for easy maintenance
4. **Quality**: Comprehensive testing and evaluation framework
