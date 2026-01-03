"""
MCP Server for Regulation RAG System.

Provides Model Context Protocol interface for:
- Searching regulations with hybrid search + reranking
- Asking questions with LLM-powered answers
- Checking sync status

Note: Database management (sync, reset) should be done via CLI for security.

Usage:
    # Run as MCP server (stdio mode)
    uv run regulation-mcp

    # Development with MCP Inspector
    uv run mcp dev src/rag/interface/mcp_server.py
"""

import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Load .env file for environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from ..application.full_view_usecase import FullViewUseCase
from ..config import get_config
from ..infrastructure.hybrid_search import Audience, QueryAnalyzer
from ..infrastructure.json_loader import JSONDocumentLoader
from .common import decide_search_mode
from .formatters import (
    clean_path_segments,
    filter_by_relevance,
    normalize_relevance_scores,
)
from .query_handler import QueryHandler, QueryOptions, QueryType

# Initialize MCP server with metadata
mcp = FastMCP(
    name="regulation-rag",
    instructions="대학 규정집 RAG 검색 및 Q&A 서버. 규정 검색, AI 질문-답변 기능을 제공합니다.",
)


def _get_store():
    """Get ChromaVectorStore instance."""
    from ..infrastructure.chroma_store import ChromaVectorStore

    config = get_config()
    return ChromaVectorStore(persist_directory=config.db_path)


def _parse_audience(value: Optional[str]) -> Optional[Audience]:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in ("교수", "faculty"):
        return Audience.FACULTY
    if normalized in ("학생", "student"):
        return Audience.STUDENT
    if normalized in ("직원", "staff"):
        return Audience.STAFF
    return None


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool()
def search_regulations(
    query: str,
    top_k: int = 5,
    include_abolished: bool = False,
    use_rerank: bool = True,
    audience: Optional[str] = None,
) -> str:
    """
    규정을 검색합니다. Hybrid Search (BM25 + Dense) 및 BGE Reranking을 사용합니다.

    Args:
        query: 검색 쿼리 (예: "교원 연구년 신청 자격", "제15조", "휴학 절차")
        top_k: 반환할 결과 수 (기본: 5)
        include_abolished: 폐지된 규정 포함 여부 (기본: False)
        use_rerank: BGE Reranker 사용 여부 (기본: True)
        audience: 대상 선택 (교수/학생/직원). 미지정 시 모호하면 재질문.

    Returns:
        검색 결과 목록 (JSON 형식)
    """
    from ..application.search_usecase import SearchUseCase

    analyzer = QueryAnalyzer()
    mode = decide_search_mode(query)
    full_view_usecase = FullViewUseCase(JSONDocumentLoader())

    store = _get_store()

    if store.count() == 0:
        return json.dumps(
            {
                "success": False,
                "error": "데이터베이스가 비어 있습니다. CLI에서 'regulation-rag sync'를 실행하세요.",
            },
            ensure_ascii=False,
        )

    if mode == "full_view":
        matches = full_view_usecase.find_matches(query)
        if not matches:
            return json.dumps(
                {
                    "success": True,
                    "type": "full_view",
                    "results": [],
                    "message": "해당 규정을 찾을 수 없습니다.",
                },
                ensure_ascii=False,
            )
        if len(matches) > 1:
            return json.dumps(
                {
                    "success": True,
                    "type": "clarification",
                    "reason": "regulation_ambiguous",
                    "options": [m.title for m in matches],
                },
                ensure_ascii=False,
            )
        view = full_view_usecase.get_full_view(matches[0].rule_code)
        if not view:
            return json.dumps(
                {"success": False, "error": "규정 전문을 불러오지 못했습니다."},
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "success": True,
                "type": "full_view",
                "regulation_name": view.title,
                "rule_code": view.rule_code,
                "toc": view.toc,
                "content": view.content,
                "addenda": view.addenda,
            },
            ensure_ascii=False,
        )

    audience_override = _parse_audience(audience)
    if audience_override is None and analyzer.is_audience_ambiguous(query):
        return json.dumps(
            {
                "success": True,
                "type": "clarification",
                "reason": "audience_ambiguous",
                "options": ["교수", "학생", "직원"],
            },
            ensure_ascii=False,
        )

    # SearchUseCase가 HybridSearcher를 자동 초기화
    search = SearchUseCase(store, use_reranker=use_rerank)

    results = search.search_unique(
        query,
        top_k=top_k,
        include_abolished=include_abolished,
        audience_override=audience_override,
    )

    if not results:
        return json.dumps(
            {"success": True, "results": [], "message": "검색 결과가 없습니다."},
            ensure_ascii=False,
        )

    # Format results
    formatted_results = []
    for i, r in enumerate(results, 1):
        reg_name = r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
        path = (
            " > ".join(clean_path_segments(r.chunk.parent_path))
            if r.chunk.parent_path
            else r.chunk.title
        )

        formatted_results.append(
            {
                "rank": i,
                "regulation_name": reg_name,
                "rule_code": r.chunk.rule_code,
                "path": path,
                "text": r.chunk.text,
                "score": round(r.score, 4),
            }
        )

    return json.dumps(
        {
            "success": True,
            "query": query,
            "results": formatted_results,
        },
        ensure_ascii=False,
    )


@mcp.tool()
def ask_regulations(
    question: str,
    top_k: int = 5,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    audience: Optional[str] = None,
) -> str:
    """
    규정에 대해 질문하고 AI 답변을 받습니다.

    Args:
        question: 질문 (예: "교원 연구년 신청 자격은?", "휴학하려면 어떻게 해야 하나요?")
        top_k: 참고할 규정 수 (기본: 5)
        provider: LLM 프로바이더 (lmstudio, ollama, openai, gemini, openrouter)
        model: 모델 이름 (기본: 프로바이더별 기본값)
        base_url: 로컬 서버 URL (lmstudio, ollama용)
        audience: 대상 선택 (교수/학생/직원). 미지정 시 모호하면 재질문.

    Returns:
        AI 답변 및 참고 규정 (JSON 형식)
    """
    from ..application.search_usecase import SearchUseCase
    from ..infrastructure.llm_adapter import LLMClientAdapter

    store = _get_store()
    config = get_config()

    if store.count() == 0:
        return json.dumps(
            {
                "success": False,
                "error": "데이터베이스가 비어 있습니다. CLI에서 'regulation-rag sync'를 실행하세요.",
            },
            ensure_ascii=False,
        )

    # Use config defaults if not provided
    provider = provider or config.llm_provider
    model = model or config.llm_model
    base_url = base_url or config.llm_base_url

    try:
        llm = LLMClientAdapter(
            provider=provider,
            model=model,
            base_url=base_url,
        )
    except Exception as e:
        return json.dumps(
            {
                "success": False,
                "error": f"LLM 초기화 실패: {str(e)}",
                "hint": "로컬 LLM 서버가 실행 중인지 확인하세요."
                if provider in ("lmstudio", "ollama")
                else "API 키 설정을 확인하세요.",
            },
            ensure_ascii=False,
        )

    analyzer = QueryAnalyzer()
    audience_override = _parse_audience(audience)
    if audience_override is None and analyzer.is_audience_ambiguous(question):
        return json.dumps(
            {
                "success": True,
                "type": "clarification",
                "reason": "audience_ambiguous",
                "options": ["교수", "학생", "직원"],
            },
            ensure_ascii=False,
        )

    # SearchUseCase가 HybridSearcher를 자동 초기화
    search = SearchUseCase(store, llm_client=llm, use_reranker=config.use_reranker)

    try:
        answer = search.ask(
            question=question, top_k=top_k, audience_override=audience_override
        )
    except Exception as e:
        return json.dumps(
            {"success": False, "error": f"답변 생성 실패: {str(e)}"}, ensure_ascii=False
        )

    # Format sources using shared formatters
    sources_list = answer.sources
    norm_scores = normalize_relevance_scores(sources_list) if sources_list else {}
    display_sources = (
        filter_by_relevance(sources_list, norm_scores) if sources_list else []
    )

    sources = []
    for r in display_sources:
        reg_name = r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
        path = (
            " > ".join(clean_path_segments(r.chunk.parent_path))
            if r.chunk.parent_path
            else r.chunk.title
        )
        norm_score = norm_scores.get(r.chunk.id, 0.0)
        rel_pct = int(norm_score * 100)

        sources.append(
            {
                "regulation_name": reg_name,
                "rule_code": r.chunk.rule_code,
                "path": path,
                "text": r.chunk.text,
                "relevance_pct": rel_pct,
            }
        )

    return json.dumps(
        {
            "success": True,
            "question": question,
            "answer": answer.text,
            "confidence": round(answer.confidence, 3),
            "sources": sources,
        },
        ensure_ascii=False,
    )


@mcp.tool()
def get_sync_status() -> str:
    """
    현재 동기화 상태를 확인합니다.

    Returns:
        동기화 상태 정보 (JSON 형식)
    """
    from ..application.sync_usecase import SyncUseCase
    from ..infrastructure.json_loader import JSONDocumentLoader

    store = _get_store()
    loader = JSONDocumentLoader()
    sync = SyncUseCase(loader, store)

    status = sync.get_sync_status()

    return json.dumps(
        {
            "success": True,
            "last_sync": status["last_sync"],
            "json_file": status["json_file"],
            "state_regulations": status["state_regulations"],
            "store_chunks": status["store_chunks"],
            "store_regulations": status["store_regulations"],
        },
        ensure_ascii=False,
    )


@mcp.tool()
def get_regulation_overview(regulation: str) -> str:
    """
    규정의 개요(목차, 조항 수, 구조)를 조회합니다.

    Args:
        regulation: 규정명 또는 규정번호 (예: "교원인사규정", "3-1-5")

    Returns:
        규정 개요 정보 (JSON 형식)
    """
    handler = QueryHandler()
    result = handler.get_regulation_overview(regulation)

    if not result.success:
        return json.dumps(
            {"success": False, "error": result.content},
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "success": True,
            "type": "overview",
            **result.data,
        },
        ensure_ascii=False,
    )


@mcp.tool()
def view_article(regulation: str, article_no: int) -> str:
    """
    특정 규정의 조항 전문을 조회합니다.

    Args:
        regulation: 규정명 또는 규정번호 (예: "교원인사규정", "3-1-5")
        article_no: 조항 번호 (예: 8 → 제8조)

    Returns:
        조항 전문 (JSON 형식)
    """
    handler = QueryHandler()
    result = handler.get_article_view(regulation, article_no)

    if result.type == QueryType.CLARIFICATION:
        return json.dumps(
            {
                "success": True,
                "type": "clarification",
                "reason": result.clarification_type,
                "options": result.clarification_options,
            },
            ensure_ascii=False,
        )

    if not result.success:
        return json.dumps(
            {"success": False, "error": result.content},
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "success": True,
            "type": "article",
            **result.data,
        },
        ensure_ascii=False,
    )


@mcp.tool()
def view_chapter(regulation: str, chapter_no: int) -> str:
    """
    특정 규정의 장 전문을 조회합니다.

    Args:
        regulation: 규정명 또는 규정번호 (예: "학칙", "3-1-1")
        chapter_no: 장 번호 (예: 3 → 제3장)

    Returns:
        장 전문 (JSON 형식)
    """
    handler = QueryHandler()
    result = handler.get_chapter_view(regulation, chapter_no)

    if result.type == QueryType.CLARIFICATION:
        return json.dumps(
            {
                "success": True,
                "type": "clarification",
                "reason": result.clarification_type,
                "options": result.clarification_options,
            },
            ensure_ascii=False,
        )

    if not result.success:
        return json.dumps(
            {"success": False, "error": result.content},
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "success": True,
            "type": "chapter",
            **result.data,
        },
        ensure_ascii=False,
    )


# ============================================================================
# MCP Resources
# ============================================================================


@mcp.resource("regulation://status")
def resource_status() -> str:
    """현재 동기화 상태를 반환합니다."""
    return get_sync_status()


@mcp.resource("regulation://list")
def resource_list() -> str:
    """등록된 규정 목록을 반환합니다."""
    store = _get_store()

    if store.count() == 0:
        return json.dumps(
            {
                "success": True,
                "regulations": [],
                "message": "데이터베이스가 비어 있습니다.",
            },
            ensure_ascii=False,
        )

    rule_codes = store.get_all_rule_codes()

    return json.dumps(
        {
            "success": True,
            "total_count": len(rule_codes),
            "regulations": sorted(list(rule_codes)),
        },
        ensure_ascii=False,
    )


# ============================================================================
# Entry Point
# ============================================================================


def main():
    """Run MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
