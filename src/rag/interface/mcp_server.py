"""
MCP Server for Regulation RAG System.

Provides Model Context Protocol interface for:
- Syncing regulations to vector database
- Searching regulations with hybrid search + reranking
- Asking questions with LLM-powered answers
- Checking sync status

Usage:
    # Run as MCP server (stdio mode)
    uv run regulation-mcp
    
    # Development with MCP Inspector
    uv run mcp dev src/rag/interface/mcp_server.py
"""

import json
import os
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Load .env file for environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Initialize MCP server with metadata
mcp = FastMCP(
    name="regulation-rag",
    instructions="대학 규정집 RAG 검색 및 Q&A 서버. 규정 동기화, 검색, AI 질문-답변 기능을 제공합니다.",
)

# Default paths (can be overridden via environment variables)
DEFAULT_DB_PATH = os.getenv("RAG_DB_PATH", "data/chroma_db")
DEFAULT_JSON_PATH = os.getenv("RAG_JSON_PATH", "data/output/규정집.json")


def _get_store():
    """Get ChromaVectorStore instance."""
    from ..infrastructure.chroma_store import ChromaVectorStore
    return ChromaVectorStore(persist_directory=DEFAULT_DB_PATH)


def _get_hybrid_searcher(store):
    """Get HybridSearcher with BM25 index built."""
    from ..infrastructure.hybrid_search import HybridSearcher
    hybrid = HybridSearcher()
    documents = store.get_all_documents()
    hybrid.add_documents(documents)
    return hybrid


# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool()
def sync_regulations(
    json_path: Optional[str] = None,
    full_sync: bool = False,
) -> str:
    """
    규정 데이터베이스를 동기화합니다.
    
    Args:
        json_path: 규정집 JSON 파일 경로 (기본: data/output/규정집.json)
        full_sync: True면 전체 재동기화, False면 증분 동기화 (기본: False)
    
    Returns:
        동기화 결과 요약 (JSON 형식)
    """
    from ..infrastructure.json_loader import JSONDocumentLoader
    from ..infrastructure.chroma_store import ChromaVectorStore
    from ..application.sync_usecase import SyncUseCase
    
    json_path = json_path or DEFAULT_JSON_PATH
    path = Path(json_path)
    
    if not path.exists():
        return json.dumps({
            "success": False,
            "error": f"파일을 찾을 수 없습니다: {json_path}"
        }, ensure_ascii=False)
    
    # Initialize components
    loader = JSONDocumentLoader()
    store = ChromaVectorStore(persist_directory=DEFAULT_DB_PATH)
    sync = SyncUseCase(loader, store)
    
    # Execute sync
    if full_sync:
        result = sync.full_sync(str(path))
    else:
        result = sync.incremental_sync(str(path))
    
    if result.has_errors:
        return json.dumps({
            "success": False,
            "errors": result.errors
        }, ensure_ascii=False)
    
    return json.dumps({
        "success": True,
        "added": result.added,
        "modified": result.modified,
        "removed": result.removed,
        "unchanged": result.unchanged,
        "total_chunks": store.count(),
    }, ensure_ascii=False)


@mcp.tool()
def search_regulations(
    query: str,
    top_k: int = 5,
    include_abolished: bool = False,
    use_rerank: bool = True,
) -> str:
    """
    규정을 검색합니다. Hybrid Search (BM25 + Dense) 및 BGE Reranking을 사용합니다.
    
    Args:
        query: 검색 쿼리 (예: "교원 연구년 신청 자격", "제15조", "휴학 절차")
        top_k: 반환할 결과 수 (기본: 5)
        include_abolished: 폐지된 규정 포함 여부 (기본: False)
        use_rerank: BGE Reranker 사용 여부 (기본: True)
    
    Returns:
        검색 결과 목록 (JSON 형식)
    """
    from ..application.search_usecase import SearchUseCase
    
    store = _get_store()
    
    if store.count() == 0:
        return json.dumps({
            "success": False,
            "error": "데이터베이스가 비어 있습니다. 먼저 sync_regulations를 실행하세요."
        }, ensure_ascii=False)
    
    hybrid = _get_hybrid_searcher(store)
    search = SearchUseCase(store, use_reranker=use_rerank, hybrid_searcher=hybrid)
    
    results = search.search_unique(
        query,
        top_k=top_k,
        include_abolished=include_abolished,
    )
    
    if not results:
        return json.dumps({
            "success": True,
            "results": [],
            "message": "검색 결과가 없습니다."
        }, ensure_ascii=False)
    
    # Format results
    formatted_results = []
    for i, r in enumerate(results, 1):
        reg_name = r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
        path = " > ".join(r.chunk.parent_path) if r.chunk.parent_path else r.chunk.title
        
        formatted_results.append({
            "rank": i,
            "regulation_name": reg_name,
            "rule_code": r.chunk.rule_code,
            "path": path,
            "text": r.chunk.text,
            "score": round(r.score, 4),
        })
    
    return json.dumps({
        "success": True,
        "query": query,
        "results": formatted_results,
    }, ensure_ascii=False)


@mcp.tool()
def ask_regulations(
    question: str,
    top_k: int = 5,
    provider: str = "lmstudio",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """
    규정에 대해 질문하고 AI 답변을 받습니다.
    
    Args:
        question: 질문 (예: "교원 연구년 신청 자격은?", "휴학하려면 어떻게 해야 하나요?")
        top_k: 참고할 규정 수 (기본: 5)
        provider: LLM 프로바이더 (lmstudio, ollama, openai, gemini, openrouter)
        model: 모델 이름 (기본: 프로바이더별 기본값)
        base_url: 로컬 서버 URL (lmstudio, ollama용)
    
    Returns:
        AI 답변 및 참고 규정 (JSON 형식)
    """
    from ..infrastructure.llm_adapter import LLMClientAdapter
    from ..application.search_usecase import SearchUseCase
    
    store = _get_store()
    
    if store.count() == 0:
        return json.dumps({
            "success": False,
            "error": "데이터베이스가 비어 있습니다. 먼저 sync_regulations를 실행하세요."
        }, ensure_ascii=False)
    
    # Use environment variables as fallback
    provider = provider or os.getenv("LLM_PROVIDER", "lmstudio")
    model = model or os.getenv("LLM_MODEL")
    base_url = base_url or os.getenv("LLM_BASE_URL")
    
    try:
        llm = LLMClientAdapter(
            provider=provider,
            model=model,
            base_url=base_url,
        )
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"LLM 초기화 실패: {str(e)}",
            "hint": "로컬 LLM 서버가 실행 중인지 확인하세요." if provider in ("lmstudio", "ollama") else "API 키 설정을 확인하세요."
        }, ensure_ascii=False)
    
    hybrid = _get_hybrid_searcher(store)
    search = SearchUseCase(store, llm_client=llm, use_reranker=True, hybrid_searcher=hybrid)
    
    try:
        answer = search.ask(question=question, top_k=top_k)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"답변 생성 실패: {str(e)}"
        }, ensure_ascii=False)
    
    # Format sources
    sources = []
    for r in answer.sources:
        reg_name = r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
        path = " > ".join(r.chunk.parent_path) if r.chunk.parent_path else r.chunk.title
        
        sources.append({
            "regulation_name": reg_name,
            "rule_code": r.chunk.rule_code,
            "path": path,
            "text": r.chunk.text,
            "score": round(r.score, 4),
        })
    
    return json.dumps({
        "success": True,
        "question": question,
        "answer": answer.text,
        "confidence": round(answer.confidence, 3),
        "sources": sources,
    }, ensure_ascii=False)


@mcp.tool()
def get_sync_status() -> str:
    """
    현재 동기화 상태를 확인합니다.
    
    Returns:
        동기화 상태 정보 (JSON 형식)
    """
    from ..infrastructure.json_loader import JSONDocumentLoader
    from ..application.sync_usecase import SyncUseCase
    
    store = _get_store()
    loader = JSONDocumentLoader()
    sync = SyncUseCase(loader, store)
    
    status = sync.get_sync_status()
    
    return json.dumps({
        "success": True,
        "last_sync": status["last_sync"],
        "json_file": status["json_file"],
        "state_regulations": status["state_regulations"],
        "store_chunks": status["store_chunks"],
        "store_regulations": status["store_regulations"],
    }, ensure_ascii=False)


@mcp.tool()
def reset_database(confirm: bool = False) -> str:
    """
    데이터베이스를 초기화합니다 (모든 데이터 삭제).
    
    Args:
        confirm: 초기화 확인 (True로 설정해야 실행됨)
    
    Returns:
        초기화 결과 (JSON 형식)
    """
    if not confirm:
        return json.dumps({
            "success": False,
            "error": "초기화를 수행하려면 confirm=True로 설정하세요."
        }, ensure_ascii=False)
    
    from ..infrastructure.json_loader import JSONDocumentLoader
    from ..application.sync_usecase import SyncUseCase
    
    store = _get_store()
    loader = JSONDocumentLoader()
    sync = SyncUseCase(loader, store)
    
    chunk_count = store.count()
    
    if chunk_count == 0:
        return json.dumps({
            "success": True,
            "message": "데이터베이스가 이미 비어 있습니다."
        }, ensure_ascii=False)
    
    # Clear vector store
    deleted = store.clear_all()
    
    # Clear sync state
    sync.reset_state()
    
    return json.dumps({
        "success": True,
        "message": f"데이터베이스 초기화 완료! {deleted}개 청크 삭제됨"
    }, ensure_ascii=False)


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
        return json.dumps({
            "success": True,
            "regulations": [],
            "message": "데이터베이스가 비어 있습니다."
        }, ensure_ascii=False)
    
    rule_codes = store.get_all_rule_codes()
    
    return json.dumps({
        "success": True,
        "total_count": len(rule_codes),
        "regulations": sorted(list(rule_codes)),
    }, ensure_ascii=False)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Run MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
