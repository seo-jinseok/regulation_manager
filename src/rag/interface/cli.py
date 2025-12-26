"""
CLI Interface for Regulation RAG System.

Provides command-line tools for:
- Syncing regulations
- Searching regulations
- Asking questions

Usage:
    uv run python -m src.rag.interface.cli sync data/output/규정집.json
    uv run python -m src.rag.interface.cli search "교원 연구년"
    uv run python -m src.rag.interface.cli ask "교원 연구년 신청 자격은?"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Rich for pretty output (optional)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_info(msg: str) -> None:
    """Print info message."""
    if RICH_AVAILABLE:
        console.print(f"[blue]ℹ[/blue] {msg}")
    else:
        print(f"[INFO] {msg}")


def print_success(msg: str) -> None:
    """Print success message."""
    if RICH_AVAILABLE:
        console.print(f"[green]✓[/green] {msg}")
    else:
        print(f"[OK] {msg}")


def print_error(msg: str) -> None:
    """Print error message."""
    if RICH_AVAILABLE:
        console.print(f"[red]✗[/red] {msg}")
    else:
        print(f"[ERROR] {msg}")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    providers = ["ollama", "lmstudio", "mlx", "local", "openai", "gemini", "openrouter"]
    default_provider = os.getenv("LLM_PROVIDER") or "ollama"
    if default_provider not in providers:
        default_provider = "ollama"
    default_model = os.getenv("LLM_MODEL") or None
    default_base_url = os.getenv("LLM_BASE_URL") or None
    parser = argparse.ArgumentParser(
        prog="rag",
        description="규정집 RAG 시스템 CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # sync command
    sync_parser = subparsers.add_parser(
        "sync",
        help="규정 데이터베이스 동기화",
    )
    sync_parser.add_argument(
        "json_path",
        type=str,
        help="규정집 JSON 파일 경로",
    )
    sync_parser.add_argument(
        "--full",
        action="store_true",
        help="전체 재동기화 (기본: 증분 동기화)",
    )
    sync_parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )

    # search command
    search_parser = subparsers.add_parser(
        "search",
        help="규정 검색",
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="검색 쿼리",
    )
    search_parser.add_argument(
        "-n", "--top-k",
        type=int,
        default=5,
        help="결과 개수 (기본: 5)",
    )
    search_parser.add_argument(
        "--include-abolished",
        action="store_true",
        help="폐지 규정 포함",
    )
    search_parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )
    search_parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="BGE Reranker 비활성화 (기본: reranking 사용)",
    )

    # ask command
    ask_parser = subparsers.add_parser(
        "ask",
        help="규정 질문 (LLM 답변)",
    )
    ask_parser.add_argument(
        "question",
        type=str,
        help="질문",
    )
    ask_parser.add_argument(
        "-n", "--top-k",
        type=int,
        default=5,
        help="참고 규정 수 (기본: 5)",
    )
    ask_parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )
    ask_parser.add_argument(
        "--provider",
        type=str,
        default=default_provider,
        choices=providers,
        help="LLM 프로바이더 (ollama, lmstudio, mlx, local, openai, gemini, openrouter)",
    )
    ask_parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="모델 이름 (기본: 프로바이더별 기본값)",
    )
    ask_parser.add_argument(
        "--base-url",
        type=str,
        default=default_base_url,
        help="로컬 서버 URL (ollama, lmstudio, mlx, local용)",
    )
    ask_parser.add_argument(
        "--show-sources",
        action="store_true",
        help="관련 규정 전문 출력",
    )
    ask_parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="BGE Reranker 비활성화 (기본: reranking 사용)",
    )

    # status command
    status_parser = subparsers.add_parser(
        "status",
        help="동기화 상태 확인",
    )
    status_parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )

    # reset command
    reset_parser = subparsers.add_parser(
        "reset",
        help="데이터베이스 초기화 (모든 데이터 삭제)",
    )
    reset_parser.add_argument(
        "--confirm",
        action="store_true",
        required=True,
        help="초기화 확인 (필수)",
    )
    reset_parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )

    return parser


def cmd_sync(args) -> int:
    """Execute sync command."""
    from ..infrastructure.json_loader import JSONDocumentLoader
    from ..infrastructure.chroma_store import ChromaVectorStore
    from ..application.sync_usecase import SyncUseCase

    json_path = Path(args.json_path)
    if not json_path.exists():
        print_error(f"파일을 찾을 수 없습니다: {json_path}")
        return 1

    print_info(f"데이터베이스: {args.db_path}")
    print_info(f"JSON 파일: {json_path.name}")

    # Initialize components
    loader = JSONDocumentLoader()
    store = ChromaVectorStore(persist_directory=args.db_path)
    sync = SyncUseCase(loader, store)

    # Execute sync
    if args.full:
        print_info("전체 동기화 실행 중...")
        result = sync.full_sync(str(json_path))
    else:
        print_info("증분 동기화 실행 중...")
        result = sync.incremental_sync(str(json_path))

    # Print results
    if result.has_errors:
        for error in result.errors:
            print_error(error)
        return 1

    print_success(str(result))
    print_info(f"총 청크 수: {store.count()}")
    return 0


def cmd_search(args) -> int:
    """Execute search command."""
    from ..infrastructure.chroma_store import ChromaVectorStore
    from ..infrastructure.hybrid_search import HybridSearcher
    from ..application.search_usecase import SearchUseCase

    store = ChromaVectorStore(persist_directory=args.db_path)

    if store.count() == 0:
        print_error("데이터베이스가 비어 있습니다. 먼저 sync를 실행하세요.")
        return 1

    use_reranker = not args.no_rerank
    if use_reranker:
        print_info("🎯 BGE Reranker 활성화 (비활성화: --no-rerank)")

    # Initialize HybridSearcher with BM25 index
    print_info("🔄 Hybrid Search 인덱스 구축 중...")
    hybrid = HybridSearcher()
    documents = store.get_all_documents()
    hybrid.add_documents(documents)
    print_info(f"✓ BM25 인덱스 구축 완료 ({len(documents):,}개 문서)")

    search = SearchUseCase(store, use_reranker=use_reranker, hybrid_searcher=hybrid)
    results = search.search_unique(
        args.query,
        top_k=args.top_k,
        include_abolished=args.include_abolished,
    )

    if not results:
        print_info("검색 결과가 없습니다.")
        return 0

    # Print results
    if RICH_AVAILABLE:
        table = Table(title=f"검색 결과: '{args.query}'")
        table.add_column("#", style="dim", width=3)
        table.add_column("규정명", style="cyan")
        table.add_column("코드", style="magenta")
        table.add_column("조항", style="green")
        table.add_column("점수", justify="right", style="magenta")

        for i, r in enumerate(results, 1):
            path = " > ".join(r.chunk.parent_path[-2:]) if r.chunk.parent_path else ""
            reg_title = r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
            table.add_row(
                str(i),
                reg_title or r.chunk.rule_code,
                r.chunk.rule_code,
                path or r.chunk.title,
                f"{r.score:.2f}",
            )
        console.print(table)

        # Print first result details
        if results:
            top = results[0]
            console.print(Panel(
                top.chunk.text[:500] + "..." if len(top.chunk.text) > 500 else top.chunk.text,
                title=f"[1위] {top.chunk.rule_code}",
                border_style="green",
            ))
    else:
        print(f"\n검색 결과: '{args.query}'")
        print("-" * 60)
        for i, r in enumerate(results, 1):
            reg_title = r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
            print(f"{i}. {reg_title} [{r.chunk.rule_code}] (점수: {r.score:.2f})")
            print(f"   {r.chunk.text[:100]}...")

    return 0


def cmd_ask(args) -> int:
    """Execute ask command with LLM."""
    from ..infrastructure.chroma_store import ChromaVectorStore
    from ..infrastructure.hybrid_search import HybridSearcher
    from ..infrastructure.llm_adapter import LLMClientAdapter
    from ..application.search_usecase import SearchUseCase

    # Check if DB has data
    store = ChromaVectorStore(persist_directory=args.db_path)
    if store.count() == 0:
        print_error("데이터베이스가 비어 있습니다. 먼저 sync를 실행하세요.")
        return 1

    # Initialize LLM client
    print_info(f"LLM 프로바이더: {args.provider}")
    if args.model:
        print_info(f"모델: {args.model}")
    
    try:
        llm = LLMClientAdapter(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
        )
    except Exception as e:
        print_error(f"LLM 초기화 실패: {e}")
        if args.provider in ("ollama", "lmstudio", "local", "mlx"):
            print_info("로컬 LLM 서버가 실행 중인지 확인하세요.")
        else:
            print_info("API 키 설정을 확인하세요.")
        return 1

    use_reranker = not args.no_rerank
    if use_reranker:
        print_info("🎯 BGE Reranker 활성화 (비활성화: --no-rerank)")

    # Initialize HybridSearcher with BM25 index
    print_info("🔄 Hybrid Search 인덱스 구축 중...")
    hybrid = HybridSearcher()
    documents = store.get_all_documents()
    hybrid.add_documents(documents)
    print_info(f"✓ BM25 인덱스 구축 완료 ({len(documents):,}개 문서)")

    # Create search use case with LLM and hybrid searcher
    search = SearchUseCase(store, llm_client=llm, use_reranker=use_reranker, hybrid_searcher=hybrid)

    print_info(f"질문: {args.question}")
    print_info("답변 생성 중...")

    try:
        answer = search.ask(
            question=args.question,
            top_k=args.top_k,
        )
    except Exception as e:
        print_error(f"답변 생성 실패: {e}")
        return 1

    # Print answer
    if RICH_AVAILABLE:
        console.print()
        console.print(Panel(
            Markdown(answer.text),
            title="🤖 LLM 답변",
            border_style="green",
        ))

        # Print sources
        if answer.sources:
            console.print()
            console.print("[bold cyan]📚 참고 규정:[/bold cyan]")
            for i, result in enumerate(answer.sources, 1):
                chunk = result.chunk
                path = " > ".join(chunk.parent_path[-3:]) if chunk.parent_path else chunk.title
                # Show more text (400 chars instead of 150)
                text_preview = chunk.text[:400] + "..." if len(chunk.text) > 400 else chunk.text
                
                console.print(Panel(
                    f"{text_preview}\n\n[dim](출처: {chunk.rule_code}, 점수: {result.score:.2f})[/dim]",
                    title=f"[{i}] {path}",
                    border_style="blue",
                ))

        # Show full sources if requested
        if args.show_sources and answer.sources:
            console.print()
            console.print("[bold yellow]━━ 관련 규정 전문 ━━[/bold yellow]")
            for result in answer.sources:
                chunk = result.chunk
                console.print(Panel(
                    chunk.text,
                    title=f"{chunk.rule_code} - {' > '.join(chunk.parent_path[-2:]) if chunk.parent_path else ''}",
                    border_style="yellow",
                ))

        # Print confidence
        console.print()
        confidence_color = "green" if answer.confidence > 0.7 else "yellow" if answer.confidence > 0.4 else "red"
        console.print(f"[dim]신뢰도: [{confidence_color}]{answer.confidence:.0%}[/{confidence_color}][/dim]")
    else:
        print(f"\n=== LLM 답변 ===")
        print(answer.text)
        print(f"\n=== 참고 규정 ===")
        for i, result in enumerate(answer.sources, 1):
            print(f"[{i}] {result.chunk.rule_code}: {result.chunk.text[:100]}...")
        if args.show_sources:
            print(f"\n=== 규정 전문 ===")
            for result in answer.sources:
                print(f"\n--- {result.chunk.rule_code} ---")
                print(result.chunk.text)

    return 0


def cmd_status(args) -> int:
    """Execute status command."""
    from ..infrastructure.chroma_store import ChromaVectorStore
    from ..infrastructure.json_loader import JSONDocumentLoader
    from ..application.sync_usecase import SyncUseCase

    store = ChromaVectorStore(persist_directory=args.db_path)
    loader = JSONDocumentLoader()
    sync = SyncUseCase(loader, store)

    status = sync.get_sync_status()

    if RICH_AVAILABLE:
        table = Table(title="동기화 상태")
        table.add_column("항목", style="cyan")
        table.add_column("값", style="green")

        table.add_row("마지막 동기화", status["last_sync"] or "없음")
        table.add_row("JSON 파일", status["json_file"] or "없음")
        table.add_row("상태 파일 규정 수", str(status["state_regulations"]))
        table.add_row("DB 청크 수", str(status["store_chunks"]))
        table.add_row("DB 규정 수", str(status["store_regulations"]))

        console.print(table)
    else:
        print("동기화 상태")
        print("-" * 40)
        for k, v in status.items():
            print(f"  {k}: {v}")

    return 0


def cmd_reset(args) -> int:
    """Execute reset command - delete all data."""
    from ..infrastructure.chroma_store import ChromaVectorStore
    from ..application.sync_usecase import SyncUseCase
    from ..infrastructure.json_loader import JSONDocumentLoader

    if not args.confirm:
        print_error("초기화를 수행하려면 --confirm 플래그를 사용하세요.")
        return 1

    store = ChromaVectorStore(persist_directory=args.db_path)
    loader = JSONDocumentLoader()
    sync = SyncUseCase(loader, store)

    # Get current count
    chunk_count = store.count()
    
    if chunk_count == 0:
        print_info("데이터베이스가 이미 비어 있습니다.")
        return 0

    print_info(f"데이터베이스: {args.db_path}")
    print_info(f"삭제 예정 청크 수: {chunk_count}")

    # Clear vector store
    deleted = store.clear_all()
    
    # Clear sync state
    sync.reset_state()

    print_success(f"데이터베이스 초기화 완료! {deleted}개 청크 삭제됨")
    return 0


def main(argv: Optional[list] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    commands = {
        "sync": cmd_sync,
        "search": cmd_search,
        "ask": cmd_ask,
        "status": cmd_status,
        "reset": cmd_reset,
    }

    if args.command in commands:
        return commands[args.command](args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
