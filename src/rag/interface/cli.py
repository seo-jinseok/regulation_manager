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

# Formatters for output formatting
from .formatters import (
    normalize_relevance_scores,
    filter_by_relevance,
    get_relevance_label_combined,
    clean_path_segments,
    extract_display_text,
    build_display_path,
    get_confidence_info,
)

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


def print_query_rewrite(search, original_query: str) -> None:
    """Print query rewrite info when available."""
    info = search.get_last_query_rewrite()
    if not info:
        return

    if RICH_AVAILABLE:
        console.print()
        console.print("[bold cyan]🔄 쿼리 분석 결과[/bold cyan]")
    else:
        print("\n=== 쿼리 분석 결과 ===")

    if not info.used:
        print_info(f"쿼리 리라이팅: (적용 안됨) '{original_query}'")
        return

    # 방법 표시
    if info.method == "llm":
        method_label = "LLM 기반 리라이팅"
        method_icon = "🤖"
    elif info.method == "rules":
        method_label = "규칙 기반 확장"
        method_icon = "📋"
    else:
        method_label = "알수없음"
        method_icon = "❓"

    # 추가 상태
    extras = []
    if info.from_cache:
        extras.append("캐시 히트")
    if info.fallback:
        extras.append("LLM 실패→폴백")
    extra_text = f" ({', '.join(extras)})" if extras else ""

    # 원본 → 변환 쿼리
    if info.original == info.rewritten:
        print_info(f"{method_icon} {method_label}{extra_text}: 변경 없음")
        print_info(f"   원본: '{info.original}'")
    else:
        print_info(f"{method_icon} {method_label}{extra_text}")
        print_info(f"   원본: '{info.original}'")
        print_info(f"   변환: '{info.rewritten}'")

    # 동의어 사용 여부
    if info.used_synonyms is not None:
        if info.used_synonyms:
            print_info("📚 동의어 사전: ✅ 적용됨 (유사어로 확장)")
        else:
            print_info("📚 동의어 사전: ➖ 미적용")

    # 인텐트 사용 여부
    if info.used_intent is not None:
        if info.used_intent:
            print_info("🎯 의도 인식: ✅ 매칭됨")
            if info.matched_intents:
                intents_str = ", ".join(info.matched_intents)
                print_info(f"   매칭된 의도: [{intents_str}]")
        else:
           print_info("🎯 의도 인식: ➖ 미매칭")

    if RICH_AVAILABLE:
        console.print()




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
        help="규정 검색 (자동으로 답변 생성 또는 문서 검색)",
        description="질문이면 AI 답변을, 키워드면 문서 검색 결과를 자동으로 보여줍니다. (혹은 -a/-q 옵션으로 강제)"
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="검색 쿼리 또는 질문",
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
        help="폐지 규정 포함 (검색 모드일 때만 유효)",
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
        help="BGE Reranker 비활성화",
    )
    search_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="상세 정보 출력",
    )
    search_parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 정보 출력",
    )
    search_parser.add_argument(
        "--feedback",
        action="store_true",
        help="결과에 대한 피드백 남기기",
    )
    # Unified specific arguments
    mode_group = search_parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-a", "--answer",
        action="store_true",
        help="AI 답변 생성 강제 (Ask 모드)",
    )
    mode_group.add_argument(
        "-q", "--quick",
        action="store_true",
        help="문서 검색만 수행 (Search 모드)",
    )
    # LLM options for when answer is triggered
    search_parser.add_argument(
        "--provider",
        type=str,
        default=default_provider,
        choices=providers,
        help="LLM 프로바이더 (답변 생성 시)",
    )
    search_parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="모델 이름",
    )
    search_parser.add_argument(
        "--base-url",
        type=str,
        default=default_base_url,
        help="로컬 서버 URL",
    )
    search_parser.add_argument(
        "--show-sources",
        action="store_true",
        help="관련 규정 전문 출력 (답변 생성 시)",
    )

    # ask command (Legacy Wrapper)
    ask_parser = subparsers.add_parser(
        "ask",
        help="규정 질문 (search -a와 동일)",
    )
    ask_parser.add_argument(
        "query",
        type=str,
        help="질문",
    )
    ask_parser.add_argument(
        "-n", "--top-k",
        type=int,
        default=5,
        help="참고 규정 수",
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
    ask_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="상세 정보 출력 (LLM 설정, 인덱스 구축 현황 등)",
    )
    ask_parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 정보 출력 (쿼리 리라이팅 등)",
    )
    ask_parser.add_argument(
        "--feedback",
        action="store_true",
        help="결과에 대한 피드백 남기기 (인터랙티브)",
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



def _decide_search_mode(args) -> str:
    """Wrapper for shared decide_search_mode."""
    from .common import decide_search_mode
    
    # Check flags first
    force_mode = None
    if hasattr(args, 'answer') and args.answer:
        force_mode = "ask"
    elif hasattr(args, 'quick') and args.quick:
        force_mode = "search"
        
    return decide_search_mode(args.query, force_mode)


def _perform_unified_search(args, force_mode: Optional[str] = None) -> int:
    """Core logic for unified search/ask."""
    from ..infrastructure.chroma_store import ChromaVectorStore
    from ..infrastructure.llm_adapter import LLMClientAdapter
    from ..application.search_usecase import SearchUseCase
    from rich.panel import Panel

    mode = force_mode or _decide_search_mode(args)
    if args.verbose:
        print_info(f"실행 모드: {mode.upper()} (쿼리: '{args.query}')")

    # Step 1: Check database
    store = ChromaVectorStore(persist_directory=args.db_path)
    if store.count() == 0:
        print_error("데이터베이스가 비어 있습니다. 먼저 sync를 실행하세요.")
        return 1

    use_reranker = not args.no_rerank

    # Initialize LLM only if needed
    llm = None
    if mode == "ask":
        if RICH_AVAILABLE:
            with console.status("[bold blue]⏳ LLM 클라이언트 초기화 중...[/bold blue]"):
                try:
                    llm = LLMClientAdapter(
                        provider=args.provider,
                        model=args.model,
                        base_url=args.base_url,
                    )
                except Exception as e:
                    print_error(f"LLM 초기화 실패: {e}")
                    return 1
        else:
             try:
                llm = LLMClientAdapter(
                    provider=args.provider,
                    model=args.model,
                    base_url=args.base_url,
                )
             except Exception as e:
                print_error(f"LLM 초기화 실패: {e}")
                return 1

    # Step 2: Build Search Interface
    # SearchUseCase initializes HybridSearcher automatically
    if RICH_AVAILABLE:
        status_msg = "[bold blue]🔍 검색 엔진 준비 중...[/bold blue]"
        with console.status(status_msg):
            search = SearchUseCase(store, llm_client=llm, use_reranker=use_reranker)
    else:
        search = SearchUseCase(store, llm_client=llm, use_reranker=use_reranker)

    # Step 3: Execute Logic based on Mode
    if mode == "search":
        # Retrieval Only
        results = search.search_unique(
            args.query,
            top_k=args.top_k,
            include_abolished=args.include_abolished if hasattr(args, 'include_abolished') else False,
        )
        
        if args.verbose or args.debug:
            print_query_rewrite(search, args.query)

        if not results:
            print_info("검색 결과가 없습니다.")
            return 0
            
        # Display Results (Search Style)
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
                    str(reg_title or r.chunk.rule_code),
                    str(r.chunk.rule_code),
                    str(path or r.chunk.title),
                    f"{r.score:.2f}",
                )
            console.print(table)
            
            # Print first result detail
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
                
        if args.feedback and results:
             _collect_cli_feedback(args.query, results[0].chunk.rule_code)

    else:
        # Ask (LLM Answer)
        if RICH_AVAILABLE:
            with console.status("[bold green]🤖 AI 답변 생성 중... (10-30초 소요)[/bold green]"):
                try:
                    answer = search.ask(
                        question=args.query,
                        top_k=args.top_k,
                    )
                except Exception as e:
                    print_error(f"답변 생성 실패: {e}")
                    return 1
        else:
            print("AI 답변 생성 중...")
            try:
                answer = search.ask(
                    question=args.query,
                    top_k=args.top_k,
                )
            except Exception as e:
                print_error(f"답변 생성 실패: {e}")
                return 1

        if args.verbose or args.debug:
            print_query_rewrite(search, args.query)

        # Display Answer (Ask Style)
        if RICH_AVAILABLE:
            console.print()
            console.print(Panel(
                Markdown(answer.text),
                title="🤖 AI 답변",
                border_style="green",
            ))
            
            if answer.sources:
                console.print()
                console.print("[bold cyan]📚 참고 규정:[/bold cyan]")
                
                # Shared formatting logic
                norm_scores = normalize_relevance_scores(answer.sources)
                display_sources = filter_by_relevance(answer.sources, norm_scores)
                
                for i, result in enumerate(display_sources, 1):
                    chunk = result.chunk
                    reg_name = chunk.parent_path[0] if chunk.parent_path else chunk.title
                    path = build_display_path(chunk.parent_path, chunk.text, chunk.title)
                    norm_score = norm_scores.get(chunk.id, 0.0)
                    rel_score = int(norm_score * 100)
                    rel_label = get_relevance_label_combined(rel_score)
                    display_text = extract_display_text(chunk.text)
                    
                    content_parts = [
                        f"[bold blue]📖 {reg_name}[/bold blue]",
                        f"[dim]📍 {path}[/dim]",
                        "",
                        display_text,
                        "",
                        f"[dim]📋 규정번호: {chunk.rule_code} | 관련도: {rel_score}% {rel_label}[/dim]" + (f" [dim]| AI 신뢰도: {result.score:.3f}[/dim]" if args.verbose else ""),
                    ]
                    
                    console.print(Panel(
                        "\n".join(content_parts),
                        title=f"[{i}]",
                        border_style="blue",
                    ))
            
            # Confidence Info
            console.print()
            conf_icon, conf_label, conf_detail = get_confidence_info(answer.confidence)
            console.print(Panel(
                f"[bold]{conf_icon} {conf_label}[/bold] (신뢰도 {answer.confidence:.0%})\n\n{conf_detail}",
                title="📊 답변 신뢰도",
                border_style="dim",
            ))

        else:
            print(f"\n=== AI 답변 ===")
            print(answer.text)
            print(f"\n=== 참고 규정 ===")
            for i, result in enumerate(answer.sources, 1):
                print(f"[{i}] {result.chunk.rule_code}: {result.chunk.text[:100]}...")
            
            if getattr(args, 'show_sources', False):
                print(f"\n=== 규정 전문 ===")
                for result in answer.sources:
                    print(f"\n--- {result.chunk.rule_code} ---")
                    print(result.chunk.text)

        if args.feedback and answer.sources:
            _collect_cli_feedback(args.query, answer.sources[0].chunk.rule_code)

    return 0


def cmd_search(args) -> int:
    """Execute search command (Unified)."""
    return _perform_unified_search(args)


def cmd_ask(args) -> int:
    """Execute ask command (Legacy Wrapper)."""
    # Map 'question' arg to 'query' expected by unified logic
    if hasattr(args, 'question'):
        args.query = args.question
    return _perform_unified_search(args, force_mode="ask")


def _collect_cli_feedback(query: str, rule_code: str):
    """Interactively collect feedback from CLI."""
    from ..infrastructure.feedback import FeedbackCollector
    
    print("\n" + "="*30)
    print("📢 이 답변이 도움이 되었나요?")
    print("1: 👍 도움이 됨 (Positive)")
    print("2: 😐 보통 (Neutral)")
    print("3: 👎 도움이 안 됨 (Negative)")
    print("0: 건너뛰기")
    
    try:
        choice = input("\n선택 (0-3): ").strip()
        if choice == "0" or not choice:
            return
            
        rating_map = {"1": 1, "2": 0, "3": -1}
        if choice not in rating_map:
            print("올바른 선택이 아닙니다.")
            return
            
        rating = rating_map[choice]
        comment = input("의견이 있다면 남겨주세요 (선택사항, Enter로 스킵): ").strip()
        
        collector = FeedbackCollector()
        collector.record_feedback(
            query=query,
            rule_code=rule_code,
            rating=rating,
            comment=comment or None,
            source="cli"
        )
        print("✅ 소중한 피드백 감사합니다!")
    except (KeyboardInterrupt, EOFError):
        print("\n건너뜁니다.")


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
        table.add_row("📚 규정집", status["json_file"] or "없음")
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
