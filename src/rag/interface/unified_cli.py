"""
Unified CLI Interface for Regulation Manager.

Provides a single entry point for all regulation management tasks:
- convert: HWP → JSON conversion
- sync: Database synchronization
- search: Regulation search
- ask: LLM-powered Q&A
- status: Sync status check
- reset: Database reset
- serve: Start Web UI or MCP Server

Usage:
    uv run regulation convert "규정집.hwp"
    uv run regulation search "교원 연구년"
    uv run regulation ask "휴학 절차"
    uv run regulation serve --web
"""

import argparse
import os
import sys
from typing import Optional

# Load .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _get_default_llm_settings():
    """Get default LLM settings from centralized config."""
    from ..config import get_config

    config = get_config()
    return (
        config.llm_providers,
        config.llm_provider,
        config.llm_model,
        config.llm_base_url,
    )


def _add_convert_parser(subparsers):
    """Add convert subcommand parser."""
    convert_providers = [
        "openai",
        "gemini",
        "openrouter",
        "ollama",
        "lmstudio",
        "local",
        "mlx",
    ]
    default_provider = os.getenv("LLM_PROVIDER") or "openai"
    if default_provider not in convert_providers:
        default_provider = "openai"
    default_model = os.getenv("LLM_MODEL") or None
    default_base_url = os.getenv("LLM_BASE_URL") or None

    parser = subparsers.add_parser(
        "convert",
        help="HWP 파일을 JSON으로 변환",
        description="HWP 규정집을 구조화된 JSON으로 변환합니다.",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="HWP 파일 또는 디렉토리 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output",
        help="출력 디렉토리 (기본: data/output)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="LLM 전처리 활성화",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=default_provider,
        choices=convert_providers,
        help="LLM 프로바이더",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="LLM 모델",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=default_base_url,
        help="LLM API URL (로컬 서버용)",
    )
    parser.add_argument(
        "--allow-llm-fallback",
        action="store_true",
        help="LLM 실패 시 정규식 폴백 허용",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="캐시 무시하고 강제 재변환",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache",
        help="캐시 디렉토리",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="상세 로그 출력",
    )
    parser.add_argument(
        "--no-enhance-rag",
        action="store_false",
        dest="enhance_rag",
        help="RAG 최적화 비활성화",
    )
    parser.set_defaults(enhance_rag=True)


def _add_sync_parser(subparsers):
    """Add sync subcommand parser."""
    parser = subparsers.add_parser(
        "sync",
        help="규정 데이터베이스 동기화",
        description="JSON 파일을 ChromaDB에 동기화합니다.",
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="규정집 JSON 파일 경로",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="전체 재동기화 (기본: 증분 동기화)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )


def _add_search_parser(subparsers):
    """Add search subcommand parser."""

    # Get default settings
    providers, default_provider, default_model, default_base_url = (
        _get_default_llm_settings()
    )

    parser = subparsers.add_parser(
        "search",
        help="규정 검색 (자동으로 답변 생성 또는 문서 검색)",
        description="질문이면 AI 답변을, 키워드면 문서 검색 결과를 자동으로 보여줍니다. (혹은 -a/-q 옵션으로 강제)",
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="검색 쿼리 또는 질문",
    )
    parser.add_argument(
        "-n",
        "--top-k",
        type=int,
        default=5,
        help="결과 개수 (기본: 5)",
    )
    parser.add_argument(
        "--include-abolished",
        action="store_true",
        help="폐지 규정 포함 (검색 모드일 때만 유효)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="BGE Reranker 비활성화",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="상세 정보 출력",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 정보 출력",
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        help="결과에 대한 피드백 남기기 (인터랙티브)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="대화형 모드로 연속 질의",
    )

    # Unified specific arguments
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-a",
        "--answer",
        action="store_true",
        help="AI 답변 생성 강제 (Ask 모드)",
    )
    mode_group.add_argument(
        "-q",
        "--quick",
        action="store_true",
        help="문서 검색만 수행 (Search 모드)",
    )

    # LLM options (for answer mode)
    parser.add_argument(
        "--provider",
        type=str,
        default=default_provider,
        choices=providers,
        help="LLM 프로바이더 (답변 생성 시)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="모델 이름",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=default_base_url,
        help="로컬 서버 URL",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="관련 규정 전문 출력 (답변 생성 시)",
    )

    # Tool calling is now DEFAULT (FunctionGemma for routing, base LLM for answers)
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Tool Calling 비활성화 (기존 방식 사용)",
    )
    parser.add_argument(
        "--tool-mode",
        type=str,
        choices=["auto", "mlx", "openai", "ollama"],
        default="auto",
        help="Tool Calling 백엔드 (auto: OpenAI API 우선, mlx: Apple Silicon Experimental)",
    )


def _add_ask_parser(subparsers):
    """Add ask subcommand parser (Legacy Wrapper)."""
    providers, default_provider, default_model, default_base_url = (
        _get_default_llm_settings()
    )

    parser = subparsers.add_parser(
        "ask",
        help="규정 질문 (search -a와 동일)",
        description="LLM을 사용하여 규정에 대한 질문에 답변합니다.",
    )
    parser.add_argument(
        "question",
        type=str,
        help="질문",
    )
    parser.add_argument(
        "-n",
        "--top-k",
        type=int,
        default=5,
        help="참고 규정 수",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=default_provider,
        choices=providers,
        help="LLM 프로바이더",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="모델 이름",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=default_base_url,
        help="로컬 서버 URL",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="관련 규정 전문 출력",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="BGE Reranker 비활성화",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="상세 정보 출력",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 정보 출력",
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        help="결과에 대한 피드백 남기기 (인터랙티브)",
    )


def _add_status_parser(subparsers):
    """Add status subcommand parser."""
    parser = subparsers.add_parser(
        "status",
        help="동기화 상태 확인",
        description="현재 데이터베이스 동기화 상태를 표시합니다.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )


def _add_reset_parser(subparsers):
    """Add reset subcommand parser."""
    parser = subparsers.add_parser(
        "reset",
        help="데이터베이스 초기화",
        description="모든 데이터를 삭제하고 데이터베이스를 초기화합니다.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        required=True,
        help="초기화 확인 (필수)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )


def _add_serve_parser(subparsers):
    """Add serve subcommand parser."""
    parser = subparsers.add_parser(
        "serve",
        help="서버 시작 (Web UI 또는 MCP)",
        description="Gradio Web UI 또는 MCP Server를 시작합니다.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--web",
        action="store_true",
        help="Gradio Web UI 시작",
    )
    group.add_argument(
        "--mcp",
        action="store_true",
        help="MCP Server 시작",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Web UI 포트 (기본: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Web UI 공개 링크 생성 (Gradio share)",
    )


def _add_evaluate_parser(subparsers):
    """Add evaluate subcommand parser."""
    parser = subparsers.add_parser(
        "evaluate",
        help="RAG 시스템 품질 평가",
        description="테스트 데이터셋으로 검색 품질을 평가합니다.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/config/evaluation_dataset.json",
        help="평가 데이터셋 경로",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="특정 카테고리만 평가",
    )
    parser.add_argument(
        "-n",
        "--top-k",
        type=int,
        default=5,
        help="검색 결과 수 (기본: 5)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/chroma_db",
        help="ChromaDB 저장 경로",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="상세 결과 출력",
    )


def _add_extract_keywords_parser(subparsers):
    """Add extract-keywords subcommand parser."""
    parser = subparsers.add_parser(
        "extract-keywords",
        help="규정에서 키워드 추출",
        description="규정 JSON에서 핵심 키워드를 자동으로 추출합니다.",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="data/output/규정집.json",
        help="규정 JSON 파일 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/config/regulation_keywords.json",
        help="출력 파일 경로",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="저장하지 않고 결과만 표시",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="상세 결과 출력",
    )


def _add_feedback_parser(subparsers):
    """Add feedback subcommand parser."""
    parser = subparsers.add_parser(
        "feedback",
        help="피드백 통계 확인",
        description="수집된 피드백 통계를 표시합니다.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="모든 피드백 삭제",
    )


def _add_analyze_parser(subparsers):
    """Add analyze subcommand parser."""
    parser = subparsers.add_parser(
        "analyze",
        help="피드백 기반 개선 제안",
        description="피드백을 분석하여 개선 사항을 제안합니다.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="상세 결과 출력",
    )


def _add_synonym_parser(subparsers):
    """Add synonym subcommand parser."""
    # Get LLM settings for suggest command
    providers, default_provider, default_model, default_base_url = (
        _get_default_llm_settings()
    )

    parser = subparsers.add_parser(
        "synonym",
        help="동의어 관리 (LLM 기반 자동 생성 및 수동 관리)",
        description="동의어 사전을 관리합니다. LLM으로 동의어를 자동 생성하거나 수동으로 추가/제거할 수 있습니다.",
    )
    synonym_subparsers = parser.add_subparsers(dest="synonym_cmd")

    # synonym suggest <term>
    suggest_parser = synonym_subparsers.add_parser(
        "suggest",
        help="LLM으로 동의어 후보 생성",
    )
    suggest_parser.add_argument("term", help="동의어를 생성할 용어")
    suggest_parser.add_argument(
        "--context",
        default="대학 규정",
        help="용어 맥락 (기본: 대학 규정)",
    )
    suggest_parser.add_argument(
        "--auto-add",
        action="store_true",
        help="검토 없이 바로 추가",
    )
    suggest_parser.add_argument(
        "--provider",
        type=str,
        default=default_provider,
        choices=providers,
        help="LLM 프로바이더",
    )
    suggest_parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="모델명",
    )
    suggest_parser.add_argument(
        "--base-url",
        type=str,
        default=default_base_url,
        help="로컬 서버 URL",
    )

    # synonym add <term> <synonym>
    add_parser = synonym_subparsers.add_parser("add", help="동의어 수동 추가")
    add_parser.add_argument("term", help="기준 용어")
    add_parser.add_argument("synonym", help="추가할 동의어")

    # synonym remove <term> <synonym>
    remove_parser = synonym_subparsers.add_parser("remove", help="동의어 제거")
    remove_parser.add_argument("term", help="기준 용어")
    remove_parser.add_argument("synonym", help="제거할 동의어")

    # synonym list [term]
    list_parser = synonym_subparsers.add_parser("list", help="동의어 목록 조회")
    list_parser.add_argument("term", nargs="?", help="특정 용어만 조회 (생략 시 전체)")


def create_parser() -> argparse.ArgumentParser:
    """Create main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="regulation",
        description="대학 규정 관리 시스템 - HWP 변환, RAG 검색, AI Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  regulation convert "규정집.hwp"       HWP → JSON 변환
  regulation sync data/output/규정집.json  DB 동기화
  regulation search "교원 연구년"        규정 검색
  regulation ask "휴학 절차"             AI 질문
  regulation status                      상태 확인
  regulation serve --web                 Web UI 시작
""",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        dest="global_debug",
        help="전역 디버그 모드 활성화",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="사용 가능한 명령어",
        metavar="<command>",
    )

    # Add all subcommands
    _add_convert_parser(subparsers)
    _add_sync_parser(subparsers)
    _add_search_parser(subparsers)
    _add_ask_parser(subparsers)
    _add_status_parser(subparsers)
    _add_reset_parser(subparsers)
    _add_serve_parser(subparsers)
    _add_evaluate_parser(subparsers)
    _add_extract_keywords_parser(subparsers)
    _add_feedback_parser(subparsers)
    _add_analyze_parser(subparsers)
    _add_synonym_parser(subparsers)

    return parser


# =============================================================================
# Command Handlers
# =============================================================================


def cmd_convert(args) -> int:
    """Execute convert command - HWP to JSON conversion."""

    # Convert argument names to match main.py expectations
    # (unified CLI uses kebab-case, main.py uses snake_case)
    class ConvertArgs:
        def __init__(self, args):
            self.input_path = args.input_path
            self.output_dir = getattr(args, "output_dir", "data/output")
            self.use_llm = getattr(args, "use_llm", False)
            self.provider = getattr(args, "provider", "openai")
            self.model = getattr(args, "model", None)
            self.base_url = getattr(args, "base_url", None)
            self.allow_llm_fallback = getattr(args, "allow_llm_fallback", False)
            self.force = getattr(args, "force", False)
            self.cache_dir = getattr(args, "cache_dir", ".cache")
            self.verbose = getattr(args, "verbose", False)
            self.enhance_rag = getattr(args, "enhance_rag", True)

    from ...main import run_pipeline

    return run_pipeline(ConvertArgs(args))


def cmd_sync(args) -> int:
    """Execute sync command."""
    from .cli import cmd_sync as _cmd_sync

    return _cmd_sync(args)


def cmd_search(args) -> int:
    """Execute search command."""
    from .cli import cmd_search as _cmd_search

    return _cmd_search(args)


def cmd_ask(args) -> int:
    """Execute ask command."""
    from .cli import cmd_ask as _cmd_ask

    return _cmd_ask(args)


def cmd_status(args) -> int:
    """Execute status command."""
    from .cli import cmd_status as _cmd_status

    return _cmd_status(args)


def cmd_reset(args) -> int:
    """Execute reset command."""
    from .cli import cmd_reset as _cmd_reset

    return _cmd_reset(args)


def cmd_serve(args) -> int:
    """Execute serve command - start Web UI or MCP Server."""
    import os

    # Enable warmup for server modes
    os.environ["WARMUP_ON_INIT"] = "true"

    if args.web:
        import gradio as gr

        from .gradio_app import CUSTOM_CSS, create_app

        app = create_app(db_path=args.db_path)
        app.launch(
            server_port=args.port,
            share=args.share,
            show_error=True,
            css=CUSTOM_CSS,
            theme=gr.themes.Soft(
                primary_hue=gr.themes.colors.emerald,
                neutral_hue=gr.themes.colors.neutral,
            ).set(
                body_background_fill="#0f0f0f",
                body_background_fill_dark="#0f0f0f",
                block_background_fill="#1a1a1a",
                block_background_fill_dark="#1a1a1a",
                border_color_primary="rgba(255,255,255,0.06)",
                border_color_primary_dark="rgba(255,255,255,0.06)",
            ),
        )
        return 0
    elif args.mcp:
        from .mcp_server import mcp

        mcp.run()
        return 0
    return 1


def cmd_evaluate(args) -> int:
    """Execute evaluate command - run quality evaluation."""
    from rich.console import Console

    from ..application.evaluate import EvaluationUseCase
    from ..application.search_usecase import SearchUseCase
    from ..infrastructure.chroma_store import ChromaVectorStore
    from ..infrastructure.llm_adapter import LLMClientAdapter

    console = Console()

    # Initialize components
    store = ChromaVectorStore(persist_directory=args.db_path)

    # Get default settings for LLM
    _, provider, model, base_url = _get_default_llm_settings()

    llm_client = LLMClientAdapter(
        provider=provider,
        model=model,
        base_url=base_url,
    )
    search_usecase = SearchUseCase(
        store=store,
        llm_client=llm_client,
        use_reranker=True,
    )

    # Run evaluation
    eval_usecase = EvaluationUseCase(
        search_usecase=search_usecase,
        dataset_path=args.dataset,
    )

    console.print("[bold]🔍 평가 데이터셋 로드 중...[/bold]")
    test_cases = eval_usecase.load_dataset()
    console.print(f"[dim]총 {len(test_cases)}개 테스트 케이스[/dim]\n")

    console.print("[bold]🧪 평가 실행 중...[/bold]")
    summary = eval_usecase.run_evaluation(
        top_k=args.top_k,
        category=args.category,
    )

    # Print results
    console.print(eval_usecase.format_summary(summary))

    if args.verbose:
        console.print(eval_usecase.format_details(summary))

    return 0 if summary.pass_rate >= 0.8 else 1


def cmd_extract_keywords(args) -> int:
    """Execute extract-keywords command."""
    from rich.console import Console

    from ..infrastructure.keyword_extractor import KeywordExtractor

    console = Console()

    extractor = KeywordExtractor(
        json_path=args.json_path,
        output_path=args.output,
    )

    console.print("[bold]📚 규정 키워드 추출 중...[/bold]")
    result = extractor.extract_keywords()

    console.print(extractor.format_summary(result))

    if args.verbose:
        console.print(extractor.format_details(result))

    if not args.dry_run:
        output_path = extractor.save_keywords(result)
        console.print(f"\n[green]✅ 저장됨: {output_path}[/green]")

    return 0


def cmd_feedback(args) -> int:
    """Execute feedback command."""
    from rich.console import Console

    from ..infrastructure.feedback import FeedbackCollector

    console = Console()
    collector = FeedbackCollector()

    if args.clear:
        collector.clear_feedback()
        console.print("[yellow]🗑️ 모든 피드백이 삭제되었습니다.[/yellow]")
        return 0

    stats = collector.get_statistics()
    console.print(collector.format_statistics(stats))

    return 0


def cmd_analyze(args) -> int:
    """Execute analyze command - analyze feedback for improvements."""
    from rich.console import Console

    from ..application.auto_learn import AutoLearnUseCase
    from ..infrastructure.feedback import FeedbackCollector

    console = Console()

    collector = FeedbackCollector()
    auto_learn = AutoLearnUseCase(feedback_collector=collector)

    console.print("[bold]🧠 피드백 분석 중...[/bold]")
    result = auto_learn.analyze_feedback()

    console.print(auto_learn.format_suggestions(result))

    return 0


def cmd_synonym(args) -> int:
    """Execute synonym management commands."""
    from .cli import cmd_synonym as _cmd_synonym

    return _cmd_synonym(args)


# =============================================================================
# Entry Point
# =============================================================================


def main(argv: Optional[list] = None) -> int:
    """Main entry point for the unified CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Global debug flag handling
    if hasattr(args, "global_debug") and args.global_debug:
        args.debug = True

    # 커맨드 없이 실행하면 interactive 모드로 시작
    if not args.command:
        # 기본값 설정
        args.command = "search"
        args.query = None
        args.interactive = True
        args.top_k = 5
        args.include_abolished = False
        args.db_path = "data/chroma_db"
        args.no_rerank = False
        args.verbose = False
        args.debug = False
        args.feedback = False
        args.answer = False
        args.quick = False
        args.show_sources = False
        # LLM 기본값
        providers, provider, model, base_url = _get_default_llm_settings()
        args.provider = provider
        args.model = model
        args.base_url = base_url
        return cmd_search(args)

    commands = {
        "convert": cmd_convert,
        "sync": cmd_sync,
        "search": cmd_search,
        "ask": cmd_ask,
        "status": cmd_status,
        "reset": cmd_reset,
        "serve": cmd_serve,
        "evaluate": cmd_evaluate,
        "extract-keywords": cmd_extract_keywords,
        "feedback": cmd_feedback,
        "analyze": cmd_analyze,
        "synonym": cmd_synonym,
    }

    if args.command in commands:
        try:
            return commands[args.command](args)
        except KeyboardInterrupt:
            print("\nAborted.")
            return 130

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
