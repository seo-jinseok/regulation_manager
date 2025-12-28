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
from pathlib import Path
from typing import Optional

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _get_default_llm_settings():
    """Get default LLM settings from environment."""
    providers = ["ollama", "lmstudio", "mlx", "local", "openai", "gemini", "openrouter"]
    default_provider = os.getenv("LLM_PROVIDER") or "ollama"
    if default_provider not in providers:
        default_provider = "ollama"
    default_model = os.getenv("LLM_MODEL") or None
    default_base_url = os.getenv("LLM_BASE_URL") or None
    return providers, default_provider, default_model, default_base_url


def _add_convert_parser(subparsers):
    """Add convert subcommand parser."""
    convert_providers = ["openai", "gemini", "openrouter", "ollama", "lmstudio", "local", "mlx"]
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
        "-v", "--verbose",
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
    parser = subparsers.add_parser(
        "search",
        help="규정 검색",
        description="Hybrid Search + Reranking으로 규정을 검색합니다.",
    )
    parser.add_argument(
        "query",
        type=str,
        help="검색 쿼리",
    )
    parser.add_argument(
        "-n", "--top-k",
        type=int,
        default=5,
        help="결과 개수 (기본: 5)",
    )
    parser.add_argument(
        "--include-abolished",
        action="store_true",
        help="폐지 규정 포함",
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
        "-v", "--verbose",
        action="store_true",
        help="상세 정보 출력",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 정보 출력",
    )


def _add_ask_parser(subparsers):
    """Add ask subcommand parser."""
    providers, default_provider, default_model, default_base_url = _get_default_llm_settings()
    
    parser = subparsers.add_parser(
        "ask",
        help="규정 질문 (LLM 답변)",
        description="LLM을 사용하여 규정에 대한 질문에 답변합니다.",
    )
    parser.add_argument(
        "question",
        type=str,
        help="질문",
    )
    parser.add_argument(
        "-n", "--top-k",
        type=int,
        default=5,
        help="참고 규정 수 (기본: 5)",
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
        "-v", "--verbose",
        action="store_true",
        help="상세 정보 출력",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 정보 출력",
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
            self.output_dir = getattr(args, 'output_dir', 'data/output')
            self.use_llm = getattr(args, 'use_llm', False)
            self.provider = getattr(args, 'provider', 'openai')
            self.model = getattr(args, 'model', None)
            self.base_url = getattr(args, 'base_url', None)
            self.allow_llm_fallback = getattr(args, 'allow_llm_fallback', False)
            self.force = getattr(args, 'force', False)
            self.cache_dir = getattr(args, 'cache_dir', '.cache')
            self.verbose = getattr(args, 'verbose', False)
            self.enhance_rag = getattr(args, 'enhance_rag', True)
    
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
    if args.web:
        from .gradio_app import create_app
        app = create_app(db_path=args.db_path)
        app.launch(
            server_port=args.port,
            share=args.share,
        )
        return 0
    elif args.mcp:
        from .mcp_server import mcp
        mcp.run()
        return 0
    return 1


# =============================================================================
# Entry Point
# =============================================================================

def main(argv: Optional[list] = None) -> int:
    """Main entry point for the unified CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "convert": cmd_convert,
        "sync": cmd_sync,
        "search": cmd_search,
        "ask": cmd_ask,
        "status": cmd_status,
        "reset": cmd_reset,
        "serve": cmd_serve,
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
