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
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Enable readline for better interactive input handling (backspace, arrow keys, etc.)
try:
    import readline  # noqa: F401 - imported for side effects
except ImportError:
    pass  # readline not available on some platforms

# Load .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Formatters for output formatting
from .chat_logic import (
    attachment_label_variants,
    build_history_context,
    expand_followup_query,
    extract_regulation_title,
    has_explicit_target,
    parse_attachment_request,
)
from .formatters import (
    build_display_path,
    clean_path_segments,
    extract_display_text,
    filter_by_relevance,
    format_regulation_content,
    get_confidence_info,
    get_relevance_label_combined,
    infer_attachment_label,
    infer_regulation_title_from_tables,
    normalize_markdown_emphasis,
    normalize_markdown_table,
    normalize_relevance_scores,
    render_full_view_nodes,
    strip_path_prefix,
)
from .query_handler import QueryContext, QueryHandler, QueryOptions, QueryResult, QueryType
from ..infrastructure.patterns import REGULATION_ONLY_PATTERN, RULE_CODE_PATTERN

# Rich for pretty output (optional)
try:
    from rich.console import Console, Group
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

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


def _print_sources_and_confidence(sources: list, confidence: float, verbose: bool = False):
    """Print sources and confidence panel."""
    if not sources:
        return

    if RICH_AVAILABLE:
        console.print()
        console.print("[bold cyan]📚 참고 규정:[/bold cyan]")

        for i, src in enumerate(sources, 1):
            if isinstance(src, dict):
                title = src.get("title", "규정")
                rule_code = src.get("rule_code", "")
                text = src.get("text", "")
                score = src.get("score", 0.0)
                path = src.get("path", "")
            else:
                chunk = src.chunk
                title = chunk.parent_path[0] if chunk.parent_path else chunk.title
                rule_code = chunk.rule_code
                text = extract_display_text(chunk.text)
                score = src.score
                path = build_display_path(chunk.parent_path, chunk.text, chunk.title)

            content_parts = [
                f"[bold blue]📖 {title}[/bold blue]",
                f"[dim]📍 {path}[/dim]" if path else "",
                "",
                text[:500] + "..." if len(text) > 500 else text,
                "",
                f"[dim]📋 규정번호: {rule_code} | AI 유사도: {score:.3f}[/dim]"
            ]

            console.print(
                Panel(
                    "\n".join(filter(None, content_parts)),
                    title=f"[{i}]",
                    border_style="blue",
                )
            )

        conf_icon, conf_label, conf_detail = get_confidence_info(confidence)
        console.print()
        console.print(
            Panel(
                f"[bold]{conf_icon} {conf_label}[/bold] (신뢰도 {confidence:.0%})\n\n{conf_detail}",
                title="📊 답변 신뢰도",
                border_style="dim",
            )
        )
    else:
        print("\n=== 참고 규정 ===")
        for i, src in enumerate(sources, 1):
            if isinstance(src, dict):
                print(f"[{i}] {src.get('rule_code')}: {src.get('text')[:100]}...")
            else:
                print(f"[{i}] {src.chunk.rule_code}: {src.chunk.text[:100]}...")
        print(f"\n평균 신뢰도: {confidence:.0%}")


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
        description="질문이면 AI 답변을, 키워드면 문서 검색 결과를 자동으로 보여줍니다. (혹은 -a/-q 옵션으로 강제)",
    )
    search_parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="검색 쿼리 또는 질문",
    )
    search_parser.add_argument(
        "-n",
        "--top-k",
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
        "-v",
        "--verbose",
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
    search_parser.add_argument(
        "--interactive",
        action="store_true",
        help="대화형 모드로 연속 질의",
    )
    # Unified specific arguments
    mode_group = search_parser.add_mutually_exclusive_group()
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
        "-n",
        "--top-k",
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
        "-v",
        "--verbose",
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

    # synonym command group
    synonym_parser = subparsers.add_parser(
        "synonym",
        help="동의어 관리 (LLM 기반 자동 생성 및 수동 관리)",
    )
    synonym_subparsers = synonym_parser.add_subparsers(dest="synonym_cmd")

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
    suggest_parser.add_argument("--model", type=str, default=default_model, help="모델명")
    suggest_parser.add_argument("--base-url", type=str, default=default_base_url, help="로컬 서버 URL")

    # synonym add <term> <synonym>
    add_syn_parser = synonym_subparsers.add_parser("add", help="동의어 수동 추가")
    add_syn_parser.add_argument("term", help="기준 용어")
    add_syn_parser.add_argument("synonym", help="추가할 동의어")

    # synonym remove <term> <synonym>
    remove_syn_parser = synonym_subparsers.add_parser("remove", help="동의어 제거")
    remove_syn_parser.add_argument("term", help="기준 용어")
    remove_syn_parser.add_argument("synonym", help="제거할 동의어")

    # synonym list [term]
    list_syn_parser = synonym_subparsers.add_parser("list", help="동의어 목록 조회")
    list_syn_parser.add_argument("term", nargs="?", help="특정 용어만 조회 (생략 시 전체)")

    return parser


def cmd_sync(args) -> int:
    """Execute sync command."""
    from ..application.sync_usecase import SyncUseCase
    from ..infrastructure.chroma_store import ChromaVectorStore
    from ..infrastructure.json_loader import JSONDocumentLoader

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
    if hasattr(args, "answer") and args.answer:
        force_mode = "ask"
    elif hasattr(args, "quick") and args.quick:
        force_mode = "search"

    return decide_search_mode(args.query, force_mode)


def _format_toc(toc: list[str]) -> str:
    if not toc:
        return "목차 정보가 없습니다."
    return "### 목차\n" + "\n".join([f"- {t}" for t in toc])


def _print_markdown(title: str, text: object) -> None:
    if RICH_AVAILABLE:
        renderable = text
        if isinstance(text, str):
            renderable = Markdown(text)
            
        console.print()
        console.print(Panel(renderable, title=title, border_style="green"))
    else:
        print(f"\n=== {title} ===")
        print(str(text))


def _text_from_regulation(formatted_text: str) -> object:
    """Convert formatted regulation text to Rich Text with header styling."""
    if not RICH_AVAILABLE:
        return formatted_text
        
    text_obj = Text()
    
    # Regex for markdown header: whitespace, 1-6 hashes, whitespace, text
    header_pattern = re.compile(r"^\s*(#{1,6})\s+(.*)")
    
    for line in formatted_text.splitlines():
        match = header_pattern.match(line)
        if match:
             # Extract title part
             # We ignore header level for CLI logic, just bold cyan
             title = match.group(2)
             text_obj.append(title + "\n", style="bold cyan")
        else:
            text_obj.append(line + "\n")
    return text_obj


def _print_query_result(result: QueryResult, verbose: bool = False) -> None:
    """Print QueryHandler result to CLI."""
    if not result.success:
        print_error(result.content)
        return
    
    if result.type == QueryType.ERROR:
        print_error(result.content)
        return
    
    if result.type == QueryType.CLARIFICATION:
        # Handle clarification requests
        if result.clarification_type == "audience":
            print("대상을 선택해주세요:")
            for opt in result.clarification_options:
                print(f"  - {opt}")
        elif result.clarification_type == "regulation":
            print("여러 규정이 매칭됩니다. 선택해주세요:")
            for i, opt in enumerate(result.clarification_options, 1):
                print(f"  {i}. {opt}")
        return
    
    # Map result types to titles
    title_map = {
        QueryType.OVERVIEW: "📋 규정 개요",
        QueryType.ARTICLE: "📌 조항 전문",
        QueryType.CHAPTER: "📑 장 전문",
        QueryType.ATTACHMENT: "📋 별표/서식",
        QueryType.FULL_VIEW: "📖 규정 전문",
        QueryType.SEARCH: "🔍 검색 결과",
        QueryType.ASK: "💬 AI 답변",
    }
    
    title = title_map.get(result.type, "결과")
    
    # Add regulation info to title if available
    if result.data.get("regulation_title") or result.data.get("title"):
        reg_title = result.data.get("regulation_title") or result.data.get("title")
        title = f"{title} - {reg_title}"
    
    content = result.content
    
    # Custom rendering for SEARCH to preserve indentation
    if result.type == QueryType.SEARCH and RICH_AVAILABLE:
        from rich.console import Group
        
        # Split content into Table part and Top Result part
        # We rely on the "---" separator we added in QueryHandler
        parts = result.content.split("\n\n---\n\n")
        
        renderables = []
        
        # 1. Result Table
        if parts:
            renderables.append(Markdown(parts[0]))
            
        # 2. Top Result Detail
        if len(parts) > 1:
            top_part = parts[1]
            # Split metadata and content
            # QueryHandler adds metadata lines starting with "**" or "###"
            # And then the text.
            # We want to find where the text starts.
            
            # Simple heuristic: Split by double newline, find the chunk text.
            # In QueryHandler: 
            # content += f"### 🏆 1위 결과: ...\n\n"
            # content += f"**규정명:** ...\n\n"
            # content += f"**경로:** ...\n\n{top_text}"
            
            # We can parse this manually or just render metadata as Markdown and Text as Text.
            # Let's extract metadata lines vs text lines.
            
            lines = top_part.splitlines()
            metadata_lines = []
            text_lines = []
            is_text = False
            
            for line in lines:
                if not is_text:
                    if not line.strip(): 
                        continue
                    if line.startswith("###") or line.startswith("**"):
                        metadata_lines.append(line)
                    else:
                        is_text = True
                        text_lines.append(line)
                else:
                    text_lines.append(line)
            
            if metadata_lines:
                renderables.append(Markdown("\n".join(metadata_lines)))
                renderables.append(Text("\n")) # Spacer
                
            if text_lines:
                raw_text = "\n".join(text_lines)
                formatted_text = format_regulation_content(raw_text)
                renderables.append(_text_from_regulation(formatted_text))

        content = Group(*renderables)

    elif result.type in (QueryType.ARTICLE, QueryType.CHAPTER, QueryType.FULL_VIEW):
        # Use Text to preserve exact spacing and style headers manually
        if RICH_AVAILABLE:
            content = _text_from_regulation(content)

    _print_markdown(title, content)

    # Print sources and confidence for ASK results
    if result.type == QueryType.ASK:
        sources = result.data.get("sources", [])
        confidence = result.data.get("confidence", 0.0)
        _print_sources_and_confidence(sources, confidence, verbose)

    # Print debug info if available
    if result.debug_info:
        if RICH_AVAILABLE:
            console.print()
            console.print(Panel(Markdown(result.debug_info), title="🔧 실행 과정 (Debug)", border_style="yellow"))
        else:
            print("\n[실행 과정 (Debug)]")
            print(result.debug_info)

def _print_regulation_overview(overview, other_matches: Optional[list] = None) -> None:
    """Print regulation overview in a nice format."""
    from ..domain.entities import RegulationStatus

    if RICH_AVAILABLE:
        # Build content lines
        lines = []

        # Status info
        status_label = "✅ 시행중" if overview.status == RegulationStatus.ACTIVE else "❌ 폐지"
        lines.append(f"**상태**: {status_label} | **총 조항 수**: {overview.article_count}개")
        lines.append("")

        # Table of contents
        if overview.chapters:
            lines.append("## 📖 목차")
            for ch in overview.chapters:
                article_info = f" ({ch.article_range})" if ch.article_range else ""
                lines.append(f"- **{ch.display_no}** {ch.title}{article_info}")
        else:
            lines.append("*(장 구조 없이 조항으로만 구성된 규정)*")

        # Addenda info
        if overview.has_addenda:
            lines.append("")
            lines.append("📎 **부칙** 있음")

        # Next action hint
        lines.append("")
        lines.append("---")
        lines.append(f"💡 특정 조항 검색: `{overview.title} 제N조` 또는 `{overview.rule_code} 제N조`")

        if other_matches:
            lines.append("")
            lines.append("❓ **혹시 다음 규정을 찾으셨나요?**")
            for m in other_matches:
                lines.append(f"- {m}")

        content = "\n".join(lines)
        console.print()
        console.print(
            Panel(
                Markdown(content),
                title=f"📋 {overview.title} ({overview.rule_code})",
                border_style="cyan",
            )
        )
    else:
        print(f"\n=== {overview.title} ({overview.rule_code}) ===")
        status_label = "시행중" if overview.status == RegulationStatus.ACTIVE else "폐지"
        print(f"상태: {status_label} | 총 조항 수: {overview.article_count}개")
        print("\n목차:")
        for ch in overview.chapters:
            article_info = f" ({ch.article_range})" if ch.article_range else ""
            print(f"  - {ch.display_no} {ch.title}{article_info}")
        if overview.has_addenda:
            print("\n부칙 있음")
        
        if other_matches:
            print("\n❓ 혹시 다음 규정을 찾으셨나요?")
            for m in other_matches:
                print(f"  - {m}")


def _find_json_path() -> Optional[str]:
    """Find the regulation JSON file in data/output directory."""
    output_dir = Path("data/output")
    if not output_dir.exists():
        return None
    # Find the first JSON file that looks like a regulation file
    for f in output_dir.iterdir():
        if f.suffix == ".json" and not f.name.endswith("_metadata.json"):
            if f.name != "dummy.json":
                return str(f)
    return None


_BACKSPACE_CHARS = {"\b", "\x7f"}


def _sanitize_query_input(text: Optional[str]) -> str:
    """Normalize user input by applying backspaces and trimming."""
    if text is None:
        return ""
    buffer = []
    for char in str(text):
        if char in _BACKSPACE_CHARS:
            if buffer:
                buffer.pop()
            continue
        if ord(char) < 32 or ord(char) == 127:
            continue
        buffer.append(char)
    return "".join(buffer).strip()


def _append_history(
    state: Optional[dict],
    role: str,
    content: str,
    max_messages: int = 20,
) -> None:
    if not state or not content:
        return
    history = state.setdefault("history", [])
    history.append({"role": role, "content": content})
    if len(history) > max_messages:
        del history[:-max_messages]


def _select_regulation(matches, interactive: bool):
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    print_info("여러 규정이 매칭됩니다. 번호 또는 제목으로 선택해주세요.")
    for i, match in enumerate(matches, 1):
        print(f"{i}. {match.title}")

    if not interactive:
        return None

    while True:
        try:
            choice = input("\n선택 (Enter로 취소): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n취소합니다.")
            return None

        if not choice:
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(matches):
                return matches[idx - 1]
        for match in matches:
            if match.title == choice:
                return match
        print("올바른 선택이 아닙니다.")


def _perform_unified_search(
    args,
    force_mode: Optional[str] = None,
    state: Optional[dict] = None,
    interactive: bool = False,
) -> int:
    """Core logic for unified search/ask."""
    from rich.panel import Panel
    from ..infrastructure.chroma_store import ChromaVectorStore
    from ..infrastructure.llm_adapter import LLMClientAdapter
    from ..infrastructure.function_gemma_adapter import FunctionGemmaAdapter
    from ..infrastructure.tool_executor import ToolExecutor
    from ..infrastructure.query_analyzer import QueryAnalyzer
    from ..application.search_usecase import SearchUseCase

    state = state or {}
    raw_query = _sanitize_query_input(args.query)
    query = raw_query
    
    if interactive and query:
        context_hint = state.get("last_regulation") or state.get("last_query")
        query = expand_followup_query(query, context_hint)
    
    query = _sanitize_query_input(query)
    if not query:
        if interactive:
            return 0
        print_error("검색어를 입력해주세요.")
        return 1
    
    args.query = query
    
    # Initialize components
    store = ChromaVectorStore(persist_directory=args.db_path)
    if store.count() == 0:
        print_error("데이터베이스가 비어 있습니다. 먼저 sync를 실행하세요.")
        return 1

    use_reranker = not getattr(args, "no_rerank", False)
    use_tool_calling = not getattr(args, "no_tools", False)
    
    # Initialize LLM Client
    llm_client = None
    try:
        llm_client = LLMClientAdapter(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
        )
    except Exception as e:
        if not interactive and (args.command == "ask" or force_mode == "ask"):
            print_error(f"LLM 초기화 실패: {e}")
            return 1

    # Initialize FunctionGemma if tools enabled
    function_gemma_client = None
    if use_tool_calling:
        search_uc = SearchUseCase(store, use_reranker=use_reranker)
        analyzer = QueryAnalyzer()
        executor = ToolExecutor(search_usecase=search_uc, query_analyzer=analyzer)
        tool_mode = getattr(args, "tool_mode", "auto")
        function_gemma_client = FunctionGemmaAdapter(
            tool_executor=executor,
            query_analyzer=analyzer,  # Pass analyzer for intent-aware prompts
            api_mode=tool_mode,
            llm_client=llm_client,
        )

    # Prepare Handler
    handler = QueryHandler(
        store=store,
        llm_client=llm_client,
        function_gemma_client=function_gemma_client,
        use_reranker=use_reranker,
    )
    
    context = QueryContext(
        state=state,
        interactive=interactive,
        last_regulation=state.get("last_regulation"),
        last_rule_code=state.get("last_rule_code"),
    )
    
    options = QueryOptions(
        top_k=args.top_k,
        include_abolished=getattr(args, "include_abolished", False),
        use_rerank=use_reranker,
        force_mode=force_mode,
        llm_provider=args.provider,
        llm_model=args.model,
        llm_base_url=args.base_url,
        use_function_gemma=use_tool_calling,
        show_debug=args.debug,
    )

    # Execute and Stream if it's an AI answer in interactive/ask mode
    if RICH_AVAILABLE and (force_mode == "ask" or (not force_mode and _decide_search_mode(args) == "ask")):
        answer_text = ""
        try:
            stream_gen = handler.process_query_stream(query, context, options)
            from rich.live import Live
            from rich.text import Text
            
            # Initial spacer
            console.print()
            
            with Live(Panel(Text("..."), title="💬 AI 답변 준비 중", border_style="dim"), console=console, refresh_per_second=10) as live:
                for event in stream_gen:
                    evt_type = event.get("type")
                    
                    if evt_type == "progress":
                        # Optionally show progress above the live panel? 
                        # For now, let's update title or just ignore for cleaner look
                        pass
                    
                    elif evt_type == "token":
                        answer_text += event.get("content", "")
                        live.update(Panel(Markdown(answer_text), title="💬 AI 답변", border_style="green"))
                        
                    elif evt_type == "complete":
                        answer_text = event.get("content", answer_text)
                        live.update(Panel(Markdown(answer_text), title="💬 AI 답변", border_style="green"))
                        
                        # Show sources if available
                        data = event.get("data", {})
                        if data.get("sources"):
                            # We can print sources after the live panel
                            _print_sources_and_confidence(data.get("sources", []), data.get("confidence", 0.0), args.verbose)
                            
                        _update_state_from_result(state, data, raw_query, answer_text, event.get("suggestions", []))
                        return 0
                        
                    elif evt_type == "error":
                        live.update(Panel(Text(f"⚠️ {event['content']}"), title="❌ 오류", border_style="red"))
                        return 1
                        
                    elif evt_type == "clarification":
                        live.stop()
                        _handle_cli_clarification(event)
                        return 0
            return 0
        except Exception as e:
            print_error(f"처리 중 오류 발생: {e}")
            return 1

    # Standard non-streaming path
    result = handler.process_query(query, context, options)
    
    if result.type == QueryType.CLARIFICATION:
        _handle_cli_clarification(result)
        return 0
        
    if not result.success and result.type != QueryType.ERROR:
        print_info(result.content)
        return 0
        
    if result.type == QueryType.ERROR:
        print_error(result.content)
        return 1

    # Display Result
    _print_query_result(result, args.verbose)
    
    # Update State
    _update_state_from_result(state, result.data, raw_query, result.content, result.suggestions)
    
    return 0

def _update_state_from_result(state: dict, data: dict, raw_query: str, content: str, suggestions: list):
    """Sync state with query result."""
    state["last_regulation"] = data.get("regulation_title") or data.get("title")
    state["last_rule_code"] = data.get("rule_code")
    state["last_query"] = raw_query
    state["last_answer"] = content
    state["suggestions"] = suggestions

def _handle_cli_clarification(result_or_event: Any):
    """Handle clarification requests in CLI."""
    if isinstance(result_or_event, dict):
        # Event from stream
        c_type = result_or_event.get("clarification_type")
        options = result_or_event.get("options", [])
        content = result_or_event.get("content", "")
    else:
        # QueryResult
        c_type = result_or_event.clarification_type
        options = result_or_event.clarification_options
        content = result_or_event.content

    print_info(content)
    if c_type == "audience":
        print("대상을 선택해주세요:")
        for opt in options:
            print(f"  - {opt}")
    elif c_type == "regulation":
        print("여러 규정이 매칭됩니다. 선택해주세요:")
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")


def cmd_search(args) -> int:
    """Execute search command (Unified)."""
    if getattr(args, "interactive", False):
        return _run_interactive_session(args)
    return _perform_unified_search(args)


def cmd_ask(args) -> int:
    """Execute ask command (Legacy Wrapper)."""
    # Map 'question' arg to 'query' expected by unified logic
    if hasattr(args, "question"):
        args.query = args.question
    return _perform_unified_search(args, force_mode="ask")


def _run_interactive_session(args) -> int:
    """Run an interactive CLI session with conversational turns."""
    from .query_suggestions import (
        format_examples_for_cli,
        format_suggestions_for_cli,
        get_initial_examples,
    )

    state = {
        "last_regulation": None,
        "last_rule_code": None,
        "last_query": None,
        "history": [],
    }

    # 현재 선택 가능한 예시/제안 목록
    current_suggestions = get_initial_examples()

    # 시작 시 쿼리 예시 표시
    print_info("대화형 모드입니다. 아래 예시 중 번호를 선택하거나 직접 질문하세요.\n")
    print(format_examples_for_cli(current_suggestions))
    print("\n  '/exit' 종료, '/reset' 문맥 초기화, '/help' 도움말\n")

    prompt = ">>> "
    query = (args.query or "").strip()

    while True:
        if not query:
            try:
                query = input(prompt).strip()
            except (KeyboardInterrupt, EOFError):
                print("\n종료합니다.")
                return 0

        # 번호 입력 처리
        if query.isdigit():
            idx = int(query) - 1
            if 0 <= idx < len(current_suggestions):
                query = current_suggestions[idx]
                print(f"  → {query}")
            else:
                print_error(f"1~{len(current_suggestions)} 사이의 번호를 입력하세요.")
                query = ""
                continue

        if query.lower() in ("/exit", "exit", "quit", "q"):
            print("종료합니다.")
            return 0
        if query.lower() in ("/reset", "reset"):
            state["last_regulation"] = None
            state["last_rule_code"] = None
            state["last_query"] = None
            state["history"] = []
            current_suggestions = get_initial_examples()
            print_info("문맥을 초기화했습니다.\n")
            print(format_examples_for_cli(current_suggestions))
            query = ""
            continue
        if query.lower() in ("/help", "help"):
            print("명령어: /exit, /reset, /help")
            print("번호를 입력하면 해당 예시/제안을 실행합니다.")
            query = ""
            continue

        # Sanitize and validate query before passing to search
        sanitized = _sanitize_query_input(query)
        if sanitized:
            args.query = sanitized
            _perform_unified_search(args, state=state, interactive=True)

            # Update suggestions from state (populated by QueryHandler via _perform_unified_search)
            if state.get("suggestions"):
                current_suggestions = state["suggestions"]
            else:
                # Fallback to initial examples if no specific suggestions provided
                current_suggestions = get_initial_examples()

        query = ""


def _collect_cli_feedback(query: str, rule_code: str):
    """Interactively collect feedback from CLI."""
    from ..infrastructure.feedback import FeedbackCollector

    print("\n" + "=" * 30)
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
            source="cli",
        )
        print("✅ 소중한 피드백 감사합니다!")
    except (KeyboardInterrupt, EOFError):
        print("\n건너뜁니다.")


def cmd_status(args) -> int:
    """Execute status command."""
    from ..application.sync_usecase import SyncUseCase
    from ..infrastructure.chroma_store import ChromaVectorStore
    from ..infrastructure.json_loader import JSONDocumentLoader

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
        table.add_row("저장된 조항 수", str(status["store_chunks"]))
        table.add_row("규정 수", str(status["store_regulations"]))

        console.print(table)
    else:
        print("동기화 상태")
        print("-" * 40)
        for k, v in status.items():
            print(f"  {k}: {v}")

    return 0


def cmd_reset(args) -> int:
    """Execute reset command - delete all data."""
    from ..application.sync_usecase import SyncUseCase
    from ..infrastructure.chroma_store import ChromaVectorStore
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
    print_info(f"삭제 예정 조항 수: {chunk_count}")

    # Clear vector store
    deleted = store.clear_all()

    # Clear sync state
    sync.reset_state()

    print_success(f"데이터베이스 초기화 완료! {deleted}개 조항 삭제됨")
    return 0


def cmd_synonym(args) -> int:
    """Execute synonym management commands."""
    from ..application.synonym_generator_service import SynonymGeneratorService
    from ..infrastructure.llm_adapter import LLMClientAdapter

    # Handle no subcommand
    if not args.synonym_cmd:
        print_error("synonym 서브커맨드를 지정해주세요: suggest, add, remove, list")
        print_info("예: regulation synonym suggest '정원'")
        return 1

    # Initialize service (without LLM for non-suggest commands)
    service = SynonymGeneratorService()

    if args.synonym_cmd == "list":
        # List synonyms
        if args.term:
            synonyms = service.get_synonyms(args.term)
            if synonyms:
                print_success(f"'{args.term}'의 동의어 ({len(synonyms)}개):")
                for i, syn in enumerate(synonyms, 1):
                    print(f"  {i}. {syn}")
            else:
                print_info(f"'{args.term}'에 대한 동의어가 없습니다.")
        else:
            terms = service.list_terms()
            if terms:
                print_success(f"등록된 용어 ({len(terms)}개):")
                for term in sorted(terms):
                    count = len(service.get_synonyms(term))
                    print(f"  - {term} ({count}개)")
            else:
                print_info("등록된 동의어가 없습니다.")
        return 0

    elif args.synonym_cmd == "add":
        # Add synonym manually
        if service.add_synonym(args.term, args.synonym):
            print_success(f"'{args.synonym}'이(가) '{args.term}'의 동의어로 추가되었습니다.")
        else:
            print_info(f"'{args.synonym}'은(는) 이미 '{args.term}'의 동의어입니다.")
        return 0

    elif args.synonym_cmd == "remove":
        # Remove synonym
        if service.remove_synonym(args.term, args.synonym):
            print_success(f"'{args.synonym}'이(가) '{args.term}'의 동의어에서 제거되었습니다.")
        else:
            print_error(f"'{args.synonym}'은(는) '{args.term}'의 동의어가 아닙니다.")
            return 1
        return 0

    elif args.synonym_cmd == "suggest":
        # Generate synonyms using LLM
        try:
            llm_client = LLMClientAdapter(
                provider=args.provider,
                model=args.model,
                base_url=args.base_url,
            )
            service = SynonymGeneratorService(llm_client=llm_client)
        except Exception as e:
            print_error(f"LLM 클라이언트 초기화 실패: {e}")
            return 1

        # Show existing synonyms if any
        existing = service.get_synonyms(args.term)
        if existing:
            print_info(f"현재 '{args.term}'의 동의어 ({len(existing)}개): {', '.join(existing)}")
            print()

        # Generate candidates
        print_info(f"🤖 '{args.term}'의 동의어를 LLM으로 생성 중...")
        try:
            candidates = service.generate_synonyms(args.term, context=args.context)
        except Exception as e:
            print_error(f"동의어 생성 실패: {e}")
            return 1

        if not candidates:
            print_info("생성된 동의어 후보가 없습니다.")
            return 0

        print_success(f"🤖 LLM이 제안하는 동의어 후보 ({len(candidates)}개):")
        for i, candidate in enumerate(candidates, 1):
            print(f"  {i}. {candidate}")

        # Auto-add mode
        if args.auto_add:
            added = service.add_synonyms(args.term, candidates)
            print_success(f"✅ {added}개 동의어가 자동으로 추가되었습니다.")
            return 0

        # Interactive selection
        print()
        print_info("추가할 동의어 번호를 선택하세요 (쉼표로 구분, 전체: all, 취소: q):")
        try:
            choice = input("> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n취소합니다.")
            return 0

        if choice == "q" or not choice:
            print_info("취소되었습니다.")
            return 0

        if choice == "all":
            selected = candidates
        else:
            selected = []
            for part in choice.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part)
                    if 1 <= idx <= len(candidates):
                        selected.append(candidates[idx - 1])

        if selected:
            added = service.add_synonyms(args.term, selected)
            print_success(f"✅ {added}개 동의어가 추가되었습니다.")
        else:
            print_info("추가된 동의어가 없습니다.")

        return 0

    return 1

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
        "synonym": cmd_synonym,
    }

    if args.command in commands:
        return commands[args.command](args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
