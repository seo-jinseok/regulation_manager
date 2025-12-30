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

# Rich for pretty output (optional)
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table

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


def _print_markdown(title: str, text: str) -> None:
    if RICH_AVAILABLE:
        console.print()
        console.print(Panel(Markdown(text), title=title, border_style="green"))
    else:
        print(f"\n=== {title} ===")
        print(text)


def _print_regulation_overview(overview) -> None:
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

    from ..application.full_view_usecase import FullViewUseCase
    from ..application.search_usecase import SearchUseCase
    from ..infrastructure.chroma_store import ChromaVectorStore
    from ..infrastructure.json_loader import JSONDocumentLoader
    from ..infrastructure.llm_adapter import LLMClientAdapter

    state = state or {}
    raw_query = _sanitize_query_input(args.query)
    query = raw_query
    context_hint = None
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
    history_text = (
        build_history_context(state.get("history", [])) if interactive else ""
    )
    if interactive:
        _append_history(state, "user", raw_query)
    explicit_target = has_explicit_target(raw_query)
    explicit_regulation = extract_regulation_title(raw_query)

    mode = force_mode or _decide_search_mode(args)
    if args.verbose:
        print_info(f"실행 모드: {mode.upper()} (쿼리: '{args.query}')")

    # Check if query is regulation name or code only -> show overview
    import re
    from ..application.search_usecase import REGULATION_ONLY_PATTERN, RULE_CODE_PATTERN

    is_regulation_only = REGULATION_ONLY_PATTERN.match(query) is not None
    is_rule_code_only = RULE_CODE_PATTERN.match(query) is not None

    if is_regulation_only or is_rule_code_only:
        from ..infrastructure.json_loader import JSONDocumentLoader

        loader = JSONDocumentLoader()
        json_path = os.getenv("RAG_JSON_PATH") or _find_json_path()

        if json_path and Path(json_path).exists():
            overview = loader.get_regulation_overview(json_path, query)
            if overview:
                _print_regulation_overview(overview)
                state["last_regulation"] = overview.title
                state["last_rule_code"] = overview.rule_code
                state["last_query"] = raw_query
                if interactive:
                    _append_history(
                        state,
                        "assistant",
                        f"{overview.title} 개요를 표시했습니다.",
                    )
                return 0
                return 0
            # If overview not found, fall through to normal search

    # Check if query targets a specific article (e.g. "Regulation Article 7")
    # This allows showing the full text of the article instead of just a search snippet
    article_match = re.search(r"(?:제)?\s*(\d+)\s*조", query)
    target_regulation = explicit_regulation or extract_regulation_title(query)

    if target_regulation and article_match:
        article_no = int(article_match.group(1))
        full_view = FullViewUseCase(JSONDocumentLoader())
        matches = full_view.find_matches(target_regulation)
        selected = _select_regulation(matches, interactive)
        
        if selected:
            if args.debug:
                 print_info(f"DEBUG: Smart Full View - Selected: {selected.title}, Article: {article_no}")
            
            article_node = full_view.get_article_view(selected.rule_code, article_no)
            
            if not article_node and args.debug:
                 print_info(f"DEBUG: Smart Full View - Node not found for rule_code={selected.rule_code}, article={article_no}")
            
            if article_node:
                content_text = render_full_view_nodes([article_node])
                _print_markdown(f"{selected.title} 제{article_no}조", content_text)
                
                state["last_regulation"] = selected.title
                state["last_rule_code"] = selected.rule_code
                state["last_query"] = raw_query
                if interactive:
                    _append_history(
                        state,
                        "assistant",
                        f"{selected.title} 제{article_no}조 전문을 표시했습니다.",
                    )
                return 0

    attachment_request = parse_attachment_request(
        args.query,
        state.get("last_regulation") if interactive else None,
    )
    if attachment_request:
        reg_query, table_no, label = attachment_request
        full_view = FullViewUseCase(JSONDocumentLoader())
        matches = full_view.find_matches(reg_query)
        selected = _select_regulation(matches, interactive)
        if not selected:
            return 0

        label_variants = attachment_label_variants(label)
        tables = full_view.find_tables(selected.rule_code, table_no, label_variants)
        if not tables:
            print_info(f"{label}를 찾을 수 없습니다.")
            return 0

        display_title = infer_regulation_title_from_tables(tables, selected.title)
        label_text = label or "별표"
        title_label = f"{display_title} {label_text}"
        if table_no:
            title_label = f"{display_title} {label_text} {table_no}"
        lines = []
        for idx, table in enumerate(tables, 1):
            path = clean_path_segments(table.path) if table.path else []
            heading = " > ".join(path) if path else display_title
            if table_no:
                table_label = f"{label_text} {table_no}"
            else:
                table_label = infer_attachment_label(table, label_text)
            lines.append(f"### [{idx}] {heading} ({table_label})")
            if table.text:
                lines.append(table.text)
            lines.append(normalize_markdown_table(table.markdown).strip())
        _print_markdown(title_label, "\n\n".join(lines))

        state["last_regulation"] = display_title
        state["last_rule_code"] = selected.rule_code
        state["last_query"] = raw_query
        if interactive:
            _append_history(
                state,
                "assistant",
                f"{title_label} 내용을 표시했습니다.",
            )
        return 0

    if mode == "full_view":
        full_view = FullViewUseCase(JSONDocumentLoader())
        matches = full_view.find_matches(args.query)
        selected = _select_regulation(matches, interactive)
        if not selected:
            return 0

        view = full_view.get_full_view(selected.rule_code) or full_view.get_full_view(
            selected.title
        )
        if not view:
            print_error("규정 전문을 불러오지 못했습니다.")
            return 1

        toc_text = _format_toc(view.toc)
        content_text = render_full_view_nodes(view.content)
        addenda_text = render_full_view_nodes(view.addenda)
        detail = f"{toc_text}\n\n### 본문\n\n{content_text or '본문이 없습니다.'}"
        if addenda_text:
            detail += f"\n\n### 부칙\n\n{addenda_text}"

        _print_markdown(f"{view.title} 전문", detail)
        state["last_regulation"] = view.title
        state["last_rule_code"] = view.rule_code
        state["last_query"] = raw_query
        if interactive:
            _append_history(state, "assistant", f"{view.title} 전문을 표시했습니다.")
        return 0

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
            with console.status(
                "[bold blue]⏳ LLM 클라이언트 초기화 중...[/bold blue]"
            ):
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
            include_abolished=args.include_abolished
            if hasattr(args, "include_abolished")
            else False,
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
                path_segments = (
                    clean_path_segments(r.chunk.parent_path)
                    if r.chunk.parent_path
                    else []
                )
                path = " > ".join(path_segments[-2:]) if path_segments else ""
                reg_title = (
                    r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
                )
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
                display_path = build_display_path(
                    top.chunk.parent_path or [],
                    top.chunk.text,
                    top.chunk.title,
                )
                display_text = strip_path_prefix(
                    top.chunk.text, top.chunk.parent_path or []
                )
                if display_text != top.chunk.text and display_path:
                    detail_text = f"{display_path}\n{display_text}"
                else:
                    detail_text = display_text
                if len(detail_text) > 500:
                    detail_text = detail_text[:500] + "..."
                console.print(
                    Panel(
                        detail_text,
                        title=f"[1위] {top.chunk.rule_code}",
                        border_style="green",
                    )
                )
        else:
            print(f"\n검색 결과: '{args.query}'")
            print("-" * 60)
            for i, r in enumerate(results, 1):
                reg_title = (
                    r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
                )
                display_text = strip_path_prefix(
                    r.chunk.text, r.chunk.parent_path or []
                )
                print(f"{i}. {reg_title} [{r.chunk.rule_code}] (점수: {r.score:.2f})")
                print(f"   {display_text[:100]}...")

        if args.feedback and results:
            _collect_cli_feedback(args.query, results[0].chunk.rule_code)
        if results:
            top = results[0]
            top_regulation = (
                top.chunk.parent_path[0] if top.chunk.parent_path else top.chunk.title
            )
            if explicit_regulation:
                state["last_regulation"] = explicit_regulation
            elif explicit_target or not state.get("last_regulation"):
                state["last_regulation"] = top_regulation
            elif state.get("last_regulation") == top_regulation:
                state["last_regulation"] = top_regulation
            state["last_rule_code"] = top.chunk.rule_code
            state["last_query"] = raw_query
            if interactive:
                summary_text = strip_path_prefix(
                    top.chunk.text, top.chunk.parent_path or []
                )
                summary = f"검색 결과 1위: {top.chunk.rule_code} {summary_text}".strip()
                _append_history(state, "assistant", summary)

    else:
        # Ask (LLM Answer)
        if RICH_AVAILABLE:
            with console.status(
                "[bold green]🤖 AI 답변 생성 중... (10-30초 소요)[/bold green]"
            ):
                try:
                    answer = search.ask(
                        question=raw_query,
                        top_k=args.top_k,
                        history_text=history_text or None,
                        search_query=query,
                    )
                except Exception as e:
                    print_error(f"답변 생성 실패: {e}")
                    return 1
        else:
            print("AI 답변 생성 중...")
            try:
                answer = search.ask(
                    question=raw_query,
                    top_k=args.top_k,
                    history_text=history_text or None,
                    search_query=query,
                )
            except Exception as e:
                print_error(f"답변 생성 실패: {e}")
                return 1

        if args.verbose or args.debug:
            print_query_rewrite(search, args.query)

        answer_text = normalize_markdown_emphasis(answer.text)

        # Display Answer (Ask Style)
        if RICH_AVAILABLE:
            console.print()
            console.print(
                Panel(
                    Markdown(answer_text),
                    title="🤖 AI 답변",
                    border_style="green",
                )
            )

            if answer.sources:
                console.print()
                console.print("[bold cyan]📚 참고 규정:[/bold cyan]")

                # Shared formatting logic
                norm_scores = normalize_relevance_scores(answer.sources)
                display_sources = filter_by_relevance(answer.sources, norm_scores)

                for i, result in enumerate(display_sources, 1):
                    chunk = result.chunk
                    reg_name = (
                        chunk.parent_path[0] if chunk.parent_path else chunk.title
                    )
                    path = build_display_path(
                        chunk.parent_path, chunk.text, chunk.title
                    )
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
                        f"[dim]📋 규정번호: {chunk.rule_code} | 관련도: {rel_score}% {rel_label}[/dim]"
                        + (
                            f" [dim]| AI 신뢰도: {result.score:.3f}[/dim]"
                            if args.verbose
                            else ""
                        ),
                    ]

                    console.print(
                        Panel(
                            "\n".join(content_parts),
                            title=f"[{i}]",
                            border_style="blue",
                        )
                    )

            # Confidence Info
            console.print()
            conf_icon, conf_label, conf_detail = get_confidence_info(answer.confidence)
            console.print(
                Panel(
                    f"[bold]{conf_icon} {conf_label}[/bold] (신뢰도 {answer.confidence:.0%})\n\n{conf_detail}",
                    title="📊 답변 신뢰도",
                    border_style="dim",
                )
            )

        else:
            print("\n=== AI 답변 ===")
            print(answer_text)
            print("\n=== 참고 규정 ===")
            for i, result in enumerate(answer.sources, 1):
                print(f"[{i}] {result.chunk.rule_code}: {result.chunk.text[:100]}...")

            if getattr(args, "show_sources", False):
                print("\n=== 규정 전문 ===")
                for result in answer.sources:
                    print(f"\n--- {result.chunk.rule_code} ---")
                    print(result.chunk.text)

        if args.feedback and answer.sources:
            _collect_cli_feedback(args.query, answer.sources[0].chunk.rule_code)
        if answer.sources:
            top = answer.sources[0].chunk
            top_regulation = top.parent_path[0] if top.parent_path else top.title
            if explicit_regulation:
                state["last_regulation"] = explicit_regulation
            elif explicit_target or not state.get("last_regulation"):
                state["last_regulation"] = top_regulation
            elif state.get("last_regulation") == top_regulation:
                state["last_regulation"] = top_regulation
            state["last_rule_code"] = top.rule_code
        state["last_query"] = raw_query
        if interactive:
            _append_history(state, "assistant", answer_text)

    return 0


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
        get_followup_suggestions,
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
        if not sanitized:
            query = ""
            continue
        args.query = sanitized
        _perform_unified_search(args, state=state, interactive=True)

        # 후속 쿼리 제안
        followups = get_followup_suggestions(
            sanitized,
            regulation_title=state.get("last_regulation"),
        )
        if followups:
            current_suggestions = followups
            print(format_suggestions_for_cli(followups))
        else:
            # 제안이 없으면 기본 예시로 복귀
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
