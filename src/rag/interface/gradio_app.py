"""
Gradio Web UI for Regulation RAG System - ChatGPT Style.

Provides a modern chat-style interface for:
- Searching regulations (auto-detected)
- Asking questions with LLM-generated answers (auto-detected)
- Viewing full regulation text

Usage:
    uv run python -m src.rag.interface.gradio_app
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

from ..application.full_view_usecase import FullViewUseCase, TableMatch
from ..application.search_usecase import QueryRewriteInfo, SearchUseCase
from ..application.sync_usecase import SyncUseCase
from ..domain.entities import RegulationStatus
from ..domain.value_objects import SearchFilter
from ..infrastructure.chroma_store import ChromaVectorStore
from ..infrastructure.json_loader import JSONDocumentLoader
from ..infrastructure.llm_adapter import LLMClientAdapter
from ..infrastructure.llm_client import MockLLMClient
from ..infrastructure.query_analyzer import Audience, QueryAnalyzer

try:
    from ..infrastructure.function_gemma_adapter import FunctionGemmaAdapter

    FUNCTION_GEMMA_AVAILABLE = True
except ImportError:
    FUNCTION_GEMMA_AVAILABLE = False
    FunctionGemmaAdapter = None

from .chat_logic import (
    format_clarification,
)
from .formatters import (
    clean_path_segments,
    filter_by_relevance,
    format_search_result_with_explanation,
    get_relevance_label_combined,
    infer_attachment_label,
    normalize_markdown_emphasis,
    normalize_markdown_table,
    normalize_relevance_scores,
    strip_path_prefix,
)
from .link_formatter import extract_and_format_references, format_as_markdown_links
from .query_handler import QueryContext, QueryHandler, QueryOptions, QueryResult

# Default paths
DEFAULT_DB_PATH = "data/chroma_db"
DEFAULT_JSON_PATH = "data/output/규정집-test01.json"
LLM_PROVIDERS = ["ollama", "lmstudio", "mlx", "local", "openai", "gemini", "openrouter"]
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER") or "ollama"
if DEFAULT_LLM_PROVIDER not in LLM_PROVIDERS:
    DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL") or ""
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL") or ""


# Load custom CSS from external file for better maintainability
def _load_custom_css() -> str:
    """Load CSS from external file, with fallback to minimal styles."""
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    # Fallback minimal CSS if file not found
    return """
    .gradio-container { background: #0f0f0f !important; }
    .chatbot { border-radius: 16px !important; }
    """


CUSTOM_CSS = _load_custom_css()


def _format_query_rewrite_debug(info: Optional[QueryRewriteInfo]) -> str:
    if not info:
        return ""

    lines = ["### 🔄 쿼리 분석 결과"]

    if not info.used:
        lines.append("- **상태**: 쿼리 리라이팅 미적용")
        lines.append(f"- **원본 쿼리**: `{info.original}`")
        return "\n".join(lines)

    # 방법 표시
    if info.method == "llm":
        method_label = "🤖 LLM 기반 리라이팅"
    elif info.method == "rules":
        method_label = "📋 규칙 기반 확장 (동의어/인텐트)"
    else:
        method_label = "❓ 알수없음"

    # 추가 상태 표시
    status_tags = []
    if info.from_cache:
        status_tags.append("📦 캐시 히트")
    if info.fallback:
        status_tags.append("⚠️ LLM 실패→폴백")
    status_text = " | ".join(status_tags) if status_tags else ""

    lines.append(f"**방법**: {method_label}")
    if status_text:
        lines.append(f"**상태**: {status_text}")

    # 쿼리 변환 결과
    lines.append("")
    lines.append("#### 쿼리 변환")
    lines.append(f"- **원본**: `{info.original}`")
    if info.original == info.rewritten:
        lines.append("- **결과**: (변경 없음)")
    else:
        lines.append(f"- **변환**: `{info.rewritten}`")

    # 동의어 적용 여부
    lines.append("")
    lines.append("#### 적용된 기법")
    if info.used_synonyms is not None:
        if info.used_synonyms:
            lines.append("- 📚 **동의어 사전**: ✅ 적용됨 (유사어로 검색 범위 확장)")
        else:
            lines.append("- 📚 **동의어 사전**: ➖ 미적용")

    # 인텐트 적용 여부
    if info.used_intent is not None:
        if info.used_intent:
            lines.append("- 🎯 **의도 인식**: ✅ 매칭됨")
            if info.matched_intents:
                intents_str = ", ".join([f"`{i}`" for i in info.matched_intents])
                lines.append(f"  - 매칭된 의도: {intents_str}")
        else:
            lines.append("- 🎯 **의도 인식**: ➖ 미매칭")

    return "\n".join(lines)


def _decide_search_mode_ui(query: str) -> str:
    """Auto-detect search mode without manual selection."""
    from .common import decide_search_mode

    return decide_search_mode(query, None)


def _process_with_handler(
    query: str,
    top_k: int,
    include_abolished: bool,
    llm_provider: str,
    llm_model: str,
    llm_base_url: str,
    target_db_path: str,
    audience_override: Optional[Audience],
    use_tools: bool,
    history: List[dict],
    state: dict,
    use_mock_llm: bool = False,
    default_db_path: str = DEFAULT_DB_PATH,
) -> QueryResult:
    """Process query using QueryHandler."""
    db_path_value = target_db_path or default_db_path
    store_for_query = ChromaVectorStore(persist_directory=db_path_value)

    # Initialize LLM client
    llm_client = None
    if not use_mock_llm:
        try:
            llm_client = LLMClientAdapter(
                provider=llm_provider,
                model=llm_model or None,
                base_url=llm_base_url or None,
            )
        except Exception:
            pass  # Will use search only if LLM fails
    else:
        llm_client = MockLLMClient()

    handler = QueryHandler(
        store=store_for_query,
        llm_client=llm_client,
        function_gemma_client=llm_client if use_tools else None,
        use_reranker=True,  # Default to True for Web UI
    )

    context = QueryContext(
        state=state,
        history=history,
        interactive=True,
        last_regulation=state.get("last_regulation"),
        last_rule_code=state.get("last_rule_code"),
    )

    options = QueryOptions(
        top_k=top_k,
        include_abolished=include_abolished,
        audience_override=audience_override,
        use_function_gemma=use_tools,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
    )

    return handler.process_query(query, context, options)


def create_app(
    db_path: str = DEFAULT_DB_PATH,
    use_mock_llm: bool = False,
) -> "gr.Blocks":
    """
    Create Gradio app instance with ChatGPT-style interface.

    Args:
        db_path: Path to ChromaDB storage.
        use_mock_llm: Use mock LLM for testing without API key.

    Returns:
        Gradio Blocks app.
    """
    if not GRADIO_AVAILABLE:
        raise ImportError("gradio is required. Install with: uv add gradio")

    # Initialize components
    store = ChromaVectorStore(persist_directory=db_path)
    loader = JSONDocumentLoader()

    llm_status = "LLM 사용 가능"
    if use_mock_llm:
        llm_status = "⚠️ Mock LLM (테스트 모드)"

    # Initialize llm_client for evaluation tab (P2)
    llm_client = None
    if use_mock_llm:
        llm_client = MockLLMClient()
    else:
        try:
            llm_client = LLMClientAdapter()
        except Exception:
            pass  # Will be None if initialization fails

    sync_usecase = SyncUseCase(loader, store)

    data_input_dir = Path("data/input")
    data_output_dir = Path("data/output")
    data_input_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    def _find_latest_json(output_dir: Path) -> Optional[Path]:
        json_files = [
            p
            for p in output_dir.rglob("*.json")
            if not p.name.endswith("_metadata.json")
        ]
        if not json_files:
            return None
        return max(json_files, key=lambda p: p.stat().st_mtime)

    def _list_json_files(output_dir: Path) -> List[Path]:
        return sorted(
            [
                p
                for p in output_dir.rglob("*.json")
                if not p.name.endswith("_metadata.json")
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    auto_sync_message = ""
    if store.count() == 0:
        latest_json = _find_latest_json(data_output_dir)
        if latest_json:
            try:
                result = sync_usecase.incremental_sync(str(latest_json))
                auto_sync_message = f"자동 동기화: {latest_json.name} ({result})"
            except Exception as e:
                auto_sync_message = f"자동 동기화 실패: {e}"

    # Get initial status
    def get_status_text() -> str:
        status = sync_usecase.get_sync_status()
        auto_sync_note = f"\n- {auto_sync_message}" if auto_sync_message else ""
        return f"""**동기화 상태**
- 마지막 동기화: {status["last_sync"] or "없음"}
- 규정집 파일: {status["json_file"] or "없음"}
- 인덱싱된 규정: {status["store_regulations"]}개
- 저장된 조항 수: {status["store_chunks"]}개
- LLM: {llm_status}{auto_sync_note}
"""

    # Initialize use cases
    QueryAnalyzer()
    FullViewUseCase(JSONDocumentLoader())

    def _parse_audience(selection: str) -> Optional[Audience]:
        if selection == "교수":
            return Audience.FACULTY
        if selection == "학생":
            return Audience.STUDENT
        if selection == "직원":
            return Audience.STAFF
        return None

    def _format_table_matches(
        matches: List[TableMatch],
        table_no: Optional[int],
        label: Optional[str],
    ) -> str:
        label_text = label or "별표"
        lines = []
        for idx, match in enumerate(matches, 1):
            path = clean_path_segments(match.path) if match.path else []
            heading = " > ".join(path) if path else match.title or label_text
            if table_no:
                table_label = f"{label_text} {table_no}"
            else:
                table_label = infer_attachment_label(match, label_text)
            lines.append(f"### [{idx}] {heading} ({table_label})")
            if match.text:
                lines.append(match.text)
            lines.append(normalize_markdown_table(match.markdown).strip())
        return "\n\n".join([line for line in lines if line])

    def _format_toc(toc: List[str]) -> str:
        if not toc:
            return "목차 정보가 없습니다."
        return "### 목차\n" + "\n".join([f"- {t}" for t in toc])

    def _build_sources_markdown(results, query: str, show_debug: bool) -> str:
        sources_md = ["### 📚 참고 규정\n"]
        norm_scores = normalize_relevance_scores(results) if results else {}
        display_sources = filter_by_relevance(results, norm_scores) if results else []

        for i, r in enumerate(display_sources, 1):
            reg_name = r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
            path = (
                " > ".join(clean_path_segments(r.chunk.parent_path))
                if r.chunk.parent_path
                else r.chunk.title
            )
            norm_score = norm_scores.get(r.chunk.id, 0.0)
            rel_pct = int(norm_score * 100)
            rel_label = get_relevance_label_combined(rel_pct)
            score_info = f" | AI 신뢰도: {r.score:.3f}" if show_debug else ""
            snippet = strip_path_prefix(r.chunk.text, r.chunk.parent_path or [])

            # Format regulation references in snippet as links (visual only for now)
            # We use a dummy link that doesn't go anywhere but looks like a link
            snippet_with_links = format_as_markdown_links(
                snippet,
                extract_and_format_references(snippet, "markdown")[0],
                link_template="#",
            )

            # 매칭 설명 추가
            explanation, _ = format_search_result_with_explanation(
                r, query, show_score=show_debug
            )

            sources_md.append(f"""#### [{i}] {reg_name}
**경로:** {path}
**매칭 정보:** {explanation}

{snippet_with_links[:500]}{"..." if len(snippet_with_links) > 500 else ""}

*규정번호: {r.chunk.rule_code} | 관련도: {rel_pct}% {rel_label}{score_info}*

---
""")

        return "\n".join(sources_md)

    def _run_ask_once(
        question: str,
        top_k: int,
        include_abolished: bool,
        llm_provider: str,
        llm_model: str,
        llm_base_url: str,
        target_db_path: str,
        audience_override: Optional[Audience],
        show_debug: bool,
        history_text: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> Tuple[str, str, str, str, str]:
        db_path_value = target_db_path or db_path
        store_for_ask = ChromaVectorStore(persist_directory=db_path_value)
        if store_for_ask.count() == 0:
            return (
                "데이터베이스가 비어 있습니다. CLI에서 'regulation sync'를 실행하세요.",
                "",
                "",
                "",
                "",
            )

        if use_mock_llm:
            llm_client = MockLLMClient()
        else:
            llm_client = LLMClientAdapter(
                provider=llm_provider,
                model=llm_model or None,
                base_url=llm_base_url or None,
            )

        search_with_llm = SearchUseCase(store_for_ask, llm_client)
        filter = None
        if not include_abolished:
            filter = SearchFilter(status=RegulationStatus.ACTIVE)

        answer = search_with_llm.ask(
            question,
            filter=filter,
            top_k=top_k,
            include_abolished=include_abolished,
            audience_override=audience_override,
            history_text=history_text,
            search_query=search_query,
        )

        answer_text = normalize_markdown_emphasis(answer.text)
        sources_text = _build_sources_markdown(answer.sources, question, show_debug)
        debug_text = ""
        if show_debug:
            debug_text = _format_query_rewrite_debug(
                search_with_llm.get_last_query_rewrite()
            )
        rule_code = answer.sources[0].chunk.rule_code if answer.sources else ""
        top_regulation_title = ""
        if answer.sources:
            top_chunk = answer.sources[0].chunk
            if top_chunk.parent_path:
                top_regulation_title = top_chunk.parent_path[0]
            else:
                top_regulation_title = top_chunk.title
        return answer_text, sources_text, debug_text, rule_code, top_regulation_title

    def _run_ask_stream(
        question: str,
        top_k: int,
        include_abolished: bool,
        llm_provider: str,
        llm_model: str,
        llm_base_url: str,
        target_db_path: str,
        audience_override: Optional[Audience],
        show_debug: bool,
        history_text: Optional[str] = None,
        search_query: Optional[str] = None,
    ):
        """
        Streaming version of _run_ask_once.

        Yields:
            dict: Contains type ('progress', 'token', 'sources', 'debug', 'metadata')
                  and corresponding content.
        """
        db_path_value = target_db_path or db_path
        store_for_ask = ChromaVectorStore(persist_directory=db_path_value)
        if store_for_ask.count() == 0:
            yield {
                "type": "error",
                "content": "데이터베이스가 비어 있습니다. CLI에서 'regulation sync'를 실행하세요.",
            }
            return

        # Progress: Starting search
        yield {"type": "progress", "content": "🔍 1/3 규정 검색 중..."}

        if use_mock_llm:
            llm_client = MockLLMClient()
        else:
            llm_client = LLMClientAdapter(
                provider=llm_provider,
                model=llm_model or None,
                base_url=llm_base_url or None,
            )

        search_with_llm = SearchUseCase(store_for_ask, llm_client)
        filter = None
        if not include_abolished:
            filter = SearchFilter(status=RegulationStatus.ACTIVE)

        # Progress: Reranking
        yield {
            "type": "progress",
            "content": "🔍 1/3 규정 검색 중...\n🎯 2/3 관련도 재정렬 중...",
        }

        sources = []
        rule_code = ""
        regulation_title = ""
        debug_text = ""

        # Use streaming if available
        try:
            for item in search_with_llm.ask_stream(
                question,
                filter=filter,
                top_k=top_k,
                include_abolished=include_abolished,
                audience_override=audience_override,
                history_text=history_text,
                search_query=search_query,
            ):
                if item["type"] == "metadata":
                    sources = item["sources"]
                    # Progress: LLM generating
                    yield {
                        "type": "progress",
                        "content": "🔍 1/3 규정 검색 중...\n🎯 2/3 관련도 재정렬 중...\n🤖 3/3 AI 답변 생성 중...",
                    }

                    if sources:
                        top_chunk = sources[0].chunk
                        rule_code = top_chunk.rule_code
                        regulation_title = (
                            top_chunk.parent_path[0]
                            if top_chunk.parent_path
                            else top_chunk.title
                        )
                elif item["type"] == "token":
                    yield {"type": "token", "content": item["content"]}
        except Exception:
            # Fallback to non-streaming
            answer = search_with_llm.ask(
                question,
                filter=filter,
                top_k=top_k,
                include_abolished=include_abolished,
                audience_override=audience_override,
                history_text=history_text,
                search_query=search_query,
            )
            sources = answer.sources
            if sources:
                top_chunk = sources[0].chunk
                rule_code = top_chunk.rule_code
                regulation_title = (
                    top_chunk.parent_path[0]
                    if top_chunk.parent_path
                    else top_chunk.title
                )
            yield {"type": "token", "content": answer.text}

        # Send sources and debug info at the end
        sources_text = _build_sources_markdown(sources, question, show_debug)
        if show_debug:
            debug_text = _format_query_rewrite_debug(
                search_with_llm.get_last_query_rewrite()
            )

        yield {"type": "sources", "content": sources_text}
        yield {"type": "debug", "content": debug_text}
        yield {
            "type": "metadata",
            "rule_code": rule_code,
            "regulation_title": regulation_title,
        }

    # Main chat function (stateful)
    def chat_respond(
        msg: str,
        history: List[dict],
        state: dict,
        top_k: int,
        abolished: bool,
        llm_p: str,
        llm_m: str,
        llm_b: str,
        db_path_val: str,
        target_sel: str,
        use_context: bool,
        use_tools: bool,
        show_debug: bool,
    ):
        """Handle chat interaction with streaming."""
        if not msg.strip():
            # Show helpful message for empty input
            history.append(
                {
                    "role": "assistant",
                    "content": "💡 검색어를 입력해주세요. 예시: '휴학 신청 절차', '교원 연구년 자격은?'",
                }
            )
            yield history, "", "", state
            return

        # Prepare arguments
        audience_override = (
            _parse_audience(target_sel) if target_sel != "자동" else None
        )

        # Build history context if enabled
        history_context = []
        if use_context:
            history_context = history

        # New logic inline here:
        db_path_value = db_path_val or db_path
        store_for_query = ChromaVectorStore(persist_directory=db_path_value)

        llm_client = None
        if not use_mock_llm:
            try:
                llm_client = LLMClientAdapter(
                    provider=llm_p,
                    model=llm_m or None,
                    base_url=llm_b or None,
                )
            except Exception:
                pass
        else:
            llm_client = MockLLMClient()

        handler = QueryHandler(
            store=store_for_query,
            llm_client=llm_client,
            function_gemma_client=llm_client if use_tools else None,
            use_reranker=True,  # Default true for web
        )

        context = QueryContext(
            state=state,
            history=history_context,
            interactive=True,
            last_regulation=state.get("last_regulation"),
            last_rule_code=state.get("last_rule_code"),
        )

        options = QueryOptions(
            top_k=top_k,
            include_abolished=abolished,
            audience_override=audience_override,
            use_function_gemma=use_tools,
            show_debug=show_debug,
            llm_provider=llm_p,
            llm_model=llm_m,
            llm_base_url=llm_b,
        )

        # Start streaming
        # Add user message
        history.append({"role": "user", "content": msg})
        # Initial assistant message for progress
        history.append({"role": "assistant", "content": "🔍 1/3 규정 검색 중..."})
        yield history, "", "", state

        current_response = ""
        current_debug = ""
        sources_text = ""

        for event in handler.process_query_stream(msg, context, options):
            evt_type = event["type"]

            if evt_type == "progress":
                history[-1] = {"role": "assistant", "content": event["content"]}
                yield history, "", current_debug, state

            elif evt_type == "token":
                current_response += event["content"]
                history[-1]["content"] = current_response
                yield history, "", current_debug, state

            elif evt_type == "sources":
                sources_text = event["content"]

            elif evt_type == "debug":
                current_debug += f"\n{event['content']}"
                yield (
                    history,
                    "",
                    current_debug,
                    state,
                )  # Yield debug updates immediately

            elif evt_type == "metadata":
                if event.get("rule_code"):
                    state["last_rule_code"] = event["rule_code"]
                if event.get("regulation_title"):
                    state["last_regulation"] = event["regulation_title"]

            elif evt_type == "state":
                # explicit state update
                state.update(event["update"])

            elif evt_type == "clarification":
                clarification_type = event["clarification_type"]
                clarification_options = event["options"]

                state["pending"] = {
                    "type": clarification_type,
                    "options": clarification_options,
                    "query": msg,  # Use original message for pending query
                    "mode": event.get(
                        "mode", "search"
                    ),  # Default to search if mode not specified by handler
                    "table_no": event.get("table_no"),
                    "label": event.get("label"),
                }

                clarified_content = format_clarification(
                    clarification_type, clarification_options
                )
                history[-1] = {"role": "assistant", "content": clarified_content}

                yield history, "", current_debug, state
                return  # Stop processing, waiting for user clarification

            elif evt_type == "error":
                history[-1] = {"role": "assistant", "content": f"⚠️ {event['content']}"}
                yield history, "", current_debug, state
                return  # Stop processing on error

            elif evt_type == "complete":
                # Final non-streaming content (e.g. Overview, Search Table) or final LLM answer
                content = event["content"]

                # If it's an LLM answer, sources might be separate.
                # For search results, sources are usually part of the content.
                if (
                    sources_text and "---" not in content[-50:]
                ):  # Avoid duplication if sources already appended
                    content += "\n\n---\n\n" + sources_text

                history[-1] = {
                    "role": "assistant",
                    "content": normalize_markdown_emphasis(content),
                }
                state["last_query"] = msg  # Update last_query with the original message
                # State updates for last_regulation/last_rule_code are handled by 'metadata' event
                # or by the state update from QueryHandler.
                yield history, "", current_debug, state

        # Final yield to ensure everything is settled, especially if no 'complete' event was sent
        # (e.g., if only progress updates were sent and then nothing more)
        # This also ensures the last state is yielded.
        yield history, "", current_debug, state

    def record_web_feedback(query, rule_code, rating, comment):
        """Record feedback from Web UI."""
        if not query or not rule_code:
            return gr.update(value="⚠️ 피드백을 남길 결과가 없습니다.", visible=True)

        from ..infrastructure.feedback import FeedbackCollector

        collector = FeedbackCollector()
        collector.record_feedback(
            query=query,
            rule_code=rule_code,
            rating=rating,
            comment=comment or None,
            source="web",
        )
        return gr.update(value="✅ 피드백이 저장되었습니다. 감사합니다!", visible=True)

    def _render_status(target_db_path: str) -> str:
        db_path_value = target_db_path or db_path
        try:
            store_local = ChromaVectorStore(persist_directory=db_path_value)
        except Exception as e:
            return f"❌ DB 초기화 실패: {e}"

        sync_state_path = Path("data/sync_state.json")
        last_synced = None
        if sync_state_path.exists():
            try:
                import json

                data = json.loads(sync_state_path.read_text(encoding="utf-8"))
                last_synced = data.get("json_file")
            except Exception:
                last_synced = None

        json_files = _list_json_files(data_output_dir)

        lines = []
        lines.append("## DB 상태")
        lines.append(f"- DB 경로: `{db_path_value}`")
        lines.append(f"- 저장된 조항 수: {store_local.count()}")
        lines.append(f"- 규정 수: {len(store_local.get_all_rule_codes())}")
        if last_synced:
            lines.append(f"- **규정집: `{last_synced}`**")

        lines.append("\n## JSON 파일 목록 (`data/output`)")
        if json_files:
            lines.append("| 파일 | 수정 시각 | 크기 | 마지막 동기화 |")
            lines.append("|---|---|---|---|")
            for p in json_files:
                mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime(
                    "%Y-%m-%d %H:%M"
                )
                size_kb = f"{p.stat().st_size / 1024:.1f} KB"
                is_synced = "✅" if last_synced and p.name == last_synced else ""
                lines.append(f"| `{p.name}` | {mtime} | {size_kb} | {is_synced} |")
        else:
            lines.append("- JSON 파일이 없습니다.")

        return "\n".join(lines)

    # Build UI (theme/css are passed to launch() for Gradio 6.0 compatibility)
    with gr.Blocks(
        title="📚 대학 규정집 Q&A",
    ) as app:
        # Header - Minimal & Clean
        gr.HTML("""
            <div style="text-align: center; padding: 28px 20px 20px;">
                <h1 style="font-size: 1.6rem; font-weight: 600; color: #fafafa;
                           letter-spacing: -0.025em; margin: 0;">
                    📚 대학 규정집 Q&A
                </h1>
                <p style="color: #a3a3a3; margin-top: 6px; font-size: 0.9rem; font-weight: 400;">
                    질문하면 AI가 답변하고, 검색어를 입력하면 관련 규정을 찾아드립니다
                </p>
            </div>
        """)

        with gr.Tabs():
            # Tab 1: Chat (Main)
            with gr.TabItem("💬 채팅"):
                with gr.Row():
                    # Main chat area
                    with gr.Column(scale=3):
                        # Navigation Buttons
                        with gr.Row():
                            btn_back = gr.Button("◀ 뒤로", size="sm", interactive=False)
                            btn_forward = gr.Button(
                                "앞으로 ▶", size="sm", interactive=False
                            )
                            # Spacer
                            gr.HTML("<div style='flex-grow: 1;'></div>")

                        chat_bot = gr.Chatbot(
                            label="",
                            height=500,
                            show_label=False,
                            value=[
                                {
                                    "role": "assistant",
                                    "content": "👋 안녕하세요! 대학 규정을 검색하거나 질문할 수 있습니다.\n\n💡 아래 예시 버튼을 클릭하거나 직접 질문을 입력해주세요.",
                                }
                            ],
                            avatar_images=("👤", "🤖"),
                            # bubble_full_width removed for Gradio 6.0 compatibility
                        )

                        # Input area
                        with gr.Row():
                            chat_input = gr.Textbox(
                                placeholder="질문이나 검색어를 입력하세요... (예: 휴학 신청 절차가 어떻게 되나요?)",
                                lines=1,
                                show_label=False,
                                scale=6,
                                container=False,
                            )
                            chat_send = gr.Button(
                                "전송",
                                variant="primary",
                                scale=1,
                                min_width=80,
                            )

                        # Example queries as clickable cards
                        gr.Markdown("### 💡 예시 질문")
                        with gr.Row():
                            ex1 = gr.Button(
                                "🎓 휴학 신청 절차가 어떻게 되나요?", size="sm"
                            )
                            ex2 = gr.Button("📖 교원인사규정 전문", size="sm")
                            ex3 = gr.Button("🔍 교원 연구년", size="sm")
                        with gr.Row():
                            ex4 = gr.Button("📋 학칙 별표 1", size="sm")
                            ex5 = gr.Button("😢 학교 그만두고 싶어요", size="sm")

                        # 대화 초기화 버튼을 예시 버튼과 분리
                        gr.Markdown("---")
                        chat_clear = gr.Button(
                            "🗑️ 대화 초기화", variant="secondary", size="sm"
                        )

                    # Settings sidebar
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 설정")

                        chat_top_k = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1, label="결과 수"
                        )
                        chat_abolished = gr.Checkbox(
                            label="폐지 규정 포함", value=False
                        )
                        chat_target = gr.Radio(
                            choices=["자동", "교수", "학생", "직원"],
                            value="자동",
                            label="대상 선택",
                        )
                        chat_context = gr.Checkbox(label="대화 문맥 활용", value=True)
                        chat_use_tools = gr.Checkbox(
                            label="🛠️ Tool Calling 사용",
                            value=True,
                            info="FunctionGemma를 사용하여 보다 정확한 답변을 제공합니다.",
                        )
                        chat_debug = gr.Checkbox(label="디버그 출력", value=False)

                        with gr.Accordion("🤖 LLM 설정", open=False):
                            chat_llm_p = gr.Dropdown(
                                choices=LLM_PROVIDERS,
                                value=DEFAULT_LLM_PROVIDER,
                                label="Provider",
                            )
                            chat_llm_m = gr.Textbox(
                                value=DEFAULT_LLM_MODEL, label="Model"
                            )
                            chat_llm_b = gr.Textbox(
                                value=DEFAULT_LLM_BASE_URL, label="Base URL"
                            )

                        # Detail panel은 숨김 처리 (채팅창에 직접 표시)
                        chat_detail = gr.Markdown(visible=False)

                        with gr.Accordion("🔧 디버그", open=False):
                            chat_debug_out = gr.Markdown()

                chat_state = gr.State(
                    {
                        "audience": None,
                        "pending": None,
                        "last_query": None,
                        "last_mode": None,
                        "last_regulation": None,
                        "last_rule_code": None,
                        "nav_history": [],  # List of (mode, query, regulation)
                        "nav_index": -1,
                    }
                )

                # Navigation Logic
                def update_nav_buttons(state):
                    history = state.get("nav_history", [])
                    index = state.get("nav_index", -1)
                    has_back = index > 0
                    has_forward = index < len(history) - 1
                    return (
                        gr.update(interactive=has_back),
                        gr.update(interactive=has_forward),
                        state,
                    )

                def confirm_navigation(state, direction):
                    history = state.get("nav_history", [])
                    index = state.get("nav_index", -1)

                    new_index = index + direction
                    if 0 <= new_index < len(history):
                        state["nav_index"] = new_index
                        mode, query, regulation = history[new_index]
                        return query, state
                    return None, state

                db_state = gr.State(db_path)

                # Event handlers
                def on_submit(
                    msg,
                    history,
                    state,
                    top_k,
                    abolished,
                    llm_p,
                    llm_m,
                    llm_b,
                    db,
                    target,
                    context,
                    use_tools,
                    debug,
                ):
                    # Update History for Navigation
                    # Logic: If query changes effectively (new search or view), apend to history
                    # We need to capture the FINAL state to update navigation

                    # Store previous state to detect change
                    prev_query = state.get("last_query")
                    prev_mode = state.get("last_mode")

                    final_state = state
                    for result in chat_respond(
                        msg,
                        history,
                        state,
                        top_k,
                        abolished,
                        llm_p,
                        llm_m,
                        llm_b,
                        db,
                        target,
                        context,
                        use_tools,
                        debug,
                    ):
                        # Unpack result and add empty string for input clear
                        hist, detail, dbg, st = result
                        final_state = st
                        yield hist, detail, dbg, st, ""

                    # After generation, update navigation history if meaningful change
                    curr_query = final_state.get("last_query")
                    curr_mode = final_state.get("last_mode")

                    if curr_query and (
                        curr_query != prev_query or curr_mode != prev_mode
                    ):
                        # Append to history
                        nav_history = final_state.get("nav_history", [])
                        nav_index = final_state.get("nav_index", -1)

                        # If we were back in history, truncate future
                        if nav_index < len(nav_history) - 1:
                            nav_history = nav_history[: nav_index + 1]

                        nav_history.append(
                            (curr_mode, curr_query, final_state.get("last_regulation"))
                        )
                        final_state["nav_history"] = nav_history
                        final_state["nav_index"] = len(nav_history) - 1

                        yield hist, detail, dbg, final_state, ""

                def on_back_click(
                    history,
                    state,
                    top_k,
                    abolished,
                    llm_p,
                    llm_m,
                    llm_b,
                    db,
                    target,
                    context,
                    use_tools,
                    debug,
                ):
                    query, new_state = confirm_navigation(state, -1)
                    if query:
                        # Re-run query
                        for res in on_submit(
                            query,
                            history,
                            new_state,
                            top_k,
                            abolished,
                            llm_p,
                            llm_m,
                            llm_b,
                            db,
                            target,
                            context,
                            use_tools,
                            debug,
                        ):
                            yield res
                    else:
                        yield history, "", "", state, ""

                def on_forward_click(
                    history,
                    state,
                    top_k,
                    abolished,
                    llm_p,
                    llm_m,
                    llm_b,
                    db,
                    target,
                    context,
                    use_tools,
                    debug,
                ):
                    query, new_state = confirm_navigation(state, 1)
                    if query:
                        for res in on_submit(
                            query,
                            history,
                            new_state,
                            top_k,
                            abolished,
                            llm_p,
                            llm_m,
                            llm_b,
                            db,
                            target,
                            context,
                            use_tools,
                            debug,
                        ):
                            yield res
                    else:
                        yield history, "", "", state, ""

                # Redefine on_submit to include button updates
                def on_submit_with_nav(
                    msg,
                    history,
                    state,
                    top_k,
                    abolished,
                    llm_p,
                    llm_m,
                    llm_b,
                    db,
                    target,
                    context,
                    use_tools,
                    debug,
                ):
                    # Wrap the generator
                    gen = on_submit(
                        msg,
                        history,
                        state,
                        top_k,
                        abolished,
                        llm_p,
                        llm_m,
                        llm_b,
                        db,
                        target,
                        context,
                        use_tools,
                        debug,
                    )
                    for res in gen:
                        hist, detail, dbg, st, inp = res
                        # Calc button state
                        nav_history = st.get("nav_history", [])
                        nav_index = st.get("nav_index", -1)
                        has_back = nav_index > 0
                        has_forward = nav_index < len(nav_history) - 1

                        yield (
                            hist,
                            detail,
                            dbg,
                            st,
                            inp,
                            gr.update(interactive=has_back),
                            gr.update(interactive=has_forward),
                        )

                chat_send.click(
                    fn=on_submit_with_nav,
                    inputs=[
                        chat_input,
                        chat_bot,
                        chat_state,
                        chat_top_k,
                        chat_abolished,
                        chat_llm_p,
                        chat_llm_m,
                        chat_llm_b,
                        db_state,
                        chat_target,
                        chat_context,
                        chat_use_tools,
                        chat_debug,
                    ],
                    outputs=[
                        chat_bot,
                        chat_detail,
                        chat_debug_out,
                        chat_state,
                        chat_input,
                        btn_back,
                        btn_forward,
                    ],
                )
                chat_input.submit(
                    fn=on_submit_with_nav,
                    inputs=[
                        chat_input,
                        chat_bot,
                        chat_state,
                        chat_top_k,
                        chat_abolished,
                        chat_llm_p,
                        chat_llm_m,
                        chat_llm_b,
                        db_state,
                        chat_target,
                        chat_context,
                        chat_use_tools,
                        chat_debug,
                    ],
                    outputs=[
                        chat_bot,
                        chat_detail,
                        chat_debug_out,
                        chat_state,
                        chat_input,
                        btn_back,
                        btn_forward,
                    ],
                )

                # Wire up Back/Forward
                btn_back.click(
                    fn=on_back_click,
                    inputs=[
                        chat_bot,
                        chat_state,
                        chat_top_k,
                        chat_abolished,
                        chat_llm_p,
                        chat_llm_m,
                        chat_llm_b,
                        db_state,
                        chat_target,
                        chat_context,
                        chat_use_tools,
                        chat_debug,
                    ],
                    outputs=[
                        chat_bot,
                        chat_detail,
                        chat_debug_out,
                        chat_state,
                        chat_input,
                        btn_back,
                        btn_forward,
                    ],
                )
                btn_forward.click(
                    fn=on_forward_click,
                    inputs=[
                        chat_bot,
                        chat_state,
                        chat_top_k,
                        chat_abolished,
                        chat_llm_p,
                        chat_llm_m,
                        chat_llm_b,
                        db_state,
                        chat_target,
                        chat_context,
                        chat_use_tools,
                        chat_debug,
                    ],
                    outputs=[
                        chat_bot,
                        chat_detail,
                        chat_debug_out,
                        chat_state,
                        chat_input,
                        btn_back,
                        btn_forward,
                    ],
                )

                chat_clear.click(
                    fn=lambda: (
                        [],
                        "",  # chat_detail은 이제 빈 값 (채팅창에 직접 표시)
                        "",
                        {
                            "audience": None,
                            "pending": None,
                            "last_query": None,
                            "last_mode": None,
                            "last_regulation": None,
                            "last_rule_code": None,
                        },
                    ),
                    inputs=[],
                    outputs=[chat_bot, chat_detail, chat_debug_out, chat_state],
                )

                # Example button handlers
                def fill_example(example_text):
                    return example_text

                ex1.click(
                    fn=lambda: "휴학 신청 절차가 어떻게 되나요?", outputs=[chat_input]
                )
                ex2.click(fn=lambda: "교원인사규정 전문", outputs=[chat_input])
                ex3.click(fn=lambda: "교원 연구년", outputs=[chat_input])
                ex4.click(fn=lambda: "학칙 별표 1", outputs=[chat_input])
                ex5.click(fn=lambda: "학교 그만두고 싶어요", outputs=[chat_input])

            # Tab 2: Status
            with gr.TabItem("📂 데이터 현황"):
                gr.Markdown(
                    "> DB 관리(동기화, 초기화)는 CLI에서 수행합니다: `regulation sync`, `regulation reset`"
                )

                status_db_path = gr.Textbox(
                    value=db_path,
                    label="DB 경로",
                    interactive=False,
                )
                status_markdown = gr.Markdown(_render_status(db_path))
                refresh_btn = gr.Button("🔄 새로고침", variant="secondary")

                def _refresh_status_only(target_db_path: str):
                    return _render_status(target_db_path)

                refresh_btn.click(
                    fn=_refresh_status_only,
                    inputs=[status_db_path],
                    outputs=[status_markdown],
                )

            # Tab 3: Live Monitor (Phase 4-5)
            with gr.TabItem("📡 실시간 모니터"):
                _create_live_monitor_tab(db_path)

            # Tab 4: Quality Evaluation (P2)
            with gr.TabItem("📊 품질 평가"):
                _create_evaluation_tab(db_path, llm_client if not use_mock_llm else None)

    return app


def _create_evaluation_tab(db_path: str, llm_client):
    """Create quality evaluation tab with P2 components."""
    gr.Markdown("### 🎯 RAG 시스템 품질 평가")
    gr.Markdown("BatchEvaluationExecutor, ProgressReporter, FailureClassifier를 활용한 종합 평가")

    with gr.Row():
        # Left column: Settings and controls
        with gr.Column(scale=1):
            gr.Markdown("#### ⚙️ 평가 설정")

            eval_personas = gr.Dropdown(
                choices=[
                    "all",
                    "student-undergraduate",
                    "student-graduate",
                    "professor",
                    "staff-admin",
                    "parent",
                    "student-international",
                ],
                value="all",
                label="페르소나 선택",
                multiselect=True,
            )
            eval_queries = gr.Slider(
                minimum=5, maximum=50, value=25, step=5,
                label="페르소나당 쿼리 수"
            )
            eval_batch_size = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="배치 크기"
            )
            eval_threshold = gr.Slider(
                minimum=0.4, maximum=0.8, value=0.6, step=0.05,
                label="실패 임계값"
            )

            gr.Markdown("#### 🎮 실행 제어")

            with gr.Row():
                eval_run_btn = gr.Button("▶ 평가 시작", variant="primary")
                eval_resume_btn = gr.Button("⏵ 재개", variant="secondary")
                eval_stop_btn = gr.Button("⏹ 중지", variant="stop")

            eval_session_id = gr.Textbox(
                label="세션 ID",
                placeholder="재개할 세션 ID 입력",
            )

        # Right column: Progress and results
        with gr.Column(scale=2):
            gr.Markdown("#### 📈 진행 상황")

            eval_progress_bar = gr.Textbox(
                label="진행률",
                value="평가 대기 중...",
                interactive=False,
            )
            eval_eta = gr.Textbox(
                label="예상 완료 시간",
                value="-",
                interactive=False,
            )
            eval_status = gr.Textbox(
                label="상태",
                value="대기 중",
                interactive=False,
            )

    # Results section
    gr.Markdown("---")
    gr.Markdown("#### 📋 평가 결과")

    with gr.Row():
        with gr.Column(scale=1):
            eval_metrics = gr.Dataframe(
                headers=["메트릭", "값", "목표", "상태"],
                datatype=["str", "str", "str", "str"],
                value=[
                    ["Faithfulness", "-", "0.90", "-"],
                    ["Answer Relevancy", "-", "0.85", "-"],
                    ["Contextual Precision", "-", "0.80", "-"],
                    ["Contextual Recall", "-", "0.80", "-"],
                    ["Overall Score", "-", "0.85", "-"],
                ],
                label="메트릭별 점수",
                interactive=False,
            )

        with gr.Column(scale=1):
            eval_summary = gr.Markdown(
                value="평가를 실행하면 결과가 표시됩니다.",
                label="평가 요약",
            )

    # Failure analysis and recommendations
    gr.Markdown("---")
    gr.Markdown("#### 🔍 실패 분석 및 개선 권장사항")

    with gr.Row():
        eval_failures = gr.Dataframe(
            headers=["실패 유형", "건수", "비율"],
            datatype=["str", "str", "str"],
            value=[],
            label="실패 유형 분석",
            interactive=False,
        )
        eval_recommendations = gr.Markdown(
            value="평가 완료 후 개선 권장사항이 표시됩니다.",
            label="개선 권장사항",
        )

    # SPEC Generation
    gr.Markdown("---")
    gr.Markdown("#### 📝 SPEC 문서 생성")

    with gr.Row():
        eval_gen_spec_btn = gr.Button("📄 SPEC 문서 생성", variant="secondary")
        eval_spec_output = gr.Code(
            language="markdown",
            label="생성된 SPEC",
            value="# SPEC 문서\n\n평가 완료 후 생성 버튼을 클릭하세요.",
            lines=10,
        )

    # Event handlers
    def run_evaluation(personas, queries_per_persona, batch_size, threshold, progress=gr.Progress()):
        """Run full evaluation with progress tracking."""
        try:
            from ..application.evaluation import CheckpointManager, ProgressReporter
            from ..domain.evaluation import (
                FailureClassifier,
                PersonaManager,
                RecommendationEngine,
                RAGQualityEvaluator,
            )

            # Initialize components
            persona_mgr = PersonaManager()
            checkpoint_mgr = CheckpointManager(checkpoint_dir="data/checkpoints")
            evaluator = RAGQualityEvaluator(judge_model="gpt-4o", use_ragas=True)

            # Determine personas
            if "all" in personas or not personas:
                target_personas = persona_mgr.list_personas()
            else:
                target_personas = list(personas)

            total_queries = len(target_personas) * queries_per_persona
            persona_counts = {p: queries_per_persona for p in target_personas}
            reporter = ProgressReporter(persona_counts=persona_counts)

            # Create session
            import uuid
            session_id = f"eval-{uuid.uuid4().hex[:8]}"
            checkpoint_mgr.create_session(
                session_id=session_id,
                total_queries=total_queries,
                personas=target_personas,
            )

            # Initialize RAG
            from ..application.search_usecase import SearchUseCase
            from ..infrastructure.chroma_store import ChromaVectorStore

            vector_store = ChromaVectorStore(persist_directory=db_path)
            search_usecase = SearchUseCase(
                store=vector_store,
                llm_client=llm_client,
                use_reranker=True,
            )

            results = []
            completed = 0

            for persona_id in target_personas:
                queries = persona_mgr.generate_queries(persona_id, count=queries_per_persona)

                for query in queries:
                    try:
                        # Search
                        search_results = search_usecase.search(query_text=query, top_k=5)
                        contexts = [r.chunk.text for r in search_results] if search_results else []

                        if not contexts:
                            continue

                        # Generate answer
                        from ..infrastructure.tool_executor import ToolExecutor
                        tool_executor = ToolExecutor(
                            search_usecase=search_usecase,
                            llm_client=llm_client,
                        )
                        answer = tool_executor._handle_generate_answer(
                            {"question": query, "context": "\n\n".join(contexts)}
                        )

                        if not answer:
                            continue

                        # Evaluate
                        result = evaluator.evaluate_single_turn(query, contexts, answer)
                        result.persona = persona_id
                        results.append(result)

                        # Update progress
                        reporter.update(persona=persona_id, query_id=f"q_{completed}", score=result.overall_score)
                        completed += 1
                        progress(completed / total_queries, desc=f"평가 중: {query[:30]}...")

                    except Exception:
                        pass

            # Calculate metrics
            if results:
                avg_faithfulness = sum(r.faithfulness for r in results if hasattr(r, 'faithfulness')) / len(results)
                avg_relevancy = sum(r.answer_relevancy for r in results if hasattr(r, 'answer_relevancy')) / len(results)
                avg_precision = sum(r.contextual_precision for r in results if hasattr(r, 'contextual_precision')) / len(results)
                avg_recall = sum(r.contextual_recall for r in results if hasattr(r, 'contextual_recall')) / len(results)
                avg_overall = sum(r.overall_score for r in results) / len(results)

                metrics_data = [
                    ["Faithfulness", f"{avg_faithfulness:.2f}", "0.90", "✅" if avg_faithfulness >= 0.90 else "❌"],
                    ["Answer Relevancy", f"{avg_relevancy:.2f}", "0.85", "✅" if avg_relevancy >= 0.85 else "❌"],
                    ["Contextual Precision", f"{avg_precision:.2f}", "0.80", "✅" if avg_precision >= 0.80 else "❌"],
                    ["Contextual Recall", f"{avg_recall:.2f}", "0.80", "✅" if avg_recall >= 0.80 else "❌"],
                    ["Overall Score", f"{avg_overall:.2f}", "0.85", "✅" if avg_overall >= 0.85 else "❌"],
                ]

                # Classify failures
                classifier = FailureClassifier()
                failures = classifier.classify_batch(results)

                failures_data = [
                    [f.failure_type.value, str(f.count), f"{f.count/len(results)*100:.1f}%"]
                    for f in failures
                ]

                # Generate recommendations
                engine = RecommendationEngine()
                failure_counts = {f.failure_type: f.count for f in failures}
                recommendations = engine.generate_recommendations(failure_counts, threshold=1)

                rec_text = "### 개선 권장사항\n\n"
                for rec in recommendations[:5]:
                    rec_text += f"**{rec.title}** ({rec.priority.value})\n"
                    rec_text += f"- {rec.description}\n"
                    rec_text += f"- 예상 효과: {rec.impact_estimate}\n\n"

                summary_text = f"""
### 평가 요약

- **세션 ID**: {session_id}
- **평가된 쿼리**: {len(results)}개
- **평균 점수**: {avg_overall:.2f}
- **합격률**: {sum(1 for r in results if r.overall_score >= threshold)/len(results)*100:.1f}%
"""

                return (
                    f"완료: {completed}/{total_queries} (100%)",
                    "완료",
                    f"세션 {session_id} 완료",
                    metrics_data,
                    summary_text,
                    failures_data,
                    rec_text,
                )

            return (
                f"완료: {completed}/{total_queries}",
                "-",
                "평가 완료 (결과 없음)",
                [],
                "평가 결과가 없습니다.",
                [],
                "분석할 데이터가 없습니다.",
            )

        except Exception as e:
            return (
                "오류 발생",
                "-",
                f"오류: {str(e)}",
                [],
                f"오류 발생: {str(e)}",
                [],
                "",
            )

    def generate_spec_from_results():
        """Generate SPEC document from latest failures."""
        try:
            from ..domain.evaluation import (
                FailureClassifier,
                RecommendationEngine,
                SPECGenerator,
            )
            from ..infrastructure.storage.evaluation_store import EvaluationStore

            # Get recent evaluations
            store = EvaluationStore(storage_dir="data/evaluations")
            evaluations = store.get_evaluations(max_score=0.6, limit=50)

            if not evaluations:
                return "# SPEC 문서\n\n분석할 실패 데이터가 없습니다."

            # Classify and generate SPEC
            classifier = FailureClassifier()
            failures = classifier.classify_batch(evaluations)

            engine = RecommendationEngine()
            failure_counts = {f.failure_type: f.count for f in failures}
            recommendations = engine.generate_recommendations(failure_counts, threshold=1)

            generator = SPECGenerator()
            spec = generator.generate_spec(failures=failures, recommendations=recommendations)

            return spec.to_markdown()

        except Exception as e:
            return f"# 오류\n\nSPEC 생성 실패: {str(e)}"

    def resume_evaluation(session_id):
        """Resume interrupted evaluation."""
        from ..application.evaluation import ResumeController, CheckpointManager

        if not session_id:
            return "세션 ID를 입력하세요.", "-", "대기 중", [], "", [], ""

        checkpoint_mgr = CheckpointManager(checkpoint_dir="data/checkpoints")
        resume_ctrl = ResumeController(checkpoint_manager=checkpoint_mgr)

        can_resume, reason = resume_ctrl.can_resume(session_id)
        if not can_resume:
            return f"재개 불가: {reason}", "-", "재개 실패", [], "", [], ""

        context = resume_ctrl.get_resume_context(session_id)
        if not context:
            return "세션을 찾을 수 없습니다.", "-", "재개 실패", [], "", [], ""

        return (
            f"재개: {context.completed_count}/{context.total_count}",
            f"남은 쿼리: {context.total_count - context.completed_count}",
            f"세션 {session_id} 재개 준비됨",
            [],
            f"세션 재개 정보:\n- 완료율: {context.completion_rate:.1f}%\n- 남은 페르소나: {', '.join(context.remaining_personas)}",
            [],
            "재개 후 평가를 실행하세요.",
        )

    # Connect event handlers
    eval_run_btn.click(
        fn=run_evaluation,
        inputs=[eval_personas, eval_queries, eval_batch_size, eval_threshold],
        outputs=[
            eval_progress_bar,
            eval_eta,
            eval_status,
            eval_metrics,
            eval_summary,
            eval_failures,
            eval_recommendations,
        ],
    )

    eval_resume_btn.click(
        fn=resume_evaluation,
        inputs=[eval_session_id],
        outputs=[
            eval_progress_bar,
            eval_eta,
            eval_status,
            eval_metrics,
            eval_summary,
            eval_failures,
            eval_recommendations,
        ],
    )

    eval_gen_spec_btn.click(
        fn=generate_spec_from_results,
        inputs=[],
        outputs=[eval_spec_output],
    )

    eval_stop_btn.click(
        fn=lambda: ("중지됨", "-", "사용자에 의해 중지됨", [], "", [], ""),
        inputs=[],
        outputs=[
            eval_progress_bar,
            eval_eta,
            eval_status,
            eval_metrics,
            eval_summary,
            eval_failures,
            eval_recommendations,
        ],
    )


def _create_live_monitor_tab(db_path: str):
    """Create Live Monitor tab for real-time RAG pipeline monitoring.

    This implements Phase 4-5 of SPEC-RAG-MONITOR-001.
    """
    from .web.live_monitor import LiveMonitor

    gr.Markdown("### 📡 실시간 RAG 파이프라인 모니터링")
    gr.Markdown("RAG 파이프라인의 실행 과정을 실시간으로 확인합니다.")

    # Initialize monitor
    monitor = LiveMonitor()

    # Import EventType for event filtering
    from ..infrastructure.logging.events import EventType

    with gr.Row():
        # Left column: Event timeline
        with gr.Column(scale=2):
            gr.Markdown("#### 📊 이벤트 타임라인")

            with gr.Row():
                # Event type filter
                event_filter = gr.Dropdown(
                    choices=["전체"] + [et.value for et in EventType],
                    value="전체",
                    label="이벤트 유형 필터",
                    scale=2,
                )

                # Refresh button
                refresh_btn = gr.Button("🔄 새로고침", variant="secondary", scale=1)

            # Event display (Dataframe)
            event_display = gr.Dataframe(
                headers=["시간", "유형", "요약"],
                datatype=["str", "str", "str"],
                value=[],
                label="이벤트",
                interactive=False,
                wrap=True,
                # max_rows removed for Gradio 6.0 compatibility
            )

            # Clear events button
            clear_btn = gr.Button("🗑️ 이벤트 지우기", variant="secondary", size="sm")

        # Right column: Query testing
        with gr.Column(scale=1):
            gr.Markdown("#### 🧪 쿼리 테스트")

            query_input = gr.Textbox(
                label="테스트 쿼리",
                placeholder="질문을 입력하세요... (예: 휴학 신청 절차)",
                lines=2,
            )

            query_top_k = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="결과 수 (top_k)",
            )

            submit_btn = gr.Button("▶ 실행", variant="primary")

            # Result display
            result_display = gr.Markdown(
                value="쿼리를 실행하면 결과가 여기에 표시됩니다.",
                label="결과",
            )

            # Event count
            event_count = gr.Textbox(
                label="캡처된 이벤트 수",
                value="0",
                interactive=False,
            )

    # Event handlers
    def refresh_events(filter_type: str):
        """Refresh event display."""
        events = monitor.get_events_for_gradio()

        if filter_type != "전체":
            # Filter by event type
            events = [e for e in events if e[1] == filter_type]

        return events, str(len(events))

    def run_query(query: str, top_k: int):
        """Run test query and return results."""
        if not query.strip():
            return "쿼리를 입력해주세요.", "0"

        result = monitor.submit_query(query, top_k=top_k)

        if not result.get("success"):
            return f"❌ 오류: {result.get('error', 'Unknown error')}", str(result.get("event_count", 0))

        # Format result
        output = f"### 결과\n\n"
        output += f"**쿼리**: {result['query']}\n\n"
        output += f"**응답 유형**: {result.get('result_type', 'unknown')}\n\n"
        output += f"**캡처된 이벤트**: {result.get('event_count', 0)}개\n\n"

        if result.get("result"):
            # Truncate long results
            result_text = result['result']
            if len(result_text) > 500:
                result_text = result_text[:500] + "..."
            output += f"---\n\n{result_text}"

        return output, str(result.get("event_count", 0))

    def clear_events():
        """Clear all events from buffer."""
        monitor.clear_events()
        return [], "0"

    # Wire up event handlers
    refresh_btn.click(
        fn=refresh_events,
        inputs=[event_filter],
        outputs=[event_display, event_count],
    )

    submit_btn.click(
        fn=run_query,
        inputs=[query_input, query_top_k],
        outputs=[result_display, event_count],
    )

    clear_btn.click(
        fn=clear_events,
        inputs=[],
        outputs=[event_display, event_count],
    )


# Alias for backward compatibility with tests
create_demo = create_app


def main():
    """Launch Gradio app."""

    parser = argparse.ArgumentParser(description="규정집 RAG 웹 UI")
    parser.add_argument("--port", type=int, default=7860, help="서버 포트")
    parser.add_argument("--share", action="store_true", help="공개 링크 생성")
    parser.add_argument("--mock-llm", action="store_true", help="Mock LLM 사용")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="DB 경로")

    args = parser.parse_args()

    app = create_app(db_path=args.db_path, use_mock_llm=args.mock_llm)
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


if __name__ == "__main__":
    main()
