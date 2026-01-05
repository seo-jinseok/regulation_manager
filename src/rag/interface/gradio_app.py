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
import shutil
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

from ...main import run_pipeline
from ..application.full_view_usecase import FullViewUseCase, TableMatch
from ..application.search_usecase import QueryRewriteInfo, SearchUseCase
from ..application.sync_usecase import SyncUseCase
from ..domain.entities import RegulationStatus
from ..domain.value_objects import SearchFilter
from ..infrastructure.chroma_store import ChromaVectorStore
from ..infrastructure.hybrid_search import Audience, QueryAnalyzer
from ..infrastructure.json_loader import JSONDocumentLoader
from ..infrastructure.json_loader import JSONDocumentLoader
from ..infrastructure.llm_adapter import LLMClientAdapter
from ..infrastructure.llm_client import MockLLMClient
try:
    from ..infrastructure.function_gemma_adapter import FunctionGemmaAdapter
    FUNCTION_GEMMA_AVAILABLE = True
except ImportError:
    FUNCTION_GEMMA_AVAILABLE = False
    FunctionGemmaAdapter = None

from .chat_logic import (
    attachment_label_variants,
    build_history_context,
    expand_followup_query,
    extract_regulation_title,
    format_clarification,
    has_explicit_target,
    parse_attachment_request,
    resolve_audience_choice,
    resolve_regulation_choice,
)
from .formatters import (
    clean_path_segments,
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
from .link_formatter import extract_and_format_references, format_as_markdown_links
from .query_handler import QueryContext, QueryHandler, QueryOptions, QueryResult, QueryType
from ..infrastructure.patterns import REGULATION_ONLY_PATTERN, RULE_CODE_PATTERN

# Default paths
DEFAULT_DB_PATH = "data/chroma_db"
DEFAULT_JSON_PATH = "data/output/규정집-test01.json"
LLM_PROVIDERS = ["ollama", "lmstudio", "mlx", "local", "openai", "gemini", "openrouter"]
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER") or "ollama"
if DEFAULT_LLM_PROVIDER not in LLM_PROVIDERS:
    DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL") or ""
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL") or ""


# Custom CSS for Modern 2025 UI - Glassmorphism + Minimalism
CUSTOM_CSS = """
/* ============================================
   Modern 2025 UI Theme
   - Glassmorphism with subtle blur
   - Minimalist color palette
   - Smooth micro-animations
   ============================================ */

:root {
    /* Primary palette - elegant emerald green */
    --primary: #10b981;
    --primary-light: #34d399;
    --primary-dark: #059669;
    
    /* Neutral palette - sophisticated grays */
    --bg-base: #0f0f0f;
    --bg-elevated: #1a1a1a;
    --bg-surface: rgba(255, 255, 255, 0.03);
    --bg-glass: rgba(255, 255, 255, 0.05);
    
    /* Text colors */
    --text-primary: #fafafa;
    --text-secondary: #a3a3a3;
    --text-muted: #737373;
    
    /* Accent colors */
    --accent-purple: #a855f7;
    --accent-blue: #3b82f6;
    --accent-amber: #f59e0b;
    
    /* Borders and shadows */
    --border-subtle: rgba(255, 255, 255, 0.06);
    --border-glass: rgba(255, 255, 255, 0.1);
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.4);
    --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.5);
    
    /* Animation timing */
    --ease-smooth: cubic-bezier(0.4, 0, 0.2, 1);
}

/* Global dark theme */
.gradio-container {
    background: var(--bg-base) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.dark {
    --background-fill-primary: var(--bg-base) !important;
}

/* ============================================
   Header - Clean & Minimal
   ============================================ */
.header-container {
    text-align: center;
    padding: 32px 24px 24px;
}

.header-title {
    font-size: 1.75rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.025em;
    margin: 0;
}

.header-subtitle {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-top: 8px;
    font-weight: 400;
}

/* ============================================
   Chat Container - Glassmorphism
   ============================================ */
.chatbot {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
}

/* Message bubbles - modern rounded */
.message {
    border-radius: 16px !important;
    padding: 14px 18px !important;
    margin: 6px 0 !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    transition: transform 0.15s var(--ease-smooth) !important;
}

.message.user {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    margin-left: 15% !important;
    box-shadow: var(--shadow-sm) !important;
}

.message.bot {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    color: var(--text-primary) !important;
    backdrop-filter: blur(10px) !important;
}

/* ============================================
   Input Area - Sleek & Modern
   ============================================ */
.input-container textarea {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    padding: 14px 16px !important;
    font-size: 0.95rem !important;
    transition: all 0.2s var(--ease-smooth) !important;
}

.input-container textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15) !important;
    outline: none !important;
}

.input-container textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ============================================
   Buttons - Minimal with Hover Effects
   ============================================ */
button.primary {
    background: var(--primary) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    transition: all 0.2s var(--ease-smooth) !important;
}

button.primary:hover {
    background: var(--primary-light) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
}

button.secondary {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 10px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    transition: all 0.2s var(--ease-smooth) !important;
}

button.secondary:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    color: var(--text-primary) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
}

/* ============================================
   Example Cards - Glass Effect
   ============================================ */
.example-btn {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 10px !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    padding: 10px 14px !important;
    transition: all 0.2s var(--ease-smooth) !important;
    backdrop-filter: blur(8px) !important;
}

.example-btn:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    color: var(--text-primary) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-md) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
}

/* ============================================
   Sidebar Settings - Clean Layout
   ============================================ */
.settings-panel {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    padding: 20px !important;
}

.settings-panel label {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* Accordion - Subtle styling */
.accordion {
    background: transparent !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
}

.accordion .label-wrap {
    background: var(--bg-surface) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}

/* Slider - Modern track */
input[type="range"] {
    accent-color: var(--primary) !important;
}

/* Checkbox - Subtle */
input[type="checkbox"] {
    accent-color: var(--primary) !important;
    border-radius: 4px !important;
}

/* Radio buttons */
.radio-group label {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    transition: all 0.15s var(--ease-smooth) !important;
}

.radio-group label:hover {
    background: var(--bg-glass) !important;
}

.radio-group label.selected {
    border-color: var(--primary) !important;
    background: rgba(16, 185, 129, 0.1) !important;
}

/* ============================================
   Tabs - Minimal underline style
   ============================================ */
.tabs {
    border-bottom: 1px solid var(--border-subtle) !important;
}

.tabitem {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: none !important;
    padding: 12px 20px !important;
    font-weight: 500 !important;
    transition: color 0.2s var(--ease-smooth) !important;
}

.tabitem:hover {
    color: var(--text-primary) !important;
}

.tabitem.selected {
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary) !important;
}

/* ============================================
   Dropdown & Textbox - Consistent styling
   ============================================ */
select, input[type="text"], .textbox input {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    padding: 10px 12px !important;
}

select:focus, input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.15) !important;
}

/* ============================================
   Markdown Content - Readable typography
   ============================================ */
.markdown-text {
    color: var(--text-primary) !important;
    line-height: 1.7 !important;
}

.markdown-text h1, .markdown-text h2, .markdown-text h3 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: -0.025em !important;
}

.markdown-text code {
    background: var(--bg-glass) !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-size: 0.9em !important;
}

.markdown-text pre {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
}

.markdown-text a {
    color: var(--primary-light) !important;
    text-decoration: none !important;
}

.markdown-text a:hover {
    text-decoration: underline !important;
}

/* ============================================
   Status indicators
   ============================================ */
.status-success {
    color: var(--primary) !important;
}

.status-warning {
    color: var(--accent-amber) !important;
}

.status-info {
    color: var(--accent-blue) !important;
}

/* ============================================
   Animations - Subtle & Smooth
   ============================================ */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.animate-in {
    animation: fadeIn 0.3s var(--ease-smooth) forwards;
}

/* Scrollbar - Minimal */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.2);
}
"""


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
    
    # Initialize FunctionGemmaAdapter if tools enabled
    function_gemma_adapter = None
    if use_tools and FUNCTION_GEMMA_AVAILABLE:
        try:
            # Use same llm_client for tool execution (if openai mode)
            function_gemma_adapter = FunctionGemmaAdapter(
                llm_client=llm_client,
                api_mode="auto"
            )
        except Exception:
            pass

    handler = QueryHandler(
        store=store_for_query,
        llm_client=llm_client,
        function_gemma_adapter=function_gemma_adapter,
        use_reranker=True, # Default to True for Web UI
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
    query_analyzer = QueryAnalyzer()
    full_view_usecase = FullViewUseCase(JSONDocumentLoader())



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

    def _build_sources_markdown(results, show_debug: bool) -> str:
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
                link_template="#"
            )

            sources_md.append(f"""#### [{i}] {reg_name}
**경로:** {path}

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
        sources_text = _build_sources_markdown(answer.sources, show_debug)
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
                "content": "데이터베이스가 비어 있습니다. CLI에서 'regulation sync'를 실행하세요."
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
        yield {"type": "progress", "content": "🔍 1/3 규정 검색 중...\n🎯 2/3 관련도 재정렬 중..."}

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
                    yield {"type": "progress", "content": "🔍 1/3 규정 검색 중...\n🎯 2/3 관련도 재정렬 중...\n🤖 3/3 AI 답변 생성 중..."}
                    
                    if sources:
                        top_chunk = sources[0].chunk
                        rule_code = top_chunk.rule_code
                        regulation_title = top_chunk.parent_path[0] if top_chunk.parent_path else top_chunk.title
                elif item["type"] == "token":
                    yield {"type": "token", "content": item["content"]}
        except Exception as e:
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
                regulation_title = top_chunk.parent_path[0] if top_chunk.parent_path else top_chunk.title
            yield {"type": "token", "content": answer.text}

        # Send sources and debug info at the end
        sources_text = _build_sources_markdown(sources, show_debug)
        if show_debug:
            debug_text = _format_query_rewrite_debug(
                search_with_llm.get_last_query_rewrite()
            )

        yield {"type": "sources", "content": sources_text}
        yield {"type": "debug", "content": debug_text}
        yield {"type": "metadata", "rule_code": rule_code, "regulation_title": regulation_title}

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
            history.append({
                "role": "assistant",
                "content": "💡 검색어를 입력해주세요. 예시: '휴학 신청 절차', '교원 연구년 자격은?'"
            })
            yield history, "", "", state
            return

        # Prepare arguments
        audience_override = _parse_audience(target_sel) if target_sel != "자동" else None
        
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

        function_gemma_adapter = None
        if use_tools and FUNCTION_GEMMA_AVAILABLE:
            try:
                function_gemma_adapter = FunctionGemmaAdapter(
                    llm_client=llm_client,
                    api_mode="auto"
                )
            except Exception:
                pass
        
        handler = QueryHandler(
            store=store_for_query,
            llm_client=llm_client,
            function_gemma_adapter=function_gemma_adapter,
            use_reranker=True, # Default true for web
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
                yield history, "", current_debug, state # Yield debug updates immediately
                
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
                    "query": msg, # Use original message for pending query
                    "mode": event.get("mode", "search"), # Default to search if mode not specified by handler
                    "table_no": event.get("table_no"),
                    "label": event.get("label"),
                }
                
                clarified_content = format_clarification(clarification_type, clarification_options)
                history[-1] = {"role": "assistant", "content": clarified_content}
                
                yield history, "", current_debug, state
                return # Stop processing, waiting for user clarification
            
            elif evt_type == "error":
                history[-1] = {"role": "assistant", "content": f"⚠️ {event['content']}"}
                yield history, "", current_debug, state
                return # Stop processing on error

            elif evt_type == "complete":
                # Final non-streaming content (e.g. Overview, Search Table) or final LLM answer
                content = event["content"]
                
                # If it's an LLM answer, sources might be separate.
                # For search results, sources are usually part of the content.
                if sources_text and "---" not in content[-50:]: # Avoid duplication if sources already appended
                     content += "\n\n---\n\n" + sources_text

                history[-1] = {"role": "assistant", "content": normalize_markdown_emphasis(content)}
                state["last_query"] = msg # Update last_query with the original message
                # State updates for last_regulation/last_rule_code are handled by 'metadata' event
                # or by the state update from QueryHandler.
                yield history, "", current_debug, state

        # Final yield to ensure everything is settled, especially if no 'complete' event was sent
        # (e.g., if only progress updates were sent and then nothing more)
        # This also ensures the last state is yielded.
        yield history, "", current_debug, state

    # Main chat function (stateful)
    def chat_respond_old(
        message: str,
        history: List[dict],
        state: dict,
        top_k: int,
        include_abolished: bool,
        llm_provider: str,
        llm_model: str,
        llm_base_url: str,
        target_db_path: str,
        target_audience: str,
        use_context: bool,
        show_debug: bool,
    ):
        history = history or []
        state = state or {}
        state.setdefault("audience", None)
        state.setdefault("pending", None)
        state.setdefault("last_query", None)
        state.setdefault("last_mode", None)
        state.setdefault("last_regulation", None)
        state.setdefault("last_rule_code", None)
        details = ""
        debug_text = ""

        history = history or []
        if not message or not message.strip():
            # Show helpful message for empty input
            history.append({
                "role": "assistant",
                "content": "💡 검색어를 입력해주세요. 예시: '휴학 신청 절차', '교원 연구년 자격은?'"
            })
            yield history, details, debug_text, state
            return # Added return here to match new chat_respond behavior
        if history and isinstance(history[0], (list, tuple)):
            normalized = []
            for user_text, assistant_text in history:
                normalized.append({"role": "user", "content": user_text})
                normalized.append({"role": "assistant", "content": assistant_text})
            history = normalized
        history_context = build_history_context(history)
        explicit_target = has_explicit_target(message)
        explicit_regulation = extract_regulation_title(message)

        history.append({"role": "user", "content": message})

        audience_override = _parse_audience(target_audience)
        explicit_audience = resolve_audience_choice(message)
        if audience_override:
            state["audience"] = target_audience
        elif explicit_audience:
            state["audience"] = explicit_audience

        pending = state.get("pending")
        attachment_query = None
        attachment_no = None
        attachment_label = None
        attachment_requested = False
        if pending:
            if pending["type"] == "audience":
                choice = resolve_audience_choice(message) or state.get("audience")
                if not choice:
                    response = format_clarification("audience", pending["options"])
                    history.append({"role": "assistant", "content": response})
                    yield history, details, debug_text, state
                    return # Added return
                state["audience"] = choice
                state["pending"] = None
                query = pending["query"]
                mode = pending["mode"]
            elif pending["type"] == "regulation":
                choice = resolve_regulation_choice(message, pending["options"])
                if not choice:
                    response = format_clarification("regulation", pending["options"])
                    history.append({"role": "assistant", "content": response})
                    yield history, details, debug_text, state
                    return # Added return
                state["pending"] = None
                query = choice
                mode = "full_view"
            elif pending["type"] == "regulation_table":
                choice = resolve_regulation_choice(message, pending["options"])
                if not choice:
                    response = format_clarification("regulation", pending["options"])
                    history.append({"role": "assistant", "content": response})
                    yield history, details, debug_text, state
                    return # Added return
                state["pending"] = None
                attachment_query = choice
                attachment_no = pending.get("table_no")
                attachment_label = pending.get("label")
                attachment_requested = True
                query = choice
                mode = "attachment"
            else:
                state["pending"] = None
                query = message
                mode = _decide_search_mode_ui(message)
        else:
            context_hint = None
            if use_context:
                context_hint = state.get("last_regulation") or state.get("last_query")
            query = expand_followup_query(message, context_hint)
            mode = _decide_search_mode_ui(query)
            attachment_request = parse_attachment_request(
                query,
                state.get("last_regulation") if use_context else None,
            )
            if attachment_request:
                attachment_query, attachment_no, attachment_label = attachment_request
                attachment_requested = True
                query = attachment_query
                mode = "attachment"



            mode = None # Let QueryHandler decide

        # Initialize QueryHandler
        db_path_value = target_db_path or "data/chroma_db"
        store = ChromaVectorStore(persist_directory=db_path_value)
        
        llm_client = None
        if use_mock_llm: # Captured from create_app scope
            llm_client = MockLLMClient()
        else:
            try:
                llm_client = LLMClientAdapter(
                    provider=llm_provider,
                    model=llm_model or None,
                    base_url=llm_base_url or None,
                )
            except Exception:
                pass # Handler handles None client for search, but for Ask it triggers error

        handler = QueryHandler(
            store=store,
            llm_client=llm_client,
            use_reranker=True, # Default true for web
        )

        options = QueryOptions(
            top_k=top_k,
            include_abolished=include_abolished,
            audience_override=_parse_audience(state.get("audience")),
            force_mode=mode if pending else None, # Only force if mode was determined by pending resolution
            show_debug=show_debug,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
        )

        context = QueryContext(
            state=state,
            history=history,
            interactive=True,
            last_regulation=state.get("last_regulation"),
            last_rule_code=state.get("last_rule_code"),
        )

        # Initialize assistant message for streaming
        history.append({
            "role": "assistant",
            "content": "🔍 1/3 규정 검색 중..."
        })

        # Processing loop
        answer_text = ""
        sources_text = ""
        debug_text = ""
        
        for event in handler.process_query_stream(query, context, options):
            evt_type = event["type"]
            
            if evt_type == "progress":
                history[-1] = {"role": "assistant", "content": event["content"]}
                yield history, details, debug_text, state
            
            elif evt_type == "token":
                # Accumulate text for streaming
                if not answer_text:
                    history[-1] = {"role": "assistant", "content": ""}
                answer_text += event["content"]
                history[-1] = {"role": "assistant", "content": answer_text}
                yield history, details, debug_text, state
                
            elif evt_type == "sources":
                sources_text = event["content"]
            
            elif evt_type == "debug":
                debug_text = event["content"]
                
            elif evt_type == "metadata":
                if event.get("rule_code"):
                    state["last_rule_code"] = event["rule_code"]
                if event.get("regulation_title"):
                    state["last_regulation"] = event["regulation_title"]
            
            elif evt_type == "state":
                # explicit state update
                state.update(event["update"])
                
            elif evt_type == "clarification":
                # QueryHandler returns clarification_type, options, and a generic content message.
                # We need to format it with buttons for the UI.
                clarification_type = event["clarification_type"]
                clarification_options = event["options"]

                state["pending"] = {
                    "type": clarification_type,
                    "options": clarification_options,
                    "query": query,
                    "mode": event.get("mode", "search"), # Default to search if mode not specified by handler
                    "table_no": event.get("table_no"),
                    "label": event.get("label"),
                }
                
                clarified_content = format_clarification(clarification_type, clarification_options)
                history[-1] = {"role": "assistant", "content": clarified_content}
                
                yield history, details, debug_text, state
                return # Stop processing, waiting for user clarification
            
            elif evt_type == "error":
                history[-1] = {"role": "assistant", "content": f"⚠️ {event['content']}"}
                yield history, details, debug_text, state
                return # Stop processing on error

            elif evt_type == "complete":
                # Final non-streaming content (e.g. Overview, Search Table) or final LLM answer
                content = event["content"]
                
                # If it's an LLM answer, sources might be separate.
                # For search results, sources are usually part of the content.
                if sources_text and "---" not in content[-50:]: # Avoid duplication if sources already appended
                     content += "\n\n---\n\n" + sources_text

                history[-1] = {"role": "assistant", "content": normalize_markdown_emphasis(content)}
                state["last_query"] = query
                # State updates for last_regulation/last_rule_code are handled by 'metadata' event
                # or by the state update from QueryHandler.
                yield history, details, debug_text, state

        # Final yield to ensure everything is settled, especially if no 'complete' event was sent
        # (e.g., if only progress updates were sent and then nothing more)
        # This also ensures the last state is yielded.
        yield history, details, debug_text, state

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
                            btn_forward = gr.Button("앞으로 ▶", size="sm", interactive=False)
                            # Spacer
                            gr.HTML("<div style='flex-grow: 1;'></div>")

                        chat_bot = gr.Chatbot(
                            label="",
                            height=500,
                            show_label=False,
                            value=[{"role": "assistant", "content": "👋 안녕하세요! 대학 규정을 검색하거나 질문할 수 있습니다.\n\n💡 아래 예시 버튼을 클릭하거나 직접 질문을 입력해주세요."}],
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
                            ex1 = gr.Button("🎓 휴학 신청 절차가 어떻게 되나요?", size="sm")
                            ex2 = gr.Button("📖 교원인사규정 전문", size="sm")
                            ex3 = gr.Button("🔍 교원 연구년", size="sm")
                        with gr.Row():
                            ex4 = gr.Button("📋 학칙 별표 1", size="sm")
                            ex5 = gr.Button("😢 학교 그만두고 싶어요", size="sm")
                        
                        # 대화 초기화 버튼을 예시 버튼과 분리
                        gr.Markdown("---")
                        chat_clear = gr.Button("🗑️ 대화 초기화", variant="secondary", size="sm")

                    # Settings sidebar
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 설정")
                        
                        chat_top_k = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1,
                            label="결과 수"
                        )
                        chat_abolished = gr.Checkbox(
                            label="폐지 규정 포함", value=False
                        )
                        chat_target = gr.Radio(
                            choices=["자동", "교수", "학생", "직원"],
                            value="자동",
                            label="대상 선택",
                        )
                        chat_context = gr.Checkbox(
                            label="대화 문맥 활용", value=True
                        )
                        chat_use_tools = gr.Checkbox(
                             label="🛠️ Tool Calling 사용", value=True,
                             info="FunctionGemma를 사용하여 보다 정확한 답변을 제공합니다."
                        )
                        chat_debug = gr.Checkbox(
                            label="디버그 출력", value=False
                        )
                        
                        with gr.Accordion("🤖 LLM 설정", open=False):
                            chat_llm_p = gr.Dropdown(
                                choices=LLM_PROVIDERS,
                                value=DEFAULT_LLM_PROVIDER,
                                label="Provider",
                            )
                            chat_llm_m = gr.Textbox(
                                value=DEFAULT_LLM_MODEL,
                                label="Model"
                            )
                            chat_llm_b = gr.Textbox(
                                value=DEFAULT_LLM_BASE_URL,
                                label="Base URL"
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
                        state
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

                # Event handlers
                def on_submit(msg, history, state, top_k, abolished, llm_p, llm_m, llm_b, db, target, context, use_tools, debug):
                    # Update History for Navigation
                    # Logic: If query changes effectively (new search or view), apend to history
                    # This is tricky because chat_respond is a generator. 
                    # We might need to handle history update inside chat_respond or wrapper.
                    # For now, let's rely on chat_respond updating state and we intercept it?
                    # Actually, chat_respond updates 'last_query' etc.
                    
                    # Store previous state to detect change
                    prev_query = state.get("last_query")
                    prev_mode = state.get("last_mode")
                    
                    # chat_respond is now a generator for streaming
                    # We need to capture the FINAL state to update navigation
                    final_state = state
                    for result in chat_respond(msg, history, state, top_k, abolished, llm_p, llm_m, llm_b, db, target, context, use_tools, debug):
                        # Unpack result and add empty string for input clear
                        hist, detail, dbg, st = result
                        final_state = st
                        yield hist, detail, dbg, st, ""
                    
                    # After generation, update navigation history if meaningful change
                    curr_query = final_state.get("last_query")
                    curr_mode = final_state.get("last_mode")
                    
                    if curr_query and (curr_query != prev_query or curr_mode != prev_mode):
                        # Append to history
                        nav_history = final_state.get("nav_history", [])
                        nav_index = final_state.get("nav_index", -1)
                        
                        # If we were back in history, truncate future
                        if nav_index < len(nav_history) - 1:
                            nav_history = nav_history[:nav_index + 1]
                        
                        nav_history.append((curr_mode, curr_query, final_state.get("last_regulation")))
                        final_state["nav_history"] = nav_history
                        final_state["nav_index"] = len(nav_history) - 1
                        
                        yield hist, detail, dbg, final_state, ""

                def on_back_click(history, state, top_k, abolished, llm_p, llm_m, llm_b, db, target, context, use_tools, debug):
                     query, new_state = confirm_navigation(state, -1)
                     if query:
                         # Re-run query
                         # Note: This will generate a new chat message. 
                         # Ideally, we should just show the old view, but chat interface is linear.
                         # Rerunning is acceptable for "Navigation" in a chat context (like browser history reloads).
                         for res in on_submit(query, history, new_state, top_k, abolished, llm_p, llm_m, llm_b, db, target, context, use_tools, debug):
                             yield res
                     else:
                         yield history, "", "", state, ""
                
                def on_forward_click(history, state, top_k, abolished, llm_p, llm_m, llm_b, db, target, context, use_tools, debug):
                     query, new_state = confirm_navigation(state, 1)
                     if query:
                          for res in on_submit(query, history, new_state, top_k, abolished, llm_p, llm_m, llm_b, db, target, context, use_tools, debug):
                             yield res
                     else:
                         yield history, "", "", state, ""

                # We need to wire up the buttons to also update their valid state (interactive or not)
                # But Gradio updates are sent with yields.
                # We can add a separate output for buttons to on_submit
                
                # Simplified approach: Just update buttons on every interaction end?
                # Using a separate event listener for button updates is hard.
                # Let's piggyback on on_submit to return button updates?
                # outputs=[chat_bot, chat_detail, chat_debug_out, chat_state, chat_input, btn_back, btn_forward]
                
                # Redefine on_submit to include button updates
                def on_submit_with_nav(msg, history, state, top_k, abolished, llm_p, llm_m, llm_b, db, target, context, use_tools, debug):
                     # ... (logic from on_submit) ...
                     # Wrap the generator
                    gen = on_submit(msg, history, state, top_k, abolished, llm_p, llm_m, llm_b, db, target, context, use_tools, debug)
                    for res in gen:
                        hist, detail, dbg, st, inp = res
                        # Calc button state
                        nav_history = st.get("nav_history", [])
                        nav_index = st.get("nav_index", -1)
                        has_back = nav_index > 0
                        has_forward = nav_index < len(nav_history) - 1
                        
                        yield hist, detail, dbg, st, inp, gr.update(interactive=has_back), gr.update(interactive=has_forward)

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
                        gr.State(db_path),
                        chat_target,
                        chat_context,
                        chat_use_tools,
                        chat_debug,
                    ],
                    outputs=[chat_bot, chat_detail, chat_debug_out, chat_state, chat_input, btn_back, btn_forward],
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
                        gr.State(db_path),
                        chat_target,
                        chat_context,
                        chat_use_tools,
                        chat_debug,
                    ],
                    outputs=[chat_bot, chat_detail, chat_debug_out, chat_state, chat_input, btn_back, btn_forward],
                )
                
                # Wire up Back/Forward
                btn_back.click(
                    fn=on_back_click,
                    inputs=[
                        chat_bot, chat_state, chat_top_k, chat_abolished, chat_llm_p, chat_llm_m, chat_llm_b, gr.State(db_path), chat_target, chat_context, chat_use_tools, chat_debug
                    ],
                    outputs=[chat_bot, chat_detail, chat_debug_out, chat_state, chat_input, btn_back, btn_forward]
                )
                btn_forward.click(
                    fn=on_forward_click,
                    inputs=[
                        chat_bot, chat_state, chat_top_k, chat_abolished, chat_llm_p, chat_llm_m, chat_llm_b, gr.State(db_path), chat_target, chat_context, chat_use_tools, chat_debug
                    ],
                    outputs=[chat_bot, chat_detail, chat_debug_out, chat_state, chat_input, btn_back, btn_forward]
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

                ex1.click(fn=lambda: "휴학 신청 절차가 어떻게 되나요?", outputs=[chat_input])
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

    return app


def main():
    """Launch Gradio app."""
    import argparse

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
