"""
Gradio Web UI for Regulation RAG System.

Provides a user-friendly web interface for:
- Searching regulations
- Asking questions with LLM-generated answers
- Viewing sync status

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

from ..infrastructure.chroma_store import ChromaVectorStore
from ..infrastructure.json_loader import JSONDocumentLoader
from ...main import run_pipeline
from ..infrastructure.llm_adapter import LLMClientAdapter
from ..infrastructure.llm_client import MockLLMClient
from ..infrastructure.hybrid_search import QueryAnalyzer, Audience
from ..application.sync_usecase import SyncUseCase
from ..application.search_usecase import QueryRewriteInfo, SearchUseCase
from ..application.full_view_usecase import FullViewUseCase, TableMatch
from ..domain.value_objects import SearchFilter
from ..domain.entities import RegulationStatus
from .chat_logic import (
    attachment_label_variants,
    expand_followup_query,
    format_clarification,
    parse_attachment_request,
    resolve_audience_choice,
    resolve_regulation_choice,
)
from .formatters import (
    normalize_relevance_scores,
    filter_by_relevance,
    get_relevance_label_combined,
    get_confidence_info,
    clean_path_segments,
    render_full_view_nodes,
    normalize_markdown_table,
    normalize_markdown_emphasis,
    strip_path_prefix,
)


# Default paths
DEFAULT_DB_PATH = "data/chroma_db"
DEFAULT_JSON_PATH = "data/output/규정집-test01.json"
LLM_PROVIDERS = ["ollama", "lmstudio", "mlx", "local", "openai", "gemini", "openrouter"]
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER") or "ollama"
if DEFAULT_LLM_PROVIDER not in LLM_PROVIDERS:
    DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL") or ""
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL") or ""


def _format_query_rewrite_debug(info: Optional[QueryRewriteInfo]) -> str:
    if not info:
        return ""

    lines = ["### 🔄 쿼리 분석 결과"]

    if not info.used:
        lines.append(f"- **상태**: 쿼리 리라이팅 미적용")
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





def _decide_search_mode_ui(query: str, mode_selection: str) -> str:
    """Wrapper for shared decide_search_mode in Gradio."""
    from .common import decide_search_mode
    
    force_mode = None
    if mode_selection == "검색 (Search)":
        force_mode = "search"
    elif mode_selection == "질문 (Ask)":
        force_mode = "ask"
    elif mode_selection == "전문 (Full View)":
        force_mode = "full_view"
        
    return decide_search_mode(query, force_mode)


def create_app(
    db_path: str = DEFAULT_DB_PATH,
    use_mock_llm: bool = False,
) -> "gr.Blocks":
    """
    Create Gradio app instance.

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

    llm_status = "ℹ️ 질문 탭에서 LLM 설정을 선택하세요."
    if use_mock_llm:
        llm_status = "⚠️ Mock LLM (테스트 모드)"

    search_usecase = SearchUseCase(store)  # use_reranker는 config 기본값 사용
    sync_usecase = SyncUseCase(loader, store)

    data_input_dir = Path("data/input")
    data_output_dir = Path("data/output")
    data_input_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    def _find_latest_json(output_dir: Path) -> Optional[Path]:
        json_files = [
            p for p in output_dir.rglob("*.json")
            if not p.name.endswith("_metadata.json")
        ]
        if not json_files:
            return None
        return max(json_files, key=lambda p: p.stat().st_mtime)

    def _list_json_files(output_dir: Path) -> List[Path]:
        return sorted(
            [
                p for p in output_dir.rglob("*.json")
                if not p.name.endswith("_metadata.json")
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def _list_hwp_files(input_dir: Path) -> List[Path]:
        return sorted(input_dir.rglob("*.hwp"))

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
        hwp_files = _list_hwp_files(data_input_dir)
        json_by_stem = {p.stem: p for p in json_files}

        lines = []
        lines.append("## DB 상태")
        lines.append(f"- DB 경로: `{db_path_value}`")
        lines.append(f"- 청크 수: {store_local.count()}")
        lines.append(f"- 규정 수: {len(store_local.get_all_rule_codes())}")
        if last_synced:
            lines.append(f"- **규정집: `{last_synced}`**")

        lines.append("\n## JSON 파일 목록 (`data/output`)")
        if json_files:
            lines.append("| 파일 | 수정 시각 | 크기 | 마지막 동기화 |")
            lines.append("|---|---|---|---|")
            for p in json_files:
                mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                size_kb = f"{p.stat().st_size / 1024:.1f} KB"
                is_synced = "✅" if last_synced and p.name == last_synced else ""
                lines.append(f"| `{p.name}` | {mtime} | {size_kb} | {is_synced} |")
        else:
            lines.append("- JSON 파일이 없습니다.")

        lines.append("\n## HWP 파일 목록 (`data/input`)")
        if hwp_files:
            lines.append("| 파일 | 변환 여부 | 대응 JSON |")
            lines.append("|---|---|---|")
            for p in hwp_files:
                json_path = json_by_stem.get(p.stem)
                converted = "✅" if json_path else "❌"
                json_name = f"`{json_path.name}`" if json_path else "-"
                lines.append(f"| `{p.name}` | {converted} | {json_name} |")
        else:
            lines.append("- HWP 파일이 없습니다.")

        return "\n".join(lines)

    def _json_choices() -> List[str]:
        return [str(p) for p in _list_json_files(data_output_dir)]

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
- 마지막 동기화: {status['last_sync'] or '없음'}
- JSON 파일: {status['json_file'] or '없음'}
- 인덱싱된 규정: {status['store_regulations']}개
- 청크 수: {status['store_chunks']}개
- LLM: {llm_status}{auto_sync_note}
"""

    def _persist_upload(file_path: str) -> Path:
        input_path = Path(file_path)
        data_input_dir = Path("data/input")
        data_input_dir.mkdir(parents=True, exist_ok=True)
        target_path = data_input_dir / input_path.name
        if input_path.resolve() != target_path.resolve():
            shutil.copy2(input_path, target_path)
    # Unified Search Function
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
            table_label = f"{label_text} {table_no}" if table_no else label_text
            lines.append(f"### [{idx}] {heading} ({table_label})")
            if match.text:
                lines.append(match.text)
            lines.append(normalize_markdown_table(match.markdown).strip())
        return "\n\n".join([line for line in lines if line])

    def _format_toc(toc: List[str]) -> str:
        if not toc:
            return "목차 정보가 없습니다."
        return "### 목차\n" + "\n".join([f"- {t}" for t in toc])

    def _build_search_table(results) -> str:
        table_rows = ["| # | 규정명 | 코드 | 조항 | 점수 |", "|---|------|------|------|------|"]
        for i, r in enumerate(results, 1):
            reg_title = r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
            path_segments = clean_path_segments(r.chunk.parent_path) if r.chunk.parent_path else []
            path = " > ".join(path_segments[-2:]) if path_segments else r.chunk.title
            table_rows.append(f"| {i} | {reg_title} | {r.chunk.rule_code} | {path[:40]} | {r.score:.2f} |")
        return "\n".join(table_rows)

    def _build_sources_markdown(results, show_debug: bool) -> str:
        sources_md = ["### 📚 참고 규정\n"]
        norm_scores = normalize_relevance_scores(results) if results else {}
        display_sources = filter_by_relevance(results, norm_scores) if results else []

        for i, r in enumerate(display_sources, 1):
            reg_name = r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
            path = " > ".join(clean_path_segments(r.chunk.parent_path)) if r.chunk.parent_path else r.chunk.title
            norm_score = norm_scores.get(r.chunk.id, 0.0)
            rel_pct = int(norm_score * 100)
            rel_label = get_relevance_label_combined(rel_pct)
            score_info = f" | AI 신뢰도: {r.score:.3f}" if show_debug else ""
            snippet = strip_path_prefix(r.chunk.text, r.chunk.parent_path or [])

            sources_md.append(f"""#### [{i}] {reg_name}
**경로:** {path}

{snippet[:300]}{'...' if len(snippet) > 300 else ''}

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
    ) -> Tuple[str, str, str, str, str]:
        db_path_value = target_db_path or db_path
        store_for_ask = ChromaVectorStore(persist_directory=db_path_value)
        if store_for_ask.count() == 0:
            return "데이터베이스가 비어 있습니다. CLI에서 'regulation-rag sync'를 실행하세요.", "", "", ""

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

    def unified_search(
        query: str,
        mode_selection: str,
        top_k: int,
        include_abolished: bool,
        llm_provider: str,
        llm_model: str,
        llm_base_url: str,
        target_db_path: str,
        target_audience: str,
        show_debug: bool,
    ):
        """Execute unified search/ask based on mode."""
        if not query.strip():
            yield "내용을 입력해주세요.", "", "", "", ""
            return

        attachment_request = parse_attachment_request(query, None)
        if attachment_request:
            reg_query, table_no, label = attachment_request
            matches = full_view_usecase.find_matches(reg_query)
            if not matches:
                yield "해당 규정을 찾을 수 없습니다.", "", "", query, ""
                return
            if len(matches) > 1:
                options = "\n".join([f"- {m.title}" for m in matches])
                detail = f"다음 규정 중 하나를 선택해주세요:\n{options}"
                yield "규정 후보가 여러 개입니다.", detail, "", query, ""
                return
            match = matches[0]
            label_variants = attachment_label_variants(label)
            tables = full_view_usecase.find_tables(match.rule_code, table_no, label_variants)
            if not tables:
                label_text = label or "별표"
                yield f"{label_text}를 찾을 수 없습니다.", "", "", query, match.rule_code
                return
            label_text = label or "별표"
            title_label = f"{match.title} {label_text}"
            if table_no:
                title_label = f"{match.title} {label_text} {table_no}"
            detail = _format_table_matches(tables, table_no, label_text)
            yield title_label, detail, "", query, match.rule_code
            return

        mode = _decide_search_mode_ui(query, mode_selection)
        audience_override = _parse_audience(target_audience)
        if mode in ("search", "ask") and audience_override is None:
            if query_analyzer.is_audience_ambiguous(query):
                msg = "대상이 모호합니다. 교수/학생/직원 중 하나를 선택해주세요."
                yield msg, "", "", "", ""
                return

        if mode == "full_view":
            table, detail, debug, q, code = full_view_regulations(query, show_debug)
            yield table, detail, debug, q, code
            return

        if mode == "search":
             # Search (Retrieval)
             # Reuse search_regulations logic but yield it as a generator to match interface
             table, detail, debug, q, code = search_regulations(
                 query, top_k, include_abolished, audience_override, show_debug
             )
             yield table, detail, debug, q, code
        else:
            # Ask (LLM)
            # Delegate to ask_question generator
            for result in ask_question(
                query, top_k, include_abolished, 
                llm_provider, llm_model, llm_base_url, 
                target_db_path, audience_override, show_debug
            ):
                yield result

    # Chat Function (stateful)
    def chat_respond(
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
            return history, details, debug_text, state
        if history and isinstance(history[0], (list, tuple)):
            normalized = []
            for user_text, assistant_text in history:
                normalized.append({"role": "user", "content": user_text})
                normalized.append({"role": "assistant", "content": assistant_text})
            history = normalized

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
                    return history, details, debug_text, state
                state["audience"] = choice
                state["pending"] = None
                query = pending["query"]
                mode = pending["mode"]
            elif pending["type"] == "regulation":
                choice = resolve_regulation_choice(message, pending["options"])
                if not choice:
                    response = format_clarification("regulation", pending["options"])
                    history.append({"role": "assistant", "content": response})
                    return history, details, debug_text, state
                state["pending"] = None
                query = choice
                mode = "full_view"
            elif pending["type"] == "regulation_table":
                choice = resolve_regulation_choice(message, pending["options"])
                if not choice:
                    response = format_clarification("regulation", pending["options"])
                    history.append({"role": "assistant", "content": response})
                    return history, details, debug_text, state
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
                mode = _decide_search_mode_ui(message, "자동 (Auto)")
        else:
            context_hint = None
            if use_context:
                context_hint = state.get("last_regulation") or state.get("last_query")
            query = expand_followup_query(message, context_hint)
            mode = _decide_search_mode_ui(query, "자동 (Auto)")
            attachment_request = parse_attachment_request(
                query,
                state.get("last_regulation") if use_context else None,
            )
            if attachment_request:
                attachment_query, attachment_no, attachment_label = attachment_request
                attachment_requested = True
                query = attachment_query
                mode = "attachment"

        analyzer = query_analyzer

        if attachment_requested:
            matches = full_view_usecase.find_matches(attachment_query or query)
            if not matches:
                history.append({"role": "assistant", "content": "해당 규정을 찾을 수 없습니다."})
                return history, details, debug_text, state
            if len(matches) > 1:
                options = [m.title for m in matches]
                state["pending"] = {
                    "type": "regulation_table",
                    "options": options,
                    "query": query,
                    "table_no": attachment_no,
                    "label": attachment_label,
                }
                history.append({"role": "assistant", "content": format_clarification("regulation", options)})
                return history, details, debug_text, state

            match = matches[0]
            label_variants = attachment_label_variants(attachment_label)
            tables = full_view_usecase.find_tables(match.rule_code, attachment_no, label_variants)
            if not tables:
                label_text = attachment_label or "별표"
                history.append({"role": "assistant", "content": f"{label_text}를 찾을 수 없습니다."})
                return history, details, debug_text, state
            label_text = attachment_label or "별표"
            details = _format_table_matches(tables, attachment_no, label_text)
            title_label = f"{match.title} {label_text}"
            if attachment_no:
                title_label = f"{match.title} {label_text} {attachment_no}"
            history.append({"role": "assistant", "content": f"**{title_label}** 내용을 표시합니다."})
            state["last_query"] = query
            state["last_mode"] = "attachment"
            state["last_regulation"] = match.title
            state["last_rule_code"] = match.rule_code
            return history, details, debug_text, state

        if mode == "full_view":
            matches = full_view_usecase.find_matches(query)
            if not matches:
                history.append({"role": "assistant", "content": "해당 규정을 찾을 수 없습니다."})
                return history, details, debug_text, state
            if len(matches) > 1:
                options = [m.title for m in matches]
                state["pending"] = {"type": "regulation", "options": options, "query": query, "mode": mode}
                history.append({"role": "assistant", "content": format_clarification("regulation", options)})
                return history, details, debug_text, state
            view = full_view_usecase.get_full_view(matches[0].rule_code) or full_view_usecase.get_full_view(matches[0].title)
            if not view:
                history.append({"role": "assistant", "content": "규정 전문을 불러오지 못했습니다."})
                return history, details, debug_text, state

            toc_text = _format_toc(view.toc)
            content_text = render_full_view_nodes(view.content)
            addenda_text = render_full_view_nodes(view.addenda)
            details = toc_text + "\n\n### 본문\n\n" + (content_text or "본문이 없습니다.")
            if addenda_text:
                details += "\n\n### 부칙\n\n" + addenda_text
            history.append({"role": "assistant", "content": f"**{view.title}** 전문을 표시합니다."})
            state["last_query"] = query
            state["last_mode"] = "full_view"
            state["last_regulation"] = view.title
            state["last_rule_code"] = view.rule_code
            return history, details, debug_text, state

        if state.get("audience") is None and analyzer.is_audience_ambiguous(query):
            options = ["교수", "학생", "직원"]
            state["pending"] = {"type": "audience", "options": options, "query": query, "mode": mode}
            history.append({"role": "assistant", "content": format_clarification("audience", options)})
            return history, details, debug_text, state

        audience_override = _parse_audience(state.get("audience") or "")

        if mode == "search":
            if store.count() == 0:
                history.append({"role": "assistant", "content": "데이터베이스가 비어 있습니다. CLI에서 'regulation-rag sync'를 실행하세요."})
                return history, details, debug_text, state
            search_with_hybrid = SearchUseCase(store)
            results = search_with_hybrid.search_unique(
                query,
                top_k=top_k,
                include_abolished=include_abolished,
                audience_override=audience_override,
            )
            if not results:
                history.append({"role": "assistant", "content": "검색 결과가 없습니다."})
            else:
                history.append({"role": "assistant", "content": _build_search_table(results)})
                top = results[0]
                full_path = " > ".join(clean_path_segments(top.chunk.parent_path)) if top.chunk.parent_path else top.chunk.title
                top_text = strip_path_prefix(top.chunk.text, top.chunk.parent_path or [])
                details = f"""### 🏆 1위 결과: {top.chunk.rule_code}

**규정명:** {top.chunk.parent_path[0] if top.chunk.parent_path else top.chunk.title}

**경로:** {full_path}

---

{top_text}
"""
                state["last_query"] = query
                state["last_mode"] = "search"
                state["last_regulation"] = top.chunk.parent_path[0] if top.chunk.parent_path else top.chunk.title
                state["last_rule_code"] = top.chunk.rule_code
            if show_debug:
                debug_text = _format_query_rewrite_debug(search_with_hybrid.get_last_query_rewrite())
            return history, details, debug_text, state

        answer_text, sources_text, debug_text, rule_code, regulation_title = _run_ask_once(
            query,
            top_k,
            include_abolished,
            llm_provider,
            llm_model,
            llm_base_url,
            target_db_path,
            audience_override,
            show_debug,
        )
        history.append({"role": "assistant", "content": answer_text})
        details = sources_text
        state["last_query"] = query
        state["last_mode"] = "ask"
        if regulation_title:
            state["last_regulation"] = regulation_title
        if rule_code:
            state["last_rule_code"] = rule_code
        return history, details, debug_text, state


    # Search function
    def search_regulations(
        query: str,
        top_k: int,
        include_abolished: bool,
        audience_override: Optional[Audience],
        show_debug: bool,
    ) -> Tuple[str, str, str]:
        """Execute search and return formatted results."""
        if not query.strip():
            return "검색어를 입력해주세요.", "", ""

        if store.count() == 0:
            return "데이터베이스가 비어 있습니다. CLI에서 'regulation-rag sync'를 실행하세요.", "", ""

        # SearchUseCase가 HybridSearcher를 자동 초기화
        search_with_hybrid = SearchUseCase(store)
        results = search_with_hybrid.search_unique(
            query,
            top_k=top_k,
            include_abolished=include_abolished,
            audience_override=audience_override,
        )

        if not results:
            debug_text = ""
            if show_debug:
                debug_text = _format_query_rewrite_debug(
                    search_with_hybrid.get_last_query_rewrite()
                )
            return "검색 결과가 없습니다.", "", debug_text

        # Format results as markdown table (CLI 수준)
        table_rows = ["| # | 규정명 | 코드 | 조항 | 점수 |", "|---|------|------|------|------|"]
        for i, r in enumerate(results, 1):
            reg_title = r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
            path_segments = clean_path_segments(r.chunk.parent_path) if r.chunk.parent_path else []
            path = " > ".join(path_segments[-2:]) if path_segments else r.chunk.title
            table_rows.append(f"| {i} | {reg_title} | {r.chunk.rule_code} | {path[:40]} | {r.score:.2f} |")

        table = "\n".join(table_rows)

        # Top result detail (CLI 수준)
        top = results[0]
        full_path = " > ".join(clean_path_segments(top.chunk.parent_path)) if top.chunk.parent_path else top.chunk.title
        detail = f"""### 🏆 1위 결과: {top.chunk.rule_code}

**규정명:** {top.chunk.parent_path[0] if top.chunk.parent_path else top.chunk.title}

**경로:** {full_path}

---

{top.chunk.text}
"""

        debug_text = ""
        if show_debug:
            debug_text = _format_query_rewrite_debug(
                search_with_hybrid.get_last_query_rewrite()
            )

        # Return (table, detail, debug, query, rule_code)
        top_rule_code = results[0].chunk.rule_code if results else ""
        return table, detail, debug_text, query, top_rule_code

    def full_view_regulations(
        query: str,
        show_debug: bool,
    ) -> Tuple[str, str, str, str, str]:
        """Render regulation full view for '전문' requests."""
        matches = full_view_usecase.find_matches(query)
        if not matches:
            return "해당 규정을 찾을 수 없습니다.", "", "", query, ""

        if len(matches) > 1:
            options = "\n".join([f"- {m.title}" for m in matches])
            detail = f"다음 규정 중 하나를 선택해주세요:\n{options}"
            return "규정 후보가 여러 개입니다.", detail, "", query, ""

        match = matches[0]
        view = full_view_usecase.get_full_view(match.rule_code) or full_view_usecase.get_full_view(match.title)
        if not view:
            return "규정 전문을 불러오지 못했습니다.", "", "", query, ""

        toc_text = _format_toc(view.toc)
        content_text = render_full_view_nodes(view.content)
        addenda_text = render_full_view_nodes(view.addenda)
        detail = "### 본문\n\n" + (content_text or "본문이 없습니다.")
        if addenda_text:
            detail += "\n\n### 부칙\n\n" + addenda_text
        return toc_text, detail, "", query, view.rule_code

    # Ask function (with LLM) - Generator for streaming progress
    def ask_question(
        question: str,
        top_k: int,
        include_abolished: bool,
        llm_provider: str,
        llm_model: str,
        llm_base_url: str,
        target_db_path: str,
        audience_override: Optional[Audience],
        show_debug: bool,
    ):
        """Ask question and get LLM answer with progress updates."""
        if not question.strip():
            yield "질문을 입력해주세요.", "", "", "", ""
            return

        # Step 1: Initialize
        yield "⏳ 데이터베이스 연결 중...", "", "", "", ""
        
        db_path_value = target_db_path or db_path
        store_for_ask = ChromaVectorStore(persist_directory=db_path_value)

        if store_for_ask.count() == 0:
            yield "데이터베이스가 비어 있습니다. CLI에서 'regulation-rag sync'를 실행하세요.", "", "", "", ""
            return

        # Step 2: Initialize LLM
        yield "⏳ LLM 클라이언트 초기화 중...", "", "", "", ""
        
        if use_mock_llm:
            llm_client = MockLLMClient()
        else:
            try:
                llm_client = LLMClientAdapter(
                    provider=llm_provider,
                    model=llm_model or None,
                    base_url=llm_base_url or None,
                )
            except Exception as e:
                yield f"LLM 초기화 실패: {e}", "", "", "", ""
                return

        # Step 3: Search
        yield "🔍 관련 규정 검색 중...", "", "", "", ""
        
        search_with_llm = SearchUseCase(store_for_ask, llm_client)

        filter = None
        if not include_abolished:
            filter = SearchFilter(status=RegulationStatus.ACTIVE)

        # Step 4: Generate answer
        yield "🤖 AI 답변 생성 중... (10-30초 소요)", "", "", "", ""
        
        answer = search_with_llm.ask(
            question,
            filter=filter,
            top_k=top_k,
            include_abolished=include_abolished,
            audience_override=audience_override,
        )

        answer_text = normalize_markdown_emphasis(answer.text)

        # Format sources using shared formatters
        sources_list = answer.sources
        norm_scores = normalize_relevance_scores(sources_list) if sources_list else {}
        display_sources = filter_by_relevance(sources_list, norm_scores) if sources_list else []

        sources_md = ["### 📚 참고 규정\n"]
        
        for i, r in enumerate(display_sources, 1):
            reg_name = r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
            path = " > ".join(clean_path_segments(r.chunk.parent_path)) if r.chunk.parent_path else r.chunk.title
            norm_score = norm_scores.get(r.chunk.id, 0.0)
            rel_pct = int(norm_score * 100)
            rel_label = get_relevance_label_combined(rel_pct)
            
            # AI 신뢰도는 show_debug일 때만 표시
            score_info = f" | AI 신뢰도: {r.score:.3f}" if show_debug else ""
            snippet = strip_path_prefix(r.chunk.text, r.chunk.parent_path or [])
            
            sources_md.append(f"""#### [{i}] {reg_name}
**경로:** {path}

{snippet[:300]}{'...' if len(snippet) > 300 else ''}

*규정번호: {r.chunk.rule_code} | 관련도: {rel_pct}% {rel_label}{score_info}*

---
""")

        # Confidence description using shared formatter
        conf_icon, conf_label, _ = get_confidence_info(answer.confidence)
        if answer.confidence >= 0.7:
            conf_desc = f"{conf_icon} 답변 신뢰도 {conf_label}"
        elif answer.confidence >= 0.4:
            conf_desc = f"{conf_icon} 답변 신뢰도 {conf_label} - 원문 확인 권장"
        else:
            conf_desc = f"{conf_icon} 답변 신뢰도 {conf_label} - 학교 행정실 문의 권장"

        sources_text = "\n".join(sources_md) + f"\n**{conf_desc}** (신뢰도 {answer.confidence:.0%})"

        debug_text = ""
        if show_debug:
            debug_text = _format_query_rewrite_debug(
                search_with_llm.get_last_query_rewrite()
            )

        # Return (answer, sources, debug, query, rule_code)
        rule_code = answer.sources[0].chunk.rule_code if answer.sources else ""
        yield answer_text, sources_text, debug_text, question, rule_code

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
            source="web"
        )
        return gr.update(value="✅ 피드백이 저장되었습니다. 감사합니다!", visible=True)

    # Sync function
    def run_sync(json_path: str, full_sync: bool) -> str:
        """Run synchronization."""
        if not json_path.strip():
            return "JSON 파일 경로를 입력해주세요."

        try:
            if full_sync:
                result = sync_usecase.full_sync(json_path)
            else:
                result = sync_usecase.incremental_sync(json_path)

            if result.has_errors:
                return f"❌ 오류 발생:\n" + "\n".join(result.errors)

            return f"""✅ 동기화 완료!
- 추가: {result.added}개
- 수정: {result.modified}개
- 삭제: {result.removed}개
- 변경없음: {result.unchanged}개
- 총 청크: {store.count()}개
"""
        except Exception as e:
            return f"❌ 오류: {str(e)}"

    def run_conversion_and_sync(
        hwp_file: str,
        use_llm: bool,
        llm_provider: str,
        llm_model: str,
        llm_base_url: str,
        output_dir: str,
        target_db_path: str,
        full_sync: bool,
    ) -> Tuple[str, str, str]:
        if not hwp_file:
            return "HWP 파일을 업로드하세요.", "", ""

        output_dir_value = output_dir or "data/output"
        db_path_value = target_db_path or db_path

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        input_path = _persist_upload(hwp_file)

        args = argparse.Namespace(
            input_path=str(input_path),
            output_dir=output_dir_value,
            use_llm=use_llm,
            provider=llm_provider,
            model=llm_model or None,
            base_url=llm_base_url or None,
            allow_llm_fallback=True,
            force=False,
            cache_dir=".cache",
            verbose=True,
            enhance_rag=True,
        )

        from rich.console import Console
        console = Console(record=True)
        status = run_pipeline(args, console=console)
        log_text = console.export_text() or ""

        if status != 0:
            return log_text or "변환 실패", "", ""

        json_path = Path(output_dir_value) / f"{input_path.stem}.json"
        if not json_path.exists():
            return f"{log_text}\nJSON 파일을 찾을 수 없습니다: {json_path}", "", ""

        store_local = ChromaVectorStore(persist_directory=db_path_value)
        loader_local = JSONDocumentLoader()
        sync_local = SyncUseCase(loader_local, store_local)
        if full_sync:
            sync_result = sync_local.full_sync(str(json_path))
        else:
            sync_result = sync_local.incremental_sync(str(json_path))

        sync_lines = [str(sync_result), f"총 청크 수: {store_local.count()}"]
        if sync_result.has_errors:
            sync_lines.extend(sync_result.errors)

        status_text = "\n".join([log_text, "[SYNC]", *sync_lines]).strip()
        return status_text, str(json_path), db_path_value

    # Build UI
    with gr.Blocks(
        title="📚 대학 규정집 Q&A",
    ) as app:
        gr.Markdown("# 📚 대학 규정집 Q&A 시스템")

        with gr.Tabs():
            # Tab 0: Chat
            with gr.TabItem("💬 대화형"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chat_bot = gr.Chatbot(label="대화", height=420)
                        chat_input = gr.Textbox(
                            label="메시지 입력",
                            placeholder="질문 또는 규정명을 입력하세요. 예: 교원인사규정 전문",
                            lines=2,
                        )
                        with gr.Row():
                            chat_send = gr.Button("전송", variant="primary")
                            chat_clear = gr.Button("대화 초기화")
                    with gr.Column(scale=2):
                        chat_top_k = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="결과 수")
                        chat_abolished = gr.Checkbox(label="폐지 규정 포함", value=False)
                        chat_target = gr.Radio(
                            choices=["자동", "교수", "학생", "직원"],
                            value="자동",
                            label="대상 선택",
                        )
                        chat_context = gr.Checkbox(label="문맥 활용", value=True)
                        chat_debug = gr.Checkbox(label="디버그 출력", value=False)
                        with gr.Accordion("⚙️ LLM 설정 (질문 모드용)", open=False):
                            chat_llm_p = gr.Dropdown(choices=LLM_PROVIDERS, value=DEFAULT_LLM_PROVIDER, label="Provider")
                            chat_llm_m = gr.Textbox(value=DEFAULT_LLM_MODEL, label="Model")
                            chat_llm_b = gr.Textbox(value=DEFAULT_LLM_BASE_URL, label="Base URL")
                        chat_detail = gr.Markdown(label="상세 / 근거")
                        chat_debug_out = gr.Markdown(label="디버그")

                chat_state = gr.State(
                    {
                        "audience": None,
                        "pending": None,
                        "last_query": None,
                        "last_mode": None,
                        "last_regulation": None,
                        "last_rule_code": None,
                    }
                )

                chat_send.click(
                    fn=chat_respond,
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
                        chat_debug,
                    ],
                    outputs=[chat_bot, chat_detail, chat_debug_out, chat_state],
                )
                chat_input.submit(
                    fn=chat_respond,
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
                        chat_debug,
                    ],
                    outputs=[chat_bot, chat_detail, chat_debug_out, chat_state],
                )
                chat_clear.click(
                    fn=lambda: (
                        [],
                        "",
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

            # Tab 1: Unified Search
            with gr.TabItem("🔎 통합 검색"):
                with gr.Row():
                    with gr.Column(scale=4):
                        uni_query = gr.Textbox(
                            label="검색어 또는 질문",
                            placeholder="예: 교원 연구년 자격은? (질문) / 연구년 규정 (검색)",
                            lines=2,
                        )
                        with gr.Row():
                            uni_mode = gr.Radio(
                                choices=["자동 (Auto)", "검색 (Search)", "질문 (Ask)", "전문 (Full View)"],
                                value="자동 (Auto)",
                                label="검색 모드",
                                scale=2,
                            )
                            uni_btn = gr.Button("🔍 실행", variant="primary", scale=1)

                    with gr.Column(scale=1):
                        uni_top_k = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="결과 수")
                        uni_abolished = gr.Checkbox(label="폐지 규정 포함", value=False)
                        uni_debug = gr.Checkbox(label="디버그 출력", value=False)
                        uni_target = gr.Radio(
                            choices=["자동", "교수", "학생", "직원"],
                            value="자동",
                            label="대상 선택",
                        )

                with gr.Accordion("⚙️ LLM 설정 (질문 모드용)", open=False):
                    with gr.Row():
                        llm_p = gr.Dropdown(choices=LLM_PROVIDERS, value=DEFAULT_LLM_PROVIDER, label="Provider")
                        llm_m = gr.Textbox(value=DEFAULT_LLM_MODEL, label="Model")
                        llm_b = gr.Textbox(value=DEFAULT_LLM_BASE_URL, label="Base URL")

                uni_main = gr.Markdown(label="결과 / 답변")
                uni_detail = gr.Markdown(label="상세 / 근거")

                with gr.Accordion("🔧 디버그 정보", open=False):
                    uni_debug_out = gr.Markdown()

                # Feedback State
                uni_fb_query = gr.State("")
                uni_fb_rule = gr.State("")

                # Feedback Row (Shared)
                with gr.Row(visible=False) as uni_fb_row:
                    with gr.Column(scale=4):
                        uni_fb_comment = gr.Textbox(label="피드백 의견 (선택)", placeholder="결과에 대한 의견을 남겨주세요.")
                    with gr.Column(scale=1):
                        with gr.Row():
                            uni_fb_up = gr.Button("👍", size="sm")
                            uni_fb_neu = gr.Button("😐", size="sm")
                            uni_fb_down = gr.Button("👎", size="sm")
                        uni_fb_msg = gr.Markdown(visible=False)

                # Events
                uni_btn.click(
                    fn=unified_search,
                    inputs=[
                        uni_query, uni_mode, uni_top_k, uni_abolished,
                        llm_p, llm_m, llm_b,
                        gr.State(db_path), uni_target, uni_debug
                    ],
                    outputs=[uni_main, uni_detail, uni_debug_out, uni_fb_query, uni_fb_rule],
                )

                # Feedback Events
                uni_query.change(lambda: gr.update(visible=False), None, uni_fb_row)
                uni_btn.click(lambda: gr.update(visible=True), None, uni_fb_row)

                for btn, rating in [(uni_fb_up, 1), (uni_fb_neu, 0), (uni_fb_down, -1)]:
                    btn.click(
                        fn=lambda q, r, c, rt=rating: record_web_feedback(q, r, rt, c),
                        inputs=[uni_fb_query, uni_fb_rule, uni_fb_comment],
                        outputs=[uni_fb_msg]
                    )

            # Tab 3: Status (Read-only)
            with gr.TabItem("📂 데이터 현황"):
                gr.Markdown("> DB 관리(동기화, 초기화)는 CLI에서 수행합니다: `regulation-rag sync`, `regulation-rag reset`")

                status_db_path = gr.Textbox(
                    value=db_path,
                    label="DB 경로",
                    interactive=False,
                )
                status_markdown = gr.Markdown(_render_status(db_path))
                refresh_btn = gr.Button("새로고침", variant="secondary")

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
        theme=gr.themes.Soft(),
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
