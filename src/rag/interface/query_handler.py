"""
Unified Query Handler for Regulation RAG System.

Provides a single entry point for query processing across all interfaces
(CLI, Web UI, MCP Server) to eliminate code duplication.

Supported query types:
- Overview: Show regulation structure (chapters, article count)
- Article: Show specific article full text
- Chapter: Show specific chapter full text
- Attachment: Show tables (별표/별첨/별지)
- Full View: Show entire regulation
- Search: Hybrid search with optional reranking
- Ask: LLM-generated answer with sources
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..application.full_view_usecase import FullViewUseCase
from ..application.search_usecase import (
    REGULATION_ONLY_PATTERN,
    RULE_CODE_PATTERN,
    SearchUseCase,
)
from ..domain.entities import RegulationStatus
from ..infrastructure.json_loader import JSONDocumentLoader
from ..infrastructure.query_analyzer import Audience, QueryAnalyzer

logger = logging.getLogger(__name__)
from .chat_logic import (
    attachment_label_variants,
    expand_followup_query,
    extract_regulation_title,
    parse_attachment_request,
)
from .common import decide_search_mode
from .formatters import (
    clean_path_segments,
    filter_by_relevance,
    format_regulation_citation,
    format_regulation_content,
    format_search_result_with_explanation,
    infer_attachment_label,
    infer_regulation_title_from_tables,
    normalize_markdown_emphasis,
    normalize_markdown_table,
    normalize_relevance_scores,
    render_full_view_nodes,
    strip_path_prefix,
)
from .query_suggestions import format_suggestions_for_cli, get_followup_suggestions

# Optional FunctionGemma imports
try:
    from ..infrastructure.function_gemma_adapter import FunctionGemmaAdapter
    from ..infrastructure.tool_executor import ToolExecutor

    FUNCTION_GEMMA_AVAILABLE = True
except ImportError:
    FUNCTION_GEMMA_AVAILABLE = False
    FunctionGemmaAdapter = None
    ToolExecutor = None


class QueryType(Enum):
    """Query result types."""

    OVERVIEW = "overview"
    ARTICLE = "article"
    CHAPTER = "chapter"
    ATTACHMENT = "attachment"
    FULL_VIEW = "full_view"
    SEARCH = "search"
    ASK = "ask"
    CLARIFICATION = "clarification"
    ERROR = "error"


@dataclass
class QueryContext:
    """Context for query processing."""

    state: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, str]] = field(default_factory=list)
    interactive: bool = False
    last_regulation: Optional[str] = None
    last_rule_code: Optional[str] = None


@dataclass
class QueryOptions:
    """Options for query processing."""

    top_k: int = 5
    force_mode: Optional[str] = None
    include_abolished: bool = False
    use_rerank: bool = True
    audience_override: Optional[Audience] = None
    show_debug: bool = False
    # LLM options
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None
    # FunctionGemma options
    use_function_gemma: bool = False


@dataclass
class QueryResult:
    """Unified query result."""

    type: QueryType
    success: bool
    # Markdown formatted content for CLI/Web display
    content: str = ""
    # Structured data for MCP/JSON output
    data: Dict[str, Any] = field(default_factory=dict)
    # State updates to apply
    state_update: Dict[str, Any] = field(default_factory=dict)
    # Clarification options (for type=CLARIFICATION)
    clarification_type: Optional[str] = None  # "regulation" | "audience"
    clarification_options: List[str] = field(default_factory=list)
    # Debug info
    debug_info: str = ""
    suggestions: List[str] = field(default_factory=list)


# Deletion warning patterns
DELETION_PATTERNS = [
    (
        re.compile(
            r"삭제\s*[\(\<\[]\s*(\d{4})[.\-/]?\s*(\d{1,2})?[.\-/]?\s*(\d{1,2})?"
        ),
        "삭제",
    ),
    (
        re.compile(
            r"폐지\s*[\(\<\[]\s*(\d{4})[.\-/]?\s*(\d{1,2})?[.\-/]?\s*(\d{1,2})?"
        ),
        "폐지",
    ),
    (re.compile(r"\(삭제\)"), "삭제"),
    (re.compile(r"\[삭제\]"), "삭제"),
    (re.compile(r"본 조는?\s*삭제"), "삭제"),
    (re.compile(r"본 항은?\s*삭제"), "삭제"),
]


def detect_deletion_warning(text: str) -> Optional[str]:
    """
    Detect if text contains deletion markers and extract date if available.

    Args:
        text: The chunk text to analyze.

    Returns:
        Warning message if deletion detected, None otherwise.
    """
    for pattern, deletion_type in DELETION_PATTERNS:
        match = pattern.search(text)
        if match:
            groups = match.groups() if match.groups() else ()
            if groups and groups[0]:
                year = groups[0]
                month = groups[1] if len(groups) > 1 and groups[1] else None
                day = groups[2] if len(groups) > 2 and groups[2] else None
                if month and day:
                    return f"⚠️ 이 조항은 {year}년 {month}월 {day}일에 {deletion_type}되었습니다."
                elif month:
                    return (
                        f"⚠️ 이 조항은 {year}년 {month}월에 {deletion_type}되었습니다."
                    )
                else:
                    return f"⚠️ 이 조항은 {year}년에 {deletion_type}되었습니다."
            else:
                return f"⚠️ 이 조항은 {deletion_type}되었습니다. 최신 규정을 확인하세요."
    return None


class QueryHandler:
    """
    Unified query handler for all interfaces.

    Extracts common logic from CLI's _perform_unified_search and
    Web UI's chat_respond into a single reusable module.
    """

    # Security constants for input validation
    MAX_QUERY_LENGTH = 500
    FORBIDDEN_PATTERNS = [
        r"<script[^>]*>.*?</script>",  # XSS prevention
        r"javascript:",  # JavaScript URL prevention
        r"on\w+\s*=",  # Event handler prevention
        r"DROP\s+TABLE",  # SQL Injection prevention
        r"DELETE\s+FROM",  # SQL Injection prevention
        r"\$\{.*\}",  # Template Injection prevention
        r"<iframe",  # iframe injection prevention
        r"<embed",  # embed tag prevention
    ]

    def __init__(
        self,
        store=None,
        llm_client=None,
        use_reranker: bool = True,
        function_gemma_client=None,
        function_gemma_adapter=None,
        json_path: Optional[str] = None,
    ):
        self.store = store
        self.llm_client = llm_client
        self.use_reranker = use_reranker
        self.loader = JSONDocumentLoader()
        self.full_view_usecase = FullViewUseCase(
            loader=self.loader, json_path=json_path
        )
        self.query_analyzer = QueryAnalyzer()
        self._last_query_rewrite = None

        # FunctionGemma setup
        self._function_gemma_adapter = function_gemma_adapter
        if FUNCTION_GEMMA_AVAILABLE and not self._function_gemma_adapter:
            if function_gemma_client:
                # Check if function_gemma_client is already a FunctionGemmaAdapter
                if hasattr(function_gemma_client, "process_query") and hasattr(
                    function_gemma_client, "_tool_executor"
                ):
                    # It's already an adapter, use it directly
                    self._function_gemma_adapter = function_gemma_client
                else:
                    # It's a raw LLM client, create adapter
                    self._setup_function_gemma(function_gemma_client)

    def _normalize_query(self, query: str) -> str:
        """Normalize input query to NFC for consistent matching."""
        if not query:
            return ""
        return unicodedata.normalize("NFC", query)

    def validate_query(self, query: str) -> Tuple[bool, str]:
        """
        Validate user query for security and format.

        Args:
            query: Query string to validate

        Returns:
            (is_valid, error_message) tuple
            - is_valid: True if validation passed, False otherwise
            - error_message: User-facing error message if validation failed
        """
        # 1. Length validation
        if len(query) > self.MAX_QUERY_LENGTH:
            return False, f"쿼리가 너무 깁니다 (최대 {self.MAX_QUERY_LENGTH}자)"

        # 2. Empty query validation
        if not query or not query.strip():
            return False, "검색어를 입력해주세요"

        # 3. Security pattern validation
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE | re.DOTALL):
                return False, "허용되지 않는 문자가 포함되어 있습니다"

        # 4. Control character validation (except newline, carriage return, tab)
        if any(ord(c) < 32 and c not in "\n\r\t" for c in query):
            return False, "허용되지 않는 제어 문자가 포함되어 있습니다"

        return True, ""

    def _setup_function_gemma(self, function_gemma_client) -> None:
        """Initialize FunctionGemma adapter with tool executor."""
        from ..application.search_usecase import SearchUseCase
        from ..application.sync_usecase import SyncUseCase
        from ..infrastructure.json_loader import JSONDocumentLoader

        # Create tool executor with use cases
        search_usecase = None
        sync_usecase = None

        if self.store:
            search_usecase = SearchUseCase(
                self.store,
                llm_client=self.llm_client,
                use_reranker=self.use_reranker,
            )
            loader = JSONDocumentLoader()
            sync_usecase = SyncUseCase(loader, self.store)

        tool_executor = ToolExecutor(
            search_usecase=search_usecase,
            sync_usecase=sync_usecase,
            query_analyzer=self.query_analyzer,
            llm_client=self.llm_client,
        )
        logger.debug(f"_setup_function_gemma: Created tool_executor={tool_executor}")

        self._function_gemma_adapter = FunctionGemmaAdapter(
            llm_client=function_gemma_client,
            tool_executor=tool_executor,
        )

    def _process_with_function_gemma(
        self,
        query: str,
        context: QueryContext,
        options: QueryOptions,
    ) -> QueryResult:
        """
        Process query using FunctionGemma tool-based approach.

        FunctionGemma selects and executes tools, then generates
        the final answer using the base LLM.
        """
        try:
            answer, tool_results = self._function_gemma_adapter.process_query(query)

            # Build debug info from tool results
            debug_lines = []
            regulation_title = None

            for result in tool_results:
                status = "✅" if result.success else "❌"
                debug_msg = f"{status} **{result.tool_name}**"

                if result.arguments:
                    import json

                    try:
                        args_str = json.dumps(result.arguments, ensure_ascii=False)
                        debug_msg += f"\n   🔹 Args: `{args_str}`"
                    except Exception:
                        debug_msg += f"\n   🔹 Args: {result.arguments}"

                res_str = str(result.result)
                if len(res_str) > 300:
                    res_str = res_str[:300] + "... (truncated)"
                debug_msg += f"\n   🔸 Result: {res_str}"

                debug_lines.append(debug_msg)

                # Extract regulation_title from search_regulations results
                if result.tool_name == "search_regulations" and result.success:
                    if isinstance(result.result, dict) and "results" in result.result:
                        for r in result.result["results"]:
                            if r.get("regulation_title"):
                                regulation_title = r["regulation_title"]
                                break

            debug_info = "\n\n".join(debug_lines) if options.show_debug else ""

            return QueryResult(
                type=QueryType.ASK,
                success=True,
                content=answer,
                data={
                    "tool_results": [r.to_dict() for r in tool_results],
                    "used_function_gemma": True,
                    "regulation_title": regulation_title,
                },
                debug_info=debug_info,
            )
        except Exception as e:
            # Fallback to standard processing on error
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content=f"FunctionGemma 처리 오류: {str(e)}. 기본 검색을 시도해주세요.",
            )

    def process_query(
        self,
        query: str,
        context: Optional[QueryContext] = None,
        options: Optional[QueryOptions] = None,
    ) -> QueryResult:
        """
        Main entry point for query processing.

        Analyzes the query and routes to appropriate handler.
        If use_function_gemma is enabled and adapter is available,
        uses FunctionGemma for tool-based processing.
        """
        context = context or QueryContext()
        options = options or QueryOptions()
        query = self._normalize_query(query)

        # Input validation
        is_valid, error_msg = self.validate_query(query)
        if not is_valid:
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content=f"⚠️ {error_msg}",
            )

        query = query.strip()

        # 1. Structural Pattern Checks (Prioritize over Tool Calling for consistency)

        # Check for regulation-only or rule-code-only queries first
        if self._is_overview_query(query):
            result = self.get_regulation_overview(query)
            if result.success:
                return self._enrich_with_suggestions(result, query)

        # Check for specific article query (e.g., "교원인사규정 제8조")
        article_match = re.search(r"(?:제)?\s*(\d+)\s*조", query)
        target_regulation = extract_regulation_title(query)

        if target_regulation and article_match:
            article_no = int(article_match.group(1))
            result = self.get_article_view(target_regulation, article_no, context)
            if result.type != QueryType.ERROR:
                return self._enrich_with_suggestions(result, query)

        # Check for chapter query (e.g., "학칙 제3장")
        chapter_match = re.search(r"(?:제)?\s*(\d+)\s*장", query)
        if target_regulation and chapter_match:
            chapter_no = int(chapter_match.group(1))
            result = self.get_chapter_view(target_regulation, chapter_no, context)
            if result.type != QueryType.ERROR:
                return self._enrich_with_suggestions(result, query)

        # Check for attachment query (별표/별첨/별지)
        attachment_request = parse_attachment_request(
            query,
            context.last_regulation if context.interactive else None,
        )
        if attachment_request:
            reg_query, table_no, label = attachment_request
            result = self.get_attachment_view(reg_query, label, table_no, context)
            if result.type != QueryType.ERROR:
                return self._enrich_with_suggestions(result, query)

        # Determine mode using heuristics
        mode = options.force_mode or decide_search_mode(query)

        # Full view mode (e.g., "학칙 전문")
        if mode == "full_view":
            result = self.get_full_view(query, context)
            if result.type != QueryType.ERROR:
                return self._enrich_with_suggestions(result, query)

        # 2. Tool Calling or Traditional Search/Ask

        # FunctionGemma tool-based processing
        if options.use_function_gemma and self._function_gemma_adapter:
            result = self._process_with_function_gemma(query, context, options)
            return self._enrich_with_suggestions(result, query)

        # Check audience ambiguity
        if (
            options.audience_override is None
            and self.query_analyzer.is_audience_ambiguous(query)
        ):
            return QueryResult(
                type=QueryType.CLARIFICATION,
                success=True,
                clarification_type="audience",
                clarification_options=["교수", "학생", "직원"],
                content="질문 대상을 선택해주세요: 교수, 학생, 직원",
            )

        # Search or Ask
        if mode == "search":
            result = self.search(query, options)
        else:
            result = self.ask(query, options, context)

        # Enrich with suggestions before returning
        return self._enrich_with_suggestions(result, query)

    def process_query_stream(
        self,
        query: str,
        context: Optional[QueryContext] = None,
        options: Optional[QueryOptions] = None,
    ):
        """
        Streaming version of process_query.
        Yields events: {"type": str, "content": Any, ...}
        """
        context = context or QueryContext()
        options = options or QueryOptions()
        query = self._normalize_query(query)

        # Input validation
        is_valid, error_msg = self.validate_query(query)
        if not is_valid:
            yield {"type": "error", "content": error_msg}
            return

        query = query.strip()

        # 1. Structural Pattern Checks

        # Check for regulation-only or rule-code-only queries first
        if self._is_overview_query(query):
            result = self.get_regulation_overview(query)
            result = self._enrich_with_suggestions(result, query)
            yield from self._yield_result(result)
            return

        # Check for specific article query
        article_match = re.search(r"(?:제)?\s*(\d+)\s*조", query)
        target_regulation = extract_regulation_title(query)

        if target_regulation and article_match:
            article_no = int(article_match.group(1))
            result = self.get_article_view(target_regulation, article_no, context)
            if result.type != QueryType.ERROR:
                result = self._enrich_with_suggestions(result, query)
                yield from self._yield_result(result)
                return

        # Check for chapter query
        chapter_match = re.search(r"(?:제)?\s*(\d+)\s*장", query)
        if target_regulation and chapter_match:
            chapter_no = int(chapter_match.group(1))
            result = self.get_chapter_view(target_regulation, chapter_no, context)
            if result.type != QueryType.ERROR:
                result = self._enrich_with_suggestions(result, query)
                yield from self._yield_result(result)
                return

        # Check for attachment query
        attachment_request = parse_attachment_request(
            query,
            context.last_regulation if context.interactive else None,
        )
        if attachment_request:
            reg_query, table_no, label = attachment_request
            result = self.get_attachment_view(reg_query, label, table_no, context)
            if result.type != QueryType.ERROR:
                result = self._enrich_with_suggestions(result, query)
                yield from self._yield_result(result)
                return

        # Determine mode
        mode = options.force_mode or decide_search_mode(query)

        # Full view mode
        if mode == "full_view":
            result = self.get_full_view(query, context)
            if result.type != QueryType.ERROR:
                result = self._enrich_with_suggestions(result, query)
                yield from self._yield_result(result)
                return

        # 2. Tool Calling or Traditional Search/Ask

        # FunctionGemma tool-based processing (Sync -> Event Stream)
        if options.use_function_gemma and self._function_gemma_adapter:
            # Yield progress
            yield {"type": "progress", "content": "🤖 도구 기반 답변 생성 중..."}

            # Run synchronously (Tool Calling does not support streaming yet)
            q_result = self._process_with_function_gemma(query, context, options)

            if q_result.success:
                # Enrich with suggestions
                q_result = self._enrich_with_suggestions(q_result, query)

                # Yield usage info
                if q_result.debug_info:
                    yield {
                        "type": "progress",
                        "content": f"🛠️ 사용된 도구:\n{q_result.debug_info}",
                    }

                # Yield final answer
                yield {
                    "type": "complete",
                    "content": q_result.content,
                    "data": q_result.data,
                    "suggestions": q_result.suggestions,
                }
                return
            else:
                yield {"type": "error", "content": q_result.content}
                return

        # Check audience ambiguity
        if (
            options.audience_override is None
            and self.query_analyzer.is_audience_ambiguous(query)
        ):
            yield {
                "type": "clarification",
                "clarification_type": "audience",
                "options": ["교수", "학생", "직원"],
                "content": "질문 대상을 선택해주세요: 교수, 학생, 직원",
            }
            return

        # Search or Ask
        if mode == "search":
            # Search is synchronous for now, treat as instant result
            yield {"type": "progress", "content": "🔍 규정 검색 중..."}
            result = self.search(query, options)
            result = self._enrich_with_suggestions(result, query)
            yield from self._yield_result(result)
        else:
            # Ask - Streaming
            # We can't easily enrich the stream yet without intercepting tokens,
            # but we can yield suggestions at the end.
            for event in self.ask_stream(query, options, context):
                if event.get("type") == "complete":
                    # For completed streaming questions, we can enrich the final content
                    temp_res = QueryResult(
                        type=QueryType.ASK,
                        success=True,
                        content=event["content"],
                        data=event.get("data", {}),
                    )
                    temp_res = self._enrich_with_suggestions(temp_res, query)
                    event["content"] = temp_res.content
                    event["suggestions"] = temp_res.suggestions
                yield event

    def _enrich_with_suggestions(self, result: QueryResult, query: str) -> QueryResult:
        """Calculate and append followup suggestions to result."""
        if result.type in (QueryType.CLARIFICATION, QueryType.ERROR):
            return result

        reg_title = result.data.get("regulation_title") or result.data.get("title")
        suggestions = get_followup_suggestions(
            query=query,
            regulation_title=reg_title,
            answer_text=result.content if result.type == QueryType.ASK else None,
        )

        if suggestions:
            result.suggestions = suggestions
            # Append formatted suggestions to content for standard display
            suggestion_text = format_suggestions_for_cli(suggestions)
            if suggestion_text:
                result.content += "\n\n" + suggestion_text

        return result

    def _yield_result(self, result: QueryResult):
        """Helper to yield a QueryResult as stream events."""
        if result.type == QueryType.CLARIFICATION:
            yield {
                "type": "clarification",
                "clarification_type": result.clarification_type,
                "options": result.clarification_options,
                "content": result.content,
            }
        elif result.type == QueryType.ERROR:
            yield {"type": "error", "content": result.content}
        else:
            # For standard results, we yield metadata then content
            # If state update exists, yield it logic might be handled by caller or here
            # We yield 'answer' or 'complete'
            # Gradio expects 'token' loop or 'complete'
            yield {
                "type": "metadata",
                "rule_code": result.data.get("rule_code"),
                "regulation_title": result.data.get("title")
                or result.data.get("regulation_title"),
            }
            yield {
                "type": "complete",
                "content": result.content,
                "data": result.data,
                "suggestions": result.suggestions,
            }
            if result.state_update:
                yield {"type": "state", "update": result.state_update}

    def ask_stream(
        self,
        question: str,
        options: Optional[QueryOptions] = None,
        context: Optional[QueryContext] = None,
    ):
        """Stream LLM-generated answer."""
        options = options or QueryOptions()
        context = context or QueryContext()

        if self.store is None or self.store.count() == 0 or self.llm_client is None:
            yield {
                "type": "error",
                "content": "시스템이 초기화되지 않았거나 DB가 비어있습니다.",
            }
            return

        search_usecase = SearchUseCase(
            self.store,
            llm_client=self.llm_client,
            use_reranker=options.use_rerank,
        )

        # History setup...
        history_text = None
        if context.history:
            history_lines = []
            for msg in context.history[-10:]:
                role = "사용자" if msg.get("role") == "user" else "AI"
                content = msg.get("content", "")
                if content:
                    history_lines.append(f"{role}: {content[:200]}")
            if history_lines:
                history_text = "\n".join(history_lines)

        yield {"type": "progress", "content": "🔍 규정 검색 중..."}

        # Handle multi-turn context expansion
        search_query = question
        if context.last_regulation:
            expanded = expand_followup_query(question, context.last_regulation)
            if expanded != question:
                search_query = expanded
                if options.show_debug:
                    yield {
                        "type": "progress",
                        "content": f"🔄 문맥 반영: {search_query}",
                    }

        # Use SearchUseCase.ask_stream
        sources = []
        for event in search_usecase.ask_stream(
            question=question,
            top_k=options.top_k,
            include_abolished=options.include_abolished,
            audience_override=options.audience_override,
            history_text=history_text,
            search_query=search_query,
        ):
            if event.get("type") == "metadata":
                sources = event.get("sources", [])
            yield event

        if sources:
            norm_scores = normalize_relevance_scores(sources)
            display_sources = filter_by_relevance(sources, norm_scores)

            citations_text = ""
            for i, r in enumerate(display_sources, 1):
                citation_block = format_regulation_citation(r.chunk)
                norm_score = norm_scores.get(r.chunk.id, 0.0)
                rel_pct = int(norm_score * 100)

                # Add matching explanation with keywords
                explanation, _ = format_search_result_with_explanation(
                    r, question, show_score=False
                )

                citations_text += (
                    f"\n\n{citation_block}\n*관련도: {rel_pct}%*\n\n"
                    f"**매칭 정보:** {explanation}\n\n---\n"
                )

            if citations_text:
                yield {"type": "token", "content": citations_text}

        self._last_query_rewrite = search_usecase.get_last_query_rewrite()

    def _is_overview_query(self, query: str) -> bool:
        """Check if query is requesting regulation overview."""
        return (
            REGULATION_ONLY_PATTERN.match(query) is not None
            or RULE_CODE_PATTERN.match(query) is not None
        )

    def get_regulation_overview(self, query: str) -> QueryResult:
        """Get regulation overview (structure, chapter count, etc.)."""
        json_path = self.full_view_usecase._resolve_json_path()
        if not json_path:
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content="규정 JSON 파일을 찾을 수 없습니다.",
            )

        overview = self.loader.get_regulation_overview(json_path, query)
        if not overview:
            # Try finding candidates
            candidates = self.loader.find_regulation_candidates(json_path, query)
            if len(candidates) == 1:
                # Found exactly one match
                overview = self.loader.get_regulation_overview(
                    json_path, candidates[0][0]
                )
            elif len(candidates) > 1:
                # Multiple matches found
                # Check for exact match (ignoring spaces)
                normalized_query = query.replace(" ", "")
                exact_match = None
                for code, title in candidates:
                    if title.replace(" ", "") == normalized_query:
                        exact_match = (code, title)
                        break

                if exact_match:
                    overview = self.loader.get_regulation_overview(
                        json_path, exact_match[0]
                    )
                else:
                    return QueryResult(
                        type=QueryType.CLARIFICATION,
                        success=True,
                        clarification_type="regulation",
                        clarification_options=[c[1] for c in candidates],
                        content="여러 규정이 매칭됩니다. 선택해주세요.",
                    )
            else:
                return QueryResult(
                    type=QueryType.ERROR,
                    success=False,
                    content="해당 규정을 찾을 수 없습니다.",
                )

        if not overview:
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content="해당 규정을 찾을 수 없습니다.",
            )

        # Build markdown content
        status_label = (
            "✅ 시행중" if overview.status == RegulationStatus.ACTIVE else "❌ 폐지"
        )
        lines = [f"## 📋 {overview.title} ({overview.rule_code})"]
        lines.append("")
        lines.append(
            f"**상태**: {status_label} | **총 조항 수**: {overview.article_count}개"
        )
        lines.append("")

        if overview.chapters:
            lines.append("### 📖 목차")
            for ch in overview.chapters:
                article_info = f" ({ch.article_range})" if ch.article_range else ""
                lines.append(f"- **{ch.display_no}** {ch.title}{article_info}")
        else:
            lines.append("*(장 구조 없이 조항으로만 구성된 규정)*")

        if overview.has_addenda:
            lines.append("")
            lines.append("📎 **부칙** 있음")

        lines.append("")
        lines.append("---")
        lines.append(
            f"💡 특정 조항 검색: `{overview.title} 제N조` 또는 `{overview.rule_code} 제N조`"
        )

        # Check for similar regulations
        is_regulation_only = REGULATION_ONLY_PATTERN.match(query) is not None
        other_matches = []
        if is_regulation_only:
            all_titles = self.loader.get_regulation_titles(json_path)
            other_matches = sorted(
                [t for t in all_titles.values() if query in t and t != overview.title]
            )
            if other_matches:
                lines.append("")
                lines.append("❓ **혹시 다음 규정을 찾으셨나요?**")
                for m in other_matches:
                    lines.append(f"- {m}")

        return QueryResult(
            type=QueryType.OVERVIEW,
            success=True,
            content="\n".join(lines),
            data={
                "title": overview.title,
                "rule_code": overview.rule_code,
                "status": overview.status.value
                if hasattr(overview.status, "value")
                else str(overview.status),
                "article_count": overview.article_count,
                "chapters": [
                    {
                        "display_no": ch.display_no,
                        "title": ch.title,
                        "article_range": ch.article_range,
                    }
                    for ch in (overview.chapters or [])
                ],
                "has_addenda": overview.has_addenda,
                "other_matches": other_matches,
            },
            state_update={
                "last_regulation": overview.title,
                "last_rule_code": overview.rule_code,
            },
        )

    def get_article_view(
        self,
        regulation: str,
        article_no: int,
        context: Optional[QueryContext] = None,
    ) -> QueryResult:
        """Get specific article full text."""
        matches = self.full_view_usecase.find_matches(regulation)

        if not matches:
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content="해당 규정을 찾을 수 없습니다.",
            )

        if len(matches) > 1:
            return QueryResult(
                type=QueryType.CLARIFICATION,
                success=True,
                clarification_type="regulation",
                clarification_options=[m.title for m in matches],
                content="여러 규정이 매칭됩니다. 선택해주세요.",
            )

        selected = matches[0]
        article_node = self.full_view_usecase.get_article_view(
            selected.rule_code, article_no
        )

        if not article_node:
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content=f"{selected.title}에서 제{article_no}조를 찾을 수 없습니다.",
            )

        content_text = render_full_view_nodes([article_node])
        content_text = format_regulation_content(content_text)
        full_response = f"## 📌 {selected.title} 제{article_no}조\n\n{content_text}"

        return QueryResult(
            type=QueryType.ARTICLE,
            success=True,
            content=full_response,
            data={
                "regulation_title": selected.title,
                "rule_code": selected.rule_code,
                "article_no": article_no,
                "content": content_text,
            },
            state_update={
                "last_regulation": selected.title,
                "last_rule_code": selected.rule_code,
            },
        )

    def get_chapter_view(
        self,
        regulation: str,
        chapter_no: int,
        context: Optional[QueryContext] = None,
    ) -> QueryResult:
        """Get specific chapter full text."""
        matches = self.full_view_usecase.find_matches(regulation)

        if not matches:
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content="해당 규정을 찾을 수 없습니다.",
            )

        if len(matches) > 1:
            return QueryResult(
                type=QueryType.CLARIFICATION,
                success=True,
                clarification_type="regulation",
                clarification_options=[m.title for m in matches],
                content="여러 규정이 매칭됩니다. 선택해주세요.",
            )

        selected = matches[0]
        json_path = self.full_view_usecase._resolve_json_path()

        if not json_path:
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content="규정 JSON 파일을 찾을 수 없습니다.",
            )

        doc = self.loader.get_regulation_doc(json_path, selected.rule_code)
        chapter_node = self.full_view_usecase.get_chapter_node(doc, chapter_no)

        if not chapter_node:
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content=f"{selected.title}에서 제{chapter_no}장을 찾을 수 없습니다.",
            )

        chapter_title = chapter_node.get("title", "").strip()
        chapter_disp = chapter_node.get("display_no", f"제{chapter_no}장").strip()
        full_title = f"{selected.title} {chapter_disp} {chapter_title}".strip()

        content_text = render_full_view_nodes(chapter_node.get("children", []))
        content_text = format_regulation_content(content_text)
        full_response = f"## 📑 {full_title}\n\n{content_text}"

        return QueryResult(
            type=QueryType.CHAPTER,
            success=True,
            content=full_response,
            data={
                "regulation_title": selected.title,
                "rule_code": selected.rule_code,
                "chapter_no": chapter_no,
                "chapter_title": chapter_title,
                "content": content_text,
            },
            state_update={
                "last_regulation": selected.title,
                "last_rule_code": selected.rule_code,
            },
        )

    def get_attachment_view(
        self,
        regulation: str,
        label: Optional[str] = None,
        table_no: Optional[int] = None,
        context: Optional[QueryContext] = None,
    ) -> QueryResult:
        """Get attachment (별표/별첨/별지) content."""
        matches = self.full_view_usecase.find_matches(regulation)

        if not matches:
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content="해당 규정을 찾을 수 없습니다.",
            )

        if len(matches) > 1:
            return QueryResult(
                type=QueryType.CLARIFICATION,
                success=True,
                clarification_type="regulation",
                clarification_options=[m.title for m in matches],
                content="여러 규정이 매칭됩니다. 선택해주세요.",
            )

        selected = matches[0]
        label_variants = attachment_label_variants(label)
        tables = self.full_view_usecase.find_tables(
            selected.rule_code, table_no, label_variants
        )

        if not tables:
            label_text = label or "별표"
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content=f"{label_text}를 찾을 수 없습니다.",
            )

        display_title = infer_regulation_title_from_tables(tables, selected.title)
        label_text = label or "별표"

        # Build content
        lines = []
        for idx, table in enumerate(tables, 1):
            path = clean_path_segments(table.path) if table.path else []
            heading = " > ".join(path) if path else table.title or label_text
            if table_no:
                table_label = f"{label_text} {table_no}"
            else:
                table_label = infer_attachment_label(table, label_text)
            lines.append(f"### [{idx}] {heading} ({table_label})")
            if table.text:
                lines.append(table.text)
            lines.append(normalize_markdown_table(table.markdown).strip())

        title_label = f"{display_title} {label_text}"
        if table_no:
            title_label = f"{display_title} {label_text} {table_no}"

        full_response = f"## 📋 {title_label}\n\n" + "\n\n".join(lines)

        return QueryResult(
            type=QueryType.ATTACHMENT,
            success=True,
            content=full_response,
            data={
                "regulation_title": display_title,
                "rule_code": selected.rule_code,
                "label": label_text,
                "table_no": table_no,
                "tables": [
                    {
                        "path": clean_path_segments(t.path) if t.path else [],
                        "text": t.text,
                        "markdown": t.markdown,
                    }
                    for t in tables
                ],
            },
            state_update={
                "last_regulation": display_title,
                "last_rule_code": selected.rule_code,
            },
        )

    def get_full_view(
        self,
        query: str,
        context: Optional[QueryContext] = None,
    ) -> QueryResult:
        """Get full regulation text."""
        matches = self.full_view_usecase.find_matches(query)

        if not matches:
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content="해당 규정을 찾을 수 없습니다.",
            )

        if len(matches) > 1:
            return QueryResult(
                type=QueryType.CLARIFICATION,
                success=True,
                clarification_type="regulation",
                clarification_options=[m.title for m in matches],
                content="여러 규정이 매칭됩니다. 선택해주세요.",
            )

        selected = matches[0]
        view = self.full_view_usecase.get_full_view(
            selected.rule_code
        ) or self.full_view_usecase.get_full_view(selected.title)

        if not view:
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content="규정 전문을 불러오지 못했습니다.",
            )

        # Build TOC
        toc_text = (
            "### 목차\n" + "\n".join([f"- {t}" for t in view.toc])
            if view.toc
            else "목차 정보가 없습니다."
        )

        content_text = render_full_view_nodes(view.content)
        content_text = format_regulation_content(content_text)

        # Abbreviate addenda if there are many items (e.g., historical changes)
        addenda_text = render_full_view_nodes(view.addenda, max_items=10)
        addenda_text = format_regulation_content(addenda_text)

        full_content = f"## 📖 {view.title}\n\n{toc_text}\n\n---\n\n### 본문\n\n{content_text or '본문이 없습니다.'}"
        if addenda_text:
            # Avoid redundant "부칙" title if rendered text already starts with one
            if re.match(r"^#+\s*부\s*칙", addenda_text.strip()):
                full_content += f"\n\n---\n\n{addenda_text}"
            else:
                full_content += f"\n\n---\n\n### 부칙\n\n{addenda_text}"

        return QueryResult(
            type=QueryType.FULL_VIEW,
            success=True,
            content=full_content,
            data={
                "title": view.title,
                "rule_code": view.rule_code,
                "toc": view.toc,
                "content": view.content,
                "addenda": view.addenda,
            },
            state_update={
                "last_regulation": view.title,
                "last_rule_code": view.rule_code,
            },
        )

    def search(
        self,
        query: str,
        options: Optional[QueryOptions] = None,
    ) -> QueryResult:
        """Perform hybrid search."""
        options = options or QueryOptions()

        if self.store is None:
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content="데이터베이스가 초기화되지 않았습니다.",
            )

        if self.store.count() == 0:
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content="데이터베이스가 비어 있습니다. CLI에서 'regulation sync'를 실행하세요.",
            )

        search_usecase = SearchUseCase(
            self.store,
            use_reranker=options.use_rerank,
        )

        results = search_usecase.search_unique(
            query,
            top_k=options.top_k,
            include_abolished=options.include_abolished,
            audience_override=options.audience_override,
        )

        self._last_query_rewrite = search_usecase.get_last_query_rewrite()

        if not results:
            return QueryResult(
                type=QueryType.SEARCH,
                success=True,
                content="검색 결과가 없습니다.",
                data={"results": []},
            )

        # Build table format for CLI/Web
        table_rows = [
            "| # | 규정명 | 코드 | 조항 | 매칭 이유 | 점수 |",
            "|---|------|------|------|----------|------|",
        ]
        for i, r in enumerate(results, 1):
            reg_title = r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
            path_segments = (
                clean_path_segments(r.chunk.parent_path) if r.chunk.parent_path else []
            )
            path = " > ".join(path_segments[-2:]) if path_segments else r.chunk.title

            # 새로 추가: 매칭 설명 생성
            _, matched_kw = format_search_result_with_explanation(
                r, query, show_score=False
            )
            explanation_short = matched_kw if matched_kw else "관련 내용"

            table_rows.append(
                f"| {i} | {reg_title} | {r.chunk.rule_code} | {path[:30]} | {explanation_short[:30]} | {r.score:.2f} |"
            )

        # Add top result detail
        top = results[0]
        top_text = strip_path_prefix(top.chunk.text, top.chunk.parent_path or [])
        full_path = (
            " > ".join(clean_path_segments(top.chunk.parent_path))
            if top.chunk.parent_path
            else top.chunk.title
        )

        # 매칭 설명 생성
        explanation, _ = format_search_result_with_explanation(
            top, query, show_score=options.show_debug
        )

        content = "\n".join(table_rows)
        content += f"\n\n---\n\n### 🏆 1위 결과: {top.chunk.rule_code}\n\n"
        content += f"**규정명:** {top.chunk.parent_path[0] if top.chunk.parent_path else top.chunk.title}\n\n"
        content += f"**경로:** {full_path}\n\n"
        content += f"**매칭 정보:** {explanation}\n\n"

        # Step 6: 삭제 조항 경고 추가
        deletion_warning = detect_deletion_warning(top_text)
        if deletion_warning:
            content += f"\n{deletion_warning}\n\n"

        content += f"{top_text}"

        # Build data for MCP
        formatted_results = []
        for i, r in enumerate(results, 1):
            reg_name = r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
            path = (
                " > ".join(clean_path_segments(r.chunk.parent_path))
                if r.chunk.parent_path
                else r.chunk.title
            )
            # 각 결과에도 삭제 경고 포함
            result_deletion_warning = detect_deletion_warning(r.chunk.text)
            formatted_results.append(
                {
                    "rank": i,
                    "regulation_name": reg_name,
                    "rule_code": r.chunk.rule_code,
                    "path": path,
                    "text": r.chunk.text,
                    "score": round(r.score, 4),
                    "deletion_warning": result_deletion_warning,
                }
            )

        top_regulation = (
            top.chunk.parent_path[0] if top.chunk.parent_path else top.chunk.title
        )

        return QueryResult(
            type=QueryType.SEARCH,
            success=True,
            content=content,
            data={
                "query": query,
                "results": formatted_results,
            },
            state_update={
                "last_regulation": top_regulation,
                "last_rule_code": top.chunk.rule_code,
            },
        )

    def ask(
        self,
        question: str,
        options: Optional[QueryOptions] = None,
        context: Optional[QueryContext] = None,
    ) -> QueryResult:
        """Get LLM-generated answer."""
        options = options or QueryOptions()
        context = context or QueryContext()

        if self.store is None:
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content="데이터베이스가 초기화되지 않았습니다.",
            )

        if self.store.count() == 0:
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content="데이터베이스가 비어 있습니다. CLI에서 'regulation sync'를 실행하세요.",
            )

        if self.llm_client is None:
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content="LLM 클라이언트가 초기화되지 않았습니다.",
            )

        search_usecase = SearchUseCase(
            self.store,
            llm_client=self.llm_client,
            use_reranker=options.use_rerank,
        )

        # Build history context if available
        history_text = None
        if context.history:
            history_lines = []
            for msg in context.history[-10:]:  # Last 10 messages
                role = "사용자" if msg.get("role") == "user" else "AI"
                content = msg.get("content", "")
                if content:
                    history_lines.append(f"{role}: {content[:200]}")
            if history_lines:
                history_text = "\n".join(history_lines)

        # Handle multi-turn context expansion
        search_query = question
        if context.last_regulation:
            expanded = expand_followup_query(question, context.last_regulation)
            if expanded != question:
                search_query = expanded
                # Log expansion for debug
                if options.show_debug:
                    logger.debug(f"Context expansion: '{question}' -> '{search_query}'")

        try:
            answer = search_usecase.ask(
                question=question,
                top_k=options.top_k,
                include_abolished=options.include_abolished,
                audience_override=options.audience_override,
                history_text=history_text,
                search_query=search_query,
            )
        except Exception as e:
            return QueryResult(
                type=QueryType.ERROR,
                success=False,
                content=f"답변 생성 실패: {str(e)}",
            )

        self._last_query_rewrite = search_usecase.get_last_query_rewrite()

        answer_text = normalize_markdown_emphasis(answer.text)

        # Build sources section
        sources_md = []
        if answer.sources:
            norm_scores = normalize_relevance_scores(answer.sources)
            display_sources = filter_by_relevance(answer.sources, norm_scores)

            for i, r in enumerate(display_sources, 1):
                citation_block = format_regulation_citation(r.chunk)
                norm_score = norm_scores.get(r.chunk.id, 0.0)
                rel_pct = int(norm_score * 100)

                # Add matching explanation with keywords
                explanation, _ = format_search_result_with_explanation(
                    r, question, show_score=False
                )

                sources_md.append(f"""{citation_block}
*관련도: {rel_pct}%*

**매칭 정보:** {explanation}

---
""")

        content = answer_text
        if sources_md:
            content += "\n\n---\n\n" + "\n".join(sources_md)

        # Build data for MCP
        sources_data = []
        if answer.sources:
            norm_scores = normalize_relevance_scores(answer.sources)
            display_sources = filter_by_relevance(answer.sources, norm_scores)
            for r in display_sources:
                reg_name = (
                    r.chunk.parent_path[0] if r.chunk.parent_path else r.chunk.title
                )
                path = (
                    " > ".join(clean_path_segments(r.chunk.parent_path))
                    if r.chunk.parent_path
                    else r.chunk.title
                )
                sources_data.append(
                    {
                        "regulation_name": reg_name,
                        "rule_code": r.chunk.rule_code,
                        "path": path,
                        "text": r.chunk.text,
                        "relevance_pct": int(norm_scores.get(r.chunk.id, 0.0) * 100),
                    }
                )

        top_regulation = ""
        top_rule_code = ""
        if answer.sources:
            top = answer.sources[0]
            top_regulation = (
                top.chunk.parent_path[0] if top.chunk.parent_path else top.chunk.title
            )
            top_rule_code = top.chunk.rule_code

        return QueryResult(
            type=QueryType.ASK,
            success=True,
            content=content,
            data={
                "question": question,
                "answer": answer.text,
                "confidence": round(answer.confidence, 3),
                "sources": sources_data,
            },
            state_update={
                "last_regulation": top_regulation,
                "last_rule_code": top_rule_code,
            },
        )

    def get_last_query_rewrite(self):
        """Get the last query rewrite info for debug output."""
        return self._last_query_rewrite
