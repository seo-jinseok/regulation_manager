"""
Tool Executor for FunctionGemma.

Maps tool calls from FunctionGemma to actual function implementations.
Acts as a bridge between LLM tool calls and the RAG system's use cases.
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ..application.search_usecase import SearchUseCase
    from ..application.sync_usecase import SyncUseCase
    from ..domain.repositories import ILLMClient
    from ..infrastructure.query_analyzer import QueryAnalyzer


@dataclass
class ToolResult:
    """Result of a tool execution."""

    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM context."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "arguments": self.arguments,
        }

    def to_context_string(self) -> str:
        """Convert to string suitable for LLM context."""
        if not self.success:
            return f"[{self.tool_name}] Error: {self.error}"
        if isinstance(self.result, str):
            return f"[{self.tool_name}] {self.result}"
        return f"[{self.tool_name}] {json.dumps(self.result, ensure_ascii=False, indent=2)}"


class ToolExecutor:
    """
    Executes tools called by FunctionGemma.

    Routes tool calls to the appropriate use case methods.
    """

    def __init__(
        self,
        search_usecase: Optional["SearchUseCase"] = None,
        sync_usecase: Optional["SyncUseCase"] = None,
        query_analyzer: Optional["QueryAnalyzer"] = None,
        llm_client: Optional["ILLMClient"] = None,
        json_path: str = "data/output/규정집.json",
    ):
        """
        Initialize ToolExecutor.

        Args:
            search_usecase: Search use case for regulation queries.
            sync_usecase: Sync use case for database operations.
            query_analyzer: Query analyzer for intent/audience detection.
            llm_client: LLM client for answer generation.
            json_path: Path to regulation JSON file.
        """
        self._search_usecase = search_usecase
        self._sync_usecase = sync_usecase
        self._query_analyzer = query_analyzer
        self._llm_client = llm_client
        self._json_path = json_path

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name with given arguments.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments as a dictionary.

        Returns:
            ToolResult with execution status and result.
        """
        try:
            handler = self._get_handler(tool_name)
            if handler is None:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    result=None,
                    error=f"Unknown tool: {tool_name}",
                )
            result = handler(arguments)
            return ToolResult(
                tool_name=tool_name, success=True, result=result, arguments=arguments
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                arguments=arguments,
            )

    def _get_handler(self, tool_name: str):
        """Get handler function for a tool."""
        handlers = {
            # Search tools
            "search_regulations": self._handle_search_regulations,
            "get_article": self._handle_get_article,
            "get_chapter": self._handle_get_chapter,
            "get_attachment": self._handle_get_attachment,
            "get_regulation_overview": self._handle_get_regulation_overview,
            "get_full_regulation": self._handle_get_full_regulation,
            # Analysis tools
            "expand_synonyms": self._handle_expand_synonyms,
            "detect_intent": self._handle_detect_intent,
            "detect_audience": self._handle_detect_audience,
            "analyze_query": self._handle_analyze_query,
            # Admin tools
            "sync_database": self._handle_sync_database,
            "get_sync_status": self._handle_get_sync_status,
            "reset_database": self._handle_reset_database,
            # Response tools
            "generate_answer": self._handle_generate_answer,
            "clarify_query": self._handle_clarify_query,
        }
        return handlers.get(tool_name)

    # ========== Search Tool Handlers ==========

    def _handle_search_regulations(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search_regulations tool with clarification detection."""
        if not self._search_usecase:
            raise RuntimeError("SearchUseCase not initialized")

        query = args["query"]
        top_k = args.get("top_k", 5)
        audience_str = args.get("audience", "all")

        # Apply intent/synonym expansion for better recall on colloquial queries
        if self._query_analyzer:
            query = self._query_analyzer.expand_query(query)

        # Convert audience string to enum
        from ..infrastructure.query_analyzer import Audience

        audience_map = {
            "all": Audience.ALL,
            "student": Audience.STUDENT,
            "faculty": Audience.FACULTY,
            "staff": Audience.STAFF,
        }
        audience = audience_map.get(audience_str, Audience.ALL)

        results = self._search_usecase.search(
            query_text=query,
            top_k=top_k,
            audience_override=audience,
        )

        # Check if clarification is needed
        from ..infrastructure.clarification_detector import ClarificationDetector

        detector = ClarificationDetector()

        if detector.is_clarification_needed(query, results):
            clarification = detector.generate_clarification(query, results)
            return {
                "type": "clarification_needed",
                "query": query,
                "reason": clarification.reason,
                "clarification_questions": clarification.clarification_questions,
                "suggested_options": clarification.suggested_options,
                "count": len(results),  # Include count for backward compatibility
                "results": [  # Include results for backward compatibility
                    {
                        "regulation_title": r.chunk.parent_path[0]
                        if r.chunk.parent_path
                        else r.chunk.title,
                        "title": r.chunk.title,
                        "text": r.chunk.text[:500] + "..."
                        if len(r.chunk.text) > 500
                        else r.chunk.text,
                        "rule_code": r.chunk.rule_code,
                        "parent_path": " > ".join(r.chunk.parent_path),
                        "score": round(r.score, 3),
                    }
                    for r in results
                ],
            }

        return {
            "count": len(results),
            "results": [
                {
                    "regulation_title": r.chunk.parent_path[0]
                    if r.chunk.parent_path
                    else r.chunk.title,
                    "title": r.chunk.title,
                    "text": r.chunk.text[:500] + "..."
                    if len(r.chunk.text) > 500
                    else r.chunk.text,
                    "rule_code": r.chunk.rule_code,
                    "parent_path": " > ".join(r.chunk.parent_path),
                    "score": round(r.score, 3),
                }
                for r in results
            ],
        }

    def _handle_get_article(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_article tool with direct article lookup."""
        regulation = args["regulation"]
        article_no = args["article_no"]

        # Use FullViewUseCase for direct article lookup
        from ..application.full_view_usecase import FullViewUseCase
        from ..infrastructure.json_loader import JSONDocumentLoader

        full_view_usecase = FullViewUseCase(JSONDocumentLoader())
        matches = full_view_usecase.find_matches(regulation)

        if not matches:
            # Fallback to search if no direct match
            query = f"{regulation} 제{article_no}조"
            return self._handle_search_regulations({"query": query, "top_k": 3})

        # Use first match (or refine later for multiple matches)
        selected = matches[0]
        article_node = full_view_usecase.get_article_view(
            selected.rule_code, article_no
        )

        if not article_node:
            return {
                "found": False,
                "regulation": regulation,
                "article_no": article_no,
                "error": f"{selected.title}에서 제{article_no}조를 찾을 수 없습니다.",
            }

        # Render article content
        from ..interface.formatters import render_full_view_nodes

        content = render_full_view_nodes([article_node])

        return {
            "found": True,
            "regulation": selected.title,
            "rule_code": selected.rule_code,
            "article_no": article_no,
            "title": article_node.get("title", f"제{article_no}조"),
            "content": content[:1500] + "..." if len(content) > 1500 else content,
        }

    def _handle_get_chapter(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_chapter tool."""
        regulation = args["regulation"]
        chapter_no = args["chapter_no"]
        query = f"{regulation} 제{chapter_no}장"
        return self._handle_search_regulations({"query": query, "top_k": 10})

    def _handle_get_attachment(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_attachment tool."""
        regulation = args["regulation"]
        label = args.get("label", "별표")
        query = f"{regulation} {label}"
        return self._handle_search_regulations({"query": query, "top_k": 5})

    def _handle_get_regulation_overview(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_regulation_overview tool."""
        regulation = args["regulation"]
        query = f"{regulation} 전체"
        results = self._handle_search_regulations({"query": query, "top_k": 1})
        # Extract overview info from first result
        return {
            "regulation": regulation,
            "overview": results.get("results", [{}])[0]
            if results.get("results")
            else None,
        }

    def _handle_get_full_regulation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_full_regulation tool."""
        regulation = args["regulation"]
        # Return indication that full regulation should be retrieved
        # Actual implementation would use FullViewUseCase
        return {
            "regulation": regulation,
            "note": "전문 조회 기능은 별도 구현 필요",
        }

    # ========== Analysis Tool Handlers ==========

    def _handle_expand_synonyms(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle expand_synonyms tool."""
        if not self._query_analyzer:
            raise RuntimeError("QueryAnalyzer not initialized")

        term = args["term"]
        # Access internal synonym dictionary
        synonyms = self._query_analyzer._synonyms.get(term, [])
        return {"term": term, "synonyms": synonyms}

    def _handle_detect_intent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle detect_intent tool."""
        if not self._query_analyzer:
            raise RuntimeError("QueryAnalyzer not initialized")

        query = args["query"]
        has_intent = self._query_analyzer.has_intent(query)
        # Get matched intents
        matches = self._query_analyzer._match_intents(query)
        return {
            "query": query,
            "has_intent": has_intent,
            "matched_intents": [
                {"id": m.intent_id, "label": m.label, "keywords": m.keywords}
                for m in matches
            ],
        }

    def _handle_detect_audience(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle detect_audience tool."""
        if not self._query_analyzer:
            raise RuntimeError("QueryAnalyzer not initialized")

        query = args["query"]
        audience = self._query_analyzer.detect_audience(query)
        candidates = self._query_analyzer.detect_audience_candidates(query)
        is_ambiguous = self._query_analyzer.is_audience_ambiguous(query)

        return {
            "query": query,
            "audience": audience.value,
            "candidates": [c.value for c in candidates],
            "is_ambiguous": is_ambiguous,
        }

    def _handle_analyze_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analyze_query tool."""
        if not self._query_analyzer:
            raise RuntimeError("QueryAnalyzer not initialized")

        query = args["query"]
        query_type = self._query_analyzer.analyze(query)
        weights = self._query_analyzer.get_weights(query)

        return {
            "query": query,
            "query_type": query_type.value,
            "bm25_weight": weights[0],
            "dense_weight": weights[1],
        }

    # ========== Admin Tool Handlers ==========

    def _handle_sync_database(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sync_database tool."""
        if not self._sync_usecase:
            raise RuntimeError("SyncUseCase not initialized")

        full = args.get("full", False)
        if full:
            result = self._sync_usecase.full_sync(self._json_path)
        else:
            result = self._sync_usecase.incremental_sync(self._json_path)

        return {
            "added": result.added,
            "modified": result.modified,
            "removed": result.removed,
            "unchanged": result.unchanged,
            "errors": result.errors,
        }

    def _handle_get_sync_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_sync_status tool."""
        if not self._sync_usecase:
            raise RuntimeError("SyncUseCase not initialized")

        return self._sync_usecase.get_sync_status()

    def _handle_reset_database(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reset_database tool."""
        if not self._sync_usecase:
            raise RuntimeError("SyncUseCase not initialized")

        self._sync_usecase.reset_state()
        return {"status": "reset_complete"}

    # ========== Response Tool Handlers ==========

    def _handle_generate_answer(self, args: Dict[str, Any]) -> str:
        """Handle generate_answer tool with quality enhancements."""
        if not self._llm_client:
            raise RuntimeError("LLM client not initialized")

        question = args["question"]
        context = args["context"]

        system_prompt = """당신은 동의대학교 규정을 설명하는 전문가입니다.
주어진 컨텍스트를 바탕으로 질문에 정확하고 친절하게 답변하세요.

📌 출처 표기 원칙 (필수):
1. 모든 조항 인용 시 반드시 "규정명 + 제N조"를 함께 명시하세요.
   - 좋은 예: "직원복무규정 제26조에 따르면...", "학칙 제15조 ②항에서는..."
   - 나쁜 예: "제26조에 따르면..." (규정명 누락 ❌)
2. 컨텍스트에 표시된 [규정명] 또는 regulation_title을 반드시 활용하세요.
3. 여러 규정을 인용할 경우, 각각 출처를 명시하세요.

⚠️ 절대 금지 사항:
1. 전화번호/연락처를 만들어내지 마세요 (예: 02-XXXX-XXXX 금지). 연락처는 "학교 홈페이지에서 확인하세요"라고 안내하세요.
2. 다른 학교(서울대, 연세대, 한국외대 등) 사례나 규정을 언급하지 마세요. 동의대학교 규정만 답변하세요.
3. 규정에 없는 숫자/비율/기한을 만들어내지 마세요.
4. "대학마다 다릅니다", "일반적으로" 같은 회피성 답변을 하지 마세요.
5. 컨텍스트에 없는 내용은 "해당 정보는 확인되지 않습니다"라고 솔직히 답변하세요."""

        user_message = f"""질문: {question}

컨텍스트:
{context}

위 컨텍스트를 바탕으로 질문에 답변해주세요."""

        answer = self._llm_client.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.0,
        )

        # Post-process answer with quality enhancements
        answer = self._apply_quality_enhancements(answer, context)

        return answer

    def _apply_quality_enhancements(self, answer: str, context: str) -> str:
        """
        Apply quality enhancements to the generated answer.

        Args:
            answer: The generated answer
            context: The context used for generation

        Returns:
            Enhanced answer
        """
        # Import here to avoid circular imports
        from ..automation.infrastructure.evaluation_helpers import HallucinationDetector

        # Check for hallucination (phone numbers, other universities, evasive responses)
        should_block, block_reason, issues = (
            HallucinationDetector.block_if_hallucination(answer)
        )

        if should_block:
            # Sanitize the answer instead of blocking completely
            sanitized, changes = HallucinationDetector.sanitize_answer(answer)
            if changes:
                # Log the changes for monitoring
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Answer sanitized: {changes}")
            return sanitized

        return answer

    def _handle_clarify_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clarify_query tool."""
        query = args["query"]
        options = args.get("options", [])

        return {
            "type": "clarification",
            "message": f"'{query}'에 대해 더 명확한 정보가 필요합니다. 다음 중 선택해주세요:",
            "options": options,
        }
