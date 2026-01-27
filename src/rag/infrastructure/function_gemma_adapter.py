"""
FunctionGemma Adapter for Regulation RAG System.

Handles communication with LLM models for tool calling.
Supports multiple modes:
1. MLX (Apple Silicon optimized) - mlx-lm package
2. OpenAI-compatible API (LM Studio, vLLM, etc.)
3. Native Ollama tool calling API
4. Text parsing fallback for basic LLM clients
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests

from .query_analyzer import Audience, QueryAnalyzer
from .tool_definitions import TOOL_DEFINITIONS, get_tools_prompt
from .tool_executor import ToolExecutor, ToolResult

logger = logging.getLogger(__name__)

# Try to import mlx-lm for Apple Silicon optimization
try:
    from mlx_lm import generate as mlx_generate
    from mlx_lm import load as mlx_load

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mlx_load = None
    mlx_generate = None

# Try to import ollama for native tool calling
try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None


@dataclass
class ToolCall:
    """Represents a tool call from the model."""

    name: str
    arguments: Dict[str, Any]


@dataclass
class FunctionGemmaResponse:
    """Response from FunctionGemma."""

    tool_calls: List[ToolCall] = field(default_factory=list)
    final_response: Optional[str] = None
    raw_output: str = ""


class FunctionGemmaAdapter:
    """
    Adapter for FunctionGemma function calling model.

    FunctionGemma uses specific control tokens for tool definitions and calls:
    - Tool definitions are provided in the prompt
    - Model outputs structured function calls
    - Results are fed back for final response generation
    """

    # FunctionGemma control tokens (based on official documentation)
    TOOL_START = "<tool_call>"
    TOOL_END = "</tool_call>"

    # System prompt for function calling
    SYSTEM_PROMPT = """당신은 동의대학교 규정 검색 시스템의 도구 라우터입니다.
사용자의 질문을 분석하고, 적절한 도구(tool)를 호출하여 답변에 필요한 정보를 수집하세요.

## 도구 호출 규칙
1. 먼저 search_regulations로 관련 정보를 검색하세요.
2. 검색 결과가 불충분하면 추가 도구를 호출하세요.
3. 충분한 정보가 모이면 generate_answer를 호출하여 최종 답변을 생성하세요.

## 도구 호출 형식
<tool_call>
{"name": "도구이름", "arguments": {"arg1": "값1", "arg2": "값2"}}
</tool_call>

## ⚠️ 중요 (절대 금지)
- 한 번에 하나의 도구만 호출하세요.
- 도구 결과를 받은 후 다음 행동을 결정하세요.
- 최종 답변 생성 시 반드시 generate_answer 도구를 사용하세요.
- 전화번호(02-XXXX-XXXX)를 만들어내지 마세요.
- 다른 학교 사례를 언급하지 마세요 (동의대학교 규정만 답변).
- 규정에 없는 숫자/기한을 생성하지 마세요.
"""

    # Class-level cache for MLX model to avoid reloading in Web UI
    _cached_mlx_model = None
    _cached_mlx_tokenizer = None

    def __init__(
        self,
        llm_client=None,
        tool_executor: Optional[ToolExecutor] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
        max_iterations: int = 3,  # Reduced from 5 to avoid excessive retries
        model: str = None,
        base_url: str = None,
        api_mode: str = "auto",
        mlx_model: str = "mlx-community/functiongemma-270m-it-4bit",
    ):
        """
        Initialize FunctionGemmaAdapter.

        Args:
            llm_client: LLM client for text parsing fallback mode.
            tool_executor: Executor for running tools.
            max_iterations: Maximum number of tool call iterations.
            model: Model name for API calls.
            base_url: Base URL for OpenAI-compatible API (e.g., http://localhost:1234).
            api_mode: API mode - "auto", "mlx", "openai", "ollama", or "text".
                - "auto": Try MLX first (on macOS), then OpenAI, then Ollama
                - "mlx": Use MLX (Apple Silicon optimized)
                - "openai": Use OpenAI-compatible API (LM Studio, vLLM, etc.)
                - "ollama": Use Ollama native API
                - "text": Use text parsing with provided llm_client
            mlx_model: Hugging Face model ID for MLX (default: functiongemma-270m-it-4bit)
        """
        self._llm_client = llm_client
        self._tool_executor = tool_executor
        self._query_analyzer = query_analyzer
        self._max_iterations = max_iterations

        # Load from environment if not provided
        self._model = model or os.getenv("LLM_MODEL", "functiongemma")
        raw_base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:1234")
        # Normalize base_url: remove trailing /v1 if present (we add it in API calls)
        self._base_url = raw_base_url.rstrip("/").removesuffix("/v1")
        self._mlx_model_id = mlx_model

        # MLX model/tokenizer (lazy loaded)
        self._mlx_model = None
        self._mlx_tokenizer = None

        # Determine API mode
        self._api_mode = self._resolve_api_mode(api_mode)
        logger.debug(
            f"FunctionGemmaAdapter initialized. tool_executor={self._tool_executor}, api_mode={self._api_mode}"
        )

    def _get_api_headers(self) -> dict:
        """Get headers for API calls, including Authorization if needed."""
        headers = {"Content-Type": "application/json"}

        # Add API key for cloud providers
        if "openrouter.ai" in self._base_url:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        elif "api.openai.com" in self._base_url:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        return headers

    def set_llm_client(self, llm_client) -> None:
        """Set the LLM client."""
        self._llm_client = llm_client

    def set_tool_executor(self, tool_executor: ToolExecutor) -> None:
        """Set the tool executor."""
        self._tool_executor = tool_executor

    def set_query_analyzer(self, query_analyzer: QueryAnalyzer) -> None:
        """Set the query analyzer."""
        self._query_analyzer = query_analyzer

    def _build_analysis_context(self, query: str) -> str:
        """Build context from QueryAnalyzer for LLM system prompt."""
        if not self._query_analyzer:
            return ""

        hints = []

        # Intent analysis
        intent_matches = self._query_analyzer._match_intents(query)
        if intent_matches:
            keywords = []
            for m in intent_matches[:2]:
                keywords.extend(m.keywords[:3])
            if keywords:
                hints.append(
                    f"[의도 분석] 사용자의 진짜 의도는 '{', '.join(keywords)}' 관련일 수 있습니다."
                )

        # Audience detection
        audience = self._query_analyzer.detect_audience(query)
        if audience != Audience.ALL:
            audience_map = {
                Audience.STUDENT: "학생",
                Audience.FACULTY: "교수/교원",
                Audience.STAFF: "직원",
            }
            hints.append(f"[대상] {audience_map.get(audience, audience.value)}")

        # Query expansion
        expanded = self._query_analyzer.expand_query(query)
        if expanded != query:
            hints.append(f"[검색 키워드] {expanded}")

        return "\n".join(hints)

    def _resolve_api_mode(self, mode: str) -> str:
        """Resolve API mode based on availability."""
        if mode == "mlx":
            return "mlx" if MLX_AVAILABLE else "openai"
        if mode == "openai":
            return "openai"
        if mode == "ollama":
            return "ollama" if OLLAMA_AVAILABLE else "text"
        if mode == "text":
            return "text"

        # Auto mode: try to detect best available

        # 0. Check for cloud providers (OpenRouter, OpenAI) - use openai mode
        if "openrouter.ai" in self._base_url or "api.openai.com" in self._base_url:
            return "openai"

        # 1. Prefer OpenAI-compatible server (LM Studio/vLLM) as it handles templates reliably
        try:
            resp = requests.get(f"{self._base_url}/v1/models", timeout=1)
            if resp.status_code == 200:
                return "openai"
        except Exception:
            pass

        # 2. Then MLX (if available) - Fast but experimental template support
        if MLX_AVAILABLE:
            return "mlx"

        # Then try Ollama
        if OLLAMA_AVAILABLE:
            try:
                ollama.list()
                return "ollama"
            except Exception:
                pass

        # Fallback to text parsing
        return "text"

    def _load_mlx_model(self):
        """Lazy load MLX model and tokenizer (with class-level caching)."""
        if not MLX_AVAILABLE:
            return

        if FunctionGemmaAdapter._cached_mlx_model is None:
            logger.info(f"Loading MLX model: {self._mlx_model_id}...")
            model, tokenizer = mlx_load(self._mlx_model_id)
            FunctionGemmaAdapter._cached_mlx_model = model
            FunctionGemmaAdapter._cached_mlx_tokenizer = tokenizer
            logger.info("MLX model loaded and cached")

        self._mlx_model = FunctionGemmaAdapter._cached_mlx_model
        self._mlx_tokenizer = FunctionGemmaAdapter._cached_mlx_tokenizer

    def _parse_functiongemma_response(self, response: str) -> List[ToolCall]:
        """Parse FunctionGemma's tool call format.

        FunctionGemma outputs: <start_function_call>call:func_name{arg:value}<end_function_call>
        """
        import re

        tool_calls = []

        # Match FunctionGemma format
        pattern = r"<start_function_call>call:(\w+)\{([^}]*)\}<end_function_call>"
        matches = re.findall(pattern, response)

        for name, args_str in matches:
            # Parse arguments (format: arg1:<escape>value1<escape>,arg2:<escape>value2<escape>)
            arguments = {}
            if args_str:
                # Split by comma, handle escaped values
                arg_pattern = r"(\w+):<escape>([^<]*)<escape>"
                arg_matches = re.findall(arg_pattern, args_str)
                for arg_name, arg_value in arg_matches:
                    arguments[arg_name] = arg_value

                # Also try simpler format: arg:value
                if not arguments:
                    simple_pattern = r"(\w+):([^,]+)"
                    simple_matches = re.findall(simple_pattern, args_str)
                    for arg_name, arg_value in simple_matches:
                        arguments[arg_name.strip()] = arg_value.strip().strip("\"'")

            tool_calls.append(ToolCall(name=name, arguments=arguments))

        return tool_calls

    def process_query_mlx(
        self, query: str, context: Optional[str] = None, llm_client=None
    ) -> Tuple[str, List[ToolResult]]:
        """
        Process query using MLX FunctionGemma for tool calling.

        FunctionGemma handles tool selection, base LLM generates final answer.

        Args:
            query: User question
            context: Optional additional context
            llm_client: Base LLM client for answer generation (uses env settings if None)
        """
        import json

        if not MLX_AVAILABLE:
            raise RuntimeError("MLX not available")
        if not self._tool_executor:
            raise RuntimeError("Tool executor not initialized")

        self._load_mlx_model()

        tool_results: List[ToolResult] = []

        # Prepare content
        tools_json = json.dumps(TOOL_DEFINITIONS, indent=None)
        system_content = f"You are a helpful assistant with access to the following functions. Use them if required - \n{tools_json}"

        user_content = query
        if context:
            user_content = f"Context:\n{context}\n\nQuestion:\n{query}"

        # Use tokenizer's chat template to generate correct prompt structure
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        try:
            prompt = self._mlx_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            # Fallback for models or tokenizers without proper template
            logger.warning(f"Failed to apply chat template ({e}), using manual format.")
            prompt = f"<start_of_turn>system\n{system_content}<end_of_turn>\n<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"

        for _iteration in range(self._max_iterations):
            # Generate response
            response = mlx_generate(
                self._mlx_model,
                self._mlx_tokenizer,
                prompt=prompt,
                max_tokens=512,
                verbose=False,
            )

            # Parse FunctionGemma format tool calls
            parsed_calls = self._parse_functiongemma_response(response)

            if not parsed_calls:
                # No tool calls
                return response, tool_results

            # Execute tool calls
            for tool_call in parsed_calls:
                # Handle generate_answer with base LLM
                if tool_call.name == "generate_answer":
                    context_text = tool_call.arguments.get("context", "")
                    question = tool_call.arguments.get("question", query)

                    # Use base LLM for answer generation
                    answer = self._generate_answer_with_base_llm(
                        question, context_text, llm_client
                    )
                    tool_results.append(
                        ToolResult(
                            tool_name="generate_answer",
                            success=True,
                            result=answer,
                        )
                    )
                    return answer, tool_results

                # Execute other tools
                result = self._tool_executor.execute(
                    tool_call.name, tool_call.arguments
                )
                tool_results.append(result)

                # Update prompt for next turn (Multi-turn)
                # Append model output (tool call)
                prompt += f"{response}<end_of_turn>\n"
                # Append tool result (simulate 'tool' role if model supported it, but FunctionGemma might expect user role for observation)
                # Standard FunctionGemma pattern for results isn't clearly documented for multi-turn in the same chat structure,
                # but we can try providing it as a user message or continuing the turn.
                # However, for simplicity and typical usage, we usually just append the result.
                # Let's try appending as a separated user block which serves as 'observation'.
                prompt += f"<start_of_turn>user\nObservation: {result.to_context_string()}<end_of_turn>\n<start_of_turn>model\n"

        # Max iterations reached - generate answer with accumulated results
        if tool_results:
            context_text = "\n\n".join([r.to_context_string() for r in tool_results])
            answer = self._generate_answer_with_base_llm(
                query, context_text, llm_client
            )
            return answer, tool_results

        return "검색 반복 횟수를 초과했습니다.", tool_results

    # --- Phase 2: Handle Insufficient Results ---

    def _handle_insufficient_results(
        self, query: str, tool_results: list, llm_client=None
    ) -> str:
        """
        Handle cases where search results are insufficient.

        Instead of retrying search indefinitely, generate answer from available
        partial results or provide a helpful response.

        Args:
            query: Original user query.
            tool_results: List of ToolResult from previous searches.
            llm_client: Optional LLM client for answer generation.

        Returns:
            Generated answer or informative message.
        """
        # No results at all
        if not tool_results:
            return (
                "관련 규정을 찾지 못했습니다. "
                "질문을 더 구체적으로 해주시거나 다른 키워드로 검색해 보세요."
            )

        # Filter successful search results
        search_results = [
            r for r in tool_results if r.success and r.tool_name == "search_regulations"
        ]

        if not search_results:
            return (
                "검색을 수행했으나 관련 규정을 찾지 못했습니다. "
                "다른 검색어로 시도해 보세요."
            )

        # Generate answer from partial results
        context_text = "\n\n".join([r.to_context_string() for r in search_results])
        logger.info(
            f"Generating answer from {len(search_results)} partial search results"
        )

        return self._generate_answer_with_base_llm(query, context_text, llm_client)

    def _generate_answer_with_base_llm(
        self, question: str, context: str, llm_client=None
    ) -> str:
        """Generate answer using base LLM (not FunctionGemma)."""
        anti_hallucination = """\n\n📌 출처 표기 원칙 (필수):
1. 모든 조항 인용 시 반드시 "규정명 + 제N조"를 함께 명시하세요.
   - 좋은 예: "직원복무규정 제26조에 따르면...", "학칙 제15조 ②항에서는..."
   - 나쁜 예: "제26조에 따르면..." (규정명 누락 ❌)
2. 컨텍스트에 표시된 [규정명] 또는 regulation_title을 반드시 활용하세요.

⚠️ 답변 원칙 (반드시 준수):
1. 컨텍스트에 있는 구체적인 수치(평점평균, 학점, 기간 등)를 그대로 인용하여 답변하세요.
2. 컨텍스트에 명시된 내용 외의 정보를 추측하거나 생성하지 마세요.
3. 절대 금지: 전화번호 생성, 다른 학교 사례 언급, 규정에 없는 숫자/등급(C-, B+ 등) 생성.
4. 정말 관련 정보가 없을 때만 "확인되지 않습니다"라고 답변하세요."""

        # Try provided client first
        if llm_client:
            return llm_client.generate(
                system_prompt="당신은 동의대학교 규정을 설명하는 전문가입니다. 컨텍스트를 바탕으로 질문에 답변하세요."
                + anti_hallucination,
                user_message=f"질문: {question}\n\n컨텍스트:\n{context}",
                temperature=0.0,
            )

        # Fall back to OpenAI-compatible API
        try:
            payload = {
                "model": os.getenv(
                    "LLM_MODEL", "eeve-korean-instruct-7b-v2.0-preview-mlx"
                ),
                "messages": [
                    {
                        "role": "system",
                        "content": "당신은 동의대학교 규정을 설명하는 전문가입니다."
                        + anti_hallucination,
                    },
                    {
                        "role": "user",
                        "content": f"질문: {question}\n\n컨텍스트:\n{context}\n\n답변:",
                    },
                ],
                "temperature": 0,
            }
            resp = requests.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                headers=self._get_api_headers(),
                timeout=120,
            )
            resp.raise_for_status()
            return (
                resp.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        except Exception as e:
            logger.warning(f"Answer generation failed: {e}")
            # Return a summary of available context when LLM fails
            if context and len(context) > 100:
                return f"관련 규정을 찾았습니다. 아래 내용을 참고하세요:\\n\\n{context[:2000]}..."
            return f"답변 생성 실패: {e}"

    def process_query_openai(
        self, query: str, context: Optional[str] = None, llm_client=None
    ) -> Tuple[str, List[ToolResult]]:
        """
        Process query using OpenAI-compatible API (LM Studio, vLLM, etc.).

        Uses the /v1/chat/completions endpoint with tools parameter.
        """
        if not self._tool_executor:
            raise RuntimeError("Tool executor not initialized")

        tool_results: List[ToolResult] = []

        # Build intent analysis context
        analysis_context = self._build_analysis_context(query)

        # System prompt for guiding tool usage with intent hints
        system_prompt = f"""당신은 동의대학교 규정 전문가입니다. 사용자의 질문에 답하기 위해 제공된 도구를 사용하세요.

{analysis_context}

작업 순서:
1. 위 분석 결과를 참고하여 search_regulations 도구로 관련 규정을 검색합니다.
   - [검색 키워드]가 제공된 경우, 해당 키워드를 query에 포함하세요.
   - 예: 사용자가 "학교 가기 싫어"라고 해도 [검색 키워드]가 "휴직 휴가 연구년"이면 해당 키워드로 검색하세요.
2. 검색 결과를 바탕으로 generate_answer 도구를 호출하여 최종 답변을 생성합니다.

⚠️ 답변 원칙 (중요):
1. 검색 결과에 관련 정보가 있으면 반드시 구체적으로 답변하세요 (조항 번호, 기준, 기간 포함).
2. 검색 결과를 받은 후에는 반드시 generate_answer를 호출하여 친절하고 구체적인 답변을 제공하세요.
3. 절대 금지:
   - 전화번호(02-XXXX-XXXX) 생성 → "학교 홈페이지에서 확인하세요"로 대체
   - 다른 학교 사례 언급 → 동의대학교 규정만 답변
   - 규정에 없는 숫자/비율/기한 생성
   - "대학마다 다릅니다", "일반적으로" 같은 회피성 표현
4. 정말로 관련 규정이 검색되지 않았을 때만 "확인되지 않습니다"라고 답변하세요."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        if context:
            messages[1]["content"] = f"{context}\n\n질문: {query}"

        for _iteration in range(self._max_iterations):
            payload = {
                "model": self._model,
                "messages": messages,
                "tools": TOOL_DEFINITIONS,
                "tool_choice": "auto",
                "temperature": 0,
            }

            try:
                resp = requests.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                    headers=self._get_api_headers(),
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                return f"API 오류: {str(e)}", tool_results

            message = data.get("choices", [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls", [])

            if tool_calls:
                # Process tool calls
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")

                    # Parse arguments (may be string or dict)
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}

                    # Special handling for generate_answer
                    if tool_name == "generate_answer":
                        # If llm_client is provided, use it for answer generation
                        if llm_client:
                            context_text = args.get("context", "")
                            question = args.get("question", query)
                            answer = self._generate_answer_with_base_llm(
                                question, context_text, llm_client
                            )
                            tool_results.append(
                                ToolResult(
                                    tool_name="generate_answer",
                                    success=True,
                                    result=answer,
                                    arguments=args,
                                )
                            )
                            return answer, tool_results

                        # Try to execute with LLM client (legacy fallback)
                        result = self._tool_executor.execute(tool_name, args)
                        tool_results.append(result)

                        if result.success:
                            return result.result, tool_results

                        # If generate_answer fails (no LLM client),
                        # ask the model to generate answer directly
                        context = args.get("context", "")
                        question = args.get("question", query)

                        # Make a simple completion request (no tools)
                        answer_system = """당신은 동의대학교 규정을 설명하는 전문가입니다.

📌 출처 표기 원칙 (필수):
1. 모든 조항 인용 시 반드시 "「규정명」 제N조"를 함께 명시하세요.
   - 좋은 예: "「직원복무규정」 제26조에 따르면..."
   - 나쁜 예: "제26조에 따르면..." (규정명 누락 ❌)
2. 컨텍스트의 regulation_title 또는 parent_path에서 규정명을 확인하세요.

⚠️ 절대 금지: 전화번호 생성, 다른 학교 사례 언급, 규정에 없는 숫자 생성.
정말 관련 정보가 없을 때만 '확인되지 않습니다'라고 답변."""
                        answer_payload = {
                            "model": self._model,
                            "messages": [
                                {"role": "system", "content": answer_system},
                                {
                                    "role": "user",
                                    "content": f"질문: {question}\n\n컨텍스트:\n{context}\n\n위 컨텍스트를 바탕으로 답변해주세요.",
                                },
                            ],
                            "temperature": 0,
                        }

                        try:
                            answer_resp = requests.post(
                                f"{self._base_url}/v1/chat/completions",
                                json=answer_payload,
                                headers=self._get_api_headers(),
                                timeout=120,
                            )
                            answer_resp.raise_for_status()
                            answer_data = answer_resp.json()
                            answer_content = (
                                answer_data.get("choices", [{}])[0]
                                .get("message", {})
                                .get("content", "")
                            )
                            return (
                                answer_content
                                if answer_content
                                else "답변을 생성하지 못했습니다.",
                                tool_results,
                            )
                        except Exception:
                            pass

                        return "답변 생성에 실패했습니다.", tool_results

                    # Normal tool execution
                    result = self._tool_executor.execute(tool_name, args)
                    tool_results.append(result)

                    # Add assistant message and tool response
                    messages.append(message)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": result.to_context_string(),
                        }
                    )
            else:
                # No tool calls, return the response content
                content = message.get("content", "")
                return (
                    content if content else "응답을 생성하지 못했습니다.",
                    tool_results,
                )

        # Max iterations reached - try to generate answer from accumulated results
        return self._handle_insufficient_results(query, tool_results, llm_client), tool_results

    def _get_ollama_tools(self) -> List[callable]:
        """Convert tool definitions to Ollama-compatible function list."""

        # Ollama expects actual Python functions for tools
        # We'll create wrapper functions that call our tool executor
        def make_tool_function(tool_name: str):
            def tool_func(**kwargs):
                if self._tool_executor:
                    result = self._tool_executor.execute(tool_name, kwargs)
                    return result.result if result.success else f"Error: {result.error}"
                return "Error: Tool executor not available"

            # Copy docstring and annotations from tool definition
            tool_def = next(
                (t for t in TOOL_DEFINITIONS if t["function"]["name"] == tool_name),
                None,
            )
            if tool_def:
                tool_func.__doc__ = tool_def["function"]["description"]
                tool_func.__name__ = tool_name
            return tool_func

        return [make_tool_function(t["function"]["name"]) for t in TOOL_DEFINITIONS]

    def process_query_native(
        self, query: str, context: Optional[str] = None
    ) -> Tuple[str, List[ToolResult]]:
        """
        Process query using Ollama native tool calling API.

        This method uses the `ollama.chat()` function with `tools` parameter.
        """
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama package not installed")
        if not self._tool_executor:
            raise RuntimeError("Tool executor not initialized")

        tool_results: List[ToolResult] = []
        messages = [{"role": "user", "content": query}]

        if context:
            messages[0]["content"] = f"{context}\n\n질문: {query}"

        # Get tool functions for Ollama
        tools = self._get_ollama_tools()

        for _iteration in range(self._max_iterations):
            response = ollama.chat(
                model=self._model,
                messages=messages,
                tools=tools,
            )

            if response.message.tool_calls:
                # Execute tool calls
                for tool_call in response.message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments

                    result = self._tool_executor.execute(tool_name, tool_args)
                    tool_results.append(result)

                    # Check if this is generate_answer (final step)
                    if tool_name == "generate_answer" and result.success:
                        return result.result, tool_results

                    # Add tool response to messages
                    messages.append(response.message)
                    messages.append(
                        {
                            "role": "tool",
                            "content": result.to_context_string(),
                        }
                    )
            else:
                # No tool calls, return the response
                return response.message.content, tool_results

        # Max iterations reached - try to generate answer from accumulated results
        return self._handle_insufficient_results(query, tool_results), tool_results

    def process_query(
        self, query: str, context: Optional[str] = None, llm_client=None
    ) -> Tuple[str, List[ToolResult]]:
        """
        Process a user query using tool calling.

        Automatically selects the best available API mode.

        Args:
            query: User's question.
            context: Optional additional context.
            llm_client: Optional base LLM client for answer generation.

        Returns:
            Tuple of (final_answer, list_of_tool_results)
        """
        # Route to appropriate API based on mode
        if self._api_mode == "mlx":
            return self.process_query_mlx(query, context, llm_client=llm_client)
        if self._api_mode == "openai":
            return self.process_query_openai(query, context, llm_client=llm_client)
        if self._api_mode == "ollama":
            return self.process_query_native(query, context)

        # Fallback to text parsing mode
        if not self._llm_client and not llm_client:
            raise RuntimeError("LLM client not initialized")

        # Use provided client or fallback to internal client
        client = llm_client or self._llm_client
        if not client:
            raise RuntimeError("LLM client not initialized")

        if not self._tool_executor:
            raise RuntimeError("Tool executor not initialized")

        tool_results: List[ToolResult] = []
        conversation = self._build_initial_prompt(query, context)

        for _iteration in range(self._max_iterations):
            # Get model response
            response = client.generate(
                system_prompt=self.SYSTEM_PROMPT,
                user_message=conversation,
                temperature=0.0,
            )

            # Parse tool calls
            parsed = self._parse_response(response)

            if not parsed.tool_calls:
                # No tool calls, return the response as final answer
                return parsed.final_response or response, tool_results

            # Execute tool calls
            for tool_call in parsed.tool_calls:
                result = self._tool_executor.execute(
                    tool_call.name, tool_call.arguments
                )
                tool_results.append(result)

                # Check if this is generate_answer (final step)
                if tool_call.name == "generate_answer" and result.success:
                    return result.result, tool_results

            # Add tool results to conversation for next iteration
            conversation = self._append_tool_results(
                conversation, tool_results[-len(parsed.tool_calls) :]
            )

        # Max iterations reached - try to generate answer from accumulated results
        return self._handle_insufficient_results(query, tool_results), tool_results

    def process_query_stream(
        self, query: str, context: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream process a user query, yielding events.

        Yields:
            Dict with event type and content:
            - {"type": "tool_call", "tool": name, "args": arguments}
            - {"type": "tool_result", "result": ToolResult}
            - {"type": "progress", "message": str}
            - {"type": "answer", "content": str}
            - {"type": "error", "message": str}
        """
        if not self._llm_client:
            yield {"type": "error", "message": "LLM client not initialized"}
            return
        if not self._tool_executor:
            yield {"type": "error", "message": "Tool executor not initialized"}
            return

        tool_results: List[ToolResult] = []
        conversation = self._build_initial_prompt(query, context)

        for iteration in range(self._max_iterations):
            yield {
                "type": "progress",
                "message": f"분석 중... (반복 {iteration + 1}/{self._max_iterations})",
            }

            # Get model response
            response = self._llm_client.generate(
                system_prompt=self.SYSTEM_PROMPT,
                user_message=conversation,
                temperature=0.0,
            )

            # Parse tool calls
            parsed = self._parse_response(response)

            if not parsed.tool_calls:
                yield {"type": "answer", "content": parsed.final_response or response}
                return

            # Execute tool calls
            for tool_call in parsed.tool_calls:
                yield {
                    "type": "tool_call",
                    "tool": tool_call.name,
                    "args": tool_call.arguments,
                }

                result = self._tool_executor.execute(
                    tool_call.name, tool_call.arguments
                )
                tool_results.append(result)

                yield {"type": "tool_result", "result": result.to_dict()}

                # Check if this is generate_answer (final step)
                if tool_call.name == "generate_answer" and result.success:
                    yield {"type": "answer", "content": result.result}
                    return

            # Add tool results to conversation for next iteration
            conversation = self._append_tool_results(
                conversation, tool_results[-len(parsed.tool_calls) :]
            )

        yield {"type": "error", "message": "검색 반복 횟수를 초과했습니다."}

    def _build_initial_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Build the initial prompt with tools and query."""
        tools_desc = get_tools_prompt()

        prompt_parts = [tools_desc, "", f"사용자 질문: {query}"]

        if context:
            prompt_parts.insert(2, f"추가 컨텍스트: {context}")

        return "\n".join(prompt_parts)

    def _append_tool_results(self, conversation: str, results: List[ToolResult]) -> str:
        """Append tool results to conversation."""
        result_texts = []
        for result in results:
            result_texts.append(f"도구 결과 ({result.tool_name}):")
            result_texts.append(result.to_context_string())

        return conversation + "\n\n" + "\n".join(result_texts)

    def _parse_response(self, response: str) -> FunctionGemmaResponse:
        """
        Parse FunctionGemma response to extract tool calls.

        Handles both structured tool calls and natural text responses.
        """
        tool_calls: List[ToolCall] = []

        # Try to find tool calls in the response
        # Pattern 1: <tool_call>JSON</tool_call>
        pattern1 = re.compile(
            rf"{re.escape(self.TOOL_START)}(.*?){re.escape(self.TOOL_END)}",
            re.DOTALL,
        )
        matches = pattern1.findall(response)

        for match in matches:
            try:
                call_data = json.loads(match.strip())
                if "name" in call_data:
                    tool_calls.append(
                        ToolCall(
                            name=call_data["name"],
                            arguments=call_data.get("arguments", {}),
                        )
                    )
            except json.JSONDecodeError:
                continue

        # Pattern 2: {"name": "...", "arguments": {...}} without tags
        if not tool_calls:
            pattern2 = re.compile(
                r'\{\s*"name"\s*:\s*"(\w+)".*?"arguments"\s*:\s*(\{[^}]+\})', re.DOTALL
            )
            matches2 = pattern2.findall(response)
            for name, args_str in matches2:
                try:
                    args = json.loads(args_str)
                    tool_calls.append(ToolCall(name=name, arguments=args))
                except json.JSONDecodeError:
                    continue

        # Remove tool call parts from response to get final text
        final_response = response
        for match in pattern1.findall(response):
            final_response = final_response.replace(
                f"{self.TOOL_START}{match}{self.TOOL_END}", ""
            )
        final_response = final_response.strip()

        return FunctionGemmaResponse(
            tool_calls=tool_calls,
            final_response=final_response if final_response else None,
            raw_output=response,
        )
