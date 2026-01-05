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
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests

from .tool_definitions import TOOL_DEFINITIONS, get_tools_prompt
from .tool_executor import ToolExecutor, ToolResult

# Try to import mlx-lm for Apple Silicon optimization
try:
    from mlx_lm import load as mlx_load, generate as mlx_generate
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
    SYSTEM_PROMPT = """당신은 대학 규정 검색 시스템의 도구 라우터입니다.
사용자의 질문을 분석하고, 적절한 도구(tool)를 호출하여 답변에 필요한 정보를 수집하세요.

## 도구 호출 규칙
1. 먼저 search_regulations로 관련 정보를 검색하세요.
2. 검색 결과가 불충분하면 추가 도구를 호출하세요.
3. 충분한 정보가 모이면 generate_answer를 호출하여 최종 답변을 생성하세요.

## 도구 호출 형식
<tool_call>
{"name": "도구이름", "arguments": {"arg1": "값1", "arg2": "값2"}}
</tool_call>

## 중요
- 한 번에 하나의 도구만 호출하세요.
- 도구 결과를 받은 후 다음 행동을 결정하세요.
- 최종 답변 생성 시 반드시 generate_answer 도구를 사용하세요.
"""

    def __init__(
        self,
        llm_client=None,
        tool_executor: Optional[ToolExecutor] = None,
        max_iterations: int = 5,
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
        self._max_iterations = max_iterations
        
        # Load from environment if not provided
        self._model = model or os.getenv("LLM_MODEL", "functiongemma")
        self._base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:1234")
        self._mlx_model_id = mlx_model
        
        # MLX model/tokenizer (lazy loaded)
        self._mlx_model = None
        self._mlx_tokenizer = None
        
        # Determine API mode
        self._api_mode = self._resolve_api_mode(api_mode)

    def set_llm_client(self, llm_client) -> None:
        """Set the LLM client."""
        self._llm_client = llm_client

    def set_tool_executor(self, tool_executor: ToolExecutor) -> None:
        """Set the tool executor."""
        self._tool_executor = tool_executor

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
        # On macOS with Apple Silicon, prefer MLX for speed
        if MLX_AVAILABLE:
            return "mlx"
        
        # Then check if OpenAI-compatible server is reachable
        try:
            resp = requests.get(f"{self._base_url}/v1/models", timeout=2)
            if resp.status_code == 200:
                return "openai"
        except Exception:
            pass
        
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
        """Lazy load MLX model and tokenizer."""
        if self._mlx_model is None and MLX_AVAILABLE:
            print(f"Loading MLX model: {self._mlx_model_id}...")
            self._mlx_model, self._mlx_tokenizer = mlx_load(self._mlx_model_id)
            print("MLX model loaded!")
    
    def process_query_mlx(
        self, query: str, context: Optional[str] = None
    ) -> Tuple[str, List[ToolResult]]:
        """
        Process query using MLX (Apple Silicon optimized).
        
        Uses mlx-lm for fast local inference on M-series chips.
        """
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX not available")
        if not self._tool_executor:
            raise RuntimeError("Tool executor not initialized")
        
        self._load_mlx_model()
        
        tool_results: List[ToolResult] = []
        tools_prompt = get_tools_prompt()
        
        # Build initial prompt
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"{tools_prompt}\n\n사용자 질문: {query}"}
        ]
        
        if context:
            messages[1]["content"] = f"{tools_prompt}\n\n추가 컨텍스트: {context}\n\n사용자 질문: {query}"
        
        for iteration in range(self._max_iterations):
            # Apply chat template
            prompt = self._mlx_tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            
            # Generate response
            response = mlx_generate(
                self._mlx_model,
                self._mlx_tokenizer,
                prompt=prompt,
                max_tokens=1024,
                verbose=False,
            )
            
            # Parse tool calls from response
            parsed = self._parse_response(response)
            
            if not parsed.tool_calls:
                # No tool calls, return the response
                return parsed.final_response or response, tool_results
            
            # Execute tool calls
            for tool_call in parsed.tool_calls:
                # Special handling for generate_answer
                if tool_call.name == "generate_answer":
                    result = self._tool_executor.execute(tool_call.name, tool_call.arguments)
                    tool_results.append(result)
                    
                    if result.success:
                        return result.result, tool_results
                    
                    # Generate answer directly using MLX
                    context_text = tool_call.arguments.get("context", "")
                    question = tool_call.arguments.get("question", query)
                    
                    answer_messages = [
                        {"role": "system", "content": "당신은 대학 규정을 설명하는 전문가입니다."},
                        {"role": "user", "content": f"질문: {question}\n\n컨텍스트:\n{context_text}\n\n답변:"}
                    ]
                    answer_prompt = self._mlx_tokenizer.apply_chat_template(
                        answer_messages, add_generation_prompt=True
                    )
                    answer = mlx_generate(
                        self._mlx_model,
                        self._mlx_tokenizer,
                        prompt=answer_prompt,
                        max_tokens=2048,
                        verbose=False,
                    )
                    return answer, tool_results
                
                # Normal tool execution
                result = self._tool_executor.execute(tool_call.name, tool_call.arguments)
                tool_results.append(result)
                
                # Add tool result to messages
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"도구 결과 ({result.tool_name}):\n{result.to_context_string()}"})
        
        return "검색 반복 횟수를 초과했습니다.", tool_results

    def process_query_openai(
        self, query: str, context: Optional[str] = None
    ) -> Tuple[str, List[ToolResult]]:
        """
        Process query using OpenAI-compatible API (LM Studio, vLLM, etc.).
        
        Uses the /v1/chat/completions endpoint with tools parameter.
        """
        if not self._tool_executor:
            raise RuntimeError("Tool executor not initialized")
        
        tool_results: List[ToolResult] = []
        
        # System prompt for guiding tool usage
        system_prompt = """당신은 대학 규정 전문가입니다. 사용자의 질문에 답하기 위해 제공된 도구를 사용하세요.

작업 순서:
1. search_regulations 도구로 관련 규정을 검색합니다.
2. 검색 결과를 바탕으로 generate_answer 도구를 호출하여 최종 답변을 생성합니다.

중요: 검색 결과를 받은 후에는 반드시 generate_answer를 호출하여 사용자에게 친절한 답변을 제공하세요."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        if context:
            messages[1]["content"] = f"{context}\n\n질문: {query}"
        
        for iteration in range(self._max_iterations):
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
                    headers={"Content-Type": "application/json"},
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
                        # Try to execute with LLM client
                        result = self._tool_executor.execute(tool_name, args)
                        tool_results.append(result)
                        
                        if result.success:
                            return result.result, tool_results
                        
                        # If generate_answer fails (no LLM client), 
                        # ask the model to generate answer directly
                        context = args.get("context", "")
                        question = args.get("question", query)
                        
                        # Make a simple completion request (no tools)
                        answer_payload = {
                            "model": self._model,
                            "messages": [
                                {"role": "system", "content": "당신은 대학 규정을 설명하는 전문가입니다. 주어진 컨텍스트를 바탕으로 질문에 정확하고 친절하게 답변하세요."},
                                {"role": "user", "content": f"질문: {question}\n\n컨텍스트:\n{context}\n\n위 컨텍스트를 바탕으로 답변해주세요."}
                            ],
                            "temperature": 0,
                        }
                        
                        try:
                            answer_resp = requests.post(
                                f"{self._base_url}/v1/chat/completions",
                                json=answer_payload,
                                headers={"Content-Type": "application/json"},
                                timeout=120,
                            )
                            answer_resp.raise_for_status()
                            answer_data = answer_resp.json()
                            answer_content = answer_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                            return answer_content if answer_content else "답변을 생성하지 못했습니다.", tool_results
                        except Exception:
                            pass
                        
                        return "답변 생성에 실패했습니다.", tool_results
                    
                    # Normal tool execution
                    result = self._tool_executor.execute(tool_name, args)
                    tool_results.append(result)
                    
                    # Add assistant message and tool response
                    messages.append(message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": result.to_context_string(),
                    })
            else:
                # No tool calls, return the response content
                content = message.get("content", "")
                return content if content else "응답을 생성하지 못했습니다.", tool_results
        
        return "검색 반복 횟수를 초과했습니다.", tool_results


    def _get_ollama_tools(self) -> List[callable]:
        """Convert tool definitions to Ollama-compatible function list."""
        # Ollama expects actual Python functions for tools
        # We'll create wrapper functions that call our tool executor
        def make_tool_function(tool_name: str):
            def tool_func(**kwargs):
                if self._tool_executor:
                    result = self._tool_executor.execute(tool_name, kwargs)
                    return result.result if result.success else f"Error: {result.error}"
                return f"Error: Tool executor not available"
            # Copy docstring and annotations from tool definition
            tool_def = next((t for t in TOOL_DEFINITIONS if t["function"]["name"] == tool_name), None)
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
        
        for iteration in range(self._max_iterations):
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
                    messages.append({
                        "role": "tool",
                        "content": result.to_context_string(),
                    })
            else:
                # No tool calls, return the response
                return response.message.content, tool_results
        
        return "검색 반복 횟수를 초과했습니다.", tool_results


    def process_query(
        self, query: str, context: Optional[str] = None
    ) -> Tuple[str, List[ToolResult]]:
        """
        Process a user query using tool calling.

        Automatically selects the best available API mode.

        Args:
            query: User's question.
            context: Optional additional context.

        Returns:
            Tuple of (final_answer, list_of_tool_results)
        """
        # Route to appropriate API based on mode
        if self._api_mode == "mlx":
            return self.process_query_mlx(query, context)
        if self._api_mode == "openai":
            return self.process_query_openai(query, context)
        if self._api_mode == "ollama":
            return self.process_query_native(query, context)
        
        # Fallback to text parsing mode
        if not self._llm_client:
            raise RuntimeError("LLM client not initialized")
        if not self._tool_executor:
            raise RuntimeError("Tool executor not initialized")

        tool_results: List[ToolResult] = []
        conversation = self._build_initial_prompt(query, context)

        for iteration in range(self._max_iterations):
            # Get model response
            response = self._llm_client.generate(
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
            conversation = self._append_tool_results(conversation, tool_results[-len(parsed.tool_calls):])

        # Max iterations reached
        return "검색 반복 횟수를 초과했습니다. 질문을 더 구체적으로 해주세요.", tool_results

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
            yield {"type": "progress", "message": f"분석 중... (반복 {iteration + 1}/{self._max_iterations})"}

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
                conversation, tool_results[-len(parsed.tool_calls):]
            )

        yield {"type": "error", "message": "검색 반복 횟수를 초과했습니다."}

    def _build_initial_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Build the initial prompt with tools and query."""
        tools_desc = get_tools_prompt()
        
        prompt_parts = [tools_desc, "", f"사용자 질문: {query}"]
        
        if context:
            prompt_parts.insert(2, f"추가 컨텍스트: {context}")

        return "\n".join(prompt_parts)

    def _append_tool_results(
        self, conversation: str, results: List[ToolResult]
    ) -> str:
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
            pattern2 = re.compile(r'\{\s*"name"\s*:\s*"(\w+)".*?"arguments"\s*:\s*(\{[^}]+\})', re.DOTALL)
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
