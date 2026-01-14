"""
Tool Definitions for FunctionGemma.

Defines JSON Schema for all tools that FunctionGemma can call.
Based on the OpenAI function calling format for compatibility.
"""

from typing import Any, Dict, List

# Tool schemas in OpenAI-compatible format
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    # ========== 검색 도구 (Search Tools) ==========
    {
        "type": "function",
        "function": {
            "name": "search_regulations",
            "description": "규정 검색. 자연어 질문이나 키워드로 관련 규정 조항을 찾습니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색 쿼리 (자연어 또는 키워드)",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "반환할 결과 수 (기본값: 5)",
                        "default": 5,
                    },
                    "audience": {
                        "type": "string",
                        "enum": ["all", "student", "faculty", "staff"],
                        "description": "대상 (학생/교원/직원/전체)",
                        "default": "all",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_article",
            "description": "특정 규정의 특정 조항을 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "regulation": {
                        "type": "string",
                        "description": "규정 이름 (예: 교원인사규정, 학칙)",
                    },
                    "article_no": {
                        "type": "integer",
                        "description": "조항 번호 (예: 15)",
                    },
                },
                "required": ["regulation", "article_no"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chapter",
            "description": "특정 규정의 특정 장(章)을 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "regulation": {
                        "type": "string",
                        "description": "규정 이름",
                    },
                    "chapter_no": {
                        "type": "integer",
                        "description": "장 번호",
                    },
                },
                "required": ["regulation", "chapter_no"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_attachment",
            "description": "규정의 별표, 별지, 별첨 등 첨부자료를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "regulation": {
                        "type": "string",
                        "description": "규정 이름",
                    },
                    "label": {
                        "type": "string",
                        "description": "첨부자료 라벨 (예: 별표1, 별지제1호서식)",
                    },
                    "table_no": {
                        "type": "integer",
                        "description": "표 번호 (라벨 대신 사용 가능)",
                    },
                },
                "required": ["regulation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_regulation_overview",
            "description": "규정의 개요, 목차, 구조를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "regulation": {
                        "type": "string",
                        "description": "규정 이름",
                    },
                },
                "required": ["regulation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_full_regulation",
            "description": "규정 전문(全文)을 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "regulation": {
                        "type": "string",
                        "description": "규정 이름",
                    },
                },
                "required": ["regulation"],
            },
        },
    },
    # ========== 분석 도구 (Analysis Tools) ==========
    {
        "type": "function",
        "function": {
            "name": "expand_synonyms",
            "description": "용어의 동의어를 확장합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "확장할 용어",
                    },
                },
                "required": ["term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_intent",
            "description": "사용자 질문의 의도를 분석합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "분석할 질문",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_audience",
            "description": "질문의 대상(학생/교원/직원)을 감지합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "분석할 질문",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_query",
            "description": "쿼리 유형을 분석합니다 (조항참조/규정명/자연어질문/의도표현).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "분석할 쿼리",
                    },
                },
                "required": ["query"],
            },
        },
    },
    # ========== 관리 도구 (Admin Tools) ==========
    {
        "type": "function",
        "function": {
            "name": "sync_database",
            "description": "벡터 데이터베이스를 JSON 파일과 동기화합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "full": {
                        "type": "boolean",
                        "description": "전체 재동기화 여부 (기본값: 증분 동기화)",
                        "default": False,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sync_status",
            "description": "데이터베이스 동기화 상태를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_database",
            "description": "벡터 데이터베이스를 초기화합니다.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    # ========== 응답 도구 (Response Tools) ==========
    {
        "type": "function",
        "function": {
            "name": "generate_answer",
            "description": "검색된 컨텍스트를 바탕으로 자연어 답변을 생성합니다. 기본 LLM을 사용합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "사용자 질문",
                    },
                    "context": {
                        "type": "string",
                        "description": "답변 생성에 사용할 컨텍스트 (검색 결과)",
                    },
                },
                "required": ["question", "context"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clarify_query",
            "description": "모호한 질문에 대해 사용자에게 명확화를 요청합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "원래 질문",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "선택지 목록",
                    },
                },
                "required": ["query", "options"],
            },
        },
    },
]


def get_tool_names() -> List[str]:
    """Get list of all tool names."""
    return [t["function"]["name"] for t in TOOL_DEFINITIONS]


def get_tool_by_name(name: str) -> Dict[str, Any] | None:
    """Get tool definition by name."""
    for tool in TOOL_DEFINITIONS:
        if tool["function"]["name"] == name:
            return tool
    return None


def get_tools_prompt() -> str:
    """Generate a prompt describing all available tools for FunctionGemma."""
    lines = ["Available tools:"]
    for tool in TOOL_DEFINITIONS:
        func = tool["function"]
        params = func.get("parameters", {}).get("properties", {})
        param_names = list(params.keys())
        lines.append(
            f"- {func['name']}({', '.join(param_names)}): {func['description']}"
        )
    return "\n".join(lines)
