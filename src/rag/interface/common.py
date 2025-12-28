
from typing import Optional

def decide_search_mode(query: str, force_mode: Optional[str] = None) -> str:
    """
    Decide whether to use 'search' (retrieval) or 'ask' (LLM answer) mode.
    
    Args:
        query: The user's query string.
        force_mode: Explicit mode ('search', 'ask', or None).
        
    Returns:
        "ask" or "search"
    """
    if force_mode:
        return force_mode.lower()
        
    query = query.strip()
    query_lower = query.lower()

    # 0. Full-view request signals
    full_view_markers = ["전문", "전체", "원문", "全文", "full text", "fullview"]
    if any(marker in query_lower for marker in full_view_markers):
        return "full_view"
    
    # 1. Strong signals for Question
    if query.endswith("?"):
        return "ask"
        
    # 2. Question words (Interrogatives & Request verbs)
    question_words = [
        "어떻게", "언제", "무엇", "누가", "어디서", "얼마나", "왜",
        "방법", "절차", "자격", "요건", "기준", 
        "알려줘", "해줘", "인가요", "나요?", "가요?", "까?", "까요"
    ]
    if any(word in query for word in question_words):
        return "ask"
        
    # 3. Descriptive/Complex Sentences (likely 'Ask' intent)
    # If the queries are long natural language sentences, we should treat them as Ask.
    # E.g. "교수님이 수업시간에 정치적인 발언을 하고 자주 화도 내"
    # Heuristic: Length > 15 chars AND has spaces (sentence-like)
    if len(query) > 15 and " " in query:
        return "ask"
        
    # Default to Search (Keyword-based)
    return "search"
