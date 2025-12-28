import re
from typing import List, Optional


def _normalize_text(text: str) -> str:
    return "".join(str(text).strip().lower().split())


def resolve_audience_choice(text: str) -> Optional[str]:
    """
    Resolve audience choice from user text.

    Returns: "교수" | "학생" | "직원" | None
    """
    if not text:
        return None

    normalized = text.strip().lower()
    matches = []

    if any(k in normalized for k in ("교수", "교원", "faculty")):
        matches.append("교수")
    if any(k in normalized for k in ("학생", "student")):
        matches.append("학생")
    if any(k in normalized for k in ("직원", "staff", "행정")):
        matches.append("직원")

    if len(matches) == 1:
        return matches[0]
    return None


def resolve_regulation_choice(text: str, options: List[str]) -> Optional[str]:
    """Resolve regulation title choice from user text."""
    if not text or not options:
        return None

    query_norm = _normalize_text(text)
    if not query_norm:
        return None

    normalized_options = [(opt, _normalize_text(opt)) for opt in options]

    for opt, opt_norm in normalized_options:
        if query_norm == opt_norm:
            return opt

    candidates = [opt for opt, opt_norm in normalized_options if query_norm in opt_norm]
    if len(candidates) == 1:
        return candidates[0]

    return None


def has_explicit_target(text: str) -> bool:
    """Detect explicit regulation targets like rule codes or named regulations."""
    if not text:
        return False

    normalized = _normalize_text(text)
    if not normalized:
        return False

    if re.search(r"\d+-\d+-\d+", text):
        return True
    if re.search(r"제\s*\d+\s*조", text):
        return True

    if any(token in normalized for token in ("그규정", "이규정", "저규정", "해당규정")):
        return False

    target_keywords = (
        "규정",
        "학칙",
        "정관",
        "규칙",
        "준칙",
        "지침",
        "요강",
        "규정집",
    )
    return any(keyword in text for keyword in target_keywords)


def is_followup_message(text: str) -> bool:
    """Heuristic for follow-up messages that rely on prior context."""
    if not text:
        return False

    normalized = _normalize_text(text)
    if not normalized:
        return False

    followup_tokens = (
        "그럼",
        "그러면",
        "그거",
        "그것",
        "이거",
        "이것",
        "저거",
        "저것",
        "해당",
        "방금",
        "앞서",
        "추가",
        "더",
        "자세히",
        "계속",
        "다시",
        "전문",
        "원문",
        "전체",
        "fulltext",
        "fullview",
    )
    if any(token in normalized for token in followup_tokens):
        return True

    return False


def expand_followup_query(message: str, context: Optional[str]) -> str:
    """Expand follow-up queries with last context when safe."""
    if not context:
        return message
    if has_explicit_target(message):
        return message
    if not is_followup_message(message):
        return message
    return f"{context} {message}".strip()


def format_clarification(kind: str, options: List[str]) -> str:
    if kind == "audience":
        return "대상이 모호합니다. 교수/학생/직원 중 하나를 선택해주세요."
    if kind == "regulation":
        if options:
            choices = "\n".join([f"- {o}" for o in options])
            return f"여러 규정이 매칭됩니다. 아래 중 하나를 선택해주세요:\n{choices}"
        return "여러 규정이 매칭됩니다. 규정명을 구체적으로 입력해주세요."
    return "추가 선택이 필요합니다."
