import re
from typing import List, Optional, Tuple


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
        "추가로",
        "더",
        "다른",
        "또",
        "나머지",
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

    # If user explicitly names a regulation (e.g. "School Regulations Article 7"),
    # we should NOT use the context (e.g. "Faculty Personnel Regulation").
    # This is a context switch.
    if extract_regulation_title(message):
        return message

    # If it's a simple article reference (e.g. "Article 7") without a regulation name,
    # we should prepend the context.
    # If it's a simple article reference (e.g. "Article 7") without a regulation name,
    # we should prepend the context.
    # Note: re.search checks for "Article N" pattern (allowing optional "제")
    if re.search(r"(?:제)?\s*\d+\s*조", message):
        return f"{context} {message}".strip()

    # For other patterns, rely on is_followup_message heuristic
    if has_explicit_target(message):
        return message

    if not is_followup_message(message):
        return message

    return f"{context} {message}".strip()


def build_history_context(
    history: List[dict],
    max_turns: int = 6,
    max_chars: int = 1200,
) -> str:
    if not history:
        return ""

    messages = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": str(content)})
            continue
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_text, assistant_text = item
            if user_text:
                messages.append({"role": "user", "content": str(user_text)})
            if assistant_text:
                messages.append({"role": "assistant", "content": str(assistant_text)})

    if max_turns and max_turns > 0:
        messages = messages[-max_turns * 2 :]

    lines = []
    for msg in messages:
        label = "사용자" if msg["role"] == "user" else "어시스턴트"
        snippet = re.sub(r"\s+", " ", msg["content"]).strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        lines.append(f"{label}: {snippet}")

    context = "\n".join(lines)
    if max_chars and len(context) > max_chars:
        context = context[-max_chars:]
        cut = context.find("\n")
        if cut != -1:
            context = context[cut + 1 :]

    return context


def format_clarification(kind: str, options: List[str]) -> str:
    if kind == "audience":
        return "대상이 모호합니다. 교수/학생/직원 중 하나를 선택해주세요."
    if kind == "regulation":
        if options:
            choices = "\n".join([f"- {o}" for o in options])
            return f"여러 규정이 매칭됩니다. 아래 중 하나를 선택해주세요:\n{choices}"
        return "여러 규정이 매칭됩니다. 규정명을 구체적으로 입력해주세요."
    return "추가 선택이 필요합니다."


_REGULATION_SUFFIXES = (
    "규정",
    "규칙",
    "학칙",
    "정관",
    "지침",
    "요강",
    "준칙",
    "세칙",
    "규정집",
)
_REGULATION_PATTERN = re.compile(
    rf"([A-Za-z0-9가-힣·\s]*?(?:{'|'.join(_REGULATION_SUFFIXES)}))"
)
_ATTACHMENT_PATTERN = re.compile(r"(별표|별첨|별지)\s*(\d+)?")
_TRAILING_PARTICLE_PATTERN = re.compile(r"(의|을|를|은|는|이|가|에|에서|으로|로)$")


def extract_regulation_title(text: str) -> Optional[str]:
    if not text:
        return None

    matches = list(_REGULATION_PATTERN.finditer(text))
    if not matches:
        return None

    def score(match: re.Match) -> int:
        return len(match.group(1).replace(" ", ""))

    best = max(matches, key=score).group(1).strip()
    best = _TRAILING_PARTICLE_PATTERN.sub("", best).strip()
    return best or None


def parse_attachment_request(
    text: str,
    fallback_regulation: Optional[str],
) -> Optional[Tuple[str, Optional[int], str]]:
    match = _ATTACHMENT_PATTERN.search(text or "")
    if not match:
        return None

    label = match.group(1)
    table_no = int(match.group(2)) if match.group(2) else None
    if table_no is None:
        nearby = re.search(rf"{label}\D{{0,12}}(\d+)\s*번?", text)
        if nearby:
            table_no = int(nearby.group(1))
    regulation = extract_regulation_title(text)
    if not regulation:
        cleaned = _ATTACHMENT_PATTERN.sub("", text).strip()
        regulation = extract_regulation_title(cleaned)
    if not regulation:
        regulation = fallback_regulation
    if not regulation:
        return None

    return regulation, table_no, label


def attachment_label_variants(label: Optional[str]) -> List[str]:
    if label:
        return [label]
    return ["별표", "별첨", "별지"]
