"""
Period Keyword Detector for deadline/date-related queries.

Detects Korean period-related keywords to identify queries about
deadlines, dates, and schedules.
"""

from typing import List

# Period-related keywords in Korean
PERIOD_KEYWORDS = frozenset([
    "기간",      # period, duration
    "언제",      # when
    "기한",      # deadline, time limit
    "날짜",      # date
    "까지",      # until, by
    "마감",      # deadline, closing
    "신청일",    # application date
    "등록일",    # registration date
    "시작",      # start, beginning
    "종료",      # end, termination
    "개강",      # start of semester
    "종강",      # end of semester
])


class PeriodKeywordDetector:
    """
    Detects period-related keywords in Korean text queries.

    Used to identify queries that are asking about deadlines, dates,
    schedules, or time periods in the academic context.
    """

    def detect_period_keywords(self, query: str) -> List[str]:
        """
        Detect all period-related keywords in the given query.

        Args:
            query: The input query string (Korean text).

        Returns:
            List of detected keywords in the order they appear.
            Each keyword appears only once in the result.
        """
        if not query:
            return []

        detected = []
        seen = set()

        for keyword in PERIOD_KEYWORDS:
            if keyword in query and keyword not in seen:
                detected.append(keyword)
                seen.add(keyword)

        return detected

    def is_period_related(self, query: str) -> bool:
        """
        Check if the query contains any period-related keywords.

        Args:
            query: The input query string (Korean text).

        Returns:
            True if the query contains at least one period keyword,
            False otherwise.
        """
        if not query:
            return False

        return any(keyword in query for keyword in PERIOD_KEYWORDS)
