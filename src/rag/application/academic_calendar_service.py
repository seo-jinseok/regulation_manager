"""
Academic Calendar Service for deadline/date-related query enhancement.

This service enhances search context with academic calendar information
when queries are related to deadlines, dates, and schedules.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.rag.domain.entities import AcademicCalendar, CalendarEvent
from src.rag.infrastructure.period_keyword_detector import PeriodKeywordDetector


class AcademicCalendarService:
    """
    Service for managing academic calendar data and enhancing queries.

    Integrates with PeriodKeywordDetector to identify period-related queries
    and enriches context with relevant calendar events.
    """

    # Default path to academic calendar JSON file
    DEFAULT_CALENDAR_PATH = "data/academic_calendar/academic_calendar.json"

    # Maximum number of events to add to context
    MAX_EVENTS_IN_CONTEXT = 3

    # Category keywords mapping for event matching
    CATEGORY_KEYWORDS: Dict[str, List[str]] = {
        "수강신청": ["수강신청", "수강", "신청"],
        "개강": ["개강", "개학"],
        "종강": ["종강", "종료"],
        "시험": ["시험", "고사", "중간", "기말"],
        "성적": ["성적", "공시", "점수"],
        "휴학": ["휴학"],
        "복학": ["복학"],
        "등록": ["등록", "납부"],
    }

    def __init__(self, calendar_path: Optional[str] = None):
        """
        Initialize AcademicCalendarService.

        Args:
            calendar_path: Path to academic calendar JSON file.
                          Defaults to DEFAULT_CALENDAR_PATH if not provided.
        """
        self.calendar_path = calendar_path or self.DEFAULT_CALENDAR_PATH
        self.period_detector = PeriodKeywordDetector()
        self._calendars: Optional[List[AcademicCalendar]] = None

    def load_calendars(self) -> List[AcademicCalendar]:
        """
        Load all calendar data from JSON file.

        Returns:
            List of AcademicCalendar objects.
            Returns empty list if file not found or invalid.
        """
        try:
            path = Path(self.calendar_path)
            if not path.exists():
                return []

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            calendars = []
            for calendar_data in data.get("calendars", []):
                calendar = AcademicCalendar.from_dict(calendar_data)
                calendars.append(calendar)

            self._calendars = calendars
            return calendars

        except (json.JSONDecodeError, KeyError, OSError):
            return []

    def get_semester_dates(self, year: int, semester: str) -> List[CalendarEvent]:
        """
        Get events for specific year and semester.

        Args:
            year: Academic year (e.g., 2024, 2025).
            semester: Semester string (e.g., "1학기", "2학기", "여름학기", "겨울학기").

        Returns:
            List of CalendarEvent objects for the specified semester.
            Returns empty list if semester not found.
        """
        calendars = self._get_or_load_calendars()

        for calendar in calendars:
            if calendar.year == year and calendar.semester == semester:
                return calendar.events

        return []

    def get_relevant_events(self, query: str) -> List[CalendarEvent]:
        """
        Get events relevant to the query.

        Uses PeriodKeywordDetector to check if query is period-related.
        Extracts year and semester from query to find matching events.
        If no year is specified, searches across all years.

        Args:
            query: User query string (Korean text).

        Returns:
            List of relevant CalendarEvent objects.
            Returns empty list if query is not period-related.
        """
        # Check if query is period-related
        if not self.period_detector.is_period_related(query):
            return []

        # Extract year and semester from query
        year, semester = self._extract_year_semester(query)

        # Get calendars for matching
        calendars = self._get_or_load_calendars()

        # Find matching calendar and get relevant events
        matching_events: List[CalendarEvent] = []

        for calendar in calendars:
            # Match by year if specified (and semester if specified)
            year_matches = year is None or calendar.year == year
            semester_matches = semester is None or calendar.semester == semester

            if year_matches and semester_matches:
                # Filter events by query keywords
                relevant_events = self._filter_events_by_query(
                    calendar.events, query
                )
                matching_events.extend(relevant_events)

        return matching_events[: self.MAX_EVENTS_IN_CONTEXT]

    def enhance_context(self, query: str, context: List[str]) -> List[str]:
        """
        Enhance search context with calendar information.

        For period-related queries, adds relevant calendar events
        to the context to provide deadline/date information.

        Args:
            query: User query string.
            context: Original search context list.

        Returns:
            Enhanced context list with calendar information added.
            Original context is preserved at the beginning.
            Limits to MAX_EVENTS_IN_CONTEXT events to avoid bloat.
        """
        # Check if query is period-related
        if not self.period_detector.is_period_related(query):
            return context.copy() if context else []

        # Get relevant events
        events = self.get_relevant_events(query)

        if not events:
            return context.copy() if context else []

        # Build enhanced context
        enhanced = list(context)  # Preserve original context

        # Format and add calendar events
        for event in events[: self.MAX_EVENTS_IN_CONTEXT]:
            event_info = self._format_event_for_context(event)
            enhanced.append(event_info)

        return enhanced

    def _get_or_load_calendars(self) -> List[AcademicCalendar]:
        """Get calendars from cache or load from file."""
        if self._calendars is None:
            self._calendars = self.load_calendars()
        return self._calendars

    def _extract_year_semester(self, query: str) -> tuple[Optional[int], Optional[str]]:
        """
        Extract year and semester from Korean query.

        Args:
            query: Korean query string.

        Returns:
            Tuple of (year, semester). Either may be None if not found.
        """
        year = None
        semester = None

        # Extract year (4 digits followed by 년)
        year_match = re.search(r"(\d{4})년", query)
        if year_match:
            year = int(year_match.group(1))

        # Extract semester
        semester_patterns = [
            (r"1학기", "1학기"),
            (r"2학기", "2학기"),
            (r"여름학기", "여름학기"),
            (r"겨울학기", "겨울학기"),
        ]

        for pattern, sem_name in semester_patterns:
            if re.search(pattern, query):
                semester = sem_name
                break

        return year, semester

    def _filter_events_by_query(
        self, events: List[CalendarEvent], query: str
    ) -> List[CalendarEvent]:
        """
        Filter events by matching query keywords to categories.

        Args:
            events: List of calendar events.
            query: Query string to match against.

        Returns:
            Filtered list of events matching query keywords.
        """
        matched_events = []

        for event in events:
            # Check if event category matches query keywords
            category_keywords = self.CATEGORY_KEYWORDS.get(event.category, [])

            # Also include event name and description for matching
            all_keywords = category_keywords + [event.name]
            if event.description:
                all_keywords.append(event.description)

            # Check if any keyword appears in query
            for keyword in all_keywords:
                if keyword and keyword in query:
                    matched_events.append(event)
                    break

        return matched_events

    def _format_event_for_context(self, event: CalendarEvent) -> str:
        """
        Format calendar event for context inclusion.

        Args:
            event: CalendarEvent to format.

        Returns:
            Formatted string for context.
        """
        if event.end_date:
            date_str = f"{event.start_date} ~ {event.end_date}"
        else:
            date_str = event.start_date

        description = f" ({event.description})" if event.description else ""

        return f"[학사일정] {event.name}: {date_str}{description}"
