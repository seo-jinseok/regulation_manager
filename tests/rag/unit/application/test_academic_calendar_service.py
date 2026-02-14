"""
Unit tests for AcademicCalendarService.

TDD approach: These tests define the expected behavior of AcademicCalendarService
which enhances search context with academic calendar information for period-related queries.
"""

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.rag.domain.entities import AcademicCalendar, CalendarEvent


class TestAcademicCalendarServiceLoadCalendars:
    """Tests for load_calendars method."""

    def test_load_calendars_from_json_file(self, tmp_path):
        """Load all calendar data from JSON file."""
        # Arrange
        calendar_data = {
            "description": "Academic Calendar",
            "version": "1.0.0",
            "calendars": [
                {
                    "year": 2024,
                    "semester": "1학기",
                    "events": [
                        {
                            "name": "개강일",
                            "start_date": "2024-03-04",
                            "category": "개강",
                        }
                    ],
                },
                {
                    "year": 2024,
                    "semester": "2학기",
                    "events": [
                        {
                            "name": "개강일",
                            "start_date": "2024-09-02",
                            "category": "개강",
                        }
                    ],
                },
            ],
        }
        json_file = tmp_path / "academic_calendar.json"
        json_file.write_text(json.dumps(calendar_data, ensure_ascii=False), encoding="utf-8")

        # Import and create service
        from src.rag.application.academic_calendar_service import AcademicCalendarService

        service = AcademicCalendarService(calendar_path=str(json_file))

        # Act
        calendars = service.load_calendars()

        # Assert
        assert len(calendars) == 2
        assert calendars[0].year == 2024
        assert calendars[0].semester == "1학기"
        assert calendars[1].semester == "2학기"

    def test_load_calendars_returns_empty_list_when_file_not_found(self, tmp_path):
        """Return empty list when calendar file does not exist."""
        from src.rag.application.academic_calendar_service import AcademicCalendarService

        service = AcademicCalendarService(calendar_path=str(tmp_path / "nonexistent.json"))

        calendars = service.load_calendars()

        assert calendars == []

    def test_load_calendars_returns_empty_list_on_invalid_json(self, tmp_path):
        """Return empty list when JSON is invalid."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("invalid json content", encoding="utf-8")

        from src.rag.application.academic_calendar_service import AcademicCalendarService

        service = AcademicCalendarService(calendar_path=str(json_file))

        calendars = service.load_calendars()

        assert calendars == []


class TestAcademicCalendarServiceGetSemesterDates:
    """Tests for get_semester_dates method."""

    @pytest.fixture
    def service_with_sample_data(self, tmp_path) -> "AcademicCalendarService":
        """Create service with sample calendar data."""
        calendar_data = {
            "calendars": [
                {
                    "year": 2024,
                    "semester": "1학기",
                    "events": [
                        {"name": "개강일", "start_date": "2024-03-04", "category": "개강"},
                        {"name": "수강신청", "start_date": "2024-02-19", "end_date": "2024-02-23", "category": "수강신청"},
                        {"name": "종강일", "start_date": "2024-06-21", "category": "종강"},
                    ],
                },
                {
                    "year": 2024,
                    "semester": "2학기",
                    "events": [
                        {"name": "개강일", "start_date": "2024-09-02", "category": "개강"},
                    ],
                },
                {
                    "year": 2025,
                    "semester": "1학기",
                    "events": [
                        {"name": "개강일", "start_date": "2025-03-03", "category": "개강"},
                    ],
                },
            ]
        }
        json_file = tmp_path / "academic_calendar.json"
        json_file.write_text(json.dumps(calendar_data, ensure_ascii=False), encoding="utf-8")

        from src.rag.application.academic_calendar_service import AcademicCalendarService

        return AcademicCalendarService(calendar_path=str(json_file))

    def test_get_semester_dates_returns_events_for_specific_semester(
        self, service_with_sample_data
    ):
        """Get events for specific year and semester."""
        events = service_with_sample_data.get_semester_dates(2024, "1학기")

        assert len(events) == 3
        assert events[0].name == "개강일"
        assert events[0].start_date == "2024-03-04"

    def test_get_semester_dates_returns_empty_for_nonexistent_semester(
        self, service_with_sample_data
    ):
        """Return empty list when semester does not exist."""
        events = service_with_sample_data.get_semester_dates(2023, "1학기")

        assert events == []

    def test_get_semester_dates_handles_summer_semester(self, service_with_sample_data):
        """Handle summer semester (여름학기) correctly."""
        # The sample data doesn't have summer semester, so should return empty
        events = service_with_sample_data.get_semester_dates(2024, "여름학기")

        assert events == []

    def test_get_semester_dates_handles_winter_semester(self, service_with_sample_data):
        """Handle winter semester (겨울학기) correctly."""
        # The sample data doesn't have winter semester, so should return empty
        events = service_with_sample_data.get_semester_dates(2024, "겨울학기")

        assert events == []


class TestAcademicCalendarServiceGetRelevantEvents:
    """Tests for get_relevant_events method."""

    @pytest.fixture
    def service_with_full_data(self, tmp_path) -> "AcademicCalendarService":
        """Create service with full calendar data including various event types."""
        calendar_data = {
            "calendars": [
                {
                    "year": 2024,
                    "semester": "1학기",
                    "events": [
                        {"name": "수강신청 기간", "start_date": "2024-02-19", "end_date": "2024-02-23", "category": "수강신청"},
                        {"name": "개강일", "start_date": "2024-03-04", "category": "개강"},
                        {"name": "휴학 신청 기간", "start_date": "2024-03-04", "end_date": "2024-03-15", "category": "휴학"},
                        {"name": "복학 신청 기간", "start_date": "2024-02-01", "end_date": "2024-02-28", "category": "복학"},
                        {"name": "중간고사 기간", "start_date": "2024-04-15", "end_date": "2024-04-19", "category": "시험"},
                        {"name": "종강일", "start_date": "2024-06-21", "category": "종강"},
                    ],
                },
                {
                    "year": 2025,
                    "semester": "1학기",
                    "events": [
                        {"name": "수강신청 기간", "start_date": "2025-02-17", "end_date": "2025-02-21", "category": "수강신청"},
                        {"name": "개강일", "start_date": "2025-03-03", "category": "개강"},
                    ],
                },
            ]
        }
        json_file = tmp_path / "academic_calendar.json"
        json_file.write_text(json.dumps(calendar_data, ensure_ascii=False), encoding="utf-8")

        from src.rag.application.academic_calendar_service import AcademicCalendarService

        return AcademicCalendarService(calendar_path=str(json_file))

    def test_get_relevant_events_for_course_registration_query(self, service_with_full_data):
        """Get events relevant to course registration query."""
        events = service_with_full_data.get_relevant_events("2024년 1학기 수강신청 언제인가요?")

        # Should return events related to 수강신청 for 2024 1학기
        assert len(events) >= 1
        assert any(e.category == "수강신청" for e in events)

    def test_get_relevant_events_for_semester_start_query(self, service_with_full_data):
        """Get events relevant to semester start query."""
        events = service_with_full_data.get_relevant_events("2025년 1학기 개강일이 언제인가요?")

        # Should return events related to 개강 for 2025 1학기
        assert len(events) >= 1
        assert any(e.category == "개강" for e in events)

    def test_get_relevant_events_returns_empty_for_non_period_query(self, service_with_full_data):
        """Return empty list when query is not period-related."""
        events = service_with_full_data.get_relevant_events("교원인사규정 제10조 내용은 무엇인가요?")

        # Non-period query should return empty list
        assert events == []

    def test_get_relevant_events_for_leave_of_absence_query(self, service_with_full_data):
        """Get events relevant to leave of absence query."""
        events = service_with_full_data.get_relevant_events("휴학 신청 기간이 언제까지인가요?")

        # Should return events related to 휴학
        assert len(events) >= 1
        assert any(e.category == "휴학" for e in events)

    def test_get_relevant_events_returns_calendar_event_objects(self, service_with_full_data):
        """Verify that get_relevant_events returns CalendarEvent objects."""
        events = service_with_full_data.get_relevant_events("2024년 1학기 수강신청 기간")

        assert len(events) >= 1
        assert isinstance(events[0], CalendarEvent)
        assert hasattr(events[0], "name")
        assert hasattr(events[0], "start_date")
        assert hasattr(events[0], "category")


class TestAcademicCalendarServiceEnhanceContext:
    """Tests for enhance_context method."""

    @pytest.fixture
    def service_for_enhance(self, tmp_path) -> "AcademicCalendarService":
        """Create service for context enhancement tests."""
        calendar_data = {
            "calendars": [
                {
                    "year": 2025,
                    "semester": "1학기",
                    "events": [
                        {"name": "수강신청 기간", "start_date": "2025-02-17", "end_date": "2025-02-21", "category": "수강신청", "description": "2025학년도 1학기 수강신청"},
                        {"name": "개강일", "start_date": "2025-03-03", "category": "개강", "description": "2025학년도 1학기 개강"},
                        {"name": "휴학 신청 기간", "start_date": "2025-03-03", "end_date": "2025-03-14", "category": "휴학"},
                        {"name": "종강일", "start_date": "2025-06-20", "category": "종강"},
                    ],
                },
            ]
        }
        json_file = tmp_path / "academic_calendar.json"
        json_file.write_text(json.dumps(calendar_data, ensure_ascii=False), encoding="utf-8")

        from src.rag.application.academic_calendar_service import AcademicCalendarService

        return AcademicCalendarService(calendar_path=str(json_file))

    def test_enhance_context_adds_calendar_info_for_period_query(self, service_for_enhance):
        """Add calendar information to context for period-related query."""
        query = "2025년 1학기 수강신청 언제인가요?"
        context = ["수강신청에 관한 규정 내용입니다."]

        enhanced = service_for_enhance.enhance_context(query, context)

        # Should add calendar event information
        assert len(enhanced) > len(context)
        # Check that calendar info was added
        calendar_info_added = any("수강신청" in c and ("2025" in c or "날짜" in c or "기간" in c) for c in enhanced)
        assert calendar_info_added

    def test_enhance_context_returns_original_context_for_non_period_query(
        self, service_for_enhance
    ):
        """Return original context unchanged for non-period query."""
        query = "교원인사규정 제10조에 대해 알려주세요."
        context = ["교원인사규정 제10조 내용입니다."]

        enhanced = service_for_enhance.enhance_context(query, context)

        # Should return original context unchanged
        assert enhanced == context

    def test_enhance_context_returns_original_when_no_matching_events_found(
        self, tmp_path
    ):
        """Return original context when period query has no matching events."""
        # Create calendar with only 2024 data
        calendar_data = {
            "calendars": [
                {
                    "year": 2024,
                    "semester": "1학기",
                    "events": [
                        {"name": "개강일", "start_date": "2024-03-04", "category": "개강"},
                    ],
                },
            ]
        }
        json_file = tmp_path / "academic_calendar.json"
        json_file.write_text(json.dumps(calendar_data, ensure_ascii=False), encoding="utf-8")

        from src.rag.application.academic_calendar_service import AcademicCalendarService

        service = AcademicCalendarService(calendar_path=str(json_file))
        query = "2026년 1학기 수강신청 기간"  # 2026 not in calendar
        context = ["수강신청 규정 내용입니다."]

        enhanced = service.enhance_context(query, context)

        # Should return original context since no matching events
        assert enhanced == context

    def test_enhance_context_limits_events_to_max_three(self, tmp_path):
        """Limit calendar events to max 3 to avoid context bloat."""
        # Create calendar with more than 3 events
        calendar_data = {
            "calendars": [
                {
                    "year": 2025,
                    "semester": "1학기",
                    "events": [
                        {"name": "수강신청 기간", "start_date": "2025-02-17", "end_date": "2025-02-21", "category": "수강신청"},
                        {"name": "등록금 납부 기간", "start_date": "2025-02-21", "end_date": "2025-02-25", "category": "등록"},
                        {"name": "개강일", "start_date": "2025-03-03", "category": "개강"},
                        {"name": "휴학 신청 기간", "start_date": "2025-03-03", "end_date": "2025-03-14", "category": "휴학"},
                        {"name": "복학 신청 기간", "start_date": "2025-02-03", "end_date": "2025-02-28", "category": "복학"},
                    ],
                },
            ]
        }
        json_file = tmp_path / "academic_calendar.json"
        json_file.write_text(json.dumps(calendar_data, ensure_ascii=False), encoding="utf-8")

        from src.rag.application.academic_calendar_service import AcademicCalendarService

        service = AcademicCalendarService(calendar_path=str(json_file))
        query = "2025년 1학기 학사일정 알려주세요"
        context = ["기본 컨텍스트"]

        enhanced = service.enhance_context(query, context)

        # Should add at most 3 calendar events (plus original context)
        # The enhanced context should have at most 4 items (1 original + 3 calendar)
        calendar_entries = [c for c in enhanced if c != context[0]]
        assert len(calendar_entries) <= 3

    def test_enhance_context_handles_empty_context(self, service_for_enhance):
        """Handle empty context list gracefully."""
        query = "2025년 1학기 개강일 언제인가요?"
        context: List[str] = []

        enhanced = service_for_enhance.enhance_context(query, context)

        # Should add calendar info even with empty context
        assert len(enhanced) >= 1

    def test_enhance_context_preserves_original_context_order(self, service_for_enhance):
        """Preserve original context entries in order."""
        query = "2025년 1학기 수강신청 기간"
        context = ["첫 번째 컨텍스트", "두 번째 컨텍스트", "세 번째 컨텍스트"]

        enhanced = service_for_enhance.enhance_context(query, context)

        # Original context should be preserved at the beginning
        for i, original in enumerate(context):
            assert enhanced[i] == original


class TestAcademicCalendarServiceIntegration:
    """Integration tests with PeriodKeywordDetector."""

    @pytest.fixture
    def service_with_real_data(self) -> "AcademicCalendarService":
        """Create service with actual academic_calendar.json file."""
        from src.rag.application.academic_calendar_service import AcademicCalendarService

        # Use the actual data file
        calendar_path = Path("data/academic_calendar/academic_calendar.json")
        if calendar_path.exists():
            return AcademicCalendarService(calendar_path=str(calendar_path))
        else:
            pytest.skip("academic_calendar.json not found")

    def test_integration_period_keyword_detection_works(self, service_with_real_data):
        """Verify PeriodKeywordDetector integration works correctly."""
        # Period-related query
        period_query = "2025년 1학기 수강신청 기간이 언제인가요?"
        events = service_with_real_data.get_relevant_events(period_query)
        assert len(events) >= 1

        # Non-period query
        non_period_query = "교원인사규정의 목적은 무엇인가요?"
        events = service_with_real_data.get_relevant_events(non_period_query)
        assert events == []

    def test_integration_year_semester_extraction(self, service_with_real_data):
        """Verify year and semester extraction from query."""
        # Query with year and semester
        query = "2025년 1학기 개강일이 언제인가요?"
        events = service_with_real_data.get_relevant_events(query)

        assert len(events) >= 1
        # Should return 2025 1학기 개강 event
        assert any(e.name == "개강일" and "2025" in e.description for e in events)

    def test_integration_enhance_context_with_real_data(self, service_with_real_data):
        """Test enhance_context with real calendar data."""
        query = "2025년 1학기 휴학 신청 기간"
        context = ["휴학에 관한 규정입니다."]

        enhanced = service_with_real_data.enhance_context(query, context)

        # Should add calendar info
        assert len(enhanced) > len(context)


class TestAcademicCalendarServiceYearSemesterExtraction:
    """Tests for year and semester extraction from Korean queries."""

    @pytest.fixture
    def service_for_extraction(self, tmp_path) -> "AcademicCalendarService":
        """Create service for extraction tests."""
        calendar_data = {
            "calendars": [
                {"year": 2024, "semester": "1학기", "events": [{"name": "개강일", "start_date": "2024-03-04", "category": "개강"}]},
                {"year": 2024, "semester": "2학기", "events": [{"name": "개강일", "start_date": "2024-09-02", "category": "개강"}]},
                {"year": 2025, "semester": "1학기", "events": [{"name": "개강일", "start_date": "2025-03-03", "category": "개강"}]},
            ]
        }
        json_file = tmp_path / "academic_calendar.json"
        json_file.write_text(json.dumps(calendar_data, ensure_ascii=False), encoding="utf-8")

        from src.rag.application.academic_calendar_service import AcademicCalendarService

        return AcademicCalendarService(calendar_path=str(json_file))

    def test_extract_year_2025(self, service_for_extraction):
        """Extract year 2025 from query."""
        events = service_for_extraction.get_relevant_events("2025년 1학기 개강일")

        assert len(events) >= 1
        assert events[0].start_date == "2025-03-03"

    def test_extract_year_2024(self, service_for_extraction):
        """Extract year 2024 from query."""
        events = service_for_extraction.get_relevant_events("2024년 2학기 개강일")

        assert len(events) >= 1
        assert events[0].start_date == "2024-09-02"

    def test_extract_semester_1hakgi(self, service_for_extraction):
        """Extract 1학기 semester from query."""
        events = service_for_extraction.get_relevant_events("2024년 1학기 개강일")

        assert len(events) >= 1

    def test_extract_semester_2hakgi(self, service_for_extraction):
        """Extract 2학기 semester from query."""
        events = service_for_extraction.get_relevant_events("2024년 2학기 개강일")

        assert len(events) >= 1

    def test_handle_query_without_year(self, service_for_extraction):
        """Handle query without explicit year (should use current or default)."""
        # Without year, should handle gracefully (return empty or use current year)
        events = service_for_extraction.get_relevant_events("1학기 개강일 언제인가요?")

        # Should not raise error
        assert isinstance(events, list)

    def test_handle_query_without_semester(self, service_for_extraction):
        """Handle query without explicit semester."""
        events = service_for_extraction.get_relevant_events("2024년 개강일 언제인가요?")

        # Should not raise error
        assert isinstance(events, list)
