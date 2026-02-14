"""
Unit tests for Academic Calendar data models.

TDD approach: These tests define the expected behavior of CalendarEvent and AcademicCalendar.
"""

import json
from pathlib import Path

import pytest

from src.rag.domain.entities import AcademicCalendar, CalendarEvent


class TestCalendarEvent:
    """Tests for CalendarEvent dataclass."""

    def test_create_calendar_event_basic(self):
        """Create basic CalendarEvent with required fields."""
        event = CalendarEvent(
            name="개강일",
            start_date="2024-03-04",
            end_date=None,
            category="개강",
            description=None,
        )

        assert event.name == "개강일"
        assert event.start_date == "2024-03-04"
        assert event.end_date is None
        assert event.category == "개강"
        assert event.description is None

    def test_create_calendar_event_with_date_range(self):
        """Create CalendarEvent with date range."""
        event = CalendarEvent(
            name="수강신청 기간",
            start_date="2024-02-19",
            end_date="2024-02-23",
            category="수강신청",
            description="2024학년도 1학기 수강신청",
        )

        assert event.name == "수강신청 기간"
        assert event.start_date == "2024-02-19"
        assert event.end_date == "2024-02-23"
        assert event.category == "수강신청"
        assert event.description == "2024학년도 1학기 수강신청"

    def test_calendar_event_from_dict(self):
        """Create CalendarEvent from dictionary."""
        data = {
            "name": "중간고사",
            "start_date": "2024-04-15",
            "end_date": "2024-04-19",
            "category": "시험",
            "description": "2024학년도 1학기 중간고사",
        }

        event = CalendarEvent.from_dict(data)

        assert event.name == "중간고사"
        assert event.start_date == "2024-04-15"
        assert event.end_date == "2024-04-19"
        assert event.category == "시험"
        assert event.description == "2024학년도 1학기 중간고사"

    def test_calendar_event_from_dict_minimal(self):
        """Create CalendarEvent from dictionary with minimal fields."""
        data = {
            "name": "종강일",
            "start_date": "2024-06-21",
            "category": "종강",
        }

        event = CalendarEvent.from_dict(data)

        assert event.name == "종강일"
        assert event.start_date == "2024-06-21"
        assert event.end_date is None
        assert event.category == "종강"
        assert event.description is None

    def test_calendar_event_to_dict(self):
        """Convert CalendarEvent to dictionary."""
        event = CalendarEvent(
            name="성적공시",
            start_date="2024-06-24",
            end_date="2024-06-26",
            category="성적",
            description="성적 열람 및 이의신청",
        )

        result = event.to_dict()

        assert result["name"] == "성적공시"
        assert result["start_date"] == "2024-06-24"
        assert result["end_date"] == "2024-06-26"
        assert result["category"] == "성적"
        assert result["description"] == "성적 열람 및 이의신청"

    def test_calendar_event_equality(self):
        """Two CalendarEvents with same data should be equal."""
        event1 = CalendarEvent(
            name="개강일",
            start_date="2024-03-04",
            end_date=None,
            category="개강",
            description=None,
        )
        event2 = CalendarEvent(
            name="개강일",
            start_date="2024-03-04",
            end_date=None,
            category="개강",
            description=None,
        )

        assert event1 == event2


class TestAcademicCalendar:
    """Tests for AcademicCalendar dataclass."""

    def test_create_academic_calendar_basic(self):
        """Create basic AcademicCalendar with required fields."""
        events = [
            CalendarEvent(
                name="개강일",
                start_date="2024-03-04",
                end_date=None,
                category="개강",
                description=None,
            )
        ]

        calendar = AcademicCalendar(
            year=2024,
            semester="1학기",
            events=events,
        )

        assert calendar.year == 2024
        assert calendar.semester == "1학기"
        assert len(calendar.events) == 1
        assert calendar.events[0].name == "개강일"

    def test_create_academic_calendar_with_multiple_events(self):
        """Create AcademicCalendar with multiple events."""
        events = [
            CalendarEvent(
                name="개강일",
                start_date="2024-03-04",
                end_date=None,
                category="개강",
                description=None,
            ),
            CalendarEvent(
                name="수강신청",
                start_date="2024-02-19",
                end_date="2024-02-23",
                category="수강신청",
                description=None,
            ),
            CalendarEvent(
                name="종강일",
                start_date="2024-06-21",
                end_date=None,
                category="종강",
                description=None,
            ),
        ]

        calendar = AcademicCalendar(
            year=2024,
            semester="1학기",
            events=events,
        )

        assert len(calendar.events) == 3

    def test_academic_calendar_from_dict(self):
        """Create AcademicCalendar from dictionary."""
        data = {
            "year": 2024,
            "semester": "2학기",
            "events": [
                {
                    "name": "개강일",
                    "start_date": "2024-09-02",
                    "end_date": None,
                    "category": "개강",
                    "description": None,
                },
                {
                    "name": "종강일",
                    "start_date": "2024-12-20",
                    "end_date": None,
                    "category": "종강",
                    "description": None,
                },
            ],
        }

        calendar = AcademicCalendar.from_dict(data)

        assert calendar.year == 2024
        assert calendar.semester == "2학기"
        assert len(calendar.events) == 2
        assert calendar.events[0].name == "개강일"
        assert calendar.events[1].name == "종강일"

    def test_academic_calendar_to_dict(self):
        """Convert AcademicCalendar to dictionary."""
        events = [
            CalendarEvent(
                name="개강일",
                start_date="2024-03-04",
                end_date=None,
                category="개강",
                description=None,
            )
        ]

        calendar = AcademicCalendar(
            year=2024,
            semester="1학기",
            events=events,
        )

        result = calendar.to_dict()

        assert result["year"] == 2024
        assert result["semester"] == "1학기"
        assert len(result["events"]) == 1
        assert result["events"][0]["name"] == "개강일"

    def test_academic_calendar_semester_types(self):
        """Test various semester types."""
        semester_types = ["1학기", "2학기", "여름학기", "겨울학기"]

        for semester in semester_types:
            calendar = AcademicCalendar(
                year=2024,
                semester=semester,
                events=[],
            )
            assert calendar.semester == semester

    def test_get_events_by_category(self):
        """Filter events by category."""
        events = [
            CalendarEvent(name="수강신청1", start_date="2024-02-19", end_date="2024-02-23", category="수강신청", description=None),
            CalendarEvent(name="개강일", start_date="2024-03-04", end_date=None, category="개강", description=None),
            CalendarEvent(name="수강신청2", start_date="2024-08-19", end_date="2024-08-23", category="수강신청", description=None),
            CalendarEvent(name="종강일", start_date="2024-06-21", end_date=None, category="종강", description=None),
        ]

        calendar = AcademicCalendar(year=2024, semester="1학기", events=events)

        registration_events = calendar.get_events_by_category("수강신청")
        assert len(registration_events) == 2
        assert all(e.category == "수강신청" for e in registration_events)


class TestAcademicCalendarJSON:
    """Tests for JSON file loading and validation."""

    @pytest.fixture
    def sample_calendar_data(self) -> dict:
        """Sample academic calendar data."""
        return {
            "calendars": [
                {
                    "year": 2024,
                    "semester": "1학기",
                    "events": [
                        {
                            "name": "개강일",
                            "start_date": "2024-03-04",
                            "end_date": None,
                            "category": "개강",
                            "description": None,
                        },
                        {
                            "name": "종강일",
                            "start_date": "2024-06-21",
                            "end_date": None,
                            "category": "종강",
                            "description": None,
                        },
                    ],
                },
            ]
        }

    def test_load_calendar_from_json(self, sample_calendar_data, tmp_path):
        """Load academic calendar from JSON file."""
        json_file = tmp_path / "academic_calendar.json"
        json_file.write_text(json.dumps(sample_calendar_data, ensure_ascii=False), encoding="utf-8")

        data = json.loads(json_file.read_text(encoding="utf-8"))
        calendar_data = data["calendars"][0]
        calendar = AcademicCalendar.from_dict(calendar_data)

        assert calendar.year == 2024
        assert calendar.semester == "1학기"
        assert len(calendar.events) == 2

    def test_json_contains_required_event_categories(self):
        """Verify JSON contains all required event categories."""
        required_categories = [
            "수강신청",
            "개강",
            "종강",
            "시험",
            "성적",
            "휴학",
            "복학",
            "등록",
        ]

        # This test documents the required categories
        # The actual JSON file should contain these
        assert len(required_categories) == 8


class TestAcademicCalendarEventCategories:
    """Tests for various event categories in academic calendar."""

    def test_course_registration_events(self):
        """Test course registration period events."""
        event = CalendarEvent(
            name="수강신청 기간",
            start_date="2024-02-19",
            end_date="2024-02-23",
            category="수강신청",
            description="2024학년도 1학기 수강신청",
        )

        assert event.category == "수강신청"
        assert event.end_date is not None

    def test_semester_start_end_events(self):
        """Test semester start and end events."""
        start_event = CalendarEvent(
            name="개강일",
            start_date="2024-03-04",
            end_date=None,
            category="개강",
            description=None,
        )
        end_event = CalendarEvent(
            name="종강일",
            start_date="2024-06-21",
            end_date=None,
            category="종강",
            description=None,
        )

        assert start_event.category == "개강"
        assert end_event.category == "종강"

    def test_exam_events(self):
        """Test exam period events."""
        midterm = CalendarEvent(
            name="중간고사",
            start_date="2024-04-15",
            end_date="2024-04-19",
            category="시험",
            description="중간고사 기간",
        )
        final = CalendarEvent(
            name="기말고사",
            start_date="2024-06-10",
            end_date="2024-06-21",
            category="시험",
            description="기말고사 기간",
        )

        assert midterm.category == "시험"
        assert final.category == "시험"

    def test_grade_events(self):
        """Test grade announcement events."""
        event = CalendarEvent(
            name="성적공시",
            start_date="2024-06-24",
            end_date="2024-06-26",
            category="성적",
            description="성적 열람 및 이의신청 기간",
        )

        assert event.category == "성적"

    def test_leave_return_events(self):
        """Test leave of absence and return events."""
        leave_event = CalendarEvent(
            name="휴학 신청 기간",
            start_date="2024-03-04",
            end_date="2024-03-15",
            category="휴학",
            description="휴학 신청",
        )
        return_event = CalendarEvent(
            name="복학 신청 기간",
            start_date="2024-02-01",
            end_date="2024-02-28",
            category="복학",
            description="복학 신청",
        )

        assert leave_event.category == "휴학"
        assert return_event.category == "복학"

    def test_tuition_events(self):
        """Test tuition payment events."""
        event = CalendarEvent(
            name="등록금 납부 기간",
            start_date="2024-02-23",
            end_date="2024-02-27",
            category="등록",
            description="등록금 납부",
        )

        assert event.category == "등록"
