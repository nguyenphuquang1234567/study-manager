"""Data models and custom exceptions for Study Manager.

StudySession and GradeRecord validate themselves when they are created, so any
instance that exists is known to be valid and safe to save.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

ASSESSMENT_TYPES: frozenset[str] = frozenset({"quiz", "exam"})
MAX_SESSION_HOURS: float = 24.0
MIN_SCORE: float = 0.0
MAX_SCORE: float = 100.0


class StudyManagerError(Exception):
    """Base class for every error raised by Study Manager."""


class InvalidEntryError(StudyManagerError):
    """A study session, grade, or prediction input contains an invalid value."""


class DataFileError(StudyManagerError):
    """A CSV file is missing, unreadable, or does not have the expected columns."""


class InsufficientDataError(StudyManagerError):
    """There is not enough usable data to fit a regression model."""


def _to_date(value: Any) -> dt.date:
    """Convert a date, datetime, or ISO string (YYYY-MM-DD) to a date.

    Raises:
        InvalidEntryError: if the value cannot be read as a date.
    """
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value.strip())
        except ValueError:
            pass
    raise InvalidEntryError(f"'{value}' is not a valid date (expected YYYY-MM-DD).")


def _to_float(value: Any, field_name: str) -> float:
    """Convert a value to float, raising InvalidEntryError instead of TypeError/ValueError."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise InvalidEntryError(f"{field_name} must be a number, got '{value}'.") from None


def _check_not_future(day: dt.date) -> None:
    if day > dt.date.today():
        raise InvalidEntryError(f"The date {day.isoformat()} is in the future.")


@dataclass
class StudySession:
    """One block of time spent studying for a course.

    Attributes:
        date: Day the session took place (not in the future).
        course: Course name, for example "Calculus".
        hours: Length of the session, greater than 0 and at most 24.
        note: Optional free-text note.
    """

    date: dt.date
    course: str
    hours: float
    note: str = ""

    def __post_init__(self) -> None:
        self.date = _to_date(self.date)
        self.course = str(self.course).strip()
        self.hours = _to_float(self.hours, "Hours")
        self.note = "" if self.note is None else str(self.note).strip()
        self.validate()

    def validate(self) -> None:
        """Check every field.

        Raises:
            InvalidEntryError: for an empty course, hours outside (0, 24], or a future date.
        """
        if not self.course:
            raise InvalidEntryError("Course name cannot be empty.")
        if not 0 < self.hours <= MAX_SESSION_HOURS:
            raise InvalidEntryError(
                f"Hours must be greater than 0 and at most {MAX_SESSION_HOURS:g}, got {self.hours:g}."
            )
        _check_not_future(self.date)

    def to_dict(self) -> dict[str, Any]:
        """Return one row for study_log.csv."""
        return {
            "date": self.date.isoformat(),
            "course": self.course,
            "hours": self.hours,
            "note": self.note,
        }


@dataclass
class GradeRecord:
    """The score received on one quiz or exam.

    Attributes:
        date: Day of the assessment (not in the future).
        course: Course name; must match the name used for study sessions.
        assessment_type: "quiz" or "exam".
        assessment_name: Label such as "Quiz 3" or "Midterm".
        score: Percentage score from 0 to 100.
    """

    date: dt.date
    course: str
    assessment_type: str
    assessment_name: str
    score: float

    def __post_init__(self) -> None:
        self.date = _to_date(self.date)
        self.course = str(self.course).strip()
        self.assessment_type = str(self.assessment_type).strip().lower()
        self.assessment_name = str(self.assessment_name).strip()
        self.score = _to_float(self.score, "Score")
        self.validate()

    def validate(self) -> None:
        """Check every field.

        Raises:
            InvalidEntryError: for empty text, an unknown type, a score outside 0-100,
                or a future date.
        """
        if not self.course:
            raise InvalidEntryError("Course name cannot be empty.")
        if not self.assessment_name:
            raise InvalidEntryError("Assessment name cannot be empty.")
        if self.assessment_type not in ASSESSMENT_TYPES:
            allowed = " or ".join(sorted(ASSESSMENT_TYPES))
            raise InvalidEntryError(
                f"Assessment type must be {allowed}, got '{self.assessment_type}'."
            )
        if not MIN_SCORE <= self.score <= MAX_SCORE:
            raise InvalidEntryError(
                f"Score must be between {MIN_SCORE:g} and {MAX_SCORE:g}, got {self.score:g}."
            )
        _check_not_future(self.date)

    def to_dict(self) -> dict[str, Any]:
        """Return one row for grades.csv."""
        return {
            "date": self.date.isoformat(),
            "course": self.course,
            "assessment_type": self.assessment_type,
            "assessment_name": self.assessment_name,
            "score": self.score,
        }
