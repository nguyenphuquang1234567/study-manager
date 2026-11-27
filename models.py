"""Data models and custom exceptions for Study Manager."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

# Limits used to validate entries and to clean rows loaded from CSV.
ASSESSMENT_TYPES: frozenset[str] = frozenset({"quiz", "exam"})
MAX_SESSION_HOURS: float = 24.0
MIN_SCORE: float = 0.0
MAX_SCORE: float = 100.0


class StudyManagerError(Exception):
    """Base class for all app errors."""


class InvalidEntryError(StudyManagerError):
    """Invalid value in a session, grade, or prediction input."""


class DataFileError(StudyManagerError):
    """CSV file is missing, unreadable, or missing columns."""


class InsufficientDataError(StudyManagerError):
    """Not enough data to fit a regression."""


def _to_date(value: Any) -> dt.date:
    """Convert a date, datetime, or YYYY-MM-DD string to a date."""
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
    """Convert to float or raise InvalidEntryError."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise InvalidEntryError(f"{field_name} must be a number, got '{value}'.") from None


def _check_not_future(day: dt.date) -> None:
    if day > dt.date.today():
        raise InvalidEntryError(f"The date {day.isoformat()} is in the future.")


@dataclass
class StudySession:
    """One study session: date, course, hours (0-24], optional note."""

    date: dt.date
    course: str
    hours: float
    note: str = ""

    def __post_init__(self) -> None:
        # Convert inputs to the right types, then check them.
        self.date = _to_date(self.date)
        self.course = str(self.course).strip()
        self.hours = _to_float(self.hours, "Hours")
        self.note = "" if self.note is None else str(self.note).strip()
        self.validate()

    def validate(self) -> None:
        """Raise InvalidEntryError if any field is invalid."""
        if not self.course:
            raise InvalidEntryError("Course name cannot be empty.")
        if not 0 < self.hours <= MAX_SESSION_HOURS:
            raise InvalidEntryError(
                f"Hours must be greater than 0 and at most {MAX_SESSION_HOURS:g}, got {self.hours:g}."
            )
        _check_not_future(self.date)

    def to_dict(self) -> dict[str, Any]:
        """Row for study_log.csv."""
        return {
            "date": self.date.isoformat(),
            "course": self.course,
            "hours": self.hours,
            "note": self.note,
        }


@dataclass
class GradeRecord:
    """One quiz or exam result: date, course, type, name, score 0-100."""

    date: dt.date
    course: str
    assessment_type: str
    assessment_name: str
    score: float

    def __post_init__(self) -> None:
        # Convert inputs to the right types, then check them.
        self.date = _to_date(self.date)
        self.course = str(self.course).strip()
        self.assessment_type = str(self.assessment_type).strip().lower()
        self.assessment_name = str(self.assessment_name).strip()
        self.score = _to_float(self.score, "Score")
        self.validate()

    def validate(self) -> None:
        """Raise InvalidEntryError if any field is invalid."""
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
        """Row for grades.csv."""
        return {
            "date": self.date.isoformat(),
            "course": self.course,
            "assessment_type": self.assessment_type,
            "assessment_name": self.assessment_name,
            "score": self.score,
        }
