"""Load, clean, save and summarize study data with Pandas."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from models import (
    ASSESSMENT_TYPES,
    MAX_SCORE,
    MAX_SESSION_HOURS,
    MIN_SCORE,
    DataFileError,
    GradeRecord,
    StudySession,
)

DATA_DIR: Path = Path(__file__).resolve().parent / "data"
DEFAULT_SESSIONS_PATH: Path = DATA_DIR / "study_log.csv"
DEFAULT_GRADES_PATH: Path = DATA_DIR / "grades.csv"
SAMPLE_SESSIONS_PATH: Path = DATA_DIR / "sample_study_log.csv"
SAMPLE_GRADES_PATH: Path = DATA_DIR / "sample_grades.csv"

SESSION_COLUMNS: tuple[str, ...] = ("date", "course", "hours", "note")
GRADE_COLUMNS: tuple[str, ...] = ("date", "course", "assessment_type", "assessment_name", "score")
ASSESSMENT_HOURS_COLUMNS: tuple[str, ...] = (
    "course", "date", "assessment_type", "assessment_name", "score", "hours",
)

# Column types for each file, passed to load_csv / clean_types as keyword arguments.
_SESSION_TYPES: dict[str, tuple[str, ...]] = {
    "date_columns": ("date",),
    "numeric_columns": ("hours",),
    "optional_columns": ("note",),
}
_GRADE_TYPES: dict[str, tuple[str, ...]] = {
    "date_columns": ("date",),
    "numeric_columns": ("score",),
}


# --- CSV helpers ---

def clean_types(
    df: pd.DataFrame,
    required_columns: tuple[str, ...] | list[str],
    *,
    date_columns: tuple[str, ...] = (),
    numeric_columns: tuple[str, ...] = (),
    optional_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, int]:
    """Convert column types and drop unusable rows; return (df, rows dropped)."""
    df = df.loc[:, list(required_columns)].copy()

    # Values that cannot be converted become NaT / NaN.
    for column in date_columns:
        parsed = pd.to_datetime(df[column], errors="coerce", format="ISO8601")
        df[column] = parsed.dt.normalize().astype("datetime64[ns]")
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)

    # Text columns: no NaN, no surrounding spaces.
    text_columns = [c for c in df.columns if c not in date_columns and c not in numeric_columns]
    for column in text_columns:
        df[column] = df[column].astype("object").fillna("").astype(str).str.strip()

    # A row is unusable if any non-optional value is missing or blank.
    must_have = [c for c in df.columns if c not in optional_columns]
    unusable = df[must_have].isna().any(axis=1)
    for column in text_columns:
        if column not in optional_columns:
            unusable |= df[column].eq("")

    dropped = int(unusable.sum())
    return df.loc[~unusable].reset_index(drop=True), dropped


def load_csv(
    path: str | Path,
    required_columns: tuple[str, ...] | list[str],
    *,
    date_columns: tuple[str, ...] = (),
    numeric_columns: tuple[str, ...] = (),
    optional_columns: tuple[str, ...] = (),
    create_if_missing: bool = False,  # True only for the user's own files
) -> tuple[pd.DataFrame, int]:
    """Read a CSV safely and return (df, rows dropped); raises DataFileError."""
    path = Path(path)
    types: dict[str, tuple[str, ...]] = {
        "date_columns": date_columns,
        "numeric_columns": numeric_columns,
        "optional_columns": optional_columns,
    }

    if not path.exists():
        if not create_if_missing:
            raise DataFileError(f"File not found: {path}")
        # First run: create the file with only the header row.
        empty = pd.DataFrame(columns=list(required_columns))
        save_csv(empty, path)
        return clean_types(empty, required_columns, **types)

    # Read everything as text; clean_types converts the types afterwards.
    try:
        raw = pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        raise DataFileError(f"{path.name} is empty; it needs at least a header row.") from None
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataFileError(f"{path.name} is not a valid CSV file ({exc}).") from exc
    except OSError as exc:
        raise DataFileError(f"{path.name} could not be opened ({exc}).") from exc

    missing = [column for column in required_columns if column not in raw.columns]
    if missing:
        raise DataFileError(f"{path.name} is missing required column(s): {', '.join(missing)}.")

    return clean_types(raw, required_columns, **types)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write a CSV without the index, dates as YYYY-MM-DD."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, date_format="%Y-%m-%d")
    except OSError as exc:
        raise DataFileError(f"Could not save {path.name} ({exc}).") from exc


def _append_row(existing: pd.DataFrame, row: pd.DataFrame) -> pd.DataFrame:
    """Append one row and keep the table sorted by date."""
    combined = row.copy() if existing.empty else pd.concat([existing, row], ignore_index=True)
    return combined.sort_values("date", kind="stable", ignore_index=True)


# --- StudyTracker ---

class StudyTracker:
    """Stores the user's sessions and grades and computes statistics."""

    def __init__(
        self,
        sessions_path: str | Path = DEFAULT_SESSIONS_PATH,
        grades_path: str | Path = DEFAULT_GRADES_PATH,
    ) -> None:
        self.sessions_path = Path(sessions_path)
        self.grades_path = Path(grades_path)
        self.sessions = pd.DataFrame(columns=list(SESSION_COLUMNS))
        self.grades = pd.DataFrame(columns=list(GRADE_COLUMNS))
        self.skipped_rows = 0  # malformed rows dropped on the last load
        self.reload()

    # Loading and saving

    def reload(self) -> None:
        """Reload both user files, creating them if missing."""
        sessions, bad_sessions = load_csv(
            self.sessions_path, SESSION_COLUMNS, create_if_missing=True, **_SESSION_TYPES
        )
        grades, bad_grades = load_csv(
            self.grades_path, GRADE_COLUMNS, create_if_missing=True, **_GRADE_TYPES
        )

        # Apply the same rules as StudySession / GradeRecord to rows from the files.
        valid_sessions = sessions["hours"].gt(0) & sessions["hours"].le(MAX_SESSION_HOURS)
        grades["assessment_type"] = grades["assessment_type"].str.lower()
        valid_grades = grades["assessment_type"].isin(sorted(ASSESSMENT_TYPES)) & grades[
            "score"
        ].between(MIN_SCORE, MAX_SCORE)

        self.sessions = sessions.loc[valid_sessions].sort_values(
            "date", kind="stable", ignore_index=True
        )
        self.grades = grades.loc[valid_grades].sort_values("date", kind="stable", ignore_index=True)
        self.skipped_rows = (
            bad_sessions + bad_grades + int((~valid_sessions).sum()) + int((~valid_grades).sum())
        )

    def add_session(self, session: StudySession) -> None:
        """Add a validated session and save it."""
        if not isinstance(session, StudySession):
            raise TypeError("add_session expects a StudySession.")
        row, _ = clean_types(pd.DataFrame([session.to_dict()]), SESSION_COLUMNS, **_SESSION_TYPES)
        self.sessions = _append_row(self.sessions, row)
        save_csv(self.sessions, self.sessions_path)

    def add_grade(self, record: GradeRecord) -> None:
        """Add a validated grade and save it."""
        if not isinstance(record, GradeRecord):
            raise TypeError("add_grade expects a GradeRecord.")
        row, _ = clean_types(pd.DataFrame([record.to_dict()]), GRADE_COLUMNS, **_GRADE_TYPES)
        self.grades = _append_row(self.grades, row)
        save_csv(self.grades, self.grades_path)

    def load_sample_data(
        self,
        sessions_sample: str | Path = SAMPLE_SESSIONS_PATH,
        grades_sample: str | Path = SAMPLE_GRADES_PATH,
    ) -> None:
        """Replace the user's files with the sample data, then reload."""
        for source, target in ((sessions_sample, self.sessions_path), (grades_sample, self.grades_path)):
            source = Path(source)
            if not source.exists():
                raise DataFileError(f"Sample file not found: {source}")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            except OSError as exc:
                raise DataFileError(f"Could not copy {source.name} ({exc}).") from exc
        self.reload()

    # Statistics

    def courses(self) -> list[str]:
        """Sorted names of every course with a session or a grade."""
        return sorted(set(self.sessions["course"]) | set(self.grades["course"]))

    def hours_by_course(self) -> pd.DataFrame:
        """Total hours, session count and average session length per course."""
        if self.sessions.empty:
            return pd.DataFrame(columns=["course", "total_hours", "sessions", "avg_session_hours"])
        return (
            self.sessions.groupby("course")["hours"]
            .agg(total_hours="sum", sessions="count", avg_session_hours="mean")
            .reset_index()
            .sort_values("total_hours", ascending=False, ignore_index=True)
        )

    def course_summary(self) -> pd.DataFrame:
        """Per-course hours plus number of grades and average score."""
        columns = ["course", "total_hours", "sessions", "avg_session_hours", "assessments", "avg_score"]
        if self.sessions.empty and self.grades.empty:
            return pd.DataFrame(columns=columns)

        hours = self.hours_by_course()
        scores = (
            self.grades.groupby("course")["score"]
            .agg(assessments="count", avg_score="mean")
            .reset_index()
            if not self.grades.empty
            else pd.DataFrame(columns=["course", "assessments", "avg_score"])
        )
        # Outer merge keeps courses that have only sessions or only grades.
        summary = hours.merge(scores, on="course", how="outer")
        summary["total_hours"] = summary["total_hours"].astype(float).fillna(0.0)
        for column in ("sessions", "assessments"):
            summary[column] = summary[column].astype(float).fillna(0).astype(int)
        summary["avg_session_hours"] = summary["avg_session_hours"].astype(float)
        summary["avg_score"] = summary["avg_score"].astype(float)
        return summary[columns].sort_values("course", ignore_index=True)

    def weekly_hours(self, course: str | None = None) -> pd.DataFrame:
        """Study hours per week (weeks end on Sunday); course=None means all courses."""
        sessions = self.sessions if course is None else self.sessions[self.sessions["course"] == course]
        if sessions.empty:
            return pd.DataFrame(columns=["week_ending", "hours"])
        weekly = sessions.groupby(pd.Grouper(key="date", freq="W"))["hours"].sum().reset_index()
        return weekly.rename(columns={"date": "week_ending"})

    def hours_per_assessment(self) -> pd.DataFrame:
        """Total study hours that went into each graded assessment."""
        if self.grades.empty:
            return pd.DataFrame(columns=list(ASSESSMENT_HOURS_COLUMNS))

        result = self.grades.copy()
        if self.sessions.empty:
            result["hours"] = 0.0
        else:
            # One due date per course and day (same-day assessments share hours).
            due_dates = (
                self.grades[["course", "date"]]
                .drop_duplicates()
                .sort_values("date", ignore_index=True)
            )
            due_dates["due_date"] = due_dates["date"]
            sessions = self.sessions[["date", "course", "hours"]].sort_values(
                "date", kind="stable", ignore_index=True
            )
            # Match each session to the next due date of the same course.
            assigned = pd.merge_asof(sessions, due_dates, on="date", by="course", direction="forward")
            # Sum per due date; sessions after the last assessment have none and are skipped.
            hours = (
                assigned.dropna(subset=["due_date"])
                .groupby(["course", "due_date"], as_index=False)["hours"]
                .sum()
                .rename(columns={"due_date": "date"})
            )
            result = result.merge(hours, on=["course", "date"], how="left")
            result["hours"] = result["hours"].fillna(0.0)

        return result[list(ASSESSMENT_HOURS_COLUMNS)].sort_values(
            ["course", "date"], kind="stable", ignore_index=True
        )

    def hours_since_last_assessment(self, course: str) -> float:
        """Hours studied for a course since its latest assessment."""
        sessions = self.sessions[self.sessions["course"] == course]
        if sessions.empty:
            return 0.0
        course_grades = self.grades[self.grades["course"] == course]
        if course_grades.empty:
            return float(sessions["hours"].sum())
        last_assessment = course_grades["date"].max()
        return float(sessions.loc[sessions["date"] > last_assessment, "hours"].sum())

    def summary_stats(self) -> dict[str, Any]:
        """Headline numbers for the Statistics tab (None when there is no data)."""
        per_assessment = self.hours_per_assessment()
        # Correlation needs 3+ points and some variation in both columns.
        correlation: float | None = None
        if (
            len(per_assessment) >= 3
            and per_assessment["hours"].nunique() > 1
            and per_assessment["score"].nunique() > 1
        ):
            correlation = float(per_assessment["hours"].corr(per_assessment["score"]))

        has_sessions = not self.sessions.empty
        has_grades = not self.grades.empty
        return {
            "total_hours": float(self.sessions["hours"].sum()) if has_sessions else 0.0,
            "session_count": int(len(self.sessions)),
            "avg_session_hours": float(self.sessions["hours"].mean()) if has_sessions else None,
            "course_count": len(self.courses()),
            "assessment_count": int(len(self.grades)),
            "avg_score": float(self.grades["score"].mean()) if has_grades else None,
            "hours_score_correlation": correlation,
        }
