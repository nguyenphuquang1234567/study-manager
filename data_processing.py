"""Loading, cleaning, saving and summarizing study data with Pandas.

The two user files are:
    data/study_log.csv  columns: date, course, hours, note
    data/grades.csv     columns: date, course, assessment_type, assessment_name, score

Both are created automatically (headers only) the first time the app runs.
"""

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


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def clean_types(
    df: pd.DataFrame,
    required_columns: tuple[str, ...] | list[str],
    *,
    date_columns: tuple[str, ...] = (),
    numeric_columns: tuple[str, ...] = (),
    optional_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, int]:
    """Convert column types and drop rows that cannot be used.

    Dates must be YYYY-MM-DD and numbers must be numeric; anything else becomes
    missing. Text is stripped of surrounding spaces. A row is dropped when any
    column that is not optional is missing or blank.

    Returns:
        The cleaned DataFrame (only the required columns, index reset) and the
        number of rows that were dropped.
    """
    df = df.loc[:, list(required_columns)].copy()

    for column in date_columns:
        parsed = pd.to_datetime(df[column], errors="coerce", format="ISO8601")
        df[column] = parsed.dt.normalize().astype("datetime64[ns]")
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)

    text_columns = [c for c in df.columns if c not in date_columns and c not in numeric_columns]
    for column in text_columns:
        df[column] = df[column].astype("object").fillna("").astype(str).str.strip()

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
    create_if_missing: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Read a CSV file safely and return (DataFrame, number of dropped rows).

    Args:
        path: File to read.
        required_columns: Columns that must be present in the header.
        date_columns, numeric_columns, optional_columns: See clean_types.
        create_if_missing: If True, a missing file is created with only the header
            row and an empty DataFrame is returned. Use this for the user's own
            files; leave it False for required input such as the Kaggle data.

    Raises:
        DataFileError: if the file is missing (and create_if_missing is False),
            empty, not valid CSV, or missing a required column.
    """
    path = Path(path)
    types: dict[str, tuple[str, ...]] = {
        "date_columns": date_columns,
        "numeric_columns": numeric_columns,
        "optional_columns": optional_columns,
    }

    if not path.exists():
        if not create_if_missing:
            raise DataFileError(f"File not found: {path}")
        empty = pd.DataFrame(columns=list(required_columns))
        save_csv(empty, path)
        return clean_types(empty, required_columns, **types)

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
    """Write a DataFrame to CSV without the index, with dates as YYYY-MM-DD.

    Raises:
        DataFileError: if the file cannot be written.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, date_format="%Y-%m-%d")
    except OSError as exc:
        raise DataFileError(f"Could not save {path.name} ({exc}).") from exc


def _append_row(existing: pd.DataFrame, row: pd.DataFrame) -> pd.DataFrame:
    """Append one cleaned row and keep the table sorted by date."""
    combined = row.copy() if existing.empty else pd.concat([existing, row], ignore_index=True)
    return combined.sort_values("date", kind="stable", ignore_index=True)


# ---------------------------------------------------------------------------
# StudyTracker
# ---------------------------------------------------------------------------

class StudyTracker:
    """Holds the user's study sessions and grades and computes statistics from them.

    Attributes:
        sessions_path: CSV file for study sessions.
        grades_path: CSV file for grades.
        sessions: DataFrame with columns date, course, hours, note.
        grades: DataFrame with columns date, course, assessment_type, assessment_name, score.
        skipped_rows: Rows dropped during the last load because they were malformed
            (bad date or number, blank required field, hours outside (0, 24],
            score outside 0-100, or an unknown assessment type). They are removed
            from the file the next time it is saved.

    Raises:
        DataFileError: from __init__ / reload if a user file exists but is corrupt.
    """

    def __init__(
        self,
        sessions_path: str | Path = DEFAULT_SESSIONS_PATH,
        grades_path: str | Path = DEFAULT_GRADES_PATH,
    ) -> None:
        self.sessions_path = Path(sessions_path)
        self.grades_path = Path(grades_path)
        self.sessions = pd.DataFrame(columns=list(SESSION_COLUMNS))
        self.grades = pd.DataFrame(columns=list(GRADE_COLUMNS))
        self.skipped_rows = 0
        self.reload()

    # ----- loading and saving -------------------------------------------

    def reload(self) -> None:
        """Read both user files again (creating them if they do not exist)."""
        sessions, bad_sessions = load_csv(
            self.sessions_path, SESSION_COLUMNS, create_if_missing=True, **_SESSION_TYPES
        )
        grades, bad_grades = load_csv(
            self.grades_path, GRADE_COLUMNS, create_if_missing=True, **_GRADE_TYPES
        )

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
        """Add a validated study session and save it to the sessions file."""
        if not isinstance(session, StudySession):
            raise TypeError("add_session expects a StudySession.")
        row, _ = clean_types(pd.DataFrame([session.to_dict()]), SESSION_COLUMNS, **_SESSION_TYPES)
        self.sessions = _append_row(self.sessions, row)
        save_csv(self.sessions, self.sessions_path)

    def add_grade(self, record: GradeRecord) -> None:
        """Add a validated grade and save it to the grades file."""
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
        """Replace the user's files with the sample files, then reload.

        Raises:
            DataFileError: if a sample file is missing or cannot be copied.
        """
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

    # ----- statistics ----------------------------------------------------

    def courses(self) -> list[str]:
        """Sorted names of every course that has a session or a grade."""
        return sorted(set(self.sessions["course"]) | set(self.grades["course"]))

    def hours_by_course(self) -> pd.DataFrame:
        """Total hours, number of sessions and average session length per course.

        Returns:
            Columns course, total_hours, sessions, avg_session_hours,
            sorted by total_hours (largest first).
        """
        if self.sessions.empty:
            return pd.DataFrame(columns=["course", "total_hours", "sessions", "avg_session_hours"])
        return (
            self.sessions.groupby("course")["hours"]
            .agg(total_hours="sum", sessions="count", avg_session_hours="mean")
            .reset_index()
            .sort_values("total_hours", ascending=False, ignore_index=True)
        )

    def course_summary(self) -> pd.DataFrame:
        """One row per course combining study hours with grade results.

        Returns:
            Columns course, total_hours, sessions, avg_session_hours, assessments,
            avg_score. A course with no grades has avg_score NaN.
        """
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
        summary = hours.merge(scores, on="course", how="outer")
        for column in ("total_hours",):
            summary[column] = summary[column].astype(float).fillna(0.0)
        for column in ("sessions", "assessments"):
            summary[column] = summary[column].astype(float).fillna(0).astype(int)
        summary["avg_session_hours"] = summary["avg_session_hours"].astype(float)
        summary["avg_score"] = summary["avg_score"].astype(float)
        return summary[columns].sort_values("course", ignore_index=True)

    def weekly_hours(self, course: str | None = None) -> pd.DataFrame:
        """Total study hours per week (weeks end on Sunday), including empty weeks.

        Args:
            course: Only count this course; None counts every course.

        Returns:
            Columns week_ending, hours.
        """
        sessions = self.sessions if course is None else self.sessions[self.sessions["course"] == course]
        if sessions.empty:
            return pd.DataFrame(columns=["week_ending", "hours"])
        weekly = sessions.groupby(pd.Grouper(key="date", freq="W"))["hours"].sum().reset_index()
        return weekly.rename(columns={"date": "week_ending"})

    def hours_per_assessment(self) -> pd.DataFrame:
        """Total study hours that went into each graded assessment.

        Each session is assigned to the next assessment of the same course on or
        after its date (pd.merge_asof, direction "forward"). Assessments of the
        same course on the same day share those hours. Sessions after the last
        assessment are not counted here (see hours_since_last_assessment).

        Returns:
            Columns course, date, assessment_type, assessment_name, score, hours,
            sorted by course then date.
        """
        if self.grades.empty:
            return pd.DataFrame(columns=list(ASSESSMENT_HOURS_COLUMNS))

        result = self.grades.copy()
        if self.sessions.empty:
            result["hours"] = 0.0
        else:
            due_dates = (
                self.grades[["course", "date"]]
                .drop_duplicates()
                .sort_values("date", ignore_index=True)
            )
            due_dates["due_date"] = due_dates["date"]
            sessions = self.sessions[["date", "course", "hours"]].sort_values(
                "date", kind="stable", ignore_index=True
            )
            assigned = pd.merge_asof(sessions, due_dates, on="date", by="course", direction="forward")
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
        """Hours studied for a course after its most recent assessment.

        These are the hours going into the next, not yet graded, assessment.
        If the course has no assessments yet, every session counts.
        """
        sessions = self.sessions[self.sessions["course"] == course]
        if sessions.empty:
            return 0.0
        course_grades = self.grades[self.grades["course"] == course]
        if course_grades.empty:
            return float(sessions["hours"].sum())
        last_assessment = course_grades["date"].max()
        return float(sessions.loc[sessions["date"] > last_assessment, "hours"].sum())

    def summary_stats(self) -> dict[str, Any]:
        """Headline numbers for the Statistics tab.

        Returns:
            dict with total_hours, session_count, avg_session_hours, course_count,
            assessment_count, avg_score and hours_score_correlation. Averages are
            None when there is no data; the correlation is None with fewer than
            3 graded assessments or when hours or scores never vary.
        """
        per_assessment = self.hours_per_assessment()
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
