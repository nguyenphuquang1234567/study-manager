"""Score prediction with SciPy linear regression.

Three models are tried in order, and the first one with enough data is used:
    1. "course"   - the user's own assessments in the selected course
    2. "personal" - all of the user's assessments, across courses
    3. "baseline" - the Kaggle dataset (5000 synthetic students)

Each model is a straight line, score = intercept + slope * hours, fitted with
scipy.stats.linregress on total study hours per assessment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from data_processing import DATA_DIR, load_csv
from models import MAX_SCORE, MIN_SCORE, DataFileError, InsufficientDataError, InvalidEntryError

BASELINE_PATH: Path = DATA_DIR / "dataset_study.csv"
BASELINE_COLUMNS: tuple[str, ...] = ("student_id", "study_hours", "grade")
KAGGLE_URL: str = "https://www.kaggle.com/datasets/andreylss/study-hours-vs-grades-dataset"
MIN_POINTS: int = 5
SIGNIFICANCE_LEVEL: float = 0.05


@dataclass(frozen=True)
class LinearModel:
    """A fitted line: score = intercept + slope * hours.

    Attributes:
        name: "course", "personal" or "baseline".
        slope: Points gained per extra hour of study.
        intercept: Predicted score at 0 hours.
        r_squared: Share of score variation explained by hours (0 to 1).
        p_value: Probability of a slope this far from 0 if there were no real trend.
        n_points: Number of data points used.
        hours_range: (min, max) hours in the data the model was fitted on.
    """

    name: str
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    n_points: int
    hours_range: tuple[float, float]

    def predict_raw(self, hours: float | np.ndarray) -> float | np.ndarray:
        """Unclipped prediction for one value or an array of hours."""
        result = self.intercept + self.slope * np.asarray(hours, dtype=float)
        return float(result) if result.ndim == 0 else result


@dataclass
class PredictionResult:
    """The outcome of GradePredictor.predict.

    Attributes:
        hours: Study hours the prediction is for.
        predicted_score: Predicted score, clipped to 0-100.
        model: The LinearModel that produced the prediction.
        warnings: Reasons to be careful with this prediction (show as warnings).
        notes: Why a model was skipped, if any (show as information).
    """

    hours: float
    predicted_score: float
    model: LinearModel
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def model_used(self) -> str:
        return self.model.name

    @property
    def slope(self) -> float:
        return self.model.slope

    @property
    def intercept(self) -> float:
        return self.model.intercept

    @property
    def r_squared(self) -> float:
        return self.model.r_squared

    @property
    def p_value(self) -> float:
        return self.model.p_value

    @property
    def n_points(self) -> int:
        return self.model.n_points

    def explanation(self) -> str:
        """A plain-language summary of the prediction and the model behind it."""
        sources = {
            "course": f"your {self.n_points} graded assessments in this course",
            "personal": f"all {self.n_points} of your graded assessments",
            "baseline": f"the Kaggle baseline of {self.n_points:,} students",
        }
        source = sources.get(self.model_used, f"{self.n_points} data points")
        if math.isnan(self.p_value):
            p_text = "p not available"
        elif self.p_value < 0.001:
            p_text = "p < 0.001"
        else:
            p_text = f"p = {self.p_value:.3f}"
        return (
            f"Based on {source}, each extra hour of study is associated with "
            f"{self.slope:+.1f} points (r squared {self.r_squared:.2f}, {p_text}). "
            f"Predicted score for {self.hours:g} hours: {self.predicted_score:.1f}."
        )


class GradePredictor:
    """Predicts a quiz or exam score from study hours.

    Attributes:
        baseline_data: Kaggle rows (study_hours, grade); useful for charts.
        baseline: LinearModel fitted on baseline_data.
        hours_range: (min, max) study hours in the Kaggle data.
        min_points: Graded assessments needed before a personal model is used.
    """

    def __init__(self, baseline_data: pd.DataFrame, min_points: int = MIN_POINTS) -> None:
        if min_points < 3:
            raise ValueError("min_points must be at least 3.")
        self.min_points = min_points
        self.baseline_data = baseline_data.loc[:, ["study_hours", "grade"]].reset_index(drop=True)
        self.baseline = self.fit(
            self.baseline_data["study_hours"],
            self.baseline_data["grade"],
            name="baseline",
            min_points=min_points,
        )
        self.hours_range = self.baseline.hours_range

    @classmethod
    def from_csv(cls, path: str | Path = BASELINE_PATH, min_points: int = MIN_POINTS) -> GradePredictor:
        """Load the Kaggle file and fit the baseline model.

        Raises:
            DataFileError: if the file is missing, unreadable, or has too few usable
                rows. The message says where to download it.
        """
        try:
            data, _ = load_csv(path, BASELINE_COLUMNS, numeric_columns=("study_hours", "grade"))
        except DataFileError as exc:
            raise DataFileError(
                f"Could not load the baseline data. {exc} "
                f"Download dataset_study.csv from {KAGGLE_URL} and put it in the data/ folder."
            ) from exc
        try:
            return cls(data, min_points=min_points)
        except InsufficientDataError as exc:
            raise DataFileError(
                f"{Path(path).name} does not contain enough usable rows. {exc}"
            ) from exc

    @staticmethod
    def fit(
        hours: Iterable[float],
        scores: Iterable[float],
        name: str = "custom",
        min_points: int = MIN_POINTS,
    ) -> LinearModel:
        """Fit score = intercept + slope * hours with scipy.stats.linregress.

        Pairs where either value is missing are ignored.

        Raises:
            InsufficientDataError: with fewer than min_points pairs, or when every
                hours value is the same (no trend can be measured).
        """
        x = np.asarray(list(hours), dtype=float)
        y = np.asarray(list(scores), dtype=float)
        if x.shape != y.shape:
            raise ValueError("hours and scores must have the same length.")
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]

        if x.size < min_points:
            verb = "is" if x.size == 1 else "are"
            raise InsufficientDataError(
                f"It needs at least {min_points} data points, but only {x.size} {verb} available."
            )
        if np.ptp(x) == 0:
            raise InsufficientDataError(
                "Every data point has the same study hours, so no trend can be measured."
            )
        try:
            result = stats.linregress(x, y)
        except ValueError as exc:  # scipy raises this when all x values are identical
            raise InsufficientDataError(f"The regression could not be fitted ({exc}).") from exc

        r_squared = float(result.rvalue) ** 2
        p_value = float(result.pvalue)
        if np.ptp(y) == 0:
            # Every score is the same: scipy returns NaN for r and p. The fitted line
            # is flat, so report that hours explain nothing and show no evidence of a trend.
            r_squared, p_value = 0.0, 1.0

        return LinearModel(
            name=name,
            slope=float(result.slope),
            intercept=float(result.intercept),
            r_squared=r_squared,
            p_value=p_value,
            n_points=int(x.size),
            hours_range=(float(x.min()), float(x.max())),
        )

    def predict(
        self,
        hours: float,
        course: str | None = None,
        personal_data: pd.DataFrame | None = None,
    ) -> PredictionResult:
        """Predict the score for a number of study hours.

        Args:
            hours: Total hours studied for the assessment (0 or more).
            course: Course to try a course-only model for; None skips that model.
            personal_data: Output of StudyTracker.hours_per_assessment()
                (needs columns course, hours, score). None or empty means
                only the baseline can be used.

        Raises:
            InvalidEntryError: if hours is not a number or is negative.
        """
        try:
            hours_value = float(hours)
        except (TypeError, ValueError):
            raise InvalidEntryError(f"Hours must be a number, got '{hours}'.") from None
        if not math.isfinite(hours_value) or hours_value < 0:
            raise InvalidEntryError(f"Hours must be 0 or more, got {hours}.")

        model, notes = self._choose_model(course, personal_data)
        raw_score = float(model.predict_raw(hours_value))
        score = float(np.clip(raw_score, MIN_SCORE, MAX_SCORE))

        warnings: list[str] = []
        low, high = model.hours_range
        if not low <= hours_value <= high:
            warnings.append(
                f"{hours_value:g} hours is outside the {low:g}-{high:g} hour range this model "
                "was fitted on, so the prediction is an extrapolation."
            )
        if score != raw_score:
            warnings.append(
                f"The model's raw estimate of {raw_score:.1f} was limited to the "
                f"{MIN_SCORE:g}-{MAX_SCORE:g} score range."
            )
        if model.name != "baseline" and not model.p_value < SIGNIFICANCE_LEVEL:
            warnings.append(
                f"The trend in your own data is not statistically significant "
                f"(p = {model.p_value:.2f}), so treat this prediction as rough."
            )

        return PredictionResult(
            hours=hours_value,
            predicted_score=score,
            model=model,
            warnings=warnings,
            notes=notes,
        )

    def _choose_model(
        self, course: str | None, personal_data: pd.DataFrame | None
    ) -> tuple[LinearModel, list[str]]:
        """Return the first model that can be fitted, plus notes on skipped models."""
        notes: list[str] = []
        if personal_data is None or len(personal_data) == 0:
            notes.append("Using the Kaggle baseline because there are no graded assessments yet.")
            return self.baseline, notes

        missing = {"course", "hours", "score"} - set(personal_data.columns)
        if missing:
            raise ValueError(f"personal_data is missing column(s): {', '.join(sorted(missing))}.")

        if course is not None:
            course_rows = personal_data[personal_data["course"] == course]
            try:
                model = self.fit(
                    course_rows["hours"], course_rows["score"], name="course", min_points=self.min_points
                )
                return model, notes
            except InsufficientDataError as exc:
                notes.append(f"A {course}-only model was not used. {exc}")

        try:
            model = self.fit(
                personal_data["hours"], personal_data["score"], name="personal", min_points=self.min_points
            )
            return model, notes
        except InsufficientDataError as exc:
            notes.append(f"A model of all your assessments was not used. {exc}")

        notes.append("Using the Kaggle baseline instead.")
        return self.baseline, notes

    @staticmethod
    def regression_line(
        model: LinearModel,
        n: int = 100,
        hours_range: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Points for drawing a model's line, clipped to 0-100.

        Args:
            model: The model to draw.
            n: Number of points.
            hours_range: (start, end) hours; defaults to the model's own data range.
                Pass a wider range to extend the line to a predicted point.
        """
        low, high = hours_range if hours_range is not None else model.hours_range
        x = np.linspace(low, high, n)
        y = np.clip(model.predict_raw(x), MIN_SCORE, MAX_SCORE)
        return x, y
