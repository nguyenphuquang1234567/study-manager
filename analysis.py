"""Score prediction with SciPy linear regression (course -> personal -> Kaggle baseline)."""

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
MIN_POINTS: int = 5  # assessments needed before a personal model is used
SIGNIFICANCE_LEVEL: float = 0.05


@dataclass(frozen=True)
class LinearModel:
    """A fitted line: score = intercept + slope * hours."""

    name: str  # "course", "personal" or "baseline"
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    n_points: int
    hours_range: tuple[float, float]  # (min, max) hours in the fitted data

    def predict_raw(self, hours: float | np.ndarray) -> float | np.ndarray:
        """Prediction without clipping, for a number or an array."""
        result = self.intercept + self.slope * np.asarray(hours, dtype=float)
        return float(result) if result.ndim == 0 else result


@dataclass
class PredictionResult:
    """Result of GradePredictor.predict."""

    hours: float
    predicted_score: float  # clipped to 0-100
    model: LinearModel
    warnings: list[str] = field(default_factory=list)  # reasons to be careful
    notes: list[str] = field(default_factory=list)  # why a model was skipped

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
        """Plain-language summary of the prediction."""
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
    """Predicts a quiz or exam score from study hours."""

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
        """Load the Kaggle file and fit the baseline; raises DataFileError."""
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
        """Fit a line with scipy.stats.linregress; raises InsufficientDataError."""
        x = np.asarray(list(hours), dtype=float)
        y = np.asarray(list(scores), dtype=float)
        if x.shape != y.shape:
            raise ValueError("hours and scores must have the same length.")
        # Ignore pairs with a missing value.
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
        # Least-squares line through the points.
        try:
            result = stats.linregress(x, y)
        except ValueError as exc:  # scipy raises this when all x values are identical
            raise InsufficientDataError(f"The regression could not be fitted ({exc}).") from exc

        r_squared = float(result.rvalue) ** 2
        p_value = float(result.pvalue)
        if np.ptp(y) == 0:
            # All scores equal: scipy gives NaN, so report no fit and no trend.
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
        """Predict the score for some study hours (personal_data = hours_per_assessment())."""
        try:
            hours_value = float(hours)
        except (TypeError, ValueError):
            raise InvalidEntryError(f"Hours must be a number, got '{hours}'.") from None
        if not math.isfinite(hours_value) or hours_value < 0:
            raise InvalidEntryError(f"Hours must be 0 or more, got {hours}.")

        # Pick a model, plug in the hours, keep the score within 0-100.
        model, notes = self._choose_model(course, personal_data)
        raw_score = float(model.predict_raw(hours_value))
        score = float(np.clip(raw_score, MIN_SCORE, MAX_SCORE))

        # Flag anything that makes this prediction less reliable.
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
        """Return the first model with enough data, plus notes on skipped models."""
        notes: list[str] = []
        if personal_data is None or len(personal_data) == 0:
            notes.append("Using the Kaggle baseline because there are no graded assessments yet.")
            return self.baseline, notes

        missing = {"course", "hours", "score"} - set(personal_data.columns)
        if missing:
            raise ValueError(f"personal_data is missing column(s): {', '.join(sorted(missing))}.")

        # 1. The selected course only.
        if course is not None:
            course_rows = personal_data[personal_data["course"] == course]
            try:
                model = self.fit(
                    course_rows["hours"], course_rows["score"], name="course", min_points=self.min_points
                )
                return model, notes
            except InsufficientDataError as exc:
                notes.append(f"A {course}-only model was not used. {exc}")

        # 2. All of the user's assessments, across courses.
        try:
            model = self.fit(
                personal_data["hours"], personal_data["score"], name="personal", min_points=self.min_points
            )
            return model, notes
        except InsufficientDataError as exc:
            notes.append(f"A model of all your assessments was not used. {exc}")

        # 3. Not enough personal data: use the Kaggle baseline.
        notes.append("Using the Kaggle baseline instead.")
        return self.baseline, notes

    @staticmethod
    def regression_line(
        model: LinearModel,
        n: int = 100,
        hours_range: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Points for drawing a model's line, clipped to 0-100."""
        low, high = hours_range if hours_range is not None else model.hours_range
        x = np.linspace(low, high, n)
        y = np.clip(model.predict_raw(x), MIN_SCORE, MAX_SCORE)
        return x, y
