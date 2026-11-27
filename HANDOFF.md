# Handoff: P1–P3 done, notes for P4–P6

P1–P3 are finished: `models.py`, `data_processing.py`, `analysis.py`, the Kaggle file and the sample data. `app.py` and `visualization.py` are empty and ready for P4.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Try it in a Python shell

```python
from data_processing import StudyTracker
from analysis import GradePredictor

tracker = StudyTracker()          # data/study_log.csv and data/grades.csv, created if missing
tracker.load_sample_data()        # overwrites those two files with the sample data
predictor = GradePredictor.from_csv()

hours = tracker.hours_since_last_assessment("Calculus")
result = predictor.predict(hours, course="Calculus", personal_data=tracker.hours_per_assessment())
print(result.explanation())
```

With the sample data, Calculus (6 assessments) uses its own `"course"` model, and Chemistry (3 assessments) falls back to `"personal"`.

## What each tab should call

**Startup (app.py)**
- `StudyTracker()` raises `DataFileError` if a user CSV exists but is corrupt. Catch it, show `st.error(str(exc))` saying which file to fix or delete, then `st.stop()`.
- Load `GradePredictor.from_csv()` inside a function decorated with `@st.cache_resource`. If it raises `DataFileError`, keep going without a predictor and show `st.error(str(exc))` only in the Prediction tab. The message already contains the Kaggle download link.

**Log Study**
- `StudySession(date, course, hours, note)` inside `try/except InvalidEntryError` → `st.warning(str(exc))`.
- `tracker.add_session(session)` saves it.
- Course dropdown: `tracker.courses()`. Table: `tracker.sessions` (date, course, hours, note).

**Log Grade**
- `GradeRecord(date, course, assessment_type, assessment_name, score)`, same pattern. `assessment_type` is `"quiz"` or `"exam"` (any capitalization).
- `tracker.add_grade(record)` saves it. Table: `tracker.grades`.

**Sidebar**
- `tracker.load_sample_data()` replaces the user's files, so put it behind a confirmation checkbox.
- `tracker.skipped_rows`: show it when it is not 0. These malformed rows are removed from the file the next time it is saved.

**Statistics**
- `tracker.summary_stats()` returns a dict: `total_hours`, `session_count`, `avg_session_hours`, `course_count`, `assessment_count`, `avg_score`, `hours_score_correlation`. Averages and the correlation can be `None`.
- `tracker.course_summary()`: per-course table (course, total_hours, sessions, avg_session_hours, assessments, avg_score).
- `tracker.hours_by_course()`: data for the bar chart (V2).
- `tracker.weekly_hours(course=None)`: week_ending, hours (V3).
- `tracker.grades`: grade history (V4).

**Prediction**
- Default hours: `tracker.hours_since_last_assessment(course)`.
- `predictor.predict(hours, course=course, personal_data=tracker.hours_per_assessment())` returns a `PredictionResult`. Catch `InvalidEntryError`. `InsufficientDataError` is already handled inside `predict` by falling back to the next model.
- `PredictionResult` fields: `predicted_score`, `model_used` (`"course"`, `"personal"` or `"baseline"`), `slope`, `intercept`, `r_squared`, `p_value`, `n_points`, `explanation()`, `warnings` (one `st.warning` each), `notes` (one `st.info` each), `model`.

**Prediction chart (V1)**
- Faded Kaggle points: `predictor.baseline_data` (columns study_hours, grade).
- Baseline line: `GradePredictor.regression_line(predictor.baseline)`.
- Personal line, when `result.model_used != "baseline"`: `GradePredictor.regression_line(result.model)`.
- User's points: `tracker.hours_per_assessment()` (columns hours, score).
- Predicted point: `(result.hours, result.predicted_score)`.
- To extend a line to the predicted point, pass `hours_range=(0, max_hours)`.

## Exceptions (all in models.py)

| Exception | Raised by | Show with |
|---|---|---|
| `InvalidEntryError` | `StudySession`, `GradeRecord`, `predictor.predict` | `st.warning` |
| `DataFileError` | `StudyTracker()`, `add_session`, `add_grade`, `load_sample_data`, `GradePredictor.from_csv` | `st.error` |
| `InsufficientDataError` | `GradePredictor.fit` (handled inside `predict`) | not needed in the app |

## Notes for tests (P5)

- `StudyTracker(tmp_path / "s.csv", tmp_path / "g.csv")` keeps tests away from the real data.
- `load_sample_data(sessions_sample, grades_sample)` accepts other sample files.
- `GradePredictor(pd.DataFrame({"study_hours": [...], "grade": [...]}))` builds a predictor without the Kaggle file.
- `GradePredictor.fit(hours, scores)` returns a `LinearModel` with `slope` and `intercept`.

## Facts for the README (P6)

- Kaggle data: synthetic, 5000 rows, no missing values. Study hours actually range 0–10 (the data card says 0–12). Generated as grade = 40 + 5 × hours + noise.
- Baseline fit: slope 5.01, intercept 39.7, r² 0.68.
- Hours per assessment: each session counts toward the next assessment of the same course on or after its date. Assessments on the same day share those hours.
- A personal model needs at least 5 graded assessments. Predictions are clipped to 0–100.
