"""Data loading for the FAHM dashboard — the reusable I/O layer (no Streamlit UI).

Mirrors the src/fahm ⟷ notebook split: this module is pure computation the app
calls; app.py handles presentation. Kept import-light so it can be unit-tested
or reused outside Streamlit.
"""
from pathlib import Path

import pandas as pd

SCORES_PATH = Path("data/processed/scores.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")
FAILURES_PATH = Path("data/processed/failure_windows.csv")
SENSORS_PATH = Path("data/processed/sensor_readings.parquet")

# analog signals shown on the live panel: (column, display label, unit)
SENSOR_DISPLAY = [
    ("Oil_temperature", "Oil temp", "\u00b0C"),
    ("Motor_current", "Motor current", "A"),
    ("TP3", "TP3 pressure", "bar"),
    ("TP2", "TP2 pressure", "bar"),
    ("DV_pressure", "DV pressure", "bar"),
    ("Reservoirs", "Reservoir", "bar"),
]


def load_scores(path: Path = SCORES_PATH) -> pd.DataFrame:
    """The pipeline's scoring artifact: window_start, zmax, alert, driver,
    label, and (post-D34) quality_bad. Sorted by time, index reset."""
    df = pd.read_parquet(path)
    df["window_start"] = pd.to_datetime(df["window_start"])
    return df.sort_values("window_start").reset_index(drop=True)


def load_features(path: Path = FEATURES_PATH):
    """The feature table (for the driver inspector). None if absent."""
    if path.exists():
        f = pd.read_parquet(path)
        f["window_start"] = pd.to_datetime(f["window_start"])
        return f.sort_values("window_start").reset_index(drop=True)
    return None


def load_failures(path: Path = FAILURES_PATH):
    """Documented failure windows (for jump buttons and timeline markers)."""
    if path.exists():
        fw = pd.read_csv(path, parse_dates=["start", "end"])
        return fw
    return None

def failure_indices(scores: pd.DataFrame, failures: pd.DataFrame) -> dict:
    """Map each failure_id -> the scores-row index nearest its start time
    (for jump-to-failure). Returns {} if failures is None."""
    if failures is None:
        return {}
    out = {}
    for _, f in failures.iterrows():
        idx = int((scores["window_start"] - f["start"]).abs().idxmin())
        out[f["failure_id"]] = idx
    return out


def build_sensor_lookup(scores: pd.DataFrame,
                        sensors_path: Path = SENSORS_PATH) -> pd.DataFrame | None:
    """Per-window LAST raw sensor reading (3b), indexed to align with `scores`.

    Reads the raw 10s sensor table once, and for each scored window takes the
    most recent raw reading at-or-before that window's window_start. Returns a
    DataFrame with one row per scored window (same order as `scores`) and the
    analog sensor columns — a fast lookup so replay never re-slices 1.5M rows.
    None if the raw file is absent.
    """
    if not sensors_path.exists():
        return None
    raw = pd.read_parquet(sensors_path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.sort_values("timestamp")

    cols = [c for c, _, _ in SENSOR_DISPLAY if c in raw.columns]
    # merge_asof: for each window_start, the last raw row at-or-before it
    left = scores[["window_start"]].sort_values("window_start")
    merged = pd.merge_asof(left, raw[["timestamp", *cols]],
                           left_on="window_start", right_on="timestamp",
                           direction="backward")
    # restore original scores order
    merged = merged.set_index(left.index).sort_index()
    return merged[cols]


def recent_alerts(scores: pd.DataFrame, t_alert: float, up_to_idx: int | None = None,
                  n: int = 8) -> pd.DataFrame:
    """The most recent windows over the alert threshold, newest first — feeds
    the 'jump to recent alarms' panel.

    If up_to_idx is given, only alerts at-or-before that replay position are
    shown (a live monitor only knows the past). Returns the alert rows plus
    their original scores-row index in an 'idx' column (for jump buttons)."""
    df = scores if up_to_idx is None else scores.iloc[: up_to_idx + 1]
    hits = df[df["zmax"] >= t_alert].copy()
    hits["idx"] = hits.index
    return hits.sort_values("window_start", ascending=False).head(n)
