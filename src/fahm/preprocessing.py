"""Preprocessing for MetroPT-3 — SKELETON. Every function body is yours to write.

Design decisions you must make (write each into your decision log):
  * dtypes per column, and how to mark missing values in digital signals
  * how to parse/validate the timestamp
  * what "canonical column order" is
  * which consistency checks the data must pass, and what to do on failure
  * output format and location
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import yaml

def load_config(path: str | Path) -> dict:
    """Load the YAML config and resolve every entry in cfg["paths"]
    to an absolute path, anchored at the project root.

    Root is derived from the config's own location: configs/ sits one
    level below the repo root, so root = config_file.parent.parent.
    This makes all paths work regardless of where Python is launched from.
    """
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f)

    root = path.parent.parent
    cfg["paths"] = {k: str(root / v) for k, v in cfg["paths"].items()}
    return cfg
    
    raise NotImplementedError

ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
          "Oil_temperature", "Motor_current"]
DIGITAL = ["COMP", "DV_eletric", "Towers", "MPG", "LPS",
           "Pressure_switch", "Oil_level", "Caudal_impulses"]
TIMESTAMP = "timestamp"

def load_raw(cfg: dict) -> pd.DataFrame:
    """Typed load of the raw CSV.

    D04: analog -> float64 (kept; precision over memory, memory not a
    constraint), digital -> int8 (verified: no missing values in data).
    D05: timestamp parsed with explicit format (fast path); Unnamed: 0
    dropped AFTER noting it revealed 10x decimation of the 1Hz original.
    """
    df = pd.read_csv(cfg["paths"]["raw_csv"])

    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], format="%Y-%m-%d %H:%M:%S")
    df[DIGITAL] = df[DIGITAL].astype("int8")   # 1.0/0.0 floats -> 1/0
    df = df.drop(columns=["Unnamed: 0"])

    return df[[TIMESTAMP] + ANALOG + DIGITAL]  # canonical column order
    raise NotImplementedError


def profile(df: pd.DataFrame) -> dict:
    """Stages 5-6: first structural look.

    Return (and/or print) at least: shape, time range, modal sampling
    interval, per-analog describe(), per-digital value_counts(),
    missingness per column.
    """
    raise NotImplementedError


def run_checks(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stages 13-14: consistency checks the data MUST pass.

    Football analogy: assign_teams verification / goal consistency.
    Physics version — implement at least four checks, e.g.:
      - timestamps strictly increasing, no duplicates
      - modal interval equals the expected sampling rate
      - TP3 and Reservoirs track each other (they measure connected volumes)
      - COMP and DV_eletric are (mostly) in antiphase
      - analog values within plausible physical ranges
    Return a small DataFrame: check_name | passed | detail.
    """
    raise NotImplementedError


def find_gaps(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stage 14: recording gaps larger than cfg's threshold.

    Return gap_start | gap_end | gap_minutes, largest first.
    """
    raise NotImplementedError


def build_failure_windows(cfg: dict) -> pd.DataFrame:
    """Stage 9/12: the ground-truth table, SEPARATE from sensor data.

    Source: the MetroPT paper's failure reports — verify the timestamps
    yourself, this is your dataset-understanding homework. Columns you'll
    want: failure_id, fault_type, start, end.
    Remember: evaluation-only. Never features, never training labels.
    """
    raise NotImplementedError


def save_processed(df: pd.DataFrame, cfg: dict) -> Path:
    """Stage 16: persist the clean typed table. Your format choice — defend it."""
    raise NotImplementedError
