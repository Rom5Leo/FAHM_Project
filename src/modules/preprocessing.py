"""Preprocessing for MetroPT-3 — SKELETON. Every function body is yours to write.

This mirrors the stages of your football preprocessing notebook (see
docs/workflow_mapping.md for the stage-by-stage mapping). The notebook should
only CALL these functions and display their results.

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


def load_config(path: str | Path) -> dict:
    """Read the YAML config into a dict.

    Hint: yaml.safe_load. Keep this the ONLY place that opens the config.
    """
    raise NotImplementedError


def load_raw(cfg: dict) -> pd.DataFrame:
    """Stage 4: read the raw CSV into a typed DataFrame.

    Your decisions: dtypes (analog vs digital), timestamp parsing, dropping
    the unnamed index column, column order. Aim: `df.info()` after this
    should show sensible dtypes and a fraction of naive-load memory.
    """
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
