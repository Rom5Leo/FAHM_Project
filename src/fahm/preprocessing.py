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
    

def load_naive(cfg: dict) -> pd.DataFrame:
    """Plain read_csv, no typing — the 'before' baseline (D02). Overview only."""
    return pd.read_csv(cfg["paths"]["raw_csv"])

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


def profile(df: pd.DataFrame) -> dict:
    """First structural look. Returns a dict; the notebook displays parts."""
    diffs = df[TIMESTAMP].diff().dropna()
    modal = diffs.mode()[0]
 
    result = {
        "shape": df.shape,
        "time_range": (df[TIMESTAMP].min(), df[TIMESTAMP].max()),
        "modal_interval": modal,
        "modal_share": (diffs == modal).mean(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
        "missing_per_column": df.isna().sum(),
        "analog_describe": df[ANALOG].describe().T,
        "digital_value_counts": {col: df[col].value_counts() for col in DIGITAL},
        "interval_distribution": diffs.value_counts(),   # full 337-row table
    }
    return result

def _check_valve_antiphase(df: pd.DataFrame, eps: float) -> dict:
    """Note: detail carries numbers even on pass."""
    dev = abs((df["COMP"] + df["DV_eletric"]).mean() - 1.0)
    return {
        "check": "valve_antiphase",
        "passed": bool(dev < eps),
        "detail": f"|mean(COMP+DV_eletric)-1| = {dev:.4f} vs eps {eps}",
    }
 
 
def _check_tp3_reservoirs(df: pd.DataFrame, eps: float, max_diff: float) -> dict:
    """two conditions, one row."""
    d = (df["TP3"] - df["Reservoirs"]).abs()
    m, mx = d.mean(), d.max()
    return {
        "check": "tp3_reservoirs_agree",
        "passed": bool(m < eps and mx < max_diff),
        "detail": f"mean {m:.4f} vs {eps}; max {mx:.3f} vs {max_diff}",
    }
 
 
def _check_timestamps_monotonic(df: pd.DataFrame) -> dict:
    diffs = df[TIMESTAMP].diff().dropna()
    n_bad = int((diffs <= pd.Timedelta(0)).sum())
    return {
        "check": "timestamps_strictly_increasing",
        "passed": bool(n_bad == 0),
        "detail": f"{n_bad} non-increasing steps in {len(diffs):,}",
    }
 
 
def _check_modal_interval(df: pd.DataFrame, expected_s: int) -> dict:
    diffs = df[TIMESTAMP].diff().dropna()
    modal = diffs.mode()[0]
    share = (diffs == modal).mean()
    return {
        "check": "modal_interval",
        "passed": bool(modal == pd.Timedelta(seconds=expected_s)),
        "detail": f"modal {modal} ({share:.1%} of steps) vs expected {expected_s}s",
    }
 
 
def _check_digitals_binary(df: pd.DataFrame) -> dict:
    offenders = {}
    for col in DIGITAL:
        s = df[col]
        n_zeros = int((s == 0).sum())
        n_ones = int((s == 1).sum())
        n_other = len(s) - n_zeros - n_ones
        if n_other:
            offenders[col] = n_other
    return {
        "check": "digitals_binary",
        "passed": not offenders,
        "detail": f"all {len(DIGITAL)} binary" if not offenders else f"violations: {offenders}",
    }
 
 
def _check_analog_ranges(df: pd.DataFrame, ranges: list[dict]) -> dict:
    offenders = {}
    for r in ranges:
        s = df[r["sensor"]]
        n_out = int(((s < r["min"]) | (s > r["max"])).sum())
        if n_out:
            offenders[r["sensor"]] = int(n_out)

    configured = {r["sensor"] for r in ranges}
    unconfigured = set(ANALOG) - configured        # data with no check
    unknown = configured - set(ANALOG)             # config with no data
    for sensor in unconfigured:
        offenders[sensor] = "unconfigured"
    for sensor in unknown:
        offenders[sensor] = "not in data"   

    return {
        "check": "analog_ranges",
        "passed": not offenders,
        "detail": "all within configured ranges" if not offenders else f"out of range: {offenders}",
    }

def run_checks(df: pd.DataFrame, cfg: dict, on_fail: str = "warn") -> pd.DataFrame:
    """All checks -> results table (check|passed|detail). Posture per D09."""
    if on_fail not in ("warn", "raise"):
        raise ValueError(f"on_fail must be 'warn' or 'raise', got {on_fail!r}")
 
    c = cfg["checks"]
    results = pd.DataFrame([
        _check_timestamps_monotonic(df),
        _check_modal_interval(df, cfg["preprocessing"]["expected_interval_seconds"]),
        _check_digitals_binary(df),
        _check_tp3_reservoirs(df, c["tp3_reservoirs_eps"], c["tp3_reservoirs_max"]),
        _check_valve_antiphase(df, c["valve_antiphase_eps"]),
        _check_analog_ranges(df, c["analog_ranges"]),
    ])
 
    failed = results[~results["passed"]]
    if on_fail == "raise" and len(failed):
        raise ValueError(
            f"{len(failed)} of {len(results)} data checks FAILED:\n"
            f"{failed.to_string(index=False)}"
        )
        
    return results

def find_gaps(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    thr = pd.Timedelta(seconds=cfg["preprocessing"]["gap_threshold_seconds"])
    diffs = df[TIMESTAMP].diff()
    mask = diffs > thr
 
    gaps = pd.DataFrame({
        "gap_start": df[TIMESTAMP].shift(1)[mask],
        "gap_end": df.loc[mask, TIMESTAMP],
        "gap_minutes": diffs[mask].dt.total_seconds()/60,
    })
    return gaps.sort_values("gap_minutes", ascending=False)

def build_failure_windows(cfg: dict) -> pd.DataFrame:
    """Ground-truth table from config -> parsed dates -> saved CSV.
 
    Source-verified failure windows (D07). Evaluation-only: never features,
    never training labels. F1's maintenance is null in config -> becomes NaT.
    """
    fw = pd.DataFrame(cfg["failure_windows"])
    for col in ("start", "end", "maintenance"):
        fw[col] = pd.to_datetime(fw[col])
    
    out = Path(cfg["paths"]["failure_windows"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fw.to_csv(out, index=False)
    return fw


def save_processed(df: pd.DataFrame, cfg: dict) -> Path:
    """Persist the typed, validated table as parquet (D11).
 
    Parquet over CSV (loses dtypes -> every load re-parses) and pickle
    (Python-only, unsafe, version-fragile): columnar, compressed, dtypes
    preserved, partial column reads. Full path lives in config -> the
    artifact's location is a documented parameter, not a code detail.
    Requires: poetry add pyarrow
    """
    out = Path(cfg["paths"]["processed"])
    if out.suffix != ".parquet":
        raise ValueError(
            f"cfg paths.processed should be a full .parquet file path, got {out!r} — "
            "update config: processed: data/processed/sensor_readings.parquet"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out
