"""Status logic for the FAHM dashboard — the decision layer (no UI).

Maps a scored window to a status string, keeping HEALTH (zmax bands) separate
from TRUST (the quality gate). This is the D34 health-vs-trust separation made
into UI state.
"""
import pandas as pd

STATUS_COLORS = {
    "healthy": "#2E8B57",    # green
    "watch": "#D4A017",      # yellow
    "alert": "#E07B39",      # orange
    "critical": "#C0392B",   # red
    "untrusted": "#808080",  # grey — data-quality gate fired
}

SUSPECTED_FAULT = {
    "tp3_decay_slope": "air leak (pressure decaying during idle)",
    "oil_median": "overwork heating (compensating for a leak)",
    "oil_std": "unstable thermal behavior",
    "oil_trend": "rising oil temperature",
    "duty": "abnormal load (compressor overworking)",
    "cycles_per_hour": "abnormal cycling rate",
    "cycle_dur_cv": "erratic cycling (short-cycling)",
    "longest_load_stretch": "sustained load without rest",
    "antiphase_share": "valve/pressure signal anomaly",
}


def window_status(scores: pd.DataFrame, i: int,
                  t_watch: float, t_alert: float, persist_k: int) -> str:
    """Status for window i, in priority order:
      1. quality gate fired  -> 'untrusted' (trust beats health)
      2. zmax >= t_alert for persist_k consecutive windows -> 'critical'
      3. zmax >= t_alert  -> 'alert'
      4. zmax >= t_watch  -> 'watch'
      5. else             -> 'healthy'
    Reads quality_bad from the scores frame (pipeline-emitted, D34)."""
    row = scores.iloc[i]
    if "quality_bad" in scores.columns and bool(row.get("quality_bad", False)):
        return "untrusted"
    z = row["zmax"]
    recent = scores["zmax"].iloc[max(0, i - persist_k + 1): i + 1]
    if len(recent) == persist_k and (recent >= t_alert).all():
        return "critical"
    if z >= t_alert:
        return "alert"
    if z >= t_watch:
        return "watch"
    return "healthy"


def suspected_fault(driver: str) -> str:
    """Human-readable fault hypothesis for a driver feature name."""
    return SUSPECTED_FAULT.get(driver, "abnormal behavior")
