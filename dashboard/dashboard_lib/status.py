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


def alarm_events(scores, t_watch: float, t_alert: float, persist_k: int,
                 up_to_idx: int | None = None):
    """Extract ALARM EVENTS — moments the status transitions INTO a non-healthy
    level (watch / alert / critical). A sustained alert is one event (its onset),
    not one per window. Returns a list of dicts newest-first:
        {idx, window_start, level, zmax, driver}
    If up_to_idx is given, only events at-or-before that replay position (a live
    monitor only knows the past).

    'Rising into a worse state' is what a person watches: healthy->watch,
    watch->alert, healthy->alert, ->critical. Staying at the same level or
    recovering is not a new event.
    """
    import pandas as pd  # local; keep module import-light

    n = len(scores) if up_to_idx is None else up_to_idx + 1
    rank = {"healthy": 0, "watch": 1, "alert": 2, "critical": 3, "untrusted": 0}
    events, prev = [], "healthy"
    for i in range(n):
        s = window_status(scores, i, t_watch, t_alert, persist_k)
        if rank[s] > rank[prev] and s in ("watch", "alert", "critical"):
            row = scores.iloc[i]
            events.append({
                "idx": i,
                "window_start": row["window_start"],
                "level": s,
                "zmax": float(row["zmax"]),
                "driver": row.get("driver", "?"),
            })
        prev = s
    return events[::-1]  # newest first


def group_alarms_by_failure(events: list, failures, lead_days: int = 14) -> dict:
    """Group alarms by the failure they precede. Each alarm belongs to at most
    ONE failure (the first, in chronological failure order, whose window
    [start - lead_days, end] contains it)."""
    import pandas as pd

    groups, remaining = {}, list(events)
    if failures is not None:
        for _, f in failures.sort_values("start").iterrows():
            start = pd.to_datetime(f["start"])
            end = pd.to_datetime(f["end"]) if pd.notna(f.get("end")) else start
            lo = start - pd.Timedelta(days=lead_days)
            grp = [e for e in remaining if lo <= e["window_start"] <= end]
            if grp:
                groups[str(f["failure_id"])] = grp
                claimed = {id(e) for e in grp}
                remaining = [e for e in remaining if id(e) not in claimed]
    if remaining:
        groups["other"] = remaining
    return groups