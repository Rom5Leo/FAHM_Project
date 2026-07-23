"""Analysis helpers — computations about the data, reused across notebooks 03+.
(plotting.py draws, preprocessing.py prepares, analysis.py measures.)"""

from __future__ import annotations

import pandas as pd

from fahm.preprocessing import TIMESTAMP


def window_mask(df: pd.DataFrame, lo, hi) -> pd.Series:
    """Boolean mask for TIMESTAMP in [lo, hi)."""
    return (df[TIMESTAMP] >= pd.Timestamp(lo)) & (df[TIMESTAMP] < pd.Timestamp(hi))


def duty(df: pd.DataFrame, lo, hi) -> float:
    """Compressor load duty (mean DV_eletric) over a time window."""
    return float(df.loc[window_mask(df, lo, hi), "DV_eletric"].mean())


def toggles_per_hour(df: pd.DataFrame, signal: str, lo, hi) -> float:
    """State changes per hour of a digital signal over a window."""
    w = df.loc[window_mask(df, lo, hi)]
    hours = (w[TIMESTAMP].max() - w[TIMESTAMP].min()).total_seconds() / 3600
    return float((w[signal] != w[signal].shift()).sum() / hours)

def antiphase_share(df: pd.DataFrame, lo, hi) -> float:
    """Fraction of samples in [lo, hi) where COMP and DV_eletric are in
    antiphase (sum == 1). ~0.99 in normal operation (D08); a drop flags
    valve/control anomalies (chatter investigation, stage 3)."""
    w = df.loc[window_mask(df, lo, hi)]
    return float((w["COMP"] + w["DV_eletric"] == 1).mean())

def find_state_episodes(df, predicate, lo, hi, coarse="6h", fine="15min") -> pd.DataFrame:
    """Episodes ([start, end]) where predicate is True (D15).
 
    Coarse scan finds windows where the predicate holds; boundaries are then
    refined at `fine` resolution. LIMITATION: an episode is only detected if
    it DOMINATES at least one coarse window, so the minimum reliably
    detectable episode length ~= `coarse` — shorten `coarse` to hunt shorter
    episodes (costlier scan). Predicate must return False on empty windows
    (gap != fault, D15c).
    """

    def scan(a, b, step):
        edges = pd.date_range(a, b, freq=step)
        if edges[-1] < pd.Timestamp(b): #tail sliver: include
            edges = edges.append(pd.DatetimeIndex([b]))

        states = [predicate(df, x, y) for x, y in zip(edges[:-1], edges[1:])]

        return states, list(edges)

    states, edges = scan(lo, hi, coarse)

    # (1) coarse flips: windows whose state differs from their predecessor
    flip_pos = [i for i in range(1, len(states)) if states[i] != states[i-1]]

    # (2) refine each flip — clamp so refined times never go backwards
    refined = []
    last_t = lo
    for i in flip_pos:
        f_states, f_edges = scan(edges[i - 1], edges[i + 1], fine)
        before = states[i - 1]
        t = next((f_edges[j] for j, s in enumerate(f_states) if s != before), edges[i])
        t = max(t, last_t)          # never earlier than the previous flip
        last_t = t
        refined.append((t, states[i]))

    # (3) pair flips into episodes (state machine)
    episodes, open_start = [], (lo if states[0] else None)
    for t, new_state in refined:
        if new_state and open_start is None:
            open_start = t                                                                     # Flase->True: open
        elif not new_state and open_start is not None:
            episodes.append((open_start, t))                                                   # True->False:close
            open_start = None
    if open_start is not None:
        episodes.append((open_start,hi))                                                       # still open at range end
    

    out = pd.DataFrame(episodes, columns=["episode_start", "episode_end"])

    # Degenerate episodes (end == start) occur when a brief flicker straddles
    # a coarse-window boundary — both flips refine into the same fine window.
    # They are below the resolution this scan can describe: drop them.
    n_degenerate = int((out["episode_end"] <= out["episode_start"]).sum())
    out = out[out["episode_end"] > out["episode_start"]].reset_index(drop=True)
    if n_degenerate:
        print(f"note: dropped {n_degenerate} sub-resolution episode(s)")
    return out
    

def stuck_instrument(df, lo, hi, var_eps=1e-4, anti_eps=0.5) -> bool:
    """Predicate: window shows the frozen-analog / broken-antiphase signature."""
    w = df.loc[window_mask(df, lo, hi)]
    if len(w) == 0:
        return False           # no data ≠ faulty instrument (it's a gap)
    return bool(w["Motor_current"].var() < var_eps and antiphase_share(df, lo, hi) < anti_eps)

def gaps_near_failures(gaps: pd.DataFrame, fw: pd.DataFrame,
                       pad_days: float = 3) -> pd.DataFrame:
    """Analysis A — gaps overlapping each failure's window ± pad_days.

    Returns one row per (failure, nearby gap) with the RELATION that matters
    for reading the story (a gap starting at a window's end = repair; one
    ending later = return to service; mere coexistence = little).
    Empty result for a failure means no gaps nearby (also informative).
    """
    rows = []
    for _, f in fw.iterrows():
        lo = f["start"] - pd.Timedelta(days=pad_days)
        hi = f["end"] + pd.Timedelta(days=pad_days)
        near = gaps[(gaps["gap_end"] >= lo) & (gaps["gap_start"] <= hi)]
        for _, g in near.iterrows():
            rows.append({
                "failure_id": f["failure_id"],
                "gap_start": g["gap_start"],
                "gap_end": g["gap_end"],
                "gap_minutes": round(g["gap_minutes"], 1),
                # negative = gap starts BEFORE the window ends (repair-shaped)
                "start_vs_window_end_h": round(
                    (g["gap_start"] - f["end"]).total_seconds() / 3600, 1),
                "end_vs_maint_h": (
                    round((g["gap_end"] - f["maintenance"]).total_seconds() / 3600, 1)
                    if pd.notna(f["maintenance"]) else None),
            })
    return pd.DataFrame(rows)


def gaps_nearest_event(gaps: pd.DataFrame, fw: pd.DataFrame,
                       top_n: int = 15) -> pd.DataFrame:
    """Analysis B — for the largest gaps, the nearest known event.

    Known events = failure starts/ends and maintenance dates. Large
    `days_away` marks an UNEXPLAINED large gap (undocumented-event candidate).
    """
    events = pd.concat([
        fw[["failure_id", "start"]].rename(columns={"start": "when"}).assign(kind="failure_start"),
        fw[["failure_id", "end"]].rename(columns={"end": "when"}).assign(kind="failure_end"),
        fw.dropna(subset=["maintenance"])[["failure_id", "maintenance"]]
          .rename(columns={"maintenance": "when"}).assign(kind="maintenance"),
    ], ignore_index=True)

    top = gaps.nlargest(top_n, "gap_minutes").copy()

    def nearest(t):
        d = (events["when"] - t).abs()
        i = d.idxmin()
        return pd.Series({
            "nearest_event": f"{events.loc[i, 'failure_id']}-{events.loc[i, 'kind']}",
            "days_away": round(d.loc[i].total_seconds() / 86400, 1),
        })

    return top.join(top["gap_start"].apply(nearest)).reset_index(drop=True)


def gap_size_context(gaps: pd.DataFrame, fw: pd.DataFrame,
                     pad_days: float = 3) -> dict:
    """Base-rate control for OQ2: are gaps near failures LARGER than typical?

    With 331 gaps over 175 days (~1.9/day) proximity alone proves nothing —
    the claim must rest on size. Compares median/max gap size near failures
    vs. everywhere else.
    """
    near_mask = pd.Series(False, index=gaps.index)
    for _, f in fw.iterrows():
        lo = f["start"] - pd.Timedelta(days=pad_days)
        hi = f["end"] + pd.Timedelta(days=pad_days)
        near_mask |= (gaps["gap_end"] >= lo) & (gaps["gap_start"] <= hi)
    near, far = gaps[near_mask], gaps[~near_mask]
    return {
        "n_near": len(near), "n_far": len(far),
        "median_near_min": round(near["gap_minutes"].median(), 1),
        "median_far_min": round(far["gap_minutes"].median(), 1),
        "max_near_min": round(near["gap_minutes"].max(), 1),
        "share_of_top20_near": round(near_mask[gaps.nlargest(20, "gap_minutes").index].mean(), 2),
    }

def signal_off_episodes(df: pd.DataFrame, signal: str,
                        lo=None, hi=None, coarse="6h", fine="15min",
                        share_threshold: float = 0.5) -> pd.DataFrame:
    """Episodes where `signal` is predominantly INACTIVE (0).

    Wraps find_state_episodes with an 'is mostly off' predicate:
    a window counts as off when the signal reads 0 for more than
    share_threshold of its samples. Empty window -> False (gap != off).
    """
    lo = lo or df[TIMESTAMP].min()
    hi = hi or df[TIMESTAMP].max()

    def mostly_off(d, a, b):
        w = d.loc[window_mask(d, a, b), signal]
        return bool(len(w) and (w == 0).mean() > share_threshold)

    return find_state_episodes(df, mostly_off, lo, hi, coarse=coarse, fine=fine)

def signal_run_summary(df: pd.DataFrame, signals: list[str],
                       state: int = 0) -> pd.DataFrame:
    """Run-length structure of a digital signal's `state` stretches.

    One row per signal: how many runs, their duration distribution, total
    time. Answers "is this state a few long episodes or thousands of
    flickers?" without any scan-parameter guessing (contrast
    signal_off_episodes, whose coarse window sets a detection floor).
    """
    rows = []
    for sig in signals:
        s = df[sig]
        runs = (s != s.shift()).cumsum()
        agg = s.groupby(runs).agg(val="first", n="size")
        agg = agg[agg["val"] == state]
        mins = agg["n"] * 10 / 60            # 10 s grid -> minutes
        rows.append({
            "signal": sig, "n_runs": len(agg),
            "median_min": round(mins.median(), 2),
            "p90_min": round(mins.quantile(0.9), 2),
            "max_min": round(mins.max(), 1),
            "total_min": round(mins.sum(), 1),
            "n_runs_over_1h": int((mins > 60).sum()),
        })
    return pd.DataFrame(rows).set_index("signal")


def episodes_nearest_event(episodes: pd.DataFrame, fw: pd.DataFrame,
                           start_col: str = "episode_start") -> pd.DataFrame:
    """For each episode, the nearest known event and its distance in days.

    Generalizes the gaps-vs-events analysis (OQ2) to any episode table.
    """
    events = pd.concat([
        fw[["failure_id", "start"]].rename(columns={"start": "when"}).assign(kind="failure_start"),
        fw[["failure_id", "end"]].rename(columns={"end": "when"}).assign(kind="failure_end"),
        fw.dropna(subset=["maintenance"])[["failure_id", "maintenance"]]
          .rename(columns={"maintenance": "when"}).assign(kind="maintenance"),
    ], ignore_index=True)

    def nearest(t):
        d = (events["when"] - t).abs()
        i = d.idxmin()
        return pd.Series({
            "nearest_event": f"{events.loc[i, 'failure_id']}-{events.loc[i, 'kind']}",
            "days_away": round(d.loc[i].total_seconds() / 86400, 1),
        })

    out = episodes.copy()
    return out.join(out[start_col].apply(nearest)).reset_index(drop=True)


def episode_state_stats(df: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """5.3 — machine state INSIDE vs OUTSIDE the episodes.

    Returns two rows (inside/outside) with duty, mean motor current, and
    sample count — enough to tell "signal off while machine idle" from
    "signal off while machine runs normally".
    """
    inside = pd.Series(False, index=df.index)
    for _, e in episodes.iterrows():
        inside |= window_mask(df, e["episode_start"], e["episode_end"])

    rows = []
    for name, m in [("inside", inside), ("outside", ~inside)]:
        w = df.loc[m]
        rows.append({
            "where": name,
            "n_samples": len(w),
            "duty": round(float(w["DV_eletric"].mean()), 3) if len(w) else None,
            "motor_mean": round(float(w["Motor_current"].mean()), 3) if len(w) else None,
            "motor_zero_share": round(float((w["Motor_current"] < 0.1).mean()), 3) if len(w) else None,
        })
    return pd.DataFrame(rows)