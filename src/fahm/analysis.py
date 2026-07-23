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

    # (2) refine each flip inside its bracket [edges[i-1], edges[i+1]]
    refined = []
    for i in flip_pos:
        f_states, f_edges = scan(edges[i-1],edges[i+1],fine)
        before = states[i-1]
        t = next((f_edges[j] for j,s in enumerate(f_states) if s != before), edges[i])           # fallback: coarse edge
        refined.append((t, states[i]))                                                         # (flip time, new state)

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
    

    return pd.DataFrame(episodes, columns=["episode_start", "episode_end"])
    

def stuck_instrument(df, lo, hi, var_eps=1e-4, anti_eps=0.5) -> bool:
    """Predicate: window shows the frozen-analog / broken-antiphase signature."""
    w = df.loc[window_mask(df, lo, hi)]
    if len(w) == 0:
        return False           # no data ≠ faulty instrument (it's a gap)
    return bool(w["Motor_current"].var() < var_eps and antiphase_share(df, lo, hi) < anti_eps)