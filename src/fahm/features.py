"""Feature engineering for FAHM — SKELETON. Bodies are yours to write.

Module role (D00 family): preprocessing prepares, analysis measures,
plotting draws — **features.py builds the model-ready table**: one row per
time window, columns = engineered health features.

Design constraints inherited from stage 3 (see 04 notebook intro):
  * gap-aware — windows never bridge a recording gap (segment logic)
  * both directions — rising AND falling signals are informative
  * state-occupancy + dynamics, not just central tendency
  * instrument-health features kept separate from machine-health features
  * every window carries its label (majority; `invalid` wins any overlap)

Suggested import block:
    from fahm.preprocessing import ANALOG, DIGITAL, TIMESTAMP
    from fahm.analysis import window_mask, antiphase_share
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fahm.preprocessing import ANALOG, DIGITAL, TIMESTAMP


# ---------------------------------------------------------------------------
# 1. The grid
# ---------------------------------------------------------------------------

def build_window_grid(df: pd.DataFrame, cfg: dict, window: str = "1h") -> pd.DataFrame:
    """The feature grid: window_start | window_end | segment_id.
    Windows never bridge a recording gap (they live inside segments)."""
    thr = pd.Timedelta(seconds=cfg["preprocessing"]["gap_threshold_seconds"])
    ts = df[TIMESTAMP]

    gap_row = ts.diff() > thr
    segment_id = gap_row.cumsum()        # 0,0,0,1,1,2,... one id per contiguous run

    # --- windows within each segment ---------------------------------------
    rows = []
    for seg, idx in df.groupby(segment_id).groups.items():
        seg_ts = ts.loc[idx]
        seg_start, seg_end = seg_ts.min(), seg_ts.max()

        edges = pd.date_range(seg_start, seg_end, freq=window)
        for w_lo, w_hi in zip(edges[:-1], edges[1:]):
            rows.append({"window_start": w_lo, "window_end": w_hi, "segment_id": seg})   

    return pd.DataFrame(rows)


def label_grid(grid: pd.DataFrame, labels: pd.Series, df: pd.DataFrame) -> pd.Series:
    """One label per window: majority row-label, but `invalid` wins ANY
    overlap (trust rule — one bad sample distrusts the whole window)."""

    ts = df[TIMESTAMP]
    out = []
    for _, w in grid.iterrows():
        mask = (ts >= w["window_start"]) & (ts < w["window_end"])
        win_labels = labels[mask]

        if len(win_labels) == 0:
            out.append("empty")                 # shouldn't happen (D20), but be safe
            continue

        if (win_labels == "invalid").any():
            out.append("invalid")
        else:
            out.append(win_labels.mode()[0])

    return pd.Series(out, index=grid.index, name="label")

# ---------------------------------------------------------------------------
# 2. Feature families — each returns a DataFrame indexed like `grid`
# ---------------------------------------------------------------------------

def duty_state_features(df: pd.DataFrame, grid: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per window: duty (valve-based load fraction) + fraction of time in each
    Motor_current mode (off / offloaded / loaded). Catches F4-up, F1-down (6.4)."""
    m = cfg["features"]["motor_modes"]
    rows = []
    for _, w in grid.iterrows():
        seg = df[(df[TIMESTAMP] >= w["window_start"]) & (df[TIMESTAMP] < w["window_end"])]
        mc = seg["Motor_current"]

        rows.append({
            "duty": seg["DV_eletric"].mean(),
            "frac_off": (mc < m["off_max"]).mean(),
            "frac_offloaded": ((mc >= m["off_max"]) & (mc < m["offloaded_max"])).mean(),
            "frac_loaded": (mc >= m["offloaded_max"]).mean(),
        })
    return pd.DataFrame(rows, index=grid.index)


def pressure_dynamics_features(df: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Per window: TP3 idle-decay slope (THE leak meter), cycles/hour.

    Idle-decay: when the compressor is OFF (COMP==0), reservoir pressure should
    hold; a leak makes it fall. The slope of TP3 vs time during idle samples is
    a near-direct leak measurement — steeper negative = worse leak.
    """
    rows = []
    for _, w in grid.iterrows():
        seg = df[(df[TIMESTAMP] >= w["window_start"]) & (df[TIMESTAMP] < w["window_end"])]

        # --- idle-decay slope of TP3 -----------------------------------------
        is_idle = seg["COMP"] == 1
        run_id = (is_idle != is_idle.shift()).cumsum()
        slopes = []
        for _, stretch in seg[is_idle].groupby(run_id[is_idle]):
            if len(stretch) >= 5:
                t = (stretch[TIMESTAMP] - stretch[TIMESTAMP].iloc[0]).dt.total_seconds()
                slopes.append(np.polyfit(t, stretch["TP3"], 1)[0])
        slope = np.median(slopes) if slopes else float("nan")

        # --- cycling rate ----------------------------------------------------
        rising_edges = (seg["DV_eletric"]== 1).sum()

        rows.append({
            "tp3_decay_slope": slope,          # bar/sec; negative = leaking
            "cycles_per_hour": rising_edges,   # window is 1h, so count ≈ rate
        })
    return pd.DataFrame(rows, index=grid.index)


def thermal_features(df: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Oil median; oil trend over the window; oil-per-duty ratio."""
    raise NotImplementedError


def variability_features(df: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Rolling/within-window std of duty & oil; cycle-duration variance;
    toggle rates. F3's only remaining chance (OQ5) — build with care."""
    raise NotImplementedError


def cycle_frequency_features(df: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Spectral features of the SLOW cycling rhythm (D21).

    Fourier for VIBRATION is impossible here — decimation to 10s caps Nyquist
    at ~1/20 Hz, so bearing/valve signatures (Hz-kHz) are gone (L02). BUT the
    load/unload cycle has a period of MINUTES, well within 10s resolution, and
    a leak makes the machine cycle FASTER to compensate — so the dominant
    cycling frequency and its drift are legitimate, physically-motivated
    features, and F3's kind of long shot (it moved no central-tendency stat).

    Hints:
      * sampling is IRREGULAR (jitter + gaps) -> a plain FFT assumes uniform
        spacing and will lie. Use Lomb-Scargle (astropy.timeseries.
        LombScargle) — the standard periodogram for uneven time series — on
        the duty or TP3 signal within each window.
      * features per window: dominant cycle frequency (peak of the
        periodogram), its power, and spectral entropy (rhythm regular vs
        erratic). A leak should raise the fundamental and/or scatter the
        spectrum.
      * cheaper robust alternative / cross-check: autocorrelation of the duty
        signal -> first-peak lag = cycle period; track its drift. Same physics,
        no spectral-leakage fuss. Build ONE first, add the other if it earns
        its columns (effect-size check, §3).
      * windows are short (1h) with a minutes-scale cycle -> only a handful of
        cycles per window. Verify the periodogram is meaningful at this length
        BEFORE trusting it; may need a longer look-back window for this family.
    """
    raise NotImplementedError


def instrument_health_features(df: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """antiphase share; stuck-analog flags (within-window variance ~ 0).

    Separate axis from machine health (stage-3 design): stage 5 trains only
    on windows that are BOTH healthy-labeled AND instrument-clean.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 3. Assembly + artifact
# ---------------------------------------------------------------------------

FAMILIES = [duty_state_features, pressure_dynamics_features,
            thermal_features, variability_features,
            cycle_frequency_features,        # D21: spectral cycle-rhythm
            instrument_health_features]


def build_features(df: pd.DataFrame, grid: pd.DataFrame,
                   cfg: dict) -> pd.DataFrame:
    """Run every family, concat columns onto the grid, return the table.

    Hint: pd.concat([grid] + [fam(df, grid) for fam in FAMILIES], axis=1);
    verify no duplicate column names and no all-NaN columns before returning
    (a family that silently returns NaNs is a bug, not a feature).
    """
    raise NotImplementedError


def save_features(feats: pd.DataFrame, cfg: dict) -> Path:
    """Persist to cfg['paths']['features'] (parquet, D11 convention).
    Add the path to config first — full filename, not a directory."""
    raise NotImplementedError
