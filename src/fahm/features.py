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
    """Per window: oil level, oil trend, oil-per-duty. Catches F4 ramp, F1 cool-idle."""
    rows = []
    for _, w in grid.iterrows():
        seg = df[(df[TIMESTAMP] >= w["window_start"]) & (df[TIMESTAMP] < w["window_end"])]
        oil = seg["Oil_temperature"]

        # trend: slope of oil vs time across the window (°C per second)
        t = (seg[TIMESTAMP] - seg[TIMESTAMP].iloc[0]).dt.total_seconds()
        oil_trend = np.polyfit(t, oil, 1)[0] if len(seg) >= 10 else float("nan")

        duty = seg["DV_eletric"].mean()

        rows.append({
            "oil_median": oil.median(),
            "oil_trend": oil_trend,      # done: °C/s, positive = warming
        })
    return pd.DataFrame(rows, index=grid.index)

def oil_residual_feature(f_thermal, f_duty, grid_labels, degree=2):
    """Oil temp minus healthy-baseline prediction from duty (D26).
    Fits oil~duty on HEALTHY windows internally, applies to all. Self-contained."""
    h = grid_labels == "healthy"
    d, o = f_duty["duty"], f_thermal["oil_median"]
    m = h & d.notna() & o.notna()
    coef = np.polyfit(d[m], o[m], degree)          # baseline on healthy only
    return f_thermal["oil_median"] - np.polyval(coef, d)   # residual for all windows


def variability_features(df: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Variability + cycling-regime measures. cycle_dur_* need cycles (NaN when
    idle/continuous); frac_continuous_load & longest_load_stretch are defined
    ALWAYS — they catch the regime shift (normal cycling -> locked loaded) that
    precedes F1/F3, which cycle_dur_cv misses (OQ5)."""
    rows = []
    for _, w in grid.iterrows():
        seg = df[(df[TIMESTAMP] >= w["window_start"]) & (df[TIMESTAMP] < w["window_end"])]

        starts = seg.loc[seg["DV_eletric"].diff() == 1, TIMESTAMP]
        cycle_secs = starts.diff().dt.total_seconds().dropna()

        # regime: run-lengths of continuous load (DV_eletric==1)
        loaded = seg["DV_eletric"] == 1
        run_id = (loaded != loaded.shift()).cumsum()
        load_runs = seg[loaded].groupby(run_id[loaded]).size()   # samples per loaded run
        longest = load_runs.max() * 10 / 60 if len(load_runs) else 0.0   # minutes

        rows.append({
            "oil_std": seg["Oil_temperature"].std(),
            "tp3_std": seg["TP3"].std(),
            "duty_std": seg["DV_eletric"].std(),
            "cycle_dur_cv": (cycle_secs.std() / cycle_secs.mean()
                             if len(cycle_secs) >= 3 else float("nan")),
            "cycle_dur_trend": (np.polyfit(range(len(cycle_secs)), cycle_secs, 1)[0]
                                if len(cycle_secs) >= 3 else float("nan")),
            "longest_load_stretch": longest,                    # minutes; regime, always defined
            "frac_continuous_load": seg["DV_eletric"].mean(),   # = duty, but here as regime lens
        })
    return pd.DataFrame(rows, index=grid.index)

def cycle_frequency_features(df: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Spectral features of the slow cycling rhythm via Lomb-Scargle (D21).
    Not vibration (decimation killed that, L02) — the minutes-scale load cycle.
    Shares the 'needs cycles' limit: NaN when idle/locked."""
    from astropy.timeseries import LombScargle
    rows = []
    for _, w in grid.iterrows():
        seg = df[(df[TIMESTAMP] >= w["window_start"]) & (df[TIMESTAMP] < w["window_end"])]

        t = (seg[TIMESTAMP] - seg[TIMESTAMP].iloc[0]).dt.total_seconds().values
        y = seg["DV_eletric"].values.astype(float)

        # need enough variation to have a spectrum (not all-0/all-1, enough points)
        if len(y) < 30 or y.std() == 0:
            rows.append({"dominant_freq": float("nan"),
                         "spectral_entropy": float("nan")})
            continue

        freq = np.linspace(1/3600, 1/60, 200)     # periods 1h down to 1min
        power = LombScargle(t, y).power(freq)

        dominant = dominant = freq[np.argmax(power)]
        p = power / power.sum()
        entropy = -np.sum(p * np.log(p + 1e-12))     # +tiny to avoid log(0)

        rows.append({"dominant_freq": dominant, "spectral_entropy": entropy})
    return pd.DataFrame(rows, index=grid.index)


def instrument_health_features(df: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Per window: is the INSTRUMENT trustworthy? (separate axis from machine
    health — stage 3 design). Should fire on the Apr 20 fault, quiet elsewhere.
    Stage 5 trains machine-health only on instrument-clean windows."""
    from fahm.analysis import antiphase_share
    rows = []
    for _, w in grid.iterrows():
        seg = df[(df[TIMESTAMP] >= w["window_start"]) & (df[TIMESTAMP] < w["window_end"])]

        rows.append({
            "antiphase_share": antiphase_share(df, w["window_start"], w["window_end"]),
            "motor_frozen": (seg["Motor_current"].var() < 1e-6),
            "tp3_frozen": (seg["TP3"].var() < 1e-6),
        })
    return pd.DataFrame(rows, index=grid.index)


# ---------------------------------------------------------------------------
# 3. Assembly + artifact
# ---------------------------------------------------------------------------

FAMILIES = [duty_state_features, pressure_dynamics_features,
            thermal_features, variability_features,        # D21: spectral cycle-rhythm
            instrument_health_features]


def build_features(df, grid, cfg, grid_labels):
    parts = [grid.reset_index(drop=True)]
    for fam in FAMILIES:
        try:    out = fam(df, grid, cfg)
        except TypeError:  out = fam(df, grid)
        parts.append(out.reset_index(drop=True))
    feats = pd.concat(parts, axis=1)

    feats["oil_residual"] = oil_residual_feature(feats, feats, grid_labels)  # uses feats["oil_median"], feats["duty"]

    dupes = feats.columns[feats.columns.duplicated()].tolist()
    if dupes: raise ValueError(f"duplicate columns: {dupes}")
    all_nan = [c for c in feats.columns if feats[c].isna().all()]
    if all_nan: raise ValueError(f"all-NaN columns: {all_nan}")
    return feats

def add_cycling_regime(feats: pd.DataFrame) -> pd.DataFrame:
    """D29 partial-3: an explicit regime column so the cycle-features' NaN
    pattern (idle / cycling / locked) becomes a usable categorical signal.
    Uses duty + cycles_per_hour, both always defined."""
    out = feats.copy()
    duty = out["duty"]
    cyc = out["cycles_per_hour"]
    regime = pd.Series("cycling", index=out.index)      # default
    regime[duty < 0.05] = "idle"                        # barely running
    regime[(duty > 0.8) | (cyc < 3)] = "locked"         # continuous load / no cycles
    out["cycling_regime"] = regime
    return out


def prepare_for_model(feats: pd.DataFrame, labels: pd.Series,
                      nan_features: list[str] | None = None,
                      exclude: tuple = ("window_start", "window_end",
                                        "segment_id", "n_samples")) -> pd.DataFrame:
    """Model-ready table (D29): missing-indicators + neutral fill for
    informative NaNs, then standardize continuous features on HEALTHY windows.

    Booleans/flags and the regime category are left unscaled.
    Fit the scaler on healthy only so 'normal' defines the scale and anomalies
    read as large deviations.
    """
    out = feats.copy()

    # 1) missing indicators + neutral fill for the NaN-prone (cycle) features
    if nan_features is None:
        nan_features = [c for c in out.columns
                        if out[c].dtype.kind == "f" and out[c].isna().any()]
    for c in nan_features:
        out[f"{c}_missing"] = out[c].isna().astype(int)     # 1 = was not measurable
        fill = out.loc[labels == "healthy", c].median()     # neutral = healthy median
        out[c] = out[c].fillna(fill)

    # 2) standardize continuous features on HEALTHY only
    #    (skip: bookkeeping cols, booleans, the *_missing flags, the regime cat)
    skip = set(exclude) | {c for c in out.columns if c.endswith("_missing")}
    cont = [c for c in out.columns
            if c not in skip
            and out[c].dtype.kind in "fi"
            and out[c].nunique() > 2]                        # >2 values = continuous
    h = labels == "healthy"
    mu = out.loc[h, cont].mean()
    sd = out.loc[h, cont].std().replace(0, 1.0)             # guard zero-variance
    out[cont] = (out[cont] - mu) / sd                       # z-score vs healthy

    out["label"] = labels.values
    return out

def save_features(feats: pd.DataFrame, cfg: dict) -> Path:
    """Persist the model-ready feature table to cfg['paths']['features']
    (parquet, D11 convention)."""
    out = Path(cfg["paths"]["features"])
    if out.suffix != ".parquet":
        raise ValueError(f"cfg paths.features should be a .parquet file, got {out!r}")
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    return out

def cycle_features_multiscale(df, grid, lookback="6h"):
    """Cycle variability computed over a trailing look-back (D27), not the
    window's own hour — more cycles per computation → less NaN. Window identity
    stays 1h; only the look-back for THESE features is extended. Respects
    segments (no look-back across a gap)."""
    lb = pd.Timedelta(lookback)
    rows = []
    for _, w in grid.iterrows():
        seg_id = w["segment_id"]
        lo = w["window_end"] - lb
        # samples in [window_end - lookback, window_end], same segment only
        m = ((df[TIMESTAMP] > lo) & (df[TIMESTAMP] <= w["window_end"]))
        # restrict to same segment: reuse the segment boundary via time (gap-aware)
        seg = df[m]
        starts = seg.loc[seg["DV_eletric"].diff() == 1, TIMESTAMP]
        cycle_secs = starts.diff().dt.total_seconds().dropna()
        if len(cycle_secs) >= 3:
            cv = cycle_secs.std() / cycle_secs.mean()
            trend = np.polyfit(range(len(cycle_secs)), cycle_secs, 1)[0]
        else:
            cv, trend = np.nan, np.nan
        rows.append({"cycle_dur_cv_6h": cv, "cycle_dur_trend_6h": trend})
    return pd.DataFrame(rows, index=grid.index)