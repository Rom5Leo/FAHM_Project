"""Build the model-ready feature table from the 1-minute base grid.

Input : data/processed/features_base.parquet  (from features.resample)
Output: data/processed/features_model.parquet (1-min rows, model features)

All rolling windows are computed WITHIN segment_id, so no window ever
bridges a recording gap.

Feature families:
  * Rolling means (30min / 4h / 24h) of the core health signals.
  * Rolling std (4h) of TP3 and Motor_current (instability).
  * duty_* : rolling mean of DV_eletric_frac — a duty cycle that keeps
    working during CONTINUOUS load, when the cycle extractor sees no
    completed cycles (the F1 blind spot found in error analysis).
  * pinned_load_2h : rolling MIN of DV_eletric_frac over 2h. Near 1.0 means
    the compressor never unloaded for two hours -> direct F1-style signature.
  * duty_trend : duty_4h minus duty_24h (short-term vs long-term baseline).

Run:
    poetry run python -m fahm_project.features.rolling_features --config configs/metropt.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

# base-table column -> short name used in feature columns
CORE_SIGNALS = {
    "TP3_mean": "tp3",
    "TP2_max": "tp2max",
    "Oil_temperature_mean": "oil",
    "Motor_current_mean": "motor",
    "DV_eletric_frac": "duty",
    "LPS_frac": "lps",
    "DV_pressure_max": "dvpmax",
    "H1_min": "h1min",
}

ROLL_WINDOWS = {"30min": 30, "4h": 240, "24h": 1440}  # in 1-min rows
STD_SIGNALS = ["TP3_mean", "Motor_current_mean"]


def build_model_features(base: pd.DataFrame) -> pd.DataFrame:
    base = base.sort_index()
    g = base.groupby("segment_id", group_keys=False)

    feats = pd.DataFrame(index=base.index)
    feats["segment_id"] = base["segment_id"]

    for col, short in CORE_SIGNALS.items():
        s = base[col].astype("float32")
        feats[f"{short}_now"] = s
        for wname, w in ROLL_WINDOWS.items():
            feats[f"{short}_{wname}"] = g[col].transform(
                lambda x, w=w: x.rolling(w, min_periods=max(2, w // 2)).mean()
            )

    for col in STD_SIGNALS:
        short = CORE_SIGNALS[col]
        feats[f"{short}_std_4h"] = g[col].transform(
            lambda x: x.rolling(240, min_periods=120).std()
        )

    # Continuous-load detector: min duty over 2h ~ 1.0 => never unloaded (F1 case)
    feats["pinned_load_2h"] = g["DV_eletric_frac"].transform(
        lambda x: x.rolling(120, min_periods=60).min()
    )
    # Short-vs-long duty divergence: rising short-term duty = developing leak
    feats["duty_trend"] = feats["duty_4h"] - feats["duty_24h"]

    before = len(feats)
    feats = feats.dropna()
    print(f"Dropped {before - len(feats):,} warm-up rows (rolling min_periods)")
    return feats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/metropt.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    src = Path(cfg["paths"]["features_base"])
    out = Path(cfg["paths"]["features_model"])
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {src} ...")
    base = pd.read_parquet(src)
    feats = build_model_features(base)
    feats.to_parquet(out)
    print(f"Wrote {len(feats):,} rows x {feats.shape[1]} cols -> {out}")


if __name__ == "__main__":
    main()
