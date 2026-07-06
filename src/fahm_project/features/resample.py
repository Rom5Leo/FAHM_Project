"""Resample the raw 10 s sensor grid into a 1-minute feature base table.

Input : data/processed/sensor_readings.parquet   (10 s grid, from load_metropt)
Output: data/processed/features_base.parquet     (1-min grid)

Per minute:
  * analog sensors  -> mean / std / min / max        (e.g. TP3_mean, TP3_max)
  * digital signals -> fraction of time active       (e.g. DV_eletric_frac)
  * n_samples       -> how many 10 s readings landed in the minute (expected 6)
  * segment_id      -> increments at every recording gap > gap threshold.
                       ALL rolling/cycle features downstream must group by
                       segment_id so windows never bridge a multi-day hole.

Design decisions (decision_log.md):
  * 1-min base grid: leak dynamics unfold over minutes-to-days; 6:1 reduction
    keeps cycle shape visible while shrinking the table to ~250k rows.
  * Digital -1 (missing marker from ingestion) is treated as NaN, not as a state.
  * Minutes with zero samples are dropped, not imputed; gaps are structural.

Run:
    poetry run python -m fahm_project.features.resample --config configs/metropt.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from fahm_project.ingestion.load_metropt import (
    ANALOG_SENSORS,
    DIGITAL_SENSORS,
    TIMESTAMP_COL,
)


def add_segment_id(index: pd.DatetimeIndex, gap_threshold_s: float) -> pd.Series:
    """Segment id that increments after every gap larger than the threshold."""
    dt = index.to_series().diff().dt.total_seconds()
    return (dt > gap_threshold_s).cumsum().rename("segment_id")


def build_base_table(df: pd.DataFrame, freq: str, gap_threshold_s: float) -> pd.DataFrame:
    df = df.set_index(TIMESTAMP_COL).sort_index()

    # Digital: -1 marks missing from ingestion -> NaN so it doesn't skew fractions.
    dig = df[DIGITAL_SENSORS].replace(-1, pd.NA).astype("Float32")

    agg_analog = df[ANALOG_SENSORS].resample(freq).agg(["mean", "std", "min", "max"])
    agg_analog.columns = [f"{c}_{stat}" for c, stat in agg_analog.columns]

    agg_digital = dig.resample(freq).mean()
    agg_digital.columns = [f"{c}_frac" for c in agg_digital.columns]

    n_samples = df[ANALOG_SENSORS[0]].resample(freq).count().rename("n_samples")

    base = pd.concat([agg_analog, agg_digital, n_samples], axis=1)
    base = base[base["n_samples"] > 0]  # drop empty minutes: gaps are structural

    base["segment_id"] = add_segment_id(base.index, gap_threshold_s).values
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/metropt.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    fcfg = cfg["features"]
    src = Path(cfg["paths"]["processed_parquet"])
    out = Path(cfg["paths"]["features_base"])
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {src} ...")
    cols = [TIMESTAMP_COL] + ANALOG_SENSORS + DIGITAL_SENSORS
    df = pd.read_parquet(src, columns=cols)
    print(f"{len(df):,} raw rows")

    base = build_base_table(
        df,
        freq=str(fcfg["base_freq"]),
        gap_threshold_s=float(fcfg["gap_threshold_seconds"]),
    )
    base.to_parquet(out)

    n_seg = base["segment_id"].nunique()
    print(f"Wrote {len(base):,} minutes across {n_seg} contiguous segments -> {out}")
    frac_partial = float((base["n_samples"] < 6).mean())
    print(f"Minutes with fewer than 6 samples: {frac_partial:.1%}")


if __name__ == "__main__":
    main()
