"""Extract compressor load/idle cycles and leak-sensitive physics features.

The APU alternates between LOAD (DV_eletric = 1, motor ~7 A, pressure builds)
and IDLE (DV_eletric = 0, pressure slowly decays as the train consumes air).
An air leak changes this rhythm in physically predictable ways:

  * idle pressure decays FASTER            -> tp3_decay_rate more negative
  * compressor must run MORE OFTEN/LONGER  -> duty and load_duration rise
  * pressure builds SLOWER under load      -> tp3_build_rate falls (big leak)

Input : data/processed/sensor_readings.parquet   (10 s grid)
Output: data/processed/cycles.parquet            (one row per load->idle cycle)
        docs/figures/cycle_health_overview.png   (duty & decay vs failure windows)

Cycle table columns:
  cycle_start, cycle_end, segment_id,
  load_duration_s, idle_duration_s, duty,
  tp3_start, tp3_peak, tp3_end,
  tp3_build_rate  (bar/min while loading; higher = healthier),
  tp3_decay_rate  (bar/min while idle; more negative = leakier),
  motor_current_load_mean, oil_temp_mean

Run AFTER load_metropt (works directly on the 10 s grid, not the 1-min table):
    poetry run python -m fahm_project.features.cycle_features --config configs/metropt.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from fahm_project.ingestion.load_metropt import TIMESTAMP_COL

NEEDED = [TIMESTAMP_COL, "DV_eletric", "TP3", "Motor_current", "Oil_temperature"]


def add_segment_id(ts: pd.Series, gap_threshold_s: float) -> pd.Series:
    dt = ts.diff().dt.total_seconds()
    return (dt > gap_threshold_s).cumsum()


def extract_cycles(df: pd.DataFrame, min_cycle_s: float) -> pd.DataFrame:
    """Run-length encode DV_eletric within each segment and pair load->idle runs."""
    load = df["DV_eletric"].replace(-1, pd.NA).ffill().fillna(0).astype("int8")

    # run id increments whenever load state or segment changes
    change = (load != load.shift(1)) | (df["segment_id"] != df["segment_id"].shift(1))
    run_id = change.cumsum()

    g = df.assign(load_state=load, run_id=run_id).groupby("run_id", sort=True)
    runs = g.agg(
        segment_id=("segment_id", "first"),
        load_state=("load_state", "first"),
        start=(TIMESTAMP_COL, "first"),
        end=(TIMESTAMP_COL, "last"),
        n=(TIMESTAMP_COL, "size"),
        tp3_first=("TP3", "first"),
        tp3_last=("TP3", "last"),
        tp3_max=("TP3", "max"),
        motor_mean=("Motor_current", "mean"),
        oil_mean=("Oil_temperature", "mean"),
    ).reset_index(drop=True)
    runs["duration_s"] = (runs["end"] - runs["start"]).dt.total_seconds() + 10.0

    # Pair each LOAD run with the IMMEDIATELY FOLLOWING idle run (same segment).
    nxt = runs.shift(-1)
    is_pair = (
        (runs["load_state"] == 1)
        & (nxt["load_state"] == 0)
        & (runs["segment_id"] == nxt["segment_id"])
    )
    lo = runs[is_pair].reset_index(drop=True)
    idl = nxt[is_pair].reset_index(drop=True)

    cycles = pd.DataFrame(
        {
            "cycle_start": lo["start"],
            "cycle_end": idl["end"],
            "segment_id": lo["segment_id"],
            "load_duration_s": lo["duration_s"],
            "idle_duration_s": idl["duration_s"],
            "tp3_start": lo["tp3_first"],
            "tp3_peak": lo["tp3_max"],
            "tp3_end": idl["tp3_last"],
            "motor_current_load_mean": lo["motor_mean"],
            "oil_temp_mean": lo["oil_mean"],
        }
    )
    cycles["duty"] = cycles["load_duration_s"] / (
        cycles["load_duration_s"] + cycles["idle_duration_s"]
    )
    cycles["tp3_build_rate"] = (
        (cycles["tp3_peak"] - cycles["tp3_start"]) / (cycles["load_duration_s"] / 60.0)
    )
    cycles["tp3_decay_rate"] = (
        (cycles["tp3_end"] - cycles["tp3_peak"]) / (cycles["idle_duration_s"] / 60.0)
    )

    # Discard degenerate blips (single-sample toggles, sensor chatter).
    keep = (cycles["load_duration_s"] >= min_cycle_s) & (
        cycles["idle_duration_s"] >= min_cycle_s
    )
    return cycles[keep].reset_index(drop=True)


def plot_overview(cycles: pd.DataFrame, fw: pd.DataFrame, out: Path) -> None:
    daily = (
        cycles.set_index("cycle_start")[["duty", "tp3_decay_rate", "load_duration_s"]]
        .resample("6h")
        .median()
    )
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    panels = [
        ("duty", "duty (load fraction)"),
        ("tp3_decay_rate", "TP3 idle decay (bar/min)"),
        ("load_duration_s", "load duration (s)"),
    ]
    for ax, (col, label) in zip(axes, panels):
        ax.plot(daily.index, daily[col], lw=0.9)
        ax.set_ylabel(label)
        for _, row in fw.iterrows():
            ax.axvspan(row["start"], row["end"], color="red", alpha=0.25)
    axes[0].set_title("Cycle health indicators (6h medians) — red = documented failures")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/metropt.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    fcfg = cfg["features"]
    src = Path(cfg["paths"]["processed_parquet"])
    out = Path(cfg["paths"]["cycles"])
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {src} ...")
    df = pd.read_parquet(src, columns=NEEDED).sort_values(TIMESTAMP_COL)
    df["segment_id"] = add_segment_id(
        df[TIMESTAMP_COL], float(fcfg["gap_threshold_seconds"])
    ).values

    cycles = extract_cycles(df, min_cycle_s=float(fcfg["min_cycle_seconds"]))
    cycles.to_parquet(out, index=False)

    med = cycles[["duty", "tp3_build_rate", "tp3_decay_rate", "load_duration_s"]].median()
    print(f"Wrote {len(cycles):,} cycles -> {out}")
    print("Median cycle profile:")
    print(med.round(4).to_string())

    fw_path = Path(cfg["paths"]["failure_windows"])
    if fw_path.exists():
        fw = pd.read_csv(fw_path, parse_dates=["start", "end"])
        fig_dir = Path(cfg["paths"]["figures_dir"])
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_overview(cycles, fw, fig_dir / "cycle_health_overview.png")


if __name__ == "__main__":
    main()
