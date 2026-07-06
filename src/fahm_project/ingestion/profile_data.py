"""Generate docs/data_profile.md from the ingested parquet.

Produces the workbook's Step 1 artifact: row counts, time coverage, sampling
regularity, recording gaps, per-sensor statistics, digital-signal duty
fractions, missingness, and two diagnostic figures:

  1. A few hours of NORMAL operation (compressor cycles visible).
  2. The day around the first documented failure window.

Run AFTER load_metropt.py and make_failure_windows.py:
    poetry run python -m fahm_project.ingestion.profile_data --config configs/metropt.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from fahm_project.ingestion.load_metropt import (
    ANALOG_SENSORS,
    DIGITAL_SENSORS,
    TIMESTAMP_COL,
)

# Physical meaning of each sensor, condensed from the MetroPT paper.
# Verify/expand these in your own words — interviewers will ask.
SENSOR_MEANING = {
    "TP2": "Pressure at the compressor (bar).",
    "TP3": "Pressure at the pneumatic panel (bar).",
    "H1": "Pressure drop at the cyclonic separator filter discharge (bar).",
    "DV_pressure": "Pressure drop when the air-drying towers discharge; ~0 while compressor works under load (bar).",
    "Reservoirs": "Downstream reservoir pressure; should track TP3 (bar).",
    "Oil_temperature": "Compressor oil temperature (deg C).",
    "Motor_current": "Current of one motor phase; ~0 A off, ~4 A offloaded, ~7 A under load.",
    "COMP": "Air-intake valve signal; active when there is no air intake (compressor off or offloaded).",
    "DV_eletric": "Compressor outlet valve signal; active when compressor works under load.",
    "Towers": "Which drying tower is active (0 = tower one, 1 = tower two).",
    "MPG": "Starts compressor under load when pressure < 8.2 bar.",
    "LPS": "Low-pressure switch; activates when pressure < 7 bar.",
    "Pressure_switch": "Detects discharge in the air-drying towers.",
    "Oil_level": "Active (1) when oil level is BELOW expected.",
    "Caudal_impulsion": "Airflow signal at the compressor output.",
}


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_gaps(ts: pd.Series, threshold_s: float) -> pd.DataFrame:
    dt = ts.diff().dt.total_seconds()
    idx = dt[dt > threshold_s].index
    gaps = pd.DataFrame(
        {
            "gap_start": ts.shift(1).loc[idx],
            "gap_end": ts.loc[idx],
            "gap_minutes": (dt.loc[idx] / 60).round(1),
        }
    )
    return gaps.sort_values("gap_minutes", ascending=False)


def plot_normal_window(df: pd.DataFrame, start: str, hours: int, out: Path) -> None:
    t0 = pd.Timestamp(start)
    win = df[(df[TIMESTAMP_COL] >= t0) & (df[TIMESTAMP_COL] < t0 + pd.Timedelta(hours=hours))]
    if win.empty:
        print(f"WARNING: normal snapshot window {start} is empty (recording gap?). "
              "Pick another start in configs/metropt.yaml and rerun.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for col in ["TP2", "TP3", "Reservoirs"]:
        axes[0].plot(win[TIMESTAMP_COL], win[col], label=col, lw=0.7)
    axes[0].set_ylabel("bar")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"Normal operation — {hours}h from {start}")

    axes[1].plot(win[TIMESTAMP_COL], win["Motor_current"], color="tab:red", lw=0.7)
    axes[1].set_ylabel("Motor_current (A)")

    axes[2].step(win[TIMESTAMP_COL], win["COMP"], label="COMP", where="post", lw=0.8)
    axes[2].step(win[TIMESTAMP_COL], win["DV_eletric"], label="DV_eletric", where="post", lw=0.8)
    axes[2].set_ylabel("digital")
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


def plot_failure_context(df: pd.DataFrame, fw: pd.DataFrame, out: Path) -> None:
    if fw.empty:
        return
    row = fw.iloc[0]
    lo = row["start"] - pd.Timedelta(days=1)
    hi = row["end"] + pd.Timedelta(days=1)
    win = df[(df[TIMESTAMP_COL] >= lo) & (df[TIMESTAMP_COL] <= hi)]
    if win.empty:
        print("WARNING: no data around first failure window — check the window dates.")
        return

    # Downsample to 1-min means for a readable multi-day plot.
    ds = (
        win.set_index(TIMESTAMP_COL)[["TP3", "Motor_current", "Oil_temperature"]]
        .resample("1min")
        .mean()
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, col in zip(axes, ds.columns):
        ax.plot(ds.index, ds[col], lw=0.8)
        ax.set_ylabel(col)
        ax.axvspan(row["start"], row["end"], color="red", alpha=0.15)
    axes[0].set_title(
        f"Around failure {row['failure_id']} ({row['fault_type']}) — red = documented window"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/metropt.yaml")
    args = parser.parse_args()
    cfg = load_cfg(args.config)

    pq_path = Path(cfg["paths"]["processed_parquet"])
    fig_dir = Path(cfg["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {pq_path} ...")
    cols = [TIMESTAMP_COL] + ANALOG_SENSORS + DIGITAL_SENSORS  # skip asset_id (heavy strings)
    df = pd.read_parquet(pq_path, columns=cols)
    print(f"{len(df):,} rows loaded")

    ts = df[TIMESTAMP_COL]
    non_monotonic = int((ts.diff().dt.total_seconds() < 0).sum())
    dt_mode = ts.diff().dt.total_seconds().mode().iat[0]
    gaps = find_gaps(ts, float(cfg["profiling"]["gap_threshold_seconds"]))

    fw_path = Path(cfg["paths"]["failure_windows"])
    fw = (
        pd.read_csv(fw_path, parse_dates=["start", "end"])
        if fw_path.exists()
        else pd.DataFrame()
    )

    # --- figures ---
    plot_normal_window(
        df,
        cfg["profiling"]["normal_snapshot_start"],
        int(cfg["profiling"]["normal_snapshot_hours"]),
        fig_dir / "normal_operation.png",
    )
    plot_failure_context(df, fw, fig_dir / "first_failure_context.png")

    # --- stats ---
    analog_stats = df[ANALOG_SENSORS].describe().T.round(3)
    digital_duty = (
        df[DIGITAL_SENSORS]
        .replace(-1, pd.NA)
        .mean()
        .rename("fraction_active")
        .to_frame()
        .round(4)
    )
    missing = df.isna().mean().rename("fraction_missing").to_frame().round(5)
    missing = missing[missing["fraction_missing"] > 0]

    # --- write markdown ---
    lines: list[str] = []
    add = lines.append
    add("# Data Profile — MetroPT-3 (APU air compressor)\n")
    add(f"Generated from `{pq_path}`.\n")
    add("## Coverage\n")
    add(f"- Rows: **{len(df):,}**")
    add(f"- Time range: **{ts.min()}** → **{ts.max()}**")
    add(f"- Modal sampling interval: **{dt_mode:.0f} s**")
    add(f"- Non-monotonic timestamp steps: **{non_monotonic}**")
    add(f"- Recording gaps > {cfg['profiling']['gap_threshold_seconds']} s: **{len(gaps)}** "
        f"(total {gaps['gap_minutes'].sum():,.0f} min)\n")
    if not gaps.empty:
        add("### Ten longest gaps\n")
        add(gaps.head(10).to_markdown(index=False))
        add("")
    add("## Sensor meaning (verify & rewrite in your own words)\n")
    meaning = pd.Series(SENSOR_MEANING, name="physical_meaning").to_frame()
    add(meaning.to_markdown())
    add("")
    add("## Analog sensor statistics\n")
    add(analog_stats.to_markdown())
    add("")
    add("## Digital signal duty (fraction of time active)\n")
    add(digital_duty.to_markdown())
    add("")
    if not missing.empty:
        add("## Missingness\n")
        add(missing.to_markdown())
        add("")
    if not fw.empty:
        add("## Documented failure windows (evaluation ground truth)\n")
        add(fw.drop(columns=["source_note"], errors="ignore").to_markdown(index=False))
        add("\n> Timestamps must be verified against the MetroPT paper before "
            "final evaluation numbers are reported.\n")
    add("## Figures\n")
    add("![Normal operation](figures/normal_operation.png)\n")
    add("![First failure context](figures/first_failure_context.png)\n")
    add("## Open questions for decision_log.md\n")
    add("- Is the chosen resample rate (10s / 1min) fine enough for leak dynamics?")
    add("- How should recording gaps be handled in rolling features (reset windows vs interpolate)?")
    add("- Which healthy period defines 'normal' for training the anomaly model?")

    out_md = Path(cfg["paths"]["profile_md"])
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
