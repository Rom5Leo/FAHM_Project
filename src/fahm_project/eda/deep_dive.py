"""EDA deep-dive: answer the open questions accumulated during A1.

Generates docs/eda_deep_dive.md plus figures in docs/figures/eda/:

  1. event_<id>.png        — raw-signal zoom around each failure AND the
                             undocumented ~March 10 event (D11).
  2. f3_oil_story.png      — Oil_temperature & Oil_level around the oil leak:
                             does oil temp drift up before F3? does Oil_level
                             flip state (polarity check, D15)?
  3. lps_events.png        — every low-pressure-switch activation across the
                             whole record vs failure windows. Rare = informative.
  4. dist_healthy_vs_prefail.png — distributions of key 1-min features in
                             healthy periods vs the 48 h before each failure.
                             If they separate, early warning is learnable.
  5. weekly_duty.png       — weekly median duty: the regime shift (D12) and
                             any slow drift, in one line.

Run AFTER resample (uses both the 10 s parquet and features_base):
    poetry run python -m fahm_project.eda.deep_dive --config configs/metropt.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from fahm_project.ingestion.load_metropt import TIMESTAMP_COL

ZOOM_SIGNALS = ["TP3", "Motor_current", "Oil_temperature", "DV_eletric"]

DIST_FEATURES = [
    "DV_eletric_frac", "TP3_mean", "Oil_temperature_mean", "Motor_current_mean"
]


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------ figures --------------------------------------

def plot_event_zoom(raw: pd.DataFrame, ev: pd.Series, pad_days: float, out: Path) -> None:
    lo = ev["start"] - pd.Timedelta(days=pad_days)
    hi = ev["end"] + pd.Timedelta(days=pad_days)
    win = raw[(raw[TIMESTAMP_COL] >= lo) & (raw[TIMESTAMP_COL] <= hi)]
    if win.empty:
        print(f"  {ev['failure_id']}: no data in window, skipped")
        return
    fig, axes = plt.subplots(len(ZOOM_SIGNALS), 1, figsize=(15, 10), sharex=True)
    for ax, col in zip(axes, ZOOM_SIGNALS):
        ax.plot(win[TIMESTAMP_COL], win[col], lw=0.6)
        ax.set_ylabel(col)
        ax.axvspan(ev["start"], ev["end"], color="red", alpha=0.15)
    axes[0].set_title(f"{ev['failure_id']} ({ev['fault_type']}) — red = event window")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_f3_oil_story(base: pd.DataFrame, fw: pd.DataFrame, out: Path) -> None:
    f3 = fw[fw["fault_type"] == "oil_leak"]
    if f3.empty:
        return
    ev = f3.iloc[0]
    lo, hi = ev["start"] - pd.Timedelta(days=10), ev["end"] + pd.Timedelta(days=5)
    win = base.loc[(base.index >= lo) & (base.index <= hi)]

    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    hourly_oil = win["Oil_temperature_mean"].resample("1h").mean()
    axes[0].plot(hourly_oil.index, hourly_oil.values, lw=0.8)
    axes[0].set_ylabel("Oil temp (°C, hourly)")

    # oil temp per unit of work: temp divided by duty — rises if cooling degrades
    duty_h = win["DV_eletric_frac"].resample("1h").mean().clip(lower=0.01)
    ratio = hourly_oil / duty_h
    axes[1].plot(ratio.index, ratio.values, lw=0.8, color="tab:purple")
    axes[1].set_ylabel("Oil temp / duty")

    lvl = win["Oil_level_frac"].resample("1h").mean()
    axes[2].step(lvl.index, lvl.values, lw=0.9, color="tab:brown", where="post")
    axes[2].set_ylabel("Oil_level frac")
    axes[2].set_ylim(-0.05, 1.05)

    for ax in axes:
        ax.axvspan(ev["start"], ev["end"], color="red", alpha=0.15)
    axes[0].set_title("F3 oil-leak story — does oil signal move before/at the red window?")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_lps_events(base: pd.DataFrame, fw: pd.DataFrame, out: Path) -> None:
    lps = base.loc[base["LPS_frac"] > 0, "LPS_frac"]
    fig, ax = plt.subplots(figsize=(15, 3.2))
    ax.plot(lps.index, lps.values, ".", ms=3, alpha=0.6)
    for _, f in fw.iterrows():
        ax.axvspan(f["start"], f["end"], color="red", alpha=0.25)
    ax.set_ylabel("LPS active fraction/min")
    ax.set_title(f"Low-pressure-switch activations ({len(lps):,} minutes) vs failures")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  saved {out.name}  |  LPS minutes: {len(lps):,}")


def _mask_periods(index: pd.DatetimeIndex, periods) -> pd.Series:
    m = pd.Series(False, index=index)
    for s, e in periods:
        m |= (index >= pd.Timestamp(s)) & (index <= pd.Timestamp(e))
    return m


def plot_distributions(
    base: pd.DataFrame,
    fw: pd.DataFrame,
    healthy_periods: list,
    prefail_hours: float,
    out: Path,
) -> None:
    prefail = [
        [str(f["start"] - pd.Timedelta(hours=prefail_hours)), str(f["start"])]
        for _, f in fw.iterrows()
    ]
    m_health = _mask_periods(base.index, healthy_periods)
    m_pre = _mask_periods(base.index, prefail)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, col in zip(axes.ravel(), DIST_FEATURES):
        h = base.loc[m_health, col].dropna()
        p = base.loc[m_pre, col].dropna()
        if len(h):
            ax.hist(h, bins=60, density=True, alpha=0.55, label="healthy")
        if len(p):
            ax.hist(p, bins=60, density=True, alpha=0.55,
                    label=f"{int(prefail_hours)}h pre-failure", color="tab:red")
        ax.set_title(col)
        ax.legend()
    fig.suptitle("Healthy vs 48h-pre-failure distributions — separation ⇒ early warning is learnable")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_weekly_duty(base: pd.DataFrame, fw: pd.DataFrame, out: Path) -> None:
    weekly = base["DV_eletric_frac"].resample("1W").median()
    fig, ax = plt.subplots(figsize=(15, 3.2))
    ax.plot(weekly.index, weekly.values, marker="o", lw=1)
    for _, f in fw.iterrows():
        ax.axvspan(f["start"], f["end"], color="red", alpha=0.25)
    ax.set_ylabel("weekly median duty")
    ax.set_title("Weekly median duty — regime shift and slow drift")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  saved {out.name}")


# ------------------------------ report ---------------------------------------

REPORT = """# EDA Deep-Dive — guided reading

Each figure answers a specific open question. Write your conclusion under each
one; unanswered boxes become error_analysis.md / decision_log.md entries.

## 1. Event zooms (`figures/eda/event_*.png`)
For every documented failure and the undocumented March event: raw TP3,
Motor_current, Oil_temperature, DV_eletric with the window shaded.
**Questions:** What does each failure look like hours before the window —
gradual drift or sudden break? Is M10_undocumented shaped like F1
(continuous load) → likely an undocumented leak, or like a maintenance test?

> My conclusion (M10): ...
> My conclusion (per failure): ...

## 2. F3 oil story (`f3_oil_story.png`)
Oil temperature (hourly), oil temp per unit duty (cooling-efficiency proxy),
Oil_level fraction, ±10 days around F3.
**Questions:** Does any oil signal move BEFORE the window? Does Oil_level
flip state — and which direction, settling the polarity question (D15)?

> My conclusion: ...

## 3. LPS activations (`lps_events.png`)
Every minute the low-pressure switch fired, across 7 months.
**Question:** Do LPS events cluster at failures (→ strong cheap feature /
threshold-rule baseline) or scatter randomly (→ noise)?

> My conclusion: ...

## 4. Healthy vs pre-failure distributions (`dist_healthy_vs_prefail.png`)
**Question:** Which features separate 48h-pre-failure from healthy — duty?
oil temp? If none separate, early warning at 48h is NOT learnable from these
features and expectations must be reset honestly.

> My conclusion: ...

## 5. Weekly duty (`weekly_duty.png`)
**Questions:** Exactly when does the regime shift happen? Is there slow drift
that could cause false alarms months after training?

> My conclusion: ...
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/metropt.yaml")
    parser.add_argument("--pad-days", type=float, default=None,
                        help="override eda.pad_days from the config")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    ecfg = cfg.get("eda", {})

    pad_days = args.pad_days if args.pad_days is not None else float(ecfg.get("pad_days", 3))
    prefail_hours = float(ecfg.get("prefail_hours", 48))
    # ONE source of truth for 'healthy': the model's training periods,
    # unless the eda block explicitly overrides them.
    healthy_periods = ecfg.get("healthy_periods", cfg["model"]["train_periods"])

    fig_dir = Path(cfg["paths"]["figures_dir"]) / "eda"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fw = pd.read_csv(cfg["paths"]["failure_windows"], parse_dates=["start", "end"])
    extra = pd.DataFrame(ecfg.get("extra_events", []))
    if not extra.empty:
        extra["start"] = pd.to_datetime(extra["start"])
        extra["end"] = pd.to_datetime(extra["end"])
        events = pd.concat([fw, extra], ignore_index=True)
    else:
        events = fw

    print("Loading 10s raw data for zooms ...")
    raw = pd.read_parquet(
        cfg["paths"]["processed_parquet"], columns=[TIMESTAMP_COL] + ZOOM_SIGNALS
    )
    print("Event zooms:")
    for _, ev in events.iterrows():
        plot_event_zoom(raw, ev, pad_days, fig_dir / f"event_{ev['failure_id']}.png")
    del raw

    print("Loading 1-min base table ...")
    base = pd.read_parquet(cfg["paths"]["features_base"])

    plot_f3_oil_story(base, fw, fig_dir / "f3_oil_story.png")
    plot_lps_events(base, fw, fig_dir / "lps_events.png")
    plot_distributions(base, fw, healthy_periods, prefail_hours,
                       fig_dir / "dist_healthy_vs_prefail.png")
    plot_weekly_duty(base, fw, fig_dir / "weekly_duty.png")

    out_md = Path("docs/eda_deep_dive.md")
    out_md.write_text(REPORT)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
