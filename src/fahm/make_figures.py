"""
Generate the three report figures the FAHM report still needs.
Run this in the project environment (where scores.parquet / features.parquet
and the fahm package live). Saves PNGs into ./docs/figures/ with the exact names
build_report.py expects.

    poetry run python make_figures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FIGDIR = Path("docs/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 7.011


plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})

NAVY, ACCENT = "#1F3A5F", "#C44E52"

# =========================================================================
# FIGURE 4.1 — health-score timeline around F1
# =========================================================================

def fig_health_timeline(
    scores_path="data/processed/scores.parquet",
    failure_windows_path="data/processed/failure_windows.csv",
    failure_id="F1",
):
    sc = pd.read_parquet(scores_path)
    sc["window_start"] = pd.to_datetime(sc["window_start"])

    fw = pd.read_csv(
        failure_windows_path,
        parse_dates=["start", "end"]
    )

    f = fw[fw["failure_id"] == failure_id].iloc[0]

    lo = f["start"] - pd.Timedelta(days=3)
    hi = f["end"] + pd.Timedelta(hours=6)

    m = (
        (sc["window_start"] >= lo)
        & (sc["window_start"] <= hi)
    )

    d = sc[m].sort_values("window_start")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(d["window_start"], d["zmax"], color=NAVY, lw=1.6, label="health score (zmax)")
    ax.axhline(THRESHOLD, color=ACCENT, ls="--", lw=1.2, label=f"alarm threshold ({THRESHOLD:.1f})")
    ax.axvspan(f["start"], f["end"], color=ACCENT, alpha=0.15, label=f"{failure_id} failure window")

    # first threshold crossing before the failure -> lead time
    pre = d[(d["window_start"] < f["start"]) & (d["zmax"] > THRESHOLD)]
    if len(pre):
        t0 = pre["window_start"].iloc[0]
        lead_h = (f["start"] - t0).total_seconds() / 3600
        ax.axvline(t0, color="green", ls=":", lw=1.2)
        ax.annotate(f"first alarm\n{lead_h:.0f} h lead",
                    xy=(t0, THRESHOLD), xytext=(t0, d["zmax"].max()*0.7),
                    color="green", fontsize=9, ha="center",
                    arrowprops=dict(arrowstyle="->", color="green"))

    ax.set_title(f"Health-score timeline — {failure_id}")
    ax.set_ylabel("zmax  (max standardized deviation)")
    ax.set_xlabel("time")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.autofmt_xdate()
    out = FIGDIR / f"health_score_timeline_{failure_id}.png"
    fig.savefig(out); plt.close(fig)
    print("saved", out)


# =========================================================================
# FIGURE 5.1 — threshold operating-point curve
# =========================================================================
def fig_threshold_curve(sweep_df=None):
    """
    sweep_df: the output of modeling.threshold_sweep — columns
    [quantile, fa_per_day, failure, lead_h]. If None, uses the recorded values.
    """
    if sweep_df is None:
        # recorded sweep values (from the notebook run)
        recs = []
        fa = {0.90:2.41, 0.95:1.12, 0.99:0.25, 0.995:0.13, 0.999:0.02}
        lead = {  # failure -> {quantile: lead_h}
            "F1": {0.90:37.5, 0.95:32.5, 0.99:32.5, 0.995:0.0, 0.999:0.0},
            "F3": {0.90:47.5, 0.95:41.5, 0.99:9.6,  0.995:0.0, 0.999:0.0},
            "F4": {0.90:42.9, 0.95:42.9, 0.99:8.0,  0.995:1.0, 0.999:0.0},
        }
        for fid, qd in lead.items():
            for q, lh in qd.items():
                recs.append({"failure": fid, "quantile": q,
                             "fa_per_day": fa[q], "lead_h": lh})
        sweep_df = pd.DataFrame(recs)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"F1": NAVY, "F3": "#4C72B0", "F4": ACCENT}
    for fid, g in sweep_df.groupby("failure"):
        g = g.sort_values("fa_per_day")
        ax.plot(g["fa_per_day"], g["lead_h"], "-o", color=colors.get(fid, "grey"),
                label=fid, lw=1.6, ms=5)

    ax.axvline(0.25, color="green", ls=":", lw=1.2)
    ax.annotate("operating point\n(q=0.99, 0.25 FA/day)",
                xy=(0.25, 20), xytext=(0.7, 30), color="green", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="green"))
    ax.set_title("Threshold operating-point tradeoff")
    ax.set_xlabel("false alarms per day")
    ax.set_ylabel("lead time (hours)")
    ax.legend(title="failure", fontsize=9, frameon=False)
    out = FIGDIR / "threshold_operating_curve.png"
    fig.savefig(out); plt.close(fig)
    print("saved", out)


# =========================================================================
# FIGURE 3.1 — per-failure effect-size heatmap
# =========================================================================
def fig_effect_heatmap(features_path="data/processed/features.parquet",
                       failure_windows_path="data/processed/failure_windows.csv",
                       prefail_hours=48):
    """
    Effect size = median z-shift of each family's KEY feature during each
    failure's LEAD-UP (prefail) vs the healthy baseline. Signed (diverging),
    so multidirectional signatures show as +/-.
    """
    feats = pd.read_parquet(features_path)
    feats["window_start"] = pd.to_datetime(feats["window_start"])
    fw = pd.read_csv(failure_windows_path, parse_dates=["start", "end"])

    # one representative feature per family (edit to your actual column names)
    family_feature = {
        "duty / state":      "duty",
        "pressure dynamics": "tp3_decay_slope",
        "thermal":           "oil_residual",
        "variability":       "cycle_dur_cv",
        "instrument health": "antiphase_share",
    }
    ts = feats["window_start"]
    healthy = feats["label"] == "healthy"

    rows = []
    for fam, col in family_feature.items():
        if col not in feats.columns:
            rows.append([np.nan]*len(fw)); continue
        base_med = feats.loc[healthy, col].median()
        base_iqr = (feats.loc[healthy, col].quantile(.75) -
                    feats.loc[healthy, col].quantile(.25)) or 1.0
        row = []
        for _, f in fw.iterrows():
            pre = (ts >= f["start"] - pd.Timedelta(hours=prefail_hours)) & (ts < f["start"])
            shift = (feats.loc[pre, col].median() - base_med) / base_iqr
            row.append(shift)
        rows.append(row)

    M = pd.DataFrame(rows, index=list(family_feature), columns=fw["failure_id"].tolist())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    vmax = np.nanmax(np.abs(M.values))
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        color="white" if abs(v) > vmax*0.5 else "black", fontsize=9)
    ax.set_title("Per-failure signatures (prefail median z-shift vs healthy)")
    fig.colorbar(im, ax=ax, label="signed effect size", shrink=0.8)
    out = FIGDIR / "effect_size_heatmap.png"
    fig.savefig(out); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    # comment out any you can't run yet; each is independent
    fig_health_timeline()
    fig_threshold_curve()      # uses recorded values if you don't pass a sweep_df
    fig_effect_heatmap()
