"""Plotting helpers for EDA — SKELETON. Bodies are yours to write.

Rules this module lives by (D00):
  * Functions RECEIVE a DataFrame and parameters — they never load data
    and never open the config themselves.
  * The notebook calls these and looks at the result; no plt code in cells.
  * Each function does ONE thing and returns the figure (returning fig lets
    the caller save it later: fig.savefig(...) — you'll want that in step 6).

Import the column lists from preprocessing so there is exactly ONE
definition of the schema in the project:

    from fahm.preprocessing import ANALOG, DIGITAL, TIMESTAMP
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from fahm.preprocessing import ANALOG, DIGITAL, TIMESTAMP

def auto_eda_pdf(df: pd.DataFrame, out_pdf: str | Path,
                 max_rows: int = 150_000, verbose: int = 1) -> Path:
    """Run AutoViz and capture ALL figures it opens into one PDF.

    Capture must happen in the same call as AutoViz (inline backend
    releases figures at cell end). Same caveats as before: sampled,
    time-blind, pandas-3 shim required.
    """
    from autoviz.AutoViz_Class import AutoViz_Class
    from matplotlib.backends.backend_pdf import PdfPages

    if not hasattr(pd.DataFrame, "applymap"):
        pd.DataFrame.applymap = pd.DataFrame.map

    av = AutoViz_Class()
    av.AutoViz(
        filename="",
        dfte=df.drop(columns=[TIMESTAMP]),
        max_rows_analyzed=max_rows,
        verbose=verbose,
        chart_format="png",
    )

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for n in plt.get_fignums():
            pdf.savefig(plt.figure(n))
    plt.close("all")
    return out_pdf

    raise NotImplementedError

def plot_analog_distributions(df: pd.DataFrame, bins: int = 60, log: bool = False):
    """One histogram per ANALOG sensor in a 2x4 grid. Returns the figure.
    log=True: log-scale y — reveals small modes dwarfed by a dominant peak."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    for ax, col in zip(axes.ravel(), ANALOG):
        ax.hist(df[col], bins=bins, log=log)
        ax.set_title(col)

    for ax in axes.ravel()[len(ANALOG):]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig

    raise NotImplementedError


def digital_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per DIGITAL signal: value counts + fraction active.

    n_other counts values that are neither 0 nor 1 (by elimination) — all
    zeros = full-data proof that the int8 cast in D04 was safe.
    frac_active = column mean (valid ONLY because n_other == 0).
    """
    rows = []
    for col in DIGITAL:
        s = df[col]
        n_zeros = int((s == 0).sum())
        n_ones = int((s == 1).sum())
        rows.append({
            "signal": col,
            "n_zeros": n_zeros,
            "n_ones": n_ones,
            "n_other": len(s) - n_zeros - n_ones,
            "frac_active": round(float(s.mean()), 4),
        })
    return pd.DataFrame(rows).set_index("signal")

    raise NotImplementedError

def plot_digital_summary(summary: pd.DataFrame):
    """Horizontal bars of frac_active per digital signal. Returns the figure.
    Takes digital_summary()'s output — compute once, plot from the result."""
    s = summary["frac_active"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(s.index, s.values)
    ax.set_xlabel("fraction of time active")
    ax.set_xlim(0, 1.05)
    for i, v in enumerate(s.values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center")
    fig.tight_layout()
    return fig

    raise NotImplementedError

def save_fig(fig, name: str, cfg: dict, dpi: int = 120) -> Path:
    """Save to cfg['paths']['figures']/<name>.png, CLOSE the figure, return path.
    Closing prevents duplicate inline rendering and memory buildup."""
    out_dir = Path(cfg["paths"]["figures"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)          # <- the fix
    return out

    raise NotImplementedError


def plot_sensor_timeline(df: pd.DataFrame, sensor: str,
                         start=None, end=None):
    """Bonus (build when needed): one sensor over a time window. Returns fig.

    You'll want this the moment a histogram makes you ask "but WHEN does it
    look like that?" — distributions hide time; this reveals it.
    Hints: mask by TIMESTAMP between start/end; plot with a thin line
    (lw=0.5) — 1.5M points at default width is an unreadable smear.
    """
    raise NotImplementedError
