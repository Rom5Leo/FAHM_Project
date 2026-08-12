"""Figure builders for the FAHM dashboard — return Plotly figures / Series,
no Streamlit calls. app.py renders them; keeping them here makes the app file
pure layout and lets these be tested or reused.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .status import STATUS_COLORS

NAVY = "#1F3A5F"


def health_timeline(view: pd.DataFrame, now: pd.Series, status: str,
                    t_watch: float, t_alert: float,
                    failures: pd.DataFrame | None = None,
                    alarms: list | None = None) -> go.Figure:
    """Rolling zmax timeline with threshold bands, a colored playhead at `now`,
    failure markers, and (if given) clickable ALARM markers.

    `alarms` is a list of event dicts (idx, window_start, level, zmax, driver)
    from status.alarm_events. They render as a selectable scatter trace so the
    app can open a detail popup when one is clicked. The trace is placed LAST
    and named 'alarms' so its curve/point numbers are stable for click handling.
    """
    from .status import STATUS_COLORS as SC

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["window_start"], y=view["zmax"],
                             mode="lines", line=dict(color=NAVY, width=2),
                             name="health score", hoverinfo="skip"))
    fig.add_hline(y=t_alert, line_dash="dash", line_color="#C0392B",
                  annotation_text="alert")
    fig.add_hline(y=t_watch, line_dash="dot", line_color="#D4A017",
                  annotation_text="watch")
    if failures is not None:
        for _, f in failures.iterrows():
            if f["start"] >= view["window_start"].min():
                fig.add_vline(x=f["start"], line_color="#C0392B",
                              line_dash="dot", opacity=0.4)
    # playhead
    fig.add_trace(go.Scatter(x=[now["window_start"]], y=[now["zmax"]],
                             mode="markers", name="now", hoverinfo="skip",
                             marker=dict(color=STATUS_COLORS[status], size=13,
                                         line=dict(color="white", width=2))))
    # clickable alarm markers (only those within the current view window)
    if alarms:
        lo_t = view["window_start"].min()
        vis = [a for a in alarms if a["window_start"] >= lo_t]
        if vis:
            fig.add_trace(go.Scatter(
                x=[a["window_start"] for a in vis],
                y=[a["zmax"] for a in vis],
                mode="markers", name="alarms",
                marker=dict(size=15, symbol="triangle-down",
                            color=[SC[a["level"]] for a in vis],
                            line=dict(color="white", width=1)),
                customdata=[a["idx"] for a in vis],
                hovertemplate="%{x|%m-%d %H:%M} · z=%{y:.1f}<extra>alarm</extra>"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                      showlegend=False, yaxis_title="health score (zmax)")
    return fig


def top_drivers(feature_row: pd.Series, k: int = 6) -> pd.Series:
    """Top-k |z| features for a window (the detector's 'why'), excluding the
    non-z columns (regime one-hots, flags, bookkeeping)."""
    drop = [c for c in feature_row.index
            if any(tag in c for tag in
                   ("regime", "missing", "frozen", "segment", "n_samples", "label"))]
    return (feature_row.abs()
            .drop(labels=drop, errors="ignore")
            .sort_values(ascending=False)
            .head(k))
