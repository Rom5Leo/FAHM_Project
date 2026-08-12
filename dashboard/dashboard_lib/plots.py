"""Figure builders for the FAHM dashboard — return Plotly figures / Series,
no Streamlit calls. app.py renders them; keeping them here makes the app file
pure layout and lets these be tested or reused.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .status import STATUS_COLORS

NAVY = "#1F3A5F"


def health_gauge(zmax: float, t_watch: float, t_alert: float,
                 status_color: str) -> go.Figure:
    """A circular gauge for the current health score — the big dial in the
    reference monitor. Colored zones: green (healthy) / yellow (watch) /
    red (alert+). The needle sits at the current zmax."""
    top = max(t_alert * 2, zmax * 1.2, 15)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=zmax,
        number={"font": {"size": 34, "color": "white"}},
        gauge={
            "axis": {"range": [0, top], "tickcolor": "white",
                     "tickfont": {"color": "white", "size": 9}},
            "bar": {"color": status_color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, t_watch], "color": "rgba(46,139,87,0.35)"},
                {"range": [t_watch, t_alert], "color": "rgba(212,160,23,0.35)"},
                {"range": [t_alert, top], "color": "rgba(192,57,43,0.35)"},
            ],
            "threshold": {"line": {"color": "white", "width": 3},
                          "thickness": 0.8, "value": zmax},
        },
    ))
    fig.update_layout(height=200, margin=dict(l=15, r=15, t=15, b=5),
                      paper_bgcolor="rgba(0,0,0,0)",
                      font={"color": "white"})
    return fig


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
                      showlegend=False, yaxis_title="health score (zmax)",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#8892A6"),
                      xaxis=dict(gridcolor="#1E2430"),
                      yaxis=dict(gridcolor="#1E2430"))
    return fig


def health_gauge(zmax: float, t_watch: float, t_alert: float,
                 status_color: str, vmax: float = 15.0) -> go.Figure:
    """A circular gauge for the current health score, with colored watch/alert
    zones — the industrial-monitor dial. Needle color follows current status."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=min(zmax, vmax),
        number=dict(font=dict(size=34, color=status_color), suffix=""),
        gauge=dict(
            axis=dict(range=[0, vmax], tickcolor="#8892A6"),
            bar=dict(color=status_color, thickness=0.28),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, t_watch], color="#16351F"),        # healthy zone
                dict(range=[t_watch, t_alert], color="#3A3416"),  # watch zone
                dict(range=[t_alert, vmax], color="#3A1A16"),     # alert zone
            ],
            threshold=dict(line=dict(color="#C0392B", width=3),
                           thickness=0.85, value=t_alert),
        )))
    fig.update_layout(height=210, margin=dict(l=15, r=15, t=10, b=5),
                      paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#E6EAF1"))
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
