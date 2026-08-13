"""Figure builders for the FAHM dashboard — return Plotly figures / Series,
no Streamlit calls. app.py renders them; keeping them here makes the app file
pure layout and lets these be tested or reused.

Styling target: an industrial ops monitor (dark navy, ring gauges with a big
centred readout, filled trend bands, thin gridlines, colored driver bars).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .status import STATUS_COLORS

# ---- shared palette (kept in sync with app.py / config.toml) ----
NAVY      = "#1F3A5F"
GRID      = "#24313f"
INK       = "#e6edf3"
MUTED     = "#8892A6"
CARD_BG   = "rgba(0,0,0,0)"   # transparent -> shows the card behind it

# translucent zone fills for the gauge ring
ZONE_HEALTHY = "rgba(46,139,87,0.30)"
ZONE_WATCH   = "rgba(212,160,23,0.30)"
ZONE_ALERT   = "rgba(192,57,43,0.30)"


def health_gauge(zmax: float, t_watch: float, t_alert: float,
                 status_color: str, vmax: float | None = None) -> go.Figure:
    """Ring gauge for the current health score — the big dial in the reference
    monitor. Colored watch/alert zones, a bar for the live value, and a thin
    threshold marker at the alert line. `vmax` auto-scales if not given so the
    needle never pins at the top when zmax spikes.

    (Single definition — the previous module had two `health_gauge`s and the
    second silently shadowed the first.)
    """
    if vmax is None:
        vmax = max(t_alert * 1.6, zmax * 1.25, 15.0)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=min(zmax, vmax),
        number=dict(font=dict(size=40, color=status_color)),
        gauge=dict(
            shape="angular",
            axis=dict(range=[0, vmax], tickcolor=MUTED,
                      tickfont=dict(color=MUTED, size=9), tickwidth=1),
            bar=dict(color=status_color, thickness=0.30),
            bgcolor=CARD_BG,
            borderwidth=0,
            steps=[
                dict(range=[0, t_watch],        color=ZONE_HEALTHY),
                dict(range=[t_watch, t_alert],  color=ZONE_WATCH),
                dict(range=[t_alert, vmax],     color=ZONE_ALERT),
            ],
            threshold=dict(line=dict(color="white", width=3),
                           thickness=0.85, value=zmax),
        )))
    fig.update_layout(height=140, margin=dict(l=10, r=10, t=4, b=2),
                      paper_bgcolor=CARD_BG, font=dict(color=INK))
    return fig


def health_timeline(view: pd.DataFrame, now: pd.Series, status: str,
                    t_watch: float, t_alert: float,
                    failures: pd.DataFrame | None = None,
                    alarms: list | None = None) -> go.Figure:
    """Rolling zmax timeline: filled area under the score, translucent threshold
    bands, a colored playhead at `now`, failure markers, and (if given)
    clickable ALARM markers.

    `alarms` is a list of event dicts (idx, window_start, level, zmax, driver)
    from status.alarm_events. They render as a selectable scatter trace placed
    LAST and named 'alarms' so its point numbers are stable for click handling.
    """
    from .status import STATUS_COLORS as SC

    x = view["window_start"]
    ytop = float(max(view["zmax"].max(), t_alert)) * 1.15

    fig = go.Figure()

    # translucent threshold bands (healthy / watch / alert) across the width
    fig.add_hrect(y0=0, y1=t_watch, fillcolor="rgba(46,139,87,0.06)",
                  line_width=0, layer="below")
    fig.add_hrect(y0=t_watch, y1=t_alert, fillcolor="rgba(212,160,23,0.07)",
                  line_width=0, layer="below")
    fig.add_hrect(y0=t_alert, y1=ytop, fillcolor="rgba(192,57,43,0.08)",
                  line_width=0, layer="below")

    # filled area under the health score line
    fig.add_trace(go.Scatter(
        x=x, y=view["zmax"], mode="lines", name="health score",
        line=dict(color="#4B8FD6", width=2),
        fill="tozeroy", fillcolor="rgba(75,143,214,0.12)", hoverinfo="skip"))

    # threshold rules
    fig.add_hline(y=t_alert, line_dash="dash", line_color="#C0392B",
                  line_width=1, annotation_text="alert",
                  annotation_font_color="#C0392B",
                  annotation_font_size=10)
    fig.add_hline(y=t_watch, line_dash="dot", line_color="#D4A017",
                  line_width=1, annotation_text="watch",
                  annotation_font_color="#D4A017",
                  annotation_font_size=10)

    # failure markers
    if failures is not None:
        for _, f in failures.iterrows():
            if f["start"] >= x.min():
                fig.add_vline(x=f["start"], line_color="#C0392B",
                              line_dash="dot", opacity=0.4, line_width=1)

    # playhead
    fig.add_trace(go.Scatter(
        x=[now["window_start"]], y=[now["zmax"]], mode="markers", name="now",
        hoverinfo="skip",
        marker=dict(color=STATUS_COLORS[status], size=14,
                    line=dict(color="white", width=2))))

    # clickable alarm markers (only those within the current view window)
    if alarms:
        lo_t = x.min()
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

    fig.update_layout(
        height=360, margin=dict(l=8, r=8, t=6, b=6), showlegend=False,
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font=dict(color=MUTED),
        yaxis_title="health score (zmax)",
        xaxis=dict(gridcolor=GRID, zeroline=False),
        yaxis=dict(gridcolor=GRID, zeroline=False, range=[0, ytop]),
        hoverlabel=dict(bgcolor="#16202c", font_color=INK,
                        bordercolor=GRID))
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


def drivers_bar(feature_row: pd.Series, k: int = 6) -> go.Figure:
    """Themed horizontal bar chart of the top-k |z| drivers — replaces the
    default st.bar_chart so it matches the dark card styling. Bars are colored
    by magnitude (hotter = more anomalous)."""
    s = top_drivers(feature_row, k)[::-1]   # smallest at top for horizontal
    vals = s.values.astype(float)
    # simple magnitude ramp: teal -> amber -> red
    def _c(v):
        if v >= 3.0:  return "#C0392B"
        if v >= 2.0:  return "#E07B39"
        if v >= 1.0:  return "#D4A017"
        return "#2E8B57"
    fig = go.Figure(go.Bar(
        x=vals, y=list(s.index), orientation="h",
        marker=dict(color=[_c(v) for v in vals]),
        text=[f"{v:.1f}" for v in vals], textposition="outside",
        textfont=dict(color=MUTED, size=18), hoverinfo="skip"))
    fig.update_layout(
        height=205, margin=dict(l=8, r=18, t=4, b=4),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font=dict(color=MUTED),
        xaxis=dict(gridcolor=GRID, zeroline=False, showticklabels=False),
        yaxis=dict(gridcolor=CARD_BG, tickfont=dict(color=MUTED, size=18)),
        showlegend=False)
    return fig
