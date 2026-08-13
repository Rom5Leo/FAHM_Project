"""FAHM — Field Asset Health Monitor dashboard (GAP2).

    poetry run streamlit run dashboard/app.py     (run from repo root)

Pure UI orchestration; computation in dashboard_lib/ (data/status/plots).

Replay clock is DECOUPLED from rendering:
  * `clock()`   — tiny fragment, ONLY mounted while playing. Ticks every 1s,
                  advances `win_idx`, renders nothing. When paused it is not
                  called at all, so there is zero background polling / flashing.
  * `monitor()` — render body; reruns (and redraws the chart) only on a real
                  event: a step from the clock, a marker click, a slider drag,
                  or a transport button.

Layout mimics a real ops dashboard: a top status bar, bordered "cards" with
section titles, and a footer status bar. Dark theme lives in
.streamlit/config.toml; card borders/accents come from the CSS block below.

Popups (alarm detail, capability) use a session_state "pending dialog" flag so
they survive reruns.
"""
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from dashboard_lib import data, status, plots

st.set_page_config(page_title="FAHM — Asset Health Monitor", layout="wide",
                   initial_sidebar_state="expanded")


# ---------------------------------------------------------------- style
st.markdown("""
<style>
  /* clear Streamlit's fixed top toolbar (Deploy + ⋮) so our bar isn't hidden */
  .block-container { padding-top: 2.6rem; padding-bottom: 1rem;
                     max-width: 100% !important; }
  header[data-testid="stHeader"] { background: rgba(0,0,0,0); }

  /* ---- compact vertical rhythm so it fits closer to one screen ---- */
  div[data-testid="stVerticalBlock"]      { gap: .55rem; }
  div[data-testid="stHorizontalBlock"]    { gap: .55rem; }
  div[data-testid="stMetric"]             { padding: 0; }
  div[data-testid="stMetricValue"]        { font-size: 1.35rem; }
  div[data-testid="stMetricLabel"]        { margin-bottom: .1rem; }
  div[data-testid="stElementContainer"]   { margin-bottom: 0; }

  /* bordered containers -> "cards" (tighter padding) */
  div[data-testid="stVerticalBlockBorderWrapper"] {
      background: #16202c;
      border: 1px solid #24313f !important;
      border-radius: 10px;
      box-shadow: 0 1px 3px rgba(0,0,0,.35);
      padding: .5rem .7rem !important;
  }
  .panel-title {
      font-size: 13px; font-weight: 700; letter-spacing: .06em;
      text-transform: uppercase; color: #9fb2c4; margin: 0 0 .3rem 0;
      border-bottom: 1px solid #24313f; padding-bottom: .3rem;
  }
  /* top + footer status bars */
  .statusbar {
      display:flex; align-items:center; justify-content:space-between;
      padding: .2rem .1rem; margin-bottom:.4rem;
  }
  .statusbar.top {
      border-bottom: 2px solid #24313f;
      /* leave room on the right for Streamlit's Deploy/⋮ controls */
      padding-right: 8rem;
  }
  .statusbar.foot  { border-top: 1px solid #24313f; margin-top:.5rem;
                     color:#7d90a3; font-size:12px; }
  .app-title { font-size: 18px; font-weight: 800; color:#e6edf3;
               letter-spacing:.02em; }
  .pill { background:#1d2a38; border:1px solid #2c3c4d; border-radius:20px;
          padding:2px 10px; font-size:12px; color:#c7d4e0; white-space:nowrap; }
  .pill.alarm { background:#3a1d1d; border-color:#5a2b2b; color:#ff8a8a; }

  /* ---- sensor readings: separated, color-accented cells ---- */
  .sensor-grid { display:grid; grid-template-columns:repeat(3,1fr);
                 gap:.5rem; }
  .sensor-cell {
      background:#111b26; border:1px solid #24313f; border-radius:8px;
      border-left:4px solid var(--accent,#3ba7e0);
      padding:.6rem .8rem;
  }
  .sensor-cell .s-label { font-size:18px; color:#9fb2c4; letter-spacing:.03em; }
  .sensor-cell .s-value { font-size:1.6rem; font-weight:700; color:#e6edf3;
                          line-height:1.3; }
  .sensor-cell .s-unit  { font-size:1rem; color:#7d90a3; font-weight:500; }
</style>
""", unsafe_allow_html=True)


def panel_title(txt):
    st.markdown(f"<div class='panel-title'>{txt}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- load (cached)
@st.cache_data
def _scores():
    return data.load_scores()

@st.cache_data
def _features():
    return data.load_features()

@st.cache_data
def _failures():
    return data.load_failures()

@st.cache_data
def _sensors(_scores_df):
    return data.build_sensor_lookup(_scores_df)

scores = _scores()
feats = _features()
fw = _failures()
sensors = _sensors(scores)
fail_idx = data.failure_indices(scores, fw)

st.session_state.setdefault("win_idx", len(scores) - 1)
st.session_state.setdefault("play", False)
st.session_state.setdefault("last_step_t", 0.0)
st.session_state.setdefault("pending_ev", None)
st.session_state.setdefault("pending_cap", False)
st.session_state.setdefault("handled_click", None)

@st.cache_data
def _all_events(tw, ta, pk):
    return status.alarm_events(scores, tw, ta, pk, up_to_idx=None)
@st.cache_data
def _events_upto(idx, tw, ta, pk):
    return status.alarm_events(scores, tw, ta, pk, up_to_idx=idx)


# ---------------------------------------------------------------- dialogs
@st.dialog("Alarm detail")
def alarm_dialog(ev):
    color = status.STATUS_COLORS[ev["level"]]
    st.markdown(
        f"<div style='background:{color};color:white;padding:8px;"
        f"border-radius:8px;text-align:center;font-size:18px;font-weight:bold'>"
        f"{ev['level'].upper()}</div>", unsafe_allow_html=True)
    st.write(f"**Raised:** {ev['window_start']:%Y-%m-%d %H:%M}  \u00b7  "
             f"**score:** {ev['zmax']:.2f}")
    st.write(f"**Driver:** `{ev['driver']}` \u2192 "
             f"{status.suspected_fault(ev['driver'])}")
    if sensors is not None:
        st.markdown("**Sensor readings at alarm:**")
        srow = sensors.iloc[ev["idx"]]
        dcols = st.columns(2)
        for k, (col, label, unit) in enumerate(data.SENSOR_DISPLAY):
            if col in sensors.columns:
                dcols[k % 2].metric(label, f"{srow[col]:.2f} {unit}")
    c1, c2 = st.columns(2)
    if c1.button("Go to this moment", use_container_width=True):
        st.session_state["win_idx"] = ev["idx"]
        st.session_state["play"] = False
        st.session_state["pending_ev"] = None
        st.rerun()
    if c2.button("Close", use_container_width=True):
        st.session_state["pending_ev"] = None
        st.rerun()


@st.dialog("Capability envelope")
def capability_dialog():
    st.caption("What this system can and cannot catch (honest, leakage-free)")
    st.table(pd.DataFrame({
        "failure": ["F4 gradual", "F1 idle-step", "F3 cycling", "F2 short",
                    "degradation"],
        "detection": ["strong .79", "good .77", "marginal .65", "weak .65",
                      "below chance .35"],
        "lead": ["14 h", "32.5 h", "9.6 h", "none", "trend"],
    }))
    if st.button("Close", use_container_width=True):
        st.session_state["pending_cap"] = False
        st.rerun()


# ---------------------------------------------------------------- sidebar
st.sidebar.markdown("### FAHM controls")
t_alert = st.sidebar.slider("Alert thr", 2.0, 15.0, 7.0, 0.5)
t_watch = st.sidebar.slider("Watch thr", 1.0, 10.0, 4.0, 0.5)
persist_k = st.sidebar.slider("Critical persist", 1, 6, 3)

st.sidebar.markdown("**Jump to failure**")
if fail_idx:
    jc = st.sidebar.columns(len(fail_idx))
    for j, (fid, idx0) in enumerate(fail_idx.items()):
        if jc[j].button(fid, key=f"jump_{fid}"):
            st.session_state["win_idx"] = max(0, idx0 - 48)
            st.session_state["play"] = False

st.sidebar.markdown("**Alarm history**")
all_events = _all_events(t_watch, t_alert, persist_k)
groups = status.group_alarms_by_failure(all_events, fw)
for gid, evs in groups.items():
    with st.sidebar.expander(f"{gid} \u2014 {len(evs)} alarm(s)"):
        for e in evs:
            lbl = (f"{e['level'][:4].upper()} {e['window_start']:%m-%d %H:%M} "
                   f"z={e['zmax']:.1f}")
            if st.button(lbl, key=f"hist_{e['idx']}", use_container_width=True):
                st.session_state["pending_ev"] = e
                st.session_state["play"] = False

if st.sidebar.button("\U0001f4cb Capability envelope", use_container_width=True):
    st.session_state["pending_cap"] = True


# ---------------------------------------------------------------- clock
@st.fragment(run_every="1s")
@st.fragment(run_every="1s")
def clock():
    """Advance the replay. Ticks once per second; the `dwell` wall-clock gate is
    the real step throttle. Guards on `play` INSIDE the fragment so the fragment
    stays mounted with a stable run_every timer instead of being re-created on
    every play/pause toggle."""
    if not st.session_state.get("play"):
        return
    if st.session_state.get("pending_ev") is not None:
        return
    dwell = st.session_state.get("secs_per_window", 2.0)
    nowt = time.time()
    if nowt - st.session_state["last_step_t"] >= dwell:
        cur = st.session_state["win_idx"]
        if cur < len(scores) - 1:
            st.session_state["win_idx"] = cur + 1
            st.session_state["last_step_t"] = nowt
            st.rerun(scope="app")
        else:
            st.session_state["play"] = False


# ---------------------------------------------------------------- top bar
def top_bar():
    idx = st.session_state["win_idx"]
    stat = status.window_status(scores, idx, t_watch, t_alert, persist_k)
    events = _events_upto(idx, t_watch, t_alert, persist_k)
    n_alerts = sum(1 for e in events if e["level"] in ("alert", "critical"))
    live = "\U0001f7e2 LIVE" if st.session_state["play"] else "\u23f8 PAUSED"
    alarm_pill = (f"<span class='pill alarm'>\u26a0 {n_alerts}</span>"
                  if n_alerts else "<span class='pill'>\u2713 0</span>")
    st.markdown(
        f"<div class='statusbar top'>"
        f"<span class='app-title'>FAHM \u2014 FIELD ASSET HEALTH MONITOR</span>"
        f"<span>{alarm_pill}"
        f"&nbsp;&nbsp;<span class='pill'>{live}</span>"
        f"&nbsp;&nbsp;<span class='pill'>STATUS: {stat.upper()}</span></span>"
        f"</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- monitor
def monitor():
    if st.session_state["pending_ev"] is not None:
        alarm_dialog(st.session_state["pending_ev"])
    elif st.session_state["pending_cap"]:
        capability_dialog()

    idx = st.session_state["win_idx"]
    now = scores.iloc[idx]
    stat = status.window_status(scores, idx, t_watch, t_alert, persist_k)
    scolor = status.STATUS_COLORS[stat]
    events = _events_upto(idx, t_watch, t_alert, persist_k)
    n_alerts = sum(1 for e in events if e["level"] in ("alert", "critical"))

    # ===== ROW 1: KPI strip — five identical cells, width-only differences =====
    # Each cell: panel title + one fixed-height value line, so the row never
    # ragged-edges regardless of content. Width vector below sizes them.
    def kpi_cell(title, value_html):
        with st.container(border=True):
            panel_title(title)
            st.markdown(
                f"<div style='height:34px;display:flex;align-items:center;'>"
                f"{value_html}</div>", unsafe_allow_html=True)

    k = st.columns([1.5, 1, 1.1, 1.3, 2.6])
    with k[0]:
        kpi_cell("System status",
                 f"<div style='width:100%;background:{scolor};color:white;"
                 f"padding:3px 6px;border-radius:6px;text-align:center;"
                 f"font-size:14px;font-weight:800;letter-spacing:.03em;'>"
                 f"{stat.upper()}</div>")
    with k[1]:
        kpi_cell("Health (z)",
                 f"<span style='font-size:1.35rem;font-weight:700;color:#e6edf3;'>"
                 f"{now['zmax']:.2f}</span>")
    with k[2]:
        delta = (f"<span style='font-size:.72rem;color:#2E8B57;font-weight:600;'>"
                 f" \u2191 {n_alerts} alert+</span>") if n_alerts else ""
        kpi_cell("Alarms",
                 f"<span style='font-size:1.35rem;font-weight:700;color:#e6edf3;'>"
                 f"{len(events)}</span>{delta}")
    with k[3]:
        kpi_cell("Window",
                 f"<span style='font-size:1.35rem;font-weight:700;color:#e6edf3;'>"
                 f"{idx+1}/{len(scores)}</span>")
    with k[4]:
        if stat in ("alert", "critical"):
            drv = now.get("driver", "?")
            a_txt = f"\u26a0 {drv} \u2192 {status.suspected_fault(drv)}"
            a_col = "#C0392B"
        elif stat == "untrusted":
            a_txt = "\u26a0 Sensor fault \u2014 reading untrusted."
            a_col = "#808080"
        else:
            a_txt = "\u2713 No active alerts"
            a_col = "#2E8B57"
        kpi_cell("Active alert",
                 f"<div style='width:100%;background:rgba(0,0,0,.15);"
                 f"border-left:4px solid {a_col};border-radius:6px;"
                 f"padding:4px 10px;color:{a_col};font-size:13px;"
                 f"font-weight:600;overflow:hidden;text-overflow:ellipsis;"
                 f"white-space:nowrap;'>{a_txt}</div>")

    # ===== ROW 2: left = gauge + playback (stacked) | right = timeline =====
    # Both cards pinned to the SAME fixed height so they bottom-align, exactly
    # like the KPI row. PANEL_H is the shared height for this row.
    PANEL_H = 430
    lo = max(0, idx - 24 * 14)
    view = scores.iloc[lo: idx + 1]
    g = st.columns([1.35, 3])

    # ---- LEFT: gauge on top, transport controls tucked below it ----
    with g[0]:
        with st.container(border=True, height=PANEL_H):
            panel_title("Health index")
            st.plotly_chart(
                plots.health_gauge(now["zmax"], t_watch, t_alert, scolor),
                use_container_width=True, key="gauge")

            # playback sub-section title, between the gauge and the controls
            st.markdown(
                "<div style='font-size:13px;font-weight:700;letter-spacing:.06em;"
                "text-transform:uppercase;color:#9fb2c4;margin:.2rem 0 .4rem 0;"
                "border-top:1px solid #24313f;padding-top:.4rem;'>Playback</div>",
                unsafe_allow_html=True)

            # transport buttons: play | back | forward (compact, no captions)
            bc = st.columns(3)
            with bc[0]:
                label = "\u23f8" if st.session_state["play"] else "\u25b6"
                if st.button(label, use_container_width=True, key="btn_play"):
                    st.session_state["play"] = not st.session_state["play"]
                    st.session_state["last_step_t"] = 0.0
                    st.rerun()
            with bc[1]:
                if st.button("\u23ee", use_container_width=True, key="btn_prev"):
                    st.session_state["play"] = False
                    st.session_state["win_idx"] = max(0, st.session_state["win_idx"] - 1)
                    st.rerun()
            with bc[2]:
                if st.button("\u23ed", use_container_width=True, key="btn_next"):
                    st.session_state["play"] = False
                    st.session_state["win_idx"] = min(len(scores) - 1,
                                                      st.session_state["win_idx"] + 1)
                    st.rerun()

            # speed + window position sliders — native labels (no overlap)
            sc = st.columns(2)
            with sc[0]:
                st.slider("Speed (s/win)", 0.5, 5.0, 2.0, 0.5,
                          key="secs_per_window")
            with sc[1]:
                seek = st.slider("Window pos", 0, len(scores) - 1, idx)
                if seek != idx:
                    st.session_state["win_idx"] = seek
                    st.rerun()

            # compact time + progress readout
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:.9rem;font-weight:700;margin-top:2px;'>"
                f"<span style='color:#e6edf3;'>\U0001f552 {now['window_start']:%m-%d %H:%M}</span>"
                f"<span style='color:#3ba7e0;'>{int(100*(idx+1)/len(scores))}%</span>"
                f"</div>", unsafe_allow_html=True)

    # ---- RIGHT: timeline alone, same fixed height as the left column ----
    with g[1]:
        with st.container(border=True, height=PANEL_H):
            panel_title("Health & alarm history")
            fig = plots.health_timeline(view, now, stat, t_watch, t_alert, fw,
                                        alarms=events)
            sel = st.plotly_chart(fig, use_container_width=True, key="timeline",
                                  on_select="rerun", selection_mode="points")

        pts = (sel.get("selection", {}) or {}).get("points", []) if sel else []
        if pts:
            clicked = pts[0].get("customdata")
            if clicked is not None and clicked != st.session_state.get("handled_click"):
                ev = next((e for e in events if e["idx"] == clicked), None)
                if ev:
                    st.session_state["handled_click"] = clicked
                    st.session_state["pending_ev"] = ev
                    st.session_state["play"] = False
                    st.rerun()

    # ===== ROW 3: sensor readings card | drivers card =====
    # per-sensor accent color, keyed by column name (falls back to blue)
    SENSOR_ACCENT = {
        "Oil_temperature": "#E07B39",   # temperature -> orange
        "Motor_current":   "#4B8FD6",   # current -> blue
        "TP3":             "#2E8B57",   # pressures -> greens/teals
        "TP2":             "#3FA796",
        "DV_pressure":     "#5B8DEF",
        "Reservoirs":      "#8E7CC3",
    }
    # both panels pinned to the same height so they bottom-align
    ROW3_H = 250
    b = st.columns(2)
    with b[0]:
        with st.container(border=True, height=ROW3_H):
            panel_title("\U0001f4e1 Sensor readings")
            if sensors is not None:
                srow = sensors.iloc[idx]
                cells = []
                for (col, label, unit) in data.SENSOR_DISPLAY:
                    if col in sensors.columns:
                        accent = SENSOR_ACCENT.get(col, "#3ba7e0")
                        val = f"{srow[col]:.1f}"
                        cells.append(
                            f"<div class='sensor-cell' style='--accent:{accent}'>"
                            f"<div class='s-label'>{label}</div>"
                            f"<div class='s-value'>{val}"
                            f"<span class='s-unit'> {unit}</span></div></div>")
                st.markdown(
                    f"<div class='sensor-grid'>{''.join(cells)}</div>",
                    unsafe_allow_html=True)
    with b[1]:
        with st.container(border=True, height=ROW3_H):
            panel_title("\u26a1 Top |z| drivers")
            if feats is not None:
                frow = feats[feats["window_start"] == now["window_start"]]
                if len(frow):
                    fr = frow.select_dtypes(include=[np.number]).iloc[0]
                    st.plotly_chart(plots.drivers_bar(fr),
                                    use_container_width=True, key="drivers")


# ---------------------------------------------------------------- footer
def footer_bar():
    idx = st.session_state["win_idx"]
    now = scores.iloc[idx]
    auto = "\U0001f7e2 ON" if st.session_state["play"] else "\u26aa OFF"
    st.markdown(
        f"<div class='statusbar foot'>"
        f"<span>Last update: {now['window_start']:%Y-%m-%d %H:%M} "
        f"&nbsp;\u00b7&nbsp; Render: {datetime.now():%H:%M:%S}</span>"
        f"<span>Auto-advance: {auto}</span>"
        f"</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- render
top_bar()
monitor()
footer_bar()

# Mount the clock unconditionally so its 1s run_every timer stays stable across
# reruns. It no-ops internally when paused, so a paused dashboard still does no
# stepping — but the timer isn't re-armed on every play/pause, which is what
# made stepping run faster than the chosen s/window.
clock()
