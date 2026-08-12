"""FAHM — Field Asset Health Monitor dashboard (GAP2).

    poetry run streamlit run dashboard/app.py

Pure UI orchestration. All computation lives in dashboard_lib/ (data, status,
plots) — mirroring the src/fahm ⟷ notebook split: logic in a package, this
file just wires widgets to it.

Architecture: one live fragment (`monitor`) reruns on its own timer via
run_every, advancing the replay cursor and redrawing badge/score/sensors/
timeline/drivers together — smoothly, no full-page refresh. Playback is
throttled (secs_per_window) so each window holds long enough to read.
"""
import time

import numpy as np
import streamlit as st

from dashboard_lib import data, status, plots

st.set_page_config(page_title="FAHM — Asset Health Monitor", layout="wide")


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
    # per-window last raw reading (3b); cached once, keyed by the scores frame
    return data.build_sensor_lookup(_scores_df)

scores = _scores()
feats = _features()
fw = _failures()
sensors = _sensors(scores)
fail_idx = data.failure_indices(scores, fw)

st.session_state.setdefault("win_idx", len(scores) - 1)
st.session_state.setdefault("play", False)
st.session_state.setdefault("last_step_t", 0.0)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("FAHM controls")
t_alert = st.sidebar.slider("Alert threshold (zmax)", 2.0, 15.0, 7.0, 0.5)
t_watch = st.sidebar.slider("Watch threshold (zmax)", 1.0, 10.0, 4.0, 0.5)
persist_k = st.sidebar.slider("Critical persistence (windows)", 1, 6, 3)
st.sidebar.slider("Seconds per window", 0.5, 5.0, 2.0, 0.5, key="secs_per_window")

st.sidebar.markdown("---")
st.sidebar.subheader("Jump to failure")
if fail_idx:
    jc = st.sidebar.columns(len(fail_idx))
    for j, (fid, idx0) in enumerate(fail_idx.items()):
        if jc[j].button(fid):
            st.session_state["win_idx"] = max(0, idx0 - 48)   # start 2 days before
            st.session_state["play"] = False


# ---------------------------------------------------------------- monitor
@st.fragment(run_every="0.5s")
def monitor():
    # throttled auto-advance
    if st.session_state["play"]:
        dwell = st.session_state.get("secs_per_window", 2.0)
        nowt = time.time()
        if nowt - st.session_state["last_step_t"] >= dwell:
            cur = st.session_state["win_idx"]
            if cur < len(scores) - 1:
                st.session_state["win_idx"] = cur + 1
                st.session_state["last_step_t"] = nowt
            else:
                st.session_state["play"] = False

    # transport row
    tcols = st.columns([1, 1, 1, 1, 3, 2])
    with tcols[0]:
        label = "\u23f8 Pause" if st.session_state["play"] else "\u25b6 Play"
        if st.button(label, use_container_width=True):
            st.session_state["play"] = not st.session_state["play"]
            st.session_state["last_step_t"] = 0.0
    with tcols[1]:
        if st.button("\u23f9 Stop", use_container_width=True):
            st.session_state["play"] = False
    with tcols[2]:
        if st.button("\u23ee Prev", use_container_width=True):
            st.session_state["play"] = False
            st.session_state["win_idx"] = max(0, st.session_state["win_idx"] - 1)
    with tcols[3]:
        if st.button("\u23ed Next", use_container_width=True):
            st.session_state["play"] = False
            st.session_state["win_idx"] = min(len(scores) - 1,
                                              st.session_state["win_idx"] + 1)
    with tcols[4]:
        seek = st.slider("seek", 0, len(scores) - 1, st.session_state["win_idx"],
                         label_visibility="collapsed")
        if seek != st.session_state["win_idx"]:
            st.session_state["win_idx"] = seek
    with tcols[5]:
        st.markdown(f"**\U0001f552 "
                    f"{scores.iloc[st.session_state['win_idx']]['window_start']:%m-%d %H:%M}**")

    idx = st.session_state["win_idx"]
    now = scores.iloc[idx]
    stat = status.window_status(scores, idx, t_watch, t_alert, persist_k)

    # status badge + metrics
    b1, b2, b3 = st.columns([2, 1, 1])
    with b1:
        st.markdown(
            f"<div style='background:{status.STATUS_COLORS[stat]};color:white;"
            f"padding:14px;border-radius:10px;text-align:center;'>"
            f"<span style='font-size:26px;font-weight:bold;'>{stat.upper()}"
            f"</span></div>", unsafe_allow_html=True)
    with b2:
        st.metric("Health score", f"{now['zmax']:.2f}")
    with b3:
        st.metric("Alert @", f"{t_alert:.1f}")

    if stat in ("alert", "critical"):
        drv = now.get("driver", "?")
        st.error(f"**{drv}** \u2192 suspected {status.suspected_fault(drv)} "
                 f"\u00b7 score {now['zmax']:.1f}")
    elif stat == "untrusted":
        st.warning("Sensor fault \u2014 reading not trusted this window.")

    # timeline with clickable alarm markers
    lo = max(0, idx - 24 * 14)
    view = scores.iloc[lo: idx + 1]
    events = status.alarm_events(scores, t_watch, t_alert, persist_k, up_to_idx=idx)

    @st.dialog("Alarm detail")
    def _alarm_dialog(ev):
        color = status.STATUS_COLORS[ev["level"]]
        st.markdown(
            f"<div style='background:{color};color:white;padding:10px;"
            f"border-radius:8px;text-align:center;font-size:20px;font-weight:bold'>"
            f"{ev['level'].upper()}</div>", unsafe_allow_html=True)
        st.write(f"**Raised:** {ev['window_start']:%Y-%m-%d %H:%M}")
        st.write(f"**Health score:** {ev['zmax']:.2f}")
        st.write(f"**Driver:** `{ev['driver']}` \u2192 "
                 f"{status.suspected_fault(ev['driver'])}")
        if sensors is not None:
            st.markdown("**Sensor readings at alarm:**")
            srow = sensors.iloc[ev["idx"]]
            dcols = st.columns(2)
            for k, (col, label, unit) in enumerate(data.SENSOR_DISPLAY):
                if col in sensors.columns:
                    dcols[k % 2].metric(label, f"{srow[col]:.2f} {unit}")
        if st.button("Go to this moment in replay", use_container_width=True):
            st.session_state["win_idx"] = ev["idx"]
            st.session_state["play"] = False
            st.rerun()

    fig = plots.health_timeline(view, now, stat, t_watch, t_alert, fw, alarms=events)
    sel = st.plotly_chart(fig, use_container_width=True, key=f"tl_{idx}",
                          on_select="rerun", selection_mode="points")

    # if an alarm marker was clicked, open its popup
    pts = (sel.get("selection", {}) or {}).get("points", []) if sel else []
    for p in pts:
        clicked_idx = p.get("customdata")
        if clicked_idx is not None:
            ev = next((e for e in events if e["idx"] == clicked_idx), None)
            if ev:
                _alarm_dialog(ev)
                break

    st.caption("\u25bc Click an alarm marker on the timeline for its details.")

    # two panels: real sensor readings (left) vs detector drivers (right)
    left, right = st.columns(2)

    with left:
        st.caption("\U0001f4e1 Sensor readings \u2014 latest in this window")
        if sensors is not None:
            srow = sensors.iloc[idx]
            gcols = st.columns(2)
            for k, (col, label, unit) in enumerate(data.SENSOR_DISPLAY):
                if col in sensors.columns:
                    val = srow[col]
                    gcols[k % 2].metric(label, f"{val:.2f} {unit}")
        else:
            st.info("sensor_readings.parquet not found \u2014 readings unavailable.")

    with right:
        st.caption("\u26a1 Driving the detector \u2014 top |z| from healthy")
        if feats is not None:
            frow = feats[feats["window_start"] == now["window_start"]]
            if len(frow):
                fr = frow.select_dtypes(include=[np.number]).iloc[0]
                st.bar_chart(plots.top_drivers(fr), horizontal=True)


monitor()

# ---------------------------------------------------------------- static footer
st.markdown("---")
st.caption("Capability envelope \u2014 what this system can and cannot catch "
           "(honest, leakage-free)")
import pandas as pd
st.table(pd.DataFrame({
    "failure": ["F4 gradual", "F1 idle-step", "F3 cycling", "F2 short", "degradation"],
    "detection": ["strong .79", "good .77", "marginal .65", "weak .65", "below chance .35"],
    "lead": ["14 h", "32.5 h", "9.6 h", "none", "trend"],
}))
