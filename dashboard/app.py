"""FAHM — Field Asset Health Monitor dashboard (GAP2).

    poetry run streamlit run dashboard/app.py

Architecture: ONE live fragment (`monitor`) reruns on its own timer via
run_every, advancing the replay cursor and redrawing the badge, score, sensor
readouts, timeline, and detector-driver panel together — smoothly, WITHOUT a
full-page refresh. The sidebar (thresholds, jump) is static, outside the
fragment. No st.rerun() anywhere.

Playback is throttled: the fragment ticks every 0.5s but the window only
advances once `secs_per_window` real seconds have passed, so each window holds
long enough to read. Prev/Next step one window (auto-pause); Stop halts.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SCORES_PATH = Path("data/processed/scores.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")
FAILURES_PATH = Path("data/processed/failure_windows.csv")

STATUS_COLORS = {
    "healthy": "#2E8B57", "watch": "#D4A017", "alert": "#E07B39",
    "critical": "#C0392B", "untrusted": "#808080",
}
SUSPECTED_FAULT = {
    "tp3_decay_slope": "air leak (pressure decaying during idle)",
    "oil_median": "overwork heating (compensating for a leak)",
    "oil_std": "unstable thermal behavior",
    "duty": "abnormal load (compressor overworking)",
    "cycles_per_hour": "abnormal cycling rate",
    "cycle_dur_cv": "erratic cycling (short-cycling)",
}
# left panel: the machine's state (edit to real raw column names when available)
RAW_SIGNALS = ["oil_median", "duty", "cycles_per_hour", "tp3_decay_slope",
               "longest_load_stretch"]

st.set_page_config(page_title="FAHM — Asset Health Monitor", layout="wide")


@st.cache_data
def load_scores():
    df = pd.read_parquet(SCORES_PATH)
    df["window_start"] = pd.to_datetime(df["window_start"])
    return df.sort_values("window_start").reset_index(drop=True)

@st.cache_data
def load_features():
    if FEATURES_PATH.exists():
        f = pd.read_parquet(FEATURES_PATH)
        f["window_start"] = pd.to_datetime(f["window_start"])
        return f.sort_values("window_start").reset_index(drop=True)
    return None

@st.cache_data
def load_failures():
    if FAILURES_PATH.exists():
        return pd.read_csv(FAILURES_PATH, parse_dates=["start"])
    return None

scores = load_scores()
feats = load_features()
fw = load_failures()

st.session_state.setdefault("win_idx", len(scores) - 1)
st.session_state.setdefault("play", False)
st.session_state.setdefault("last_step_t", 0.0)


def window_status(i, t_watch, t_alert, persist_k):
    row = scores.iloc[i]
    if "quality_bad" in scores.columns and bool(row.get("quality_bad", False)):
        return "untrusted"
    z = row["zmax"]
    recent = scores["zmax"].iloc[max(0, i - persist_k + 1): i + 1]
    if len(recent) == persist_k and (recent >= t_alert).all():
        return "critical"
    if z >= t_alert:
        return "alert"
    if z >= t_watch:
        return "watch"
    return "healthy"


# ---------------------------------------------------------------- sidebar
st.sidebar.title("FAHM controls")
t_alert = st.sidebar.slider("Alert threshold (zmax)", 2.0, 15.0, 7.0, 0.5)
t_watch = st.sidebar.slider("Watch threshold (zmax)", 1.0, 10.0, 4.0, 0.5)
persist_k = st.sidebar.slider("Critical persistence (windows)", 1, 6, 3)
st.sidebar.slider("Seconds per window", 0.5, 5.0, 2.0, 0.5, key="secs_per_window")

st.sidebar.markdown("---")
st.sidebar.subheader("Jump to failure")
if fw is not None:
    jc = st.sidebar.columns(4)
    for j, fid in enumerate(["F1", "F2", "F3", "F4"]):
        row = fw[fw["failure_id"] == fid]
        if len(row):
            tgt = int((scores["window_start"] - row["start"].iloc[0]).abs().idxmin())
            if jc[j].button(fid):
                st.session_state["win_idx"] = max(0, tgt - 48)
                st.session_state["play"] = False


# ---------------------------------------------------------------- monitor
@st.fragment(run_every="0.5s")
def monitor():
    # throttled auto-advance: step by 1 only after secs_per_window elapsed
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
            st.session_state["last_step_t"] = 0.0        # advance immediately
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

    # resolve current window AFTER all controls
    idx = st.session_state["win_idx"]
    now = scores.iloc[idx]
    status = window_status(idx, t_watch, t_alert, persist_k)

    # status badge + metrics
    b1, b2, b3 = st.columns([2, 1, 1])
    with b1:
        st.markdown(
            f"<div style='background:{STATUS_COLORS[status]};color:white;"
            f"padding:14px;border-radius:10px;text-align:center;'>"
            f"<span style='font-size:26px;font-weight:bold;'>{status.upper()}"
            f"</span></div>", unsafe_allow_html=True)
    with b2:
        st.metric("Health score", f"{now['zmax']:.2f}")
    with b3:
        st.metric("Alert @", f"{t_alert:.1f}")

    if status in ("alert", "critical"):
        drv = now.get("driver", "?")
        st.error(f"**{drv}** \u2192 suspected {SUSPECTED_FAULT.get(drv, 'anomaly')} "
                 f"\u00b7 score {now['zmax']:.1f}")
    elif status == "untrusted":
        st.warning("Sensor fault \u2014 reading not trusted this window.")

    # timeline with playhead
    lo = max(0, idx - 24 * 14)
    view = scores.iloc[lo: idx + 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["window_start"], y=view["zmax"],
                             mode="lines", line=dict(color="#1F3A5F", width=2)))
    fig.add_hline(y=t_alert, line_dash="dash", line_color="#C0392B")
    fig.add_hline(y=t_watch, line_dash="dot", line_color="#D4A017")
    fig.add_trace(go.Scatter(x=[now["window_start"]], y=[now["zmax"]],
                             mode="markers",
                             marker=dict(color=STATUS_COLORS[status], size=12,
                                         line=dict(color="white", width=2))))
    if fw is not None:
        for _, f in fw.iterrows():
            if f["start"] >= view["window_start"].min():
                fig.add_vline(x=f["start"], line_color="#C0392B",
                              line_dash="dot", opacity=0.4)
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      showlegend=False, yaxis_title="health score (zmax)")
    st.plotly_chart(fig, use_container_width=True, key=f"tl_{idx}")

    # two feature panels: raw state (left) vs detector drivers (right)
    if feats is not None:
        frow = feats[feats["window_start"] == now["window_start"]]
        if len(frow):
            fr = frow.iloc[0]
            left, right = st.columns(2)
            with left:
                st.caption("\U0001f4e1 Sensor readings — this window")
                gcols = st.columns(2)
                shown = [s for s in RAW_SIGNALS if s in fr]
                for k, sig in enumerate(shown):
                    gcols[k % 2].metric(sig, f"{fr[sig]:+.2f}")
            with right:
                st.caption("\u26a1 Driving the detector — top |z| from healthy")
                num = fr[feats.select_dtypes(include=[np.number]).columns]
                top = (num.abs()
                       .drop(labels=[c for c in num.index if any(
                            k in c for k in ("regime", "missing", "frozen",
                                             "segment", "n_samples"))],
                             errors="ignore")
                       .sort_values(ascending=False).head(6))
                st.bar_chart(top, horizontal=True)


monitor()

# ---------------------------------------------------------------- static footer
st.markdown("---")
st.caption("Capability envelope \u2014 what this system can and cannot catch")
st.table(pd.DataFrame({
    "failure": ["F1 idle-step", "F4 gradual", "F3 cycling", "F2 short", "degradation"],
    "detection": ["strong .86", "good .80", "moderate .70", "weak .64", "weak .58"],
    "lead": ["32.5 h", "8 h", "9.6 h", "none", "trend"],
}))
