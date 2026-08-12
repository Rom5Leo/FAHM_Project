"""Modeling for FAHM — SKELETON. Bodies are yours to write (worked examples
marked DONE).

Module role: features.py built the table; modeling.py turns it into scores
and evaluations. Frame per D19/D23: anomaly detection — fit on healthy +
instrument-clean, score everything, evaluate per-failure + degraded span.

Design rules:
  * NO random splits — time-aware only (adjacent windows are near-duplicates).
  * Training mask = label 'healthy' AND instrument-clean (antiphase > 0.5
    pre-scaling; after healthy-z-scaling use the _missing/flag columns as-is).
  * Every scorer has the same signature: fit on train windows, return a
    score PER WINDOW for the whole grid (higher = more anomalous).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


BOOKKEEPING = ["window_start", "window_end", "segment_id", "label"]


# ---------------------------------------------------------------------------
# 1. Matrix assembly — final dtype conversion
# ---------------------------------------------------------------------------

def make_matrix(feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into numeric model matrix X and bookkeeping meta.
    Booleans -> 0/1; cycling_regime -> one-hot; label/bounds kept in meta."""
    bk = [c for c in BOOKKEEPING if c in feats.columns]
    meta = feats[bk].copy()
    X = feats.drop(columns=bk).copy()

    # booleans -> int (motor_frozen, tp3_frozen, the _missing flags if bool)
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    # cycling_regime -> one-hot (unordered category; NOT 0/1/2)
    if "cycling_regime" in X.columns:
        dummies = pd.get_dummies(X["cycling_regime"], prefix="regime").astype(int)
        X = pd.concat([X.drop(columns="cycling_regime"), dummies], axis=1)

    # safety: the model matrix must be all-numeric
    obj = X.select_dtypes(include="object").columns
    if len(obj):
        raise ValueError(f"non-numeric columns left in X: {list(obj)}")

    return X, meta


# ---------------------------------------------------------------------------
# 2. Time-aware split + fit healthy-scaler
# ---------------------------------------------------------------------------

def time_split(feats: pd.DataFrame, train_frac: float = 0.6) -> dict:
    is_healthy = (feats["label"] == "healthy").values
    is_prefail = (feats["label"] == "prefail").values
    is_infail = (feats["label"] == "infail").values
    is_degraded = (feats["label"] == "degraded").values

    n_train = int(is_healthy.sum() * train_frac)

    # rank healthy windows chronologically; first n_train -> train
    healthy_rank = pd.Series(np.nan, index=feats.index)
    healthy_rank[is_healthy] = (feats.loc[is_healthy, "window_start"]
                                .rank(method="first").values - 1)
    is_train = is_healthy & (healthy_rank.values < n_train)
    is_val = is_healthy & (healthy_rank.values >= n_train)

    return {
        "train": is_train,
        "val_healthy": is_val,
        "eval_fail": is_prefail | is_infail,
        "eval_degr": is_degraded,
    }

def fit_healthy_scaler(X, splits, exclude=("window_start", "window_end",
                                           "segment_id", "n_samples")):
    """Fit healthy z-score stats on TRAIN-healthy only (no val leakage)."""
    skip = set(exclude) | {c for c in X.columns if c.endswith("_missing")}
    cont = [c for c in X.columns
            if c not in skip and X[c].dtype.kind in "fi" and X[c].nunique() > 2]
    train = splits["train"]
    mu = X.loc[train, cont].mean()
    sd = X.loc[train, cont].std().replace(0, 1.0)
    return mu, sd, cont

def apply_scaler(X, mu, sd, cont):
    Xs = X.copy()
    Xs[cont] = (X[cont] - mu) / sd
    return Xs

# ---------------------------------------------------------------------------
# 3. Scorers — same contract: higher score = more anomalous
# ---------------------------------------------------------------------------

def zmax_score(X, splits, cont_cols=None):
    """Max |z| across the CONTINUOUS (z-scored) features only.
    Flags/one-hots/_missing are not z-scores — including them pollutes the
    max — so score only cont_cols (the columns fit_healthy_scaler scaled).
    If cont_cols is None, falls back to all numeric (legacy behavior)."""
    if cont_cols is None:
        Xn = X.select_dtypes(include=[np.number])
    else:
        Xn = X[cont_cols]
    return Xn.abs().max(axis=1)

def mahalanobis_score(X: pd.DataFrame, splits: dict) -> pd.Series:
    """Mahalanobis distance with Ledoit-Wolf shrinkage covariance.
    Shrinkage gives a well-conditioned, invertible covariance even under the
    structural collinearity of these features (occupancy fractions, one-hots,
    _missing flags) — so no feature-dropping, and no silent pinv garbage."""
    from sklearn.covariance import LedoitWolf
    Xn = X.select_dtypes(include=[np.number])
    cols = Xn.columns                                  # freeze column order
    train = Xn.loc[splits["train"], cols].values

    lw = LedoitWolf().fit(train)
    mu = lw.location_
    inv = np.linalg.inv(lw.covariance_)                # invertible after shrinkage

    diff = Xn[cols].values - mu
    md = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))
    return pd.Series(md, index=X.index, name="mahalanobis")


def iforest_score(X: pd.DataFrame, splits: dict, **kw) -> pd.Series:
    from sklearn.ensemble import IsolationForest
    Xn = X.select_dtypes(include=[np.number])

    iso = IsolationForest(random_state=1, contamination="auto", **kw)
    iso.fit(Xn[splits["train"]])
    raw = iso.score_samples(Xn)            # higher = more NORMAL
    return pd.Series(-raw, index=X.index, name="iforest")   # TODO: flip sign -> higher = anomalous


# ---------------------------------------------------------------------------
# 4. Evaluation — per failure, lead time, false alarms
# ---------------------------------------------------------------------------

def evaluate_scores(scores: dict, feats: pd.DataFrame, fw: pd.DataFrame,
                    splits: dict) -> pd.DataFrame:
    """One row per (model, target): ROC-AUC, PR-AUC, precision, recall at the
    operating threshold, lead time, false-alarm rate.
    Targets = each failure (prefail+infail windows) and the degraded span,
    each scored against val_healthy. Accuracy is deliberately omitted
    (meaningless at this imbalance)."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    val = splits["val_healthy"]
    ts = feats["window_start"]
    rows = []

    for name, s in scores.items():
        # operating threshold = 99th percentile of healthy-validation scores
        thr = np.quantile(s[val], 0.99)
        fa_per_day = round((s[val] > thr).mean() * 24, 2)

        # --- per failure ---
        for _, f in fw.iterrows():
            pre = (ts >= f["start"] - pd.Timedelta(hours=48)) & (ts < f["start"])
            inf = (ts >= f["start"]) & (ts <= f["end"])
            target = pre | inf
            if target.sum() == 0:
                continue

            n_pos = int(target.sum())
            y = np.r_[np.ones(n_pos), np.zeros(int(val.sum()))]
            sc = np.r_[s[target].values, s[val].values]

            roc = roc_auc_score(y, sc)
            pr = average_precision_score(y, sc)

            # at the operating threshold
            pred = sc > thr
            tp = int(pred[:n_pos].sum())                 # target windows flagged
            recall = tp / n_pos
            precision = tp / int(pred.sum()) if pred.sum() else 0.0

            # lead time: earliest prefail window over threshold -> failure start
            fired = ts[pre][s[pre] > thr]
            lead = round((f["start"] - fired.min()).total_seconds() / 3600, 1) if len(fired) else 0.0

            rows.append({
                "model": name, "target": f["failure_id"],
                "roc_auc": round(roc, 3), "pr_auc": round(pr, 3),
                "precision": round(precision, 2), "recall": round(recall, 2),
                "lead_h": lead, "fa_per_day": fa_per_day,
            })

        # --- degraded span (D19 second target) ---
        degr = splits["eval_degr"]
        if degr.sum():
            n_pos = int(degr.sum())
            y = np.r_[np.ones(n_pos), np.zeros(int(val.sum()))]
            sc = np.r_[s[degr].values, s[val].values]
            pred = sc > thr
            tp = int(pred[:n_pos].sum())
            rows.append({
                "model": name, "target": "degraded",
                "roc_auc": round(roc_auc_score(y, sc), 3),
                "pr_auc": round(average_precision_score(y, sc), 3),
                "precision": round(tp / int(pred.sum()), 2) if pred.sum() else 0.0,
                "recall": round(tp / n_pos, 2),
                "lead_h": None, "fa_per_day": fa_per_day,
            })

    return pd.DataFrame(rows)

def show_evaluation(ev: pd.DataFrame, metrics=("roc_auc", "pr_auc", "recall", "lead_h")):
    """Pretty per-metric pivots (target x model) from an evaluate_scores table.
    Returns a dict of pivots; prints each. One call = the whole comparison."""
    from IPython.display import display          # explicit — no reliance on notebook globals
    pivots = {}
    for m in metrics:
        p = ev.pivot_table(index="target", columns="model", values=m)
        pivots[m] = p
        print(f"\n=== {m} ===")
        display(p.round(3))
    return pivots

def relabel_prefail(feats, fw, prefail_hours=168):
    """Recompute labels at window level with a longer prefail window.
    Keeps infail/degraded/invalid; only extends the prefail reach."""
    lab = feats["label"].copy()
    ts = feats["window_start"]
    for _, f in fw.iterrows():
        pre = (ts >= f["start"] - pd.Timedelta(hours=prefail_hours)) & (ts < f["start"])
        # only relabel windows currently 'healthy' (don't overwrite infail/degraded/invalid)
        lab[pre & (lab == "healthy")] = "prefail"
    return lab

def leave_one_failure_out(X, feats, fw, make_model, prefail_hours=48):
    """Train on healthy + 3 failures, test on the held-out 4th (unseen).
    Rotates all failures. make_model: zero-arg callable -> fresh classifier.
    Returns per-failure metrics on UNSEEN failures — the honest generalization test."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    ts = feats["window_start"]

    # window label: 1 = prefail/infail, 0 = healthy; which failure each positive belongs to
    y = pd.Series(0, index=feats.index)
    fail_of = pd.Series(pd.NA, index=feats.index, dtype="object")
    for _, f in fw.iterrows():
        m = (ts >= f["start"] - pd.Timedelta(hours=prefail_hours)) & (ts <= f["end"])
        y[m] = 1
        fail_of[m] = f["failure_id"]

    is_healthy = feats["label"] == "healthy"
    rows = []
    for _, f in fw.iterrows():
        fid = f["failure_id"]
        # TEST: this failure's windows + healthy;  TRAIN: other failures' windows + healthy
        test = (fail_of == fid) | is_healthy
        train = (fail_of.notna() & (fail_of != fid)) | is_healthy
        # keep test-healthy and train-healthy disjoint by time isn't required here since
        # the FAILURE is the held-out unit; healthy is shared context. (Note in log.)

        Xtr, ytr = X[train], y[train]
        Xte, yte = X[test], y[test]
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue

        model = make_model()
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        rows.append({
            "failure": fid,
            "roc_auc": round(roc_auc_score(yte, proba), 3),
            "pr_auc": round(average_precision_score(yte, proba), 3),
            "n_test_pos": int(yte.sum()),
        })
    return pd.DataFrame(rows)

def threshold_sweep(score, feats, fw, splits, quantiles=(0.90, 0.95, 0.99, 0.995, 0.999)):
    """How lead time and false-alarm rate trade off as the alarm threshold moves."""
    val = splits["val_healthy"]
    ts = feats["window_start"]
    rows = []
    for q in quantiles:
        thr = np.quantile(score[val], q)
        fa_day = (score[val] > thr).mean() * 24
        for _, f in fw.iterrows():
            pre = (ts >= f["start"] - pd.Timedelta(hours=48)) & (ts < f["start"])
            fired = ts[pre][score[pre] > thr]
            lead = round((f["start"] - fired.min()).total_seconds()/3600, 1) if len(fired) else 0.0
            rows.append({"quantile": q, "fa_per_day": round(fa_day, 2),
                         "failure": f["failure_id"], "lead_h": lead})
    return pd.DataFrame(rows)

def xgb_permutation_importance(X, feats, fw, prefail_hours=48, n_repeats=10):
    """Permutation importance of XGBoost (all failures vs healthy)."""
    import numpy as np, pandas as pd
    from xgboost import XGBClassifier
    from sklearn.metrics import average_precision_score

    Xn = X.select_dtypes(include=[np.number]).reset_index(drop=True)
    ts = feats["window_start"].reset_index(drop=True)

    y = np.zeros(len(Xn), dtype=int)
    for _, f in fw.iterrows():
        m = ((ts >= f["start"] - pd.Timedelta(hours=prefail_hours)) & (ts <= f["end"])).values
        y[m] = 1

    label = feats["label"].reset_index(drop=True).to_numpy()
    use = (label == "healthy") | (y == 1)

    Xu = np.ascontiguousarray(Xn[use].to_numpy(dtype=float))
    yu = np.asarray(y[use]).ravel().astype(int)          # force 1D
    assert yu.ndim == 1, yu.shape
    print("shapes — Xu:", Xu.shape, "yu:", yu.shape)

    xgb = XGBClassifier(scale_pos_weight=20, eval_metric="logloss",
                        random_state=1, n_estimators=200)
    xgb.fit(Xu, yu)

    rng = np.random.RandomState(1)
    base = average_precision_score(yu, xgb.predict_proba(Xu)[:, 1])

    importances = np.zeros(Xu.shape[1])
    for j in range(Xu.shape[1]):
        drops = np.empty(n_repeats)
        for r in range(n_repeats):
            Xp = Xu.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops[r] = base - average_precision_score(yu, xgb.predict_proba(Xp)[:, 1])
        importances[j] = drops.mean()

    return pd.Series(importances, index=Xn.columns).sort_values(ascending=False)

def plot_importance(imp, title="Feature importance", top=None, color="#4C72B0"):
    """Horizontal bar chart of an importance Series (largest at top). Returns fig."""
    import matplotlib.pyplot as plt
    s = imp.head(top) if top else imp
    s = s.iloc[::-1]                       # reverse so largest plots at top
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(s))))
    ax.barh(s.index, s.values, color=color)
    ax.set_xlabel("importance (mean AP drop when shuffled)")
    ax.set_title(title)
    ax.axvline(0, color="grey", lw=0.6)    # negatives = feature hurt the model
    fig.tight_layout()
    return fig

# importance on PREFAIL-only (drop infail) — the honestly predictive signal
def xgb_importance_prefail_only(X, feats, fw, prefail_hours=48, n_repeats=10):
    import numpy as np, pandas as pd
    from xgboost import XGBClassifier
    from sklearn.metrics import average_precision_score

    Xn = X.select_dtypes(include=[np.number]).reset_index(drop=True)
    ts = feats["window_start"].reset_index(drop=True)

    y = np.zeros(len(Xn), dtype=int)
    for _, f in fw.iterrows():
        m = ((ts >= f["start"] - pd.Timedelta(hours=prefail_hours)) & (ts < f["start"])).values  # prefail ONLY
        y[m] = 1

    label = feats["label"].reset_index(drop=True).to_numpy()
    use = (label == "healthy") | (y == 1)

    Xu = np.ascontiguousarray(Xn[use].to_numpy(dtype=float))
    yu = np.asarray(y[use]).ravel().astype(int)          # force 1D
    assert yu.ndim == 1, yu.shape
    print("shapes — Xu:", Xu.shape, "yu:", yu.shape)

    xgb = XGBClassifier(scale_pos_weight=40, eval_metric="logloss",
                        random_state=1, n_estimators=200)
    xgb.fit(Xu, yu)

    rng = np.random.RandomState(1)
    base = average_precision_score(yu, xgb.predict_proba(Xu)[:, 1])

    importances = np.zeros(Xu.shape[1])
    for j in range(Xu.shape[1]):
        drops = np.empty(n_repeats)
        for r in range(n_repeats):
            Xp = Xu.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops[r] = base - average_precision_score(yu, xgb.predict_proba(Xp)[:, 1])
        importances[j] = drops.mean()

    return pd.Series(importances, index=Xn.columns).sort_values(ascending=False)

def forecast_residual_score(feats, X, splits, signal="oil_median", n_lags=6):
    """Forecast-residual anomaly score (D23). Train a lag model on HEALTHY
    windows to predict `signal` from its previous n_lags values; the score is
    the absolute residual (|actual - predicted|). Large residual = the signal
    stopped following its own recent pattern = anomalous.

    Respects time order (lags are past-only) and segments (no lag across a gap).
    """
    from sklearn.linear_model import Ridge

    s = X[signal].reset_index(drop=True)
    seg = feats["segment_id"].reset_index(drop=True)

    # build lag matrix: row t = [s[t-1], ..., s[t-n_lags]] -> predict s[t]
    # only where all lags are in the SAME segment (no forecasting across a gap)
    rows, targets, idx = [], [], []
    for t in range(n_lags, len(s)):
        if seg[t] == seg[t - n_lags]:                 # same segment across the window
            rows.append(s[t - n_lags:t].values[::-1]) # most-recent-first
            targets.append(s[t])
            idx.append(t)
    Xlag = np.array(rows); ylag = np.array(targets); idx = np.array(idx)

    # train on healthy rows only
    healthy = (feats["label"].reset_index(drop=True).values == "healthy")[idx]
    model = Ridge().fit(Xlag[healthy], ylag[healthy])

    pred = model.predict(Xlag)
    resid = np.abs(ylag - pred)

    # map back to full-length score (NaN where no lag available -> fill 0 = "not scored")
    score = pd.Series(0.0, index=feats.index)
    score.iloc[idx] = resid
    return score.rename(f"fcast_resid_{signal}")

def healthy_anchored_residual(feats, X, splits, signals=("oil_median", "tp3_decay_slope"),
                              multi=True):
    """Healthy-anchored forecast-residual (D23, corrected).
    For each target signal, train a model on HEALTHY windows to predict it from
    the OTHER features, then score |actual - healthy_predicted| across all
    windows. Unlike lag-forecasting, this never adapts to the degrading machine
    — it holds the healthy expectation and measures departure from it.

    multi=True: combine per-signal residuals into one score (max of the
    healthy-scaled residuals), so it's comparable to zmax's multivariate score.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    h = (feats["label"] == "healthy").values
    resids = {}
    for sig in signals:
        predictors = [c for c in X.columns if c != sig]
        Xp = X[predictors].to_numpy()
        y = X[sig].to_numpy()

        model = GradientBoostingRegressor(random_state=1).fit(Xp[h], y[h])
        pred = model.predict(Xp)
        r = np.abs(y - pred)
        # scale each residual by its healthy spread so signals are comparable
        r = r / (r[h].std() or 1)
        resids[sig] = r

    R = pd.DataFrame(resids, index=feats.index)
    if multi:
        return R.max(axis=1).rename("healthy_anchored_resid")   # max = "most departed signal"
    return R