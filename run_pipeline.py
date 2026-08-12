#!/usr/bin/env python
"""FAHM pipeline runner — end-to-end, config-driven, no notebook (GAP1).

Usage:
    python run_pipeline.py --config configs/config.yaml                 # full pipeline
    python run_pipeline.py --config configs/config.yaml --stage features
    python run_pipeline.py --config configs/config.yaml --stage score --threshold-q 0.99

Stages (each reads its input artifact, writes its output artifact):
    preprocess  raw CSV -> sensor_readings.parquet (+ failure_windows.csv check)
    features    processed parquet -> features.parquet (grid, families, prep)
    score       features.parquet -> scores.parquet (zmax + threshold + alerts)
    evaluate    scores + failure windows -> evaluation report (per-failure)
    all         everything above, in order
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("fahm.pipeline")


# --------------------------------------------------------------------------
# stage implementations — thin wrappers around the src/fahm modules
# --------------------------------------------------------------------------

def stage_preprocess(cfg: dict) -> None:
    """Raw CSV -> typed, checked, parquet artifact. Mirrors notebook 01."""
    from fahm import preprocessing as pp

    log.info("preprocess: loading raw CSV ...")
    df = pp.load_raw(cfg)                       # naive+typed load
    pp.run_checks(df, cfg)                      # six integrity checks (raise on fail)
    fw = pp.build_failure_windows(cfg)          # from config (D07)
    path = pp.save_processed(df, cfg)
    log.info("preprocess: %s rows -> %s", f"{len(df):,}", path)


def stage_features(cfg: dict) -> None:
    """Processed parquet -> model-ready features.parquet. Mirrors notebook 04."""
    import pandas as pd
    from fahm import preprocessing as pp
    from fahm import analysis as an
    from fahm import features as ft

    log.info("features: loading processed data ...")
    df = pp.load_processed(cfg)
    fw = pd.read_csv(cfg["paths"]["failure_windows"],
                     parse_dates=["start", "end", "maintenance"])

    labels = an.label_windows(df, fw,
        degraded_periods=cfg["labels"]["degraded_periods"],
        invalid_periods=cfg["labels"]["invalid_periods"])

    grid = ft.build_window_grid(df, cfg)
    grid_labels = ft.label_grid(grid, labels, df)
    feats = ft.build_features(df, grid, cfg, grid_labels)
    feats = ft.add_cycling_regime(feats)
    model_table = ft.prepare_for_model(feats, grid_labels)
    path = ft.save_features(model_table, cfg)
    log.info("features: %s windows x %s cols -> %s",
             len(model_table), model_table.shape[1], path)


def stage_score(cfg: dict, threshold_q: float = 0.99) -> None:
    """features.parquet -> per-window zmax score + alert flag. Mirrors notebook 05."""
    import numpy as np
    import pandas as pd
    from fahm import modeling as mdl

    log.info("score: loading features ...")
    feats = pd.read_parquet(cfg["paths"]["features"])
    X, meta = mdl.make_matrix(feats)
    X = X.drop(columns=[c for c in ("frac_continuous_load", "frac_loaded")
                        if c in X.columns])

    splits = mdl.time_split(feats)

    # D34: features arrive UNSCALED; fit scaler on train-healthy only (no leak),
    # then score with pure zmax over the continuous columns only.
    mu, sd, cont = mdl.fit_healthy_scaler(X, splits)
    X = mdl.apply_scaler(X, mu, sd, cont)
    score = mdl.zmax_score(X, splits, cont_cols=cont)
    thr = float(np.quantile(score[splits["val_healthy"]], threshold_q))

    out = meta[["window_start", "window_end", "label"]].copy()
    out["zmax"] = score
    out["alert"] = score > thr
    out["driver"] = X[cont].abs().idxmax(axis=1)         # driver among z-scored cont cols
    # data-quality gate (trust axis, separate from health) -> dashboard "untrusted"
    # only GENUINE sensor faults -> untrusted; NOT the benign cycle-missing (idle machine)
    quality_cols = [c for c in X.columns if c.endswith("_frozen")]
    out["quality_bad"] = (X[quality_cols] == 1).any(axis=1) if quality_cols else False

    path = Path(cfg["paths"].get("scores", "data/processed/scores.parquet"))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    log.info("score: threshold(q=%.3f)=%.3f | alerts=%s/%s | quality_flags=%s -> %s",
             threshold_q, thr, int(out["alert"].sum()), len(out),
             int(out["quality_bad"].sum()), path)


def stage_evaluate(cfg: dict) -> None:
    """Scores + failure windows -> per-failure evaluation, printed + saved."""
    import pandas as pd
    from fahm import modeling as mdl

    feats = pd.read_parquet(cfg["paths"]["features"])
    fw = pd.read_csv(cfg["paths"]["failure_windows"],
                     parse_dates=["start", "end", "maintenance"])
    X, _ = mdl.make_matrix(feats)
    X = X.drop(columns=[c for c in ("frac_continuous_load", "frac_loaded")
                        if c in X.columns])
    splits = mdl.time_split(feats)
    # D34: scale on train-healthy only, pure zmax over continuous cols
    mu, sd, cont = mdl.fit_healthy_scaler(X, splits)
    X = mdl.apply_scaler(X, mu, sd, cont)
    scores = {"zmax": mdl.zmax_score(X, splits, cont_cols=cont)}
    ev = mdl.evaluate_scores(scores, feats, fw, splits)
    print(ev.to_string(index=False))
    path = Path(cfg["paths"].get("evaluation", "docs/evaluation.csv"))
    path.parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(path, index=False)
    log.info("evaluate: report -> %s", path)


STAGES = {
    "preprocess": stage_preprocess,
    "features": stage_features,
    "score": stage_score,
    "evaluate": stage_evaluate,
}


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FAHM pipeline runner")
    ap.add_argument("--config", required=True, help="path to config.yaml")
    ap.add_argument("--stage", default="all",
                    choices=[*STAGES, "all"], help="which stage to run")
    ap.add_argument("--threshold-q", type=float, default=0.99,
                    help="alert threshold quantile (score stage)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S")

    from fahm.preprocessing import load_config
    cfg = load_config(args.config)

    to_run = list(STAGES) if args.stage == "all" else [args.stage]
    for name in to_run:
        log.info("=== stage: %s ===", name)
        try:
            if name == "score":
                STAGES[name](cfg, threshold_q=args.threshold_q)
            else:
                STAGES[name](cfg)
        except Exception:
            log.exception("stage '%s' FAILED — aborting", name)
            return 1
    log.info("pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
