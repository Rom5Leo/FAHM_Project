# Progress Report — Field Asset Health Monitor (A1)

Status as of the baseline-model milestone. This document explains what was
built, why, in what order, and what every file does. Companion documents:
`decision_log.md` (why each choice was made), `docs/domain_primer.md`
(the physics and ML concepts), `docs/data_profile.md` (dataset facts).

---

## 1. The goal in one paragraph

Build a system that watches the Air Production Unit (air compressor) of a
metro train through its sensors and warns an engineer *before* a failure
forces the train out of service. Dataset: MetroPT-3 — 1,516,948 readings at
10-second intervals, Feb–Sep 2020, one asset, 7 analog sensors, 8 digital
signals, no labels except 4 failure events documented in the accompanying
paper (3 air leaks, 1 oil leak). Because there are only 4 failures, the
system is framed as **anomaly detection trained on healthy data, evaluated as
early warning** — not supervised classification (decision D01).

## 2. Pipeline overview

```
                    load_metropt.py                 make_failure_windows.py
raw CSV (Kaggle) ──────────────────► sensor_readings.parquet    failure_windows.csv
                                          │        (10 s grid)        │ (eval ground truth)
                     profile_data.py ◄────┤                           │
                     data_profile.md      │                           │
                     + 2 figures          │                           │
                                          ├────────────────┐         │
                              resample.py │                │ cycle_features.py
                                          ▼                ▼          │
                            features_base.parquet    cycles.parquet   │
                                 (1-min grid)        (physics layer)  │
                                          │                │          │
                        rolling_features.py                └── cycle_health_overview.png
                                          ▼                           │
                            features_model.parquet                    │
                                          │                           │
                          baseline_iforest.py ◄───────────────────────┘
                                          ▼
              model_outputs.parquet + anomaly_score_timeline.png
              + docs/evaluation_baseline.md  (early warning & false alarms)
```

Everything is driven by one config file, `configs/metropt.yaml` — paths,
resample rate, gap threshold, training periods, alert threshold, persistence.
Changing an experiment means changing the config, not the code.

## 3. Stage by stage

### Stage 0 — Repo scaffold
Poetry project, `src/fahm_project` package layout, git repo. Notebooks are
for exploration only; everything that matters lives in `src/` and is run as
`python -m fahm_project.<module>` (workbook rule).

### Stage 1 — Ingestion (`src/fahm_project/ingestion/load_metropt.py`)
Streams the ~1.5M-row CSV in 1M-row chunks (constant memory), assigns compact
dtypes (float32 analog, int8 digital), parses timestamps, normalizes
column-name variants across dataset releases via `COLUMN_ALIASES` (the Kaggle
release spells the flowmeter `Caudal_impulses`; we crashed on this once and
made the fix permanent, D04), validates the schema with a readable error, and
writes `data/processed/sensor_readings.parquet` (22 MB vs multi-hundred-MB
CSV). An `asset_id` column is added even though there is one asset, so the A2
simulator can later emit multiple assets into the identical schema.

### Stage 2 — Ground truth (`ingestion/make_failure_windows.py`)
Writes `failure_windows.csv` with the 4 documented failures (F1 air leak
Apr 18; F2 air leak May 29–30; F3 oil leak Jun 5–7; F4 air leak Jul 15).
Used ONLY for evaluation, never training (enforced in model code).
**Open item:** verify exact timestamps against the MetroPT paper (D05).

### Stage 3 — Profiling (`ingestion/profile_data.py`)
Generates `docs/data_profile.md`: coverage, sampling regularity, gap table,
per-sensor statistics, digital duty fractions, sensor-meaning table, and two
figures. Key findings it produced:
- Sampling is **10 s** (not 1 Hz as first assumed) → 1.5M rows = full
  Feb–Sep coverage (D06).
- **331 recording gaps ≈ 38 days** of missing time → all downstream windows
  must be gap-aware (D07).
- Compressor load duty ≈ 16% (DV_eletric fraction) — the single number an
  air leak inflates.
- Digital polarity of some signals (Oil_level 90% "active") contradicts the
  paper → verify empirically (D15).

### Stage 4 — Feature base grid (`features/resample.py`)
10 s → 1-minute aggregates: mean/std/min/max per analog sensor, fraction-
active per digital, `n_samples`, and `segment_id` which increments at every
gap > 10 min. Every rolling or cycle computation downstream groups by
`segment_id` so no window ever bridges a hole (D07, D08). Output:
`features_base.parquet` (~250k rows).

### Stage 5 — Physics layer (`features/cycle_features.py`)
Run-length encodes DV_eletric into load/idle runs, pairs each load run with
the following idle run, and emits one row per compressor cycle: duty, load and
idle durations, TP3 build rate under load, **TP3 decay rate while idle** — the
most direct air-leak measurement in the data (D09). Saves `cycles.parquet`
and `docs/figures/cycle_health_overview.png` (duty / decay / load duration
across 7 months with failure windows shaded).

What that figure taught us (the most important findings so far):
- **F4 is a textbook cycle-feature detection**: duty → 0.62, decay → −0.58
  bar/min at the failure, with deterioration visible slightly before.
- **F1 blinds the cycle features**: the motor is pinned at ~5.7 A for 24 h —
  no cycle ever completes, so the feature designed for leaks goes silent during
  the worst leak (D10). Fixed in Stage 6.
- **Undocumented event ~Mar 10**: duty ≈ 1.0, an 11-hour continuous load run —
  no documented failure. Excluded from training; to be investigated (D11).
- **Regime shift ~Mar 1**: baseline duty 0.07 (Feb) → 0.12 (after) (D12).

### Stage 6 — Model features (`features/rolling_features.py`)
Builds `features_model.parquet` (~37 columns): rolling means (30 min / 4 h /
24 h) of the core health signals, rolling stds of TP3 and motor current,
`duty_trend` (4 h duty minus 24 h duty), and `pinned_load_2h` — the rolling
2-hour **minimum** of per-minute duty, which reads ≈1.0 when the compressor
never unloads. That last feature is the direct fix for the F1 blind spot: it
requires no completed cycles (D10).

### Stage 7 — Baseline model (`models/baseline_iforest.py`)
StandardScaler + IsolationForest trained ONLY on configured healthy periods
(Feb 1–28 + Apr 5–14; rationale and tradeoff in D13), with a hard code-level
guarantee that failure windows can never enter training. Scores every minute
(higher = more anomalous), thresholds at the 99.5th percentile of *training*
scores, and applies persistence: an alert fires only when ≥30 of the last 60
minutes exceed threshold (D14). Outputs `model_outputs.parquet`, a full
timeline figure, and `docs/evaluation_baseline.md` with the two metrics that
matter to an engineer: early-warning hours per failure and false-alarm
episodes per healthy week (D02).

The whole feature→model→evaluation chain was verified end-to-end on synthetic
leak data before touching real data (simulated leak detected 11.5 h early).

## 4. Artifacts inventory

| Artifact | Produced by | Purpose |
|---|---|---|
| `data/processed/sensor_readings.parquet` | load_metropt | canonical 10 s data |
| `data/processed/failure_windows.csv` | make_failure_windows | evaluation ground truth |
| `docs/data_profile.md` + 2 figures | profile_data | dataset facts & sanity checks |
| `data/processed/features_base.parquet` | resample | 1-min gap-aware grid |
| `data/processed/cycles.parquet` + overview figure | cycle_features | physics/leak indicators |
| `data/processed/features_model.parquet` | rolling_features | model input |
| `data/processed/model_outputs.parquet` + timeline figure | baseline_iforest | scores & alerts |
| `docs/evaluation_baseline.md` | baseline_iforest | engineer-metric evaluation |
| `decision_log.md` | (manual) | every choice, defended |

## 5. What is deliberately NOT done yet

- Failure-window timestamps not verified against the paper (blocks final numbers).
- Baseline model not yet run/tuned on real data → error analysis pending.
- No stronger model (autoencoder / forecasting-residual) — baseline first,
  per workbook rule.
- F3 (oil leak) likely under-served by current features; oil-focused features
  pending EDA findings.
- March 10 event unexplained.
- model_card.md and error_analysis.md to be written after the first real
  evaluation.

## 6. Immediate next steps

1. Run the EDA deep-dive (`eda/deep_dive.py`) → resolve March 10, the F3 oil
   story, and digital polarities.
2. Verify failure timestamps against the paper; update `make_failure_windows.py`.
3. Run baseline on real data → read per-failure table + false-alarm rate.
4. Error analysis → feature/threshold iteration → model card.
