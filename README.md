# FAHM — Field Asset Health Monitor

**Predictive maintenance on industrial sensor data: forecasting air-compressor
failures before they happen.**

A time-series anomaly-detection pipeline over the MetroPT-3 metro-train air
production unit (1.5M sensor readings, 7 months, four documented air-leak
failures). Built end-to-end: ingestion → EDA → feature engineering → modeling →
a config-driven pipeline runner. The emphasis throughout is **honest evaluation**
— per-failure detectability with real lead times, not a headline accuracy number.

---

## The headline result

A simple, interpretable detector (**zmax** — maximum standardized deviation from
healthy operation) warns of failures **8–32 hours ahead** at **~1 false alarm
every 4 days**, and beats every sophisticated alternative tried (IsolationForest,
Mahalanobis, seven supervised classifiers, forecast-residual).

| Failure | Detection (ROC-AUC) | Lead time | Verdict |
|---|---|---|---|
| F1 (idle-then-step leak) | 0.86 | 32.5 h | strong |
| F4 (gradual leak) | 0.80 | 8 h | good |
| F3 (erratic cycling) | 0.70 | 9.6 h | moderate |
| F2 (sudden) | 0.64 | — | not predictable in these sensors |
| slow degradation | 0.58 | — | weak (watch as a trend) |

These are **credible predictive-maintenance numbers** — real early warning with a
real precision/recall/false-alarm tradeoff — not the leakage-driven ~100%
accuracy that a window-labeled, randomly-split setup produces.

---

## Why this is the hard version of the problem

Predicting a failure **before** it manifests is fundamentally different from
detecting one while it happens:

- **Trained on healthy data only** (anomaly detection), because four failures
  cannot supervise a classifier. Failures are *evaluation* targets, never
  training data.
- **Time-ordered splits**, never random — adjacent windows are near-duplicates,
  so shuffling would leak the answer.
- **Labels the lead-up, not the failure window** — the model learns to warn, not
  to describe a machine that is already broken.
- **Evaluated per failure**, because the four failures have distinct,
  multidirectional signatures (some announce with rising duty and heat, one with
  idle, one with erratic cycling).

The modest AUCs are the honest cost of solving the real problem.

---

## Pipeline

Raw CSV → **preprocess** → **features** → **score** → **evaluate**, one command:

```bash
poetry install
poetry run python run_pipeline.py --config configs/config.yaml --stage all
```

Each stage reads and writes a parquet artifact, so stages are resumable. The
runner reproduces the notebook results exactly — the `src/fahm` package is the
single source of truth; notebooks explore, the runner orchestrates.

| Stage | In → Out | What it does |
|---|---|---|
| preprocess | raw CSV → `sensor_readings.parquet` | typed load, six integrity checks, failure windows |
| features | processed → `features.parquet` | gap-aware 1h grid, 6 feature families, scaling |
| score | features → `scores.parquet` | zmax score, alert flag, **driving feature** per window |
| evaluate | scores → `evaluation.csv` | per-failure ROC/PR-AUC, lead time, false-alarm rate |

---

## Method, stage by stage

**1 — Preprocessing.** Typed load (224 → 104 MB), six integrity checks that
*raise* on bad data, a gap inventory (331 gaps, 18% missing time), failure
windows sourced and verified against the dataset documentation.

**2 — EDA.** Distribution and correlation analysis with a state-mixture lens;
effect sizes rather than p-values (1.5M autocorrelated samples make significance
meaningless); corrected the documented motor-current modes against the data.

**3 — Anomaly context.** Per-failure portraits; a six-category labeling scheme
(healthy / prefail / infail / postrepair / **degraded** / **invalid**) where the
last two are *findings* — a 12-day undocumented degraded span and a 20-hour
frozen-instrument fault — carved out by investigation so the healthy baseline is
trustworthy. Base-rate controls throughout to reject spurious "near-failure"
patterns.

**4 — Feature engineering.** A gap-aware 1-hour window grid (4,060 windows), six
validated feature families (duty/state-occupancy, pressure dynamics, thermal,
variability, cycle-frequency, instrument-health). Highlights: a per-idle-stretch
pressure-decay slope (the direct leak meter), an oil residual against a healthy
oil-vs-workload baseline (separating "hot because busy" from "hot because
failing"), and multi-scale cycle features. Informative missingness handled
explicitly (indicator + fill), features scaled against healthy operation so
anomalies read as large deviations.

**5 — Modeling.** Three paradigms compared under one honest protocol:
anomaly detection (zmax wins), supervised leave-one-failure-out (reveals a
*failure taxonomy* — F3 is learnable from other failures, F4 is too unique),
and forecast-residual (confirms the failures are level, not trajectory,
anomalies). Full scorecard: ROC-AUC, PR-AUC, precision/recall, lead time,
false-alarm rate. Threshold presented as a tunable operating-point menu.

---

## What drives the detection (verified predictive, not descriptive)

Feature importance, computed on the **lead-up only** (excluding during-failure
windows to avoid describing the failure rather than predicting it), identifies
two physically-grounded signals:

- **Pressure decay slope** — the air leak measured *directly* (reservoir
  pressure falling during idle).
- **Oil temperature** — the leak measured *indirectly* (the compressor overworks
  to compensate → heat).

Both detect the same root cause through different sensors — a coherent physical
story, cross-confirmed by the independent zmax and XGBoost importance methods.

---

## Deploying this

The `score` stage emits a per-window health score, an alert flag, and the
**driving feature** — so an alert is actionable, not a black box:

> ⚠️ 2020-07-08 18:00 — health score 23.9 (threshold 7.0), driver:
> `tp3_decay_slope` → suspected air leak, ~8 h to likely failure.

Interpretable features (from the physics-first analysis) are what make the alert
a work order rather than an opaque number.

---

## Repository

```
src/fahm/            preprocessing · analysis · plotting · features · modeling
notebooks/           01–05, the exploratory narrative per stage
run_pipeline.py      config-driven end-to-end runner (headless)
configs/             config.yaml (paths, thresholds, verified label spans)
decision_log.md      the full audit trail: every decision, lesson, dead-end
```

**`decision_log.md` is worth reading** — it documents the reasoning behind every
choice (D##), the lessons and corrections (L##), and the open questions (OQ##),
including the data-leakage patterns explicitly avoided and a sensor-polarity
error caught by physical reasoning.

---

## Honest limitations

- **F2 is undetectable** in these sensors — no precursor at any usable threshold.
- **Slow degradation is weak** (0.58) — better read as a trend than a binary alarm.
- **Low recall** at the conservative threshold — the system favors trustworthy
  early warning over exhaustive flagging (one early true positive suffices to warn).

Reporting these plainly is the point: a documented capability envelope is more
useful to an engineer than a suspiciously perfect score.

---

## Stack

Python 3.11 · pandas · scikit-learn · XGBoost · scipy · Poetry ·
MetroPT-3 (UCI / Davari et al., DSAA 2021)
