# Decision Log — Field Asset Health Monitor (A1)

Format per workbook: **Assumption / Alternatives / Choice / Reason / Consequence.**
Entries in build order.

---

## D01 — Problem framing: anomaly detection, not supervised classification

- **Assumption:** The 4 documented failures are too few to learn from directly.
- **Alternatives:** (a) supervised failure classification, (b) RUL regression,
  (c) unsupervised anomaly detection evaluated as early warning.
- **Choice:** (c). Train only on healthy data; use the 4 documented failure
  windows exclusively for evaluation (early-warning time, false alarms).
- **Reason:** With n_failures ≈ 4, any supervised model would memorize, not
  generalize. Anomaly detection matches the operational question: "is the APU
  behaving abnormally?"
- **Consequence:** No per-fault-type predictions in v1. Fault-type attribution
  is deferred to the copilot layer (A3) using evidence, and to A2 simulation.

## D02 — Engineer metric, not ML metric

- **Choice:** Primary metrics are (1) early-warning hours before each documented
  failure and (2) false-alarm episodes per healthy week. AUC/F1 are secondary.
- **Reason:** A field engineer acts on alerts; alert quality is what matters.
- **Consequence:** Evaluation code counts alert *episodes*, not per-minute hits.

## D03 — Ingestion: chunked CSV -> parquet, compact dtypes, no global sort

- **Alternatives:** load whole CSV; keep CSV as working format; sort globally.
- **Choice:** Stream in 1M-row chunks; float32 (analog) / int8 (digital);
  write once to snappy parquet; verify monotonic timestamps instead of sorting.
- **Reason:** Constant memory, ~10x smaller working file, one canonical source.
- **Consequence:** Profiler must check monotonicity (it did: 0 violations).

## D04 — Column-name normalization via alias map

- **Observed:** Kaggle release spells the flowmeter column `Caudal_impulses`;
  the paper uses `Caudal_impulsion`. Ingestion initially crashed on this.
- **Choice:** `COLUMN_ALIASES` map in ingestion normalizes known variants to one
  canonical schema and raises a readable error listing the real header otherwise.
- **Consequence:** Downstream code sees exactly one schema, matching the paper.

## D05 — Failure windows: evaluation-only ground truth, pending verification

- **Choice:** `failure_windows.csv` built from the commonly cited windows in the
  MetroPT paper (Veloso et al., Scientific Data 2022): F1 air leak Apr 18,
  F2 air leak May 29–30, F3 oil leak Jun 5–7, F4 air leak Jul 15.
- **Open item:** Exact timestamps NOT yet verified against the paper. Every
  early-warning number is provisional until this is done.
- **Consequence:** Windows are never used as training labels (enforced in code).

## D06 — Corrected assumptions after profiling (supersedes early planning)

- **Observed:** Sampling is **10 s**, not 1 Hz; 1,516,948 rows ≈ Feb 1 – Sep 1
  2020 (~175 days). 331 recording gaps totalling ~38 days. All 4 failure
  windows fall inside covered periods.
- **Consequence:** Feature windows sized to a 10 s grid; dataset is light
  (~250k rows after 1-min resample), so iteration is cheap.

## D07 — Gap handling: segment, never bridge

- **Alternatives:** interpolate across gaps; ignore gaps; segment.
- **Choice:** `segment_id` increments at every gap > 600 s. All rolling and
  cycle computations group by segment.
- **Reason:** A 2-day hole would poison every window spanning it. Gaps are
  structural (unit off / not recorded), not missing-at-random.
- **Consequence:** Warm-up rows at each segment start are dropped by
  `min_periods`; short segments contribute few/no rolling features.

## D08 — 1-minute base grid

- **Alternatives:** keep 10 s; resample to 10 min.
- **Choice:** 1-min aggregates (mean/std/min/max analog, fraction-active
  digital, n_samples).
- **Reason:** Leak dynamics unfold over minutes-to-days; 6:1 reduction keeps
  cycle shape visible while making the table trivially fast to iterate on.
- **Consequence:** Sub-minute transients are invisible by design.

## D09 — Cycle features as the physics layer

- **Choice:** Run-length encode DV_eletric into load/idle runs; pair each load
  run with the following idle run; compute duty, load/idle duration, TP3 build
  rate (load) and TP3 decay rate (idle). Runs < 30 s discarded as chatter.
- **Reason:** TP3 idle decay is the most direct air-leak measurement available:
  literally how fast pressure is lost when the compressor is off.
- **Validation:** On synthetic leak data, duty 0.17→0.44 and decay −0.12→−0.36.

## D10 — Error-analysis finding: continuous-load blind spot (F1)

- **Observed:** During F1 the motor is pinned at ~5.7 A for 24 h — the
  compressor never unloads, so NO load→idle cycles complete, and cycle
  features go silent exactly when failure is most severe.
- **Fix:** `pinned_load_2h` = rolling 2 h **minimum** of per-minute
  DV_eletric_frac (≈1.0 ⇒ never unloaded), plus rolling duty means from the
  1-min grid, which need no completed cycles.
- **Consequence:** The model feature set covers both leak regimes: gradual
  (duty/decay drift) and catastrophic (continuous load).

## D11 — Undocumented ~March 10 event

- **Observed:** Duty ≈ 1.0, one ~42,000 s continuous load run, decay −0.32
  around Mar 8–12, with no documented failure.
- **Choice:** Exclude the period from training. Classification (undocumented
  failure vs maintenance/test) deferred to EDA deep-dive.
- **Consequence:** Model alerts there are counted honestly as "false alarms"
  in v1 metrics but flagged in error analysis as possibly true positives.

## D12 — Operating-regime shift at ~Mar 1

- **Observed:** Baseline duty ~0.07 in February vs ~0.12 after the Feb 28–Mar 1
  gap. The usage pattern changed.
- **Consequence:** "Normal" must cover both regimes → see D13.

## D13 — Training periods for the anomaly model

- **Alternatives:** Feb only; all non-failure data; curated healthy periods.
- **Choice:** Feb 1–28 **plus** Apr 5–14 (clean post-regime-shift slice).
  Hard safety check in code: training can never include a failure window.
- **Tradeoff (accepted):** Apr 5–14 ends 4 days before F1; if degradation
  started earlier than documented, training is slightly contaminated, which
  would make the model *less* sensitive (conservative direction).

## D14 — Threshold + persistence alerting

- **Choice:** Threshold = 99.5th percentile of TRAINING scores. Alert fires
  only when ≥30 of the last 60 minutes exceed threshold (within a segment).
- **Reason:** Single-minute score spikes should not page an engineer; leaks
  persist, noise doesn't.
- **Consequence:** Detection latency ≥ ~30 min by construction — acceptable
  for leak-class faults, and tunable in config.

## D15 — Documented sensor-behavior corrections (vs paper)

- **Observed:** Motor current levels in this release: ~6 A start spike,
  ~3.8 A sustained load, ~0 A idle (paper says 7 A / 4 A). Oil_level is
  "active" 90% of the time, which contradicts "active = oil below expected" —
  digital polarity suspect and must be verified empirically around F3.
- **Consequence:** Never trust documented polarity of digital signals without
  an empirical check; sensor table in data_profile.md to be updated with
  observed values.
