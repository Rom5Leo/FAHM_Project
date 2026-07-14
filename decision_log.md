# Decision Log — FAHM (self-built pipeline)

## D00 — Architecture: notebook / src / config separation
- Choice: notebooks hold no logic (calls + displays only); all transforms
  live in src/fahm; all paths and parameters live in configs/config.yaml.
- Reason: reproducibility, testability, and one place to change experiments.
- Consequence: every notebook cell >~5 lines is a signal to move code to src.

## D01 — Path resolution anchored at the project root
- Problem: notebooks run with CWD = notebooks/, so root-relative paths in
  the config failed (FileNotFoundError).
- Alternatives: (a) ../-style paths relative to the notebook — rejected,
  breaks for any script not launched from notebooks/; (b) always launch
  from repo root — fragile, relies on discipline.
- Choice: load_config() resolves every cfg["paths"] entry to absolute,
  using the config file's own location (configs/ = root + 1) as the anchor.
- Consequence: load_config is the single place the config is opened; no
  other code ever handles relative paths.

## D02 — Naive load first, typed load second
- Choice: step-1 load_raw() is a plain pd.read_csv with no dtypes/parsing.
- Reason: establishes a measured baseline (memory, dtypes) so step 2's
  typing decisions are improvements against a number, not guesses.
- Consequence: known ugliness left visible on purpose: Unnamed: 0 index
  column, timestamp as object/string, float64 everywhere.

## D03 — Package named `fahm`
- Choice: src/fahm (short, meaningful) over src/modules (meaningless) or
  src/fahm_project (long). Import name = package name convention.

## D04 — Dtypes: analog float64, digital int8
- Observed: digitals arrive as 0.0/1.0 floats; verified ZERO missing values
  in all columns (df.isna().sum() == 0).
- Alternatives: float32 for analog (halves analog memory; sensors report 3
  decimals, well within float32 precision) — viable, not chosen.
- Choice: analog stays float64 (precision headroom for later arithmetic;
  memory is not a constraint at 1.5M rows); digital cast to int8 — safe
  ONLY because the no-missing-values check passed (plain int8 cannot hold NaN).
- Result: memory 224.2 MB -> 104.2 MB (2.15x).
- Consequence: if a future data refresh introduces NaNs in digital columns,
  the int8 cast will fail loudly — acceptable, that failure is informative.

## D05 — Timestamp parsing + canonical schema
- Choice: parse with explicit format="%Y-%m-%d %H:%M:%S" (fast path, and
  fails loudly on format surprises instead of silently guessing);
  drop "Unnamed: 0" AFTER extracting its insight (see Lessons); enforce
  canonical column order [timestamp, analog..., digital...] in load_raw,
  the single place schema is defined (ANALOG/DIGITAL module constants).
- Open micro-decision: column name kept as Kaggle's "Caudal_impulses"
  (paper spells it "Caudal_impulsion") — zero-rename convenience over
  literature alignment; revisit if it causes confusion in docs.

## D06 — EDA findings that bind later steps (step 3)
- **Digitals proven pure:** n_other == 0 for all 8 signals on the full
  1,516,948 rows -> closes the assumption D04's int8 cast depended on.
- **TP3 ≈ Reservoirs** (same connected air volume, values identical to 3
  decimals) -> step-4 check: mean(|TP3 − Reservoirs|) < ε.
- **COMP + DV_eletric ≈ 1** (observed 0.998) -> step-4 check: antiphase
  valve signals.
- **Motor_current mode mapping (corrected):** ~0 A = off; ~3.9 A =
  OFFLOADED running (motor on, intake closed, no compression);
  ~5.5–6.2 A = under load. Initial reading (3.9 A = load, 6 A = start
  transient) was WRONG — refuted by cross-referencing TP2/COMP in time.
  Consequence: any "loaded state" definition downstream keys on
  COMP/DV_eletric, with Motor_current thresholds as corroboration only.
- **TP2 idle zero-offset (~−0.012 bar):** calibration behavior, not error
  -> analog range checks must allow small negative pressures.
- **DV_pressure 9.8 bar sample:** real maintenance/test episode on
  2020-04-06 14:18 (rapid loaded cycling after a 1h50m gap -> discharge
  spike -> offloaded standby). Behavior, not error. 12 days before F1;
  presumed unrelated.

## D07 — Fault types corrected against the primary source
- Observed: the source failure table lists ALL FOUR failures as Air Leak.
  The earlier "June oil leak" label came from unverified secondary memory.
- Choice: failure_windows transcribed from the source table into
  configs/config.yaml, including the maintenance dates from the Report
  column. Source numbering reads #1,#1,#3,#4 — second row assumed a typo
  for #2.
- Consequence: OQ1's test redesigned (no oil-leak window exists to test
  against). Every future evaluation number traces to this verified table.

## D08 — Check thresholds derived from measurements (step 3)
- tp3_reservoirs_eps = 0.01: 5x observed mean |TP3-Reservoirs| (0.0019);
  tp3_reservoirs_max = 0.5: observed max 0.182.
- valve_antiphase_eps = 0.02: observed |mean(COMP+DV_eletric)-1| = 0.0024,
  ~8x headroom. Antiphase holds in 98.9% of samples; 16,762 violations
  match the expected count of load/unload transitions caught mid-switch
  by 10s snapshots — boundary sampling, not a third state.
- analog_ranges: observed 7-month min/max per sensor + margin with inline
  reason. These are "sensor broken" tripwires, NOT anomaly detection.
- gap_threshold_seconds = 60: jitter tops out ~22s, real holes are
  minutes+; 60 sits between. Revisit after step 4's gap inventory.
- REVISITED (per plan): inventory = 331 gaps, smallest 1.73 min (clearly a
  recording stop, far above the ≤22s jitter zone — no borderline cases),
  total 54,571 min ≈ 37.9 days ≈ 18% of the span unrecorded. Threshold 60s
  KEPT. Coverage fact -> Summary cell + all downstream time-based reasoning.

## D09 — Check failure policy: pure function + caller-chosen posture
- Choice: run_checks computes the full results table always; on_fail
  parameter decides posture — "warn" (return table, default; notebook era)
  or "raise" (halt; for any future unattended script). All checks run
  before raising, so one run reveals all failures.
- Reason: checks validate DATA TRUST, not failure risk (that's the model's
  job). Exploration needs the failing data alive; automation needs a gate.
- Deployment note: in production, check failures would feed an operations
  alert ("monitoring is blind") — escalation policy belongs to the operator.
- Deferred: per-check severity (corruption = always fatal vs drift = warn)
  via a severity column — add when a real case demands it, not before.
- Raise-gate sabotage-verified (TP2 ceiling→5 in deep-copied config): 
  warn mode reported the red row, raise mode halted with formatted failure list;
  violation count 230,162 ≈ the 15-16% load duty — even the sabotage was physically consistent. Real cfg unaffected.

## D10 — One notebook per pipeline stage
- Choice: stage notebooks — 01 preprocessing, 02 EDA, then (as stages
  begin) 03 anomaly context, 04 features, 05 model. Program convention.
- Contract: each notebook consumes the SAVED ARTIFACT of the previous
  (e.g. 02_eda loads the processed parquet), never re-runs its work.
- Origin: the original notebook had mixed preprocessing+EDA; split
  accordingly, EDA setup carries a TODO to switch from load_raw to the
  processed file once save_processed exists.

## D11 — Processed artifact: parquet, full path in config
- Choice: save the typed, validated table as parquet at
  data/processed/sensor_readings.parquet; the FULL path (with filename)
  lives in config, not in code.
- Why parquet: preserves dtypes (CSV would turn everything back into
  strings and force re-parsing on every load); compressed and columnar
  (fast, can read selected columns); safe and portable (pickle is
  Python-only and unsafe to load from untrusted sources).
- Why path in config: the artifact's location is a parameter other
  notebooks depend on (02_eda loads it) — parameters live in config (D00).
- Guard: save_processed refuses a directory-only path with an
  instructive error (verified live: it caught the stale config value).

## D12 — Skew→log→correlation recipe rejected for analog sensors
- Context: the DS-program EDA recipe (skewness check -> log1p transform of
  |skew|>1 columns -> Pearson vs Spearman comparison) was considered for
  the analog sensors.
- Observed: high "skewness" values here describe MODE STRUCTURE, not tails
  — TP2/Motor_current/DV_pressure are mixtures of machine states
  (idle/offloaded/load), each mode itself narrow. The recipe's assumption
  (unimodal continuous feature with a skewed tail) does not hold; log1p is
  also inapplicable mechanically (TP2/H1/DV_pressure go slightly negative
  from zero-offset, D06).
- Choice: compute BOTH Pearson and Spearman heatmaps (comparison habit
  kept), but skip the transform pipeline; interpret correlation clusters
  as shared machine state, not feature redundancy. Feature-selection-style
  conclusions are deferred to ENGINEERED features (decay slopes, duty),
  where correlations mean what the recipe assumes.
- Note: same-timestamp correlation is blind to LAGGED coupling (e.g.
  Oil_temperature follows workload with delay) — a known limitation, not
  an absence of relationship.

## D13 - Group comparison (the t-test recipe)
 - In the classification project, t-tests compared features across target classes. Here groups exist but must be constructed: rows labeled healthy / pre-failure / in-failure from the failure windows. 
 - Two recipe adjustments (L04 pattern): sensors are state mixtures → rank-based comparison (Mann-Whitney) over t-test; and 10-second samples are heavily autocorrelated → with n≈1.5M dependent samples every p-value is vanishingly small, so p-values are meaningless here. We therefore compare groups by effect size and distribution overlap — the quantity that actually predicts early-warning detectability — and defer the comparison itself to stage 3, where the labels are built. 
 - The model's early-warning evaluation is the real significance test.

---

# Open Questions

### OQ1 (redesigned) — Digital polarity vs docs
- Oil_level 0.904 active / Pressure_switch 0.991 active contradict their
  documented meanings. Test: when does the INACTIVE time occur — clustered
  (failures/maintenance/gaps) or scattered? Run after step 5 context exists.

### OQ2 — Post-gap behavior
- The one investigated anomaly began 9 min after a 1h50m gap ended. Are
  anomalies clustered near gaps? If yes, post-gap minutes may need a
  warm-up/exclusion rule in feature building.

---

# Lessons

### L01 — Environment & notebook mechanics
- IPython magics take no trailing comments.
- autoreload refreshes code but not variables already in memory (re-run
  cells that CREATED data after editing functions).
- "Python 3.11" in VSC's picker can be several different environments.

### L02 — What the raw data revealed (step 2)
- The "Unnamed: 0" column stepped 0,10,20,... — it was the ORIGINAL 1Hz row
  index, proving Kaggle downsampled by DECIMATION (every 10th sample), not
  averaging. Values are instantaneous snapshots: slightly noisier, and
  sub-20s events are invisible (Nyquist). Consequence for the project:
  spectral/vibration features are excluded by data availability; leak and
  temperature physics live at minutes-to-days, so early-warning capability
  is unaffected. Timestamps also show jitter (10s modal, some 9s/11s steps)
  -> consistency checks must test the MODAL interval, not assert exact 10s.
- Read a "junk" column before dropping it; this one held the only evidence
  of how the dataset was made.

### L03 — EDA method (step 3)
- **A tool classifying my own processed data is not independent validation**
  — AutoViz "confirming" 7 numeric + 8 boolean columns was reading back my
  D04 dtypes. Real validation = the full-data n_other check.
- **Auto-EDA tools are a 10-minute skim, not an artifact pipeline** —
  AutoViz on pandas 3 needed a shim, is time-blind and sampled, and had
  flaky export. Timebox convenience tools; cut losses.
- **Auto-DQ advice assumes i.i.d. tabular data** — "5130 duplicate rows"
  after dropping timestamp, and "cap the outliers" on a two-state machine,
  were both nonsense here.
- **Distributions hide time** — every real insight of step 3 (mode
  mapping, the maintenance episode, gap adjacency) required timeline
  plots. Histograms raise questions; timelines answer them.
- **One-sensor conclusions are provisional** — the DV_pressure story only
  resolved, and the Motor_current mode labels only got corrected, when
  TP2/COMP/Motor_current were read together in time.
- **A truncated plot window is data speaking** — the missing left edge of
  the ±2h window revealed the recording gap adjacent to the anomaly.
- **Log-scale histograms are mandatory for two-state machines** — the
  linear grid hid every small mode and every tail; log=True exposed all
  of the Q4 findings.
- **Read a "junk" signal before dismissing it** (pattern repeating from
  step 2's Unnamed: 0): the single weird DV_pressure sample led to a
  corrected mode mapping and the post-gap question.
- **Verify labels against the primary source before building tests on
  them** — one unverified word ("oil") nearly aimed an entire
  investigation at the wrong target.

### L04 — Recipes carry assumptions (step "02 EDA")
- The skew->log->Pearson pipeline assumes unimodal continuous features.
  On a state machine's sensors, "skewness" is bimodality in disguise and
  the prescribed cure (log transform) moves the modes without fixing
  anything — and can't even run (negative values).
- Habit: before running a taught recipe, check its assumptions against
  the data's NATURE (states vs distributions, time series vs i.i.d.).
  This is the second instance of the pattern — auto-DQ advice (L03)
  failed here for the same underlying reason: i.i.d. tabular assumptions.