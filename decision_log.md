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

## D14 — Failure-context plot: view clamped to data window
- Problem: axvline for F2's maintenance marker (2020-04-30, the flagged
  source oddity) stretched the x-axis a month wide, crushing the data.
- Choice: set_xlim(lo, hi) always; maintenance marker drawn only if inside
  the view, otherwise noted in the title text ("maintenance off-window").
- Principle: a marker must never dictate the zoom; source dates stay
  as transcribed (D07 rule: transcribe faithfully, annotate skeptically).

## D15 — Episode boundary finder: design before code
- Need: locate start/end of binary conditions in time (first case: the
  Apr 20 instrument fault) without eyeballing printouts.
- Design choices:
  (a) DETECTOR/FINDER SPLIT — find_state_episodes(df, predicate, ...) hunts
      transitions of ANY boolean window-predicate; stuck_instrument() is
      just one predicate. One search algorithm, many detectors.
  (b) SCAN, NOT BISECT — bisection assumes exactly one transition in the
      bracket; episodes can flicker. Coarse scan at Δ finds ALL flips;
      each flip's bracket is re-scanned at finer Δ. Flicker-proof, and
      naturally returns multiple (start, end) episodes.
  (c) GAP ≠ FAULT — windows with no data return False from the predicate;
      recording gaps are OQ2's domain, not instrument faults. Without this
      guard the search range crossing a gap would fabricate episodes.
- Consumers: section 6 labeling (invalid-sample masking), stage-4
  instrument-health features, any future monitoring use.

## D16 — OQ1 resolved: both suspect digitals are inverted event-detectors
- Pressure_switch: off-time is entirely sub-minute flickers (11,814 runs,
  median 0.17 min, max 0.7 min, none > 1h) -> brief-event detector with
  INVERTED polarity vs docs (active = normal; momentary drop = the event).
- Oil_level: same flicker behavior (~7,660 runs) PLUS 8 long episodes.
- Sensor-meaning table corrected accordingly; the 0.90/0.99 "active"
  fractions from step 3 are explained and not anomalous.

## D17 — Row labeling for group comparison (label_windows, in analysis.py)
- Six priority-ordered categories: healthy / prefail / infail / postrepair /
  degraded / invalid (later overrides earlier). degraded & invalid are passed
  BY NAME (stage-3 findings: OQ3's Apr 18-30, the Apr 20 instrument fault) —
  not derivable from fw. This is what makes 'healthy' trustworthy.
- Lives in analysis.py (it measures/derives, not prepares).
- Distribution sanity-checked: healthy 88.4%, every category's size
  physically explicable; priority override verified (Apr 20 → invalid, not
  degraded).

## D18 — Physical units on every axis (professionalism pass)
- UNITS map + axis_label() live in preprocessing.py beside the schema
  constants (ANALOG/DIGITAL) — one source of truth for sensor units:
  bar for the five pressures, °C for oil temperature, A for motor current,
  dimensionless (marked [-]) for the eight digital signals.
- All seven plotting functions retrofitted: value axes carry their unit,
  time axes labeled "time", and count vs probability-density vs log-count
  are named honestly rather than left blank.
- Tables: units appended to markdown headers where not obvious
  (e.g. "median [°C]").
- Trigger: unlabeled axes caught during the 6.4 review. For a physics-facing
  portfolio, missing units and labels read as carelessness — the fix is a
  one-map, one-helper change so every current and future plot is consistent.

## D19 — Degraded spans: evaluation ground truth, not training data
- Tension: the Apr 18-30 degraded span IS the leak — the most valuable
  signal we have — so hiding it wastes it; but it must not enter the healthy
  TRAINING set (would teach the model that leaking is normal), and it can't
  be a supervised target (n=1 degraded episode -> memorizes April, not
  "leaking").
- Choice: train the detector on clean HEALTHY only; use degraded (and, when
  found, other degraded-like spans) as a SECOND evaluation target alongside
  the failure windows. "Does a healthy-trained detector light up during known
  degradation?" is a stronger result than failure-window detection alone.
- The `degraded` label stays; its ROLE changes from excluded to evaluated.

## D20 — Feature grid design
- Window = 1h; result: 4,060 windows across 332 segments (= 331 gaps + 1).
- Gap-aware by construction (segment_id = (diff>thr).cumsum()); no window
  bridges a gap.
- Windows are 1h FROM EACH SEGMENT'S START, not clock-aligned — a consequence
  of the gap-aware design; boundaries don't match across segments.
- Partial trailing window per segment dropped (edges[:-1]/[1:]).
- Thin-window rule: none needed. All 4,060 windows are densely populated
  (min 297, median 363 samples; 0 windows < 60 samples) — the gap-aware
  segmentation excludes sparse regions before windowing, so no window is
  under-sampled. n_samples kept as a column for downstream trust weighting.
- Look-back scales for rolling features: TBD when those families are built.
- Window labels: majority row-label, invalid-wins-any-overlap (trust rule).
  Distribution: healthy 3,586 / degraded 190 / prefail 160 / infail 56 /
  postrepair 48 / invalid 20; zero empty. invalid = ~20 windows = the ~20.5h
  Apr 20 fault at 1h resolution; only ~2 boundary windows are mixed-and-dropped.

## D21 — Spectral features: cycle-rhythm, not vibration
- Fourier/FFT for VIBRATION or acoustic fault signatures is impossible on this
  data: 10s decimation caps Nyquist at ~1/20 Hz, so mechanical signatures
  (bearings, valve slap; Hz-kHz) are gone (L02). This is a DATA limitation,
  not a method choice — with kHz vibration data it would be central.
- But the load/unload CYCLE has a minutes-scale period, well within 10s
  resolution, and a leak makes the machine cycle faster/erratically. So the
  dominant cycling frequency, its drift, and spectral entropy are legitimate
  physically-motivated features (cycle_frequency_features).
- Method: Lomb-Scargle periodogram (handles irregular sampling — jitter+gaps
  break a plain FFT), cross-checked with duty autocorrelation. F3's long shot
  (OQ5): it moved no central-tendency stat, but might show as a rhythm change.

### D22 — Calendar/seasonality family (with a detrend purpose)
- A metro APU's load has daily (rush hour) and weekly rhythm — operation, not
  degradation. Calendar features serve two roles: cyclic hour/day-of-week
  encodings, and (the important one) a per-hour healthy baseline to express
  duty as a residual, separating real leaks from normal peak load.
- Gate: verify duty actually shows hour/day seasonality before adding the
  family.
- Decision: Calendar Features Dropped.

## D23 — Detection framing, with forecasting as a residual signal (not pure forecasting)
- Problem shape dictates paradigm: 4 failures in 7 months + partial labels +
  "investigate this machine" framing → anomaly/DEGRADATION DETECTION (learn
  healthy, measure distance), NOT trajectory forecasting (can't learn failure
  dynamics from n=4).
- BUT forecasting enters as a tool: a healthy-trained forecaster whose
  RESIDUAL (predicted vs actual) becomes an anomaly score — flags "the machine
  stopped being predictable" before any value crosses a threshold. A candidate
  second model family alongside the feature-based detector; possibly F3's only
  chance (OQ5), since a level-sudden failure may be preceded by rising
  unpredictability.
- Plan: build both, compare per-failure on detection lead-time.

## D24 — Feature validation via window-level per-failure effect sizes
- feature_effect_by_failure (analysis.py): the window-level sibling of
  prefail_effect_by_failure (which works on raw rows). Compares each failure's
  prefail-WINDOW feature values to healthy-window values, robust effect
  (median shift / IQR). Run after every feature family — a family that doesn't
  separate any failure doesn't earn its columns.
- Confirmed the approach adds signal: windowed duty gave F4 +3.46 / F1 −0.64
  vs raw-sensor +0.136 / −0.078 — aggregation concentrates what per-sample
  metrics dilute.

## D25 — COMP polarity is INVERTED (like the OQ1 digitals)
- Measured: mean Motor_current when COMP==0 is 5.6 A (loaded) -> COMP==0 means
  RUNNING, COMP==1 means IDLE. The docs' implicit "1=on" is wrong here, matching
  the Pressure_switch/Oil_level inversions (OQ1/D16).
- Fix: idle mask = COMP==1 everywhere. Corrects pressure_dynamics_features
  (was fitting decay over LOADED samples -> spurious positive slopes).
- Audit: duty uses DV_eletric (unaffected); antiphase_share is symmetric
  (unaffected). D06's "loaded keys on COMP" amended: loaded = COMP==0.

## D26 — Thermal residual via healthy oil-vs-duty baseline
- Problem: raw oil_median conflates "hot because working hard" with "hot
  because faulty." The first attempt (oil_per_duty ratio) failed — a ratio
  responds to numerator AND denominator (F4's ratio DROPPED because duty rose
  faster than oil) and blew up at small duty.
- Choice: fit oil ~ f(duty) on HEALTHY windows only (degree-2, the scatter
  shows a clear rising-then-plateau curve); oil_residual = observed oil minus
  what the healthy curve predicts at that window's duty. Positive = hotter than
  the workload explains = leak signature, independent of workload level.
- Result: F1 −1.23, F2 +0.64 (genuine thermal anomalies); F4 ≈0 (its heat is
  fully workload-explained — correct null). Residual is the right tool for
  "more/less than context predicts"; same pattern as forecast-residual
  detection (D23).

## D27 — Multi-scale look-back (reserved for cycle features)
- Grid stays at 1h (max rows 4,060, clean gap handling). Cycle/variability
  features suffer NaN when a 1h window has too few cycles. Planned refinement:
  compute cycle features over a longer look-back (e.g. 6h preceding each 1h
  window) to raise coverage without losing rows. Not yet implemented; noted as
  the principled fix for cycle-feature sparsity.
- (executed) — Multi-scale cycle features resolve the sparsity tradeoff: 
  6h look-back for cycle features: NaN 78%→6%. Removes the imputation-noise
  drag §7.2 measured (F1/F2/F4/degraded recover) while keeping F3 (0.689 vs
  0.676 if dropped). Net improvement across targets. Trend feature needs
  winsorizing (heavy tails from sparse-cycle polyfits).

## D28 — Spectral (cycle_frequency) family: analyzed, excluded from model set
- Lomb-Scargle dominant_freq + spectral_entropy corroborate F4 (faster, more
  regular rhythm) but are SILENT on F3 (−0.17/−0.10) — the failure they were
  meant to help. Everything they catch is already caught more cheaply. Kept in
  the notebook as analysis (demonstrates the spectral angle was tested); removed
  from FAMILIES so the model set carries no astropy-dependent dead weight.
- Calendar (D22) likewise excluded (no seasonality). Removing a non-earning
  feature is a deliberate decision, logged like any other.

## D29 — NaN handling: informative-missingness, not imputation
- Cycle/variability features are NaN in idle/locked windows — MEANINGFUL
  absence (the regime itself), not random missingness. Naive fill would lie.
- Strategy: for each NaN-prone feature, add a boolean *_missing indicator, then
  fill the value neutrally (median). Model sees both value and measurability;
  the missingness (idle/locked) becomes usable signal (it carried F3's regime).
- Rescaling: standardize continuous features on HEALTHY windows only (fit
  scaler on healthy, apply to all) — so "normal" defines the scale and
  anomalies read as large deviations. Booleans/flags left unscaled.

## D30 — Anomaly-detection results: zmax baseline wins
- Three detectors (zmax = max|z| across healthy-scaled features; IsolationForest;
  Mahalanobis), fit on early-healthy, evaluated per-failure vs val-healthy on a
  TIME-ORDERED split (no shuffling — autocorrelated windows would leak).
- zmax wins every target: F1 AUC 0.864 (32.5h lead), F4 0.795 (8h), F3 0.702
  (9.6h), F2 0.635 (0h), degraded 0.575. IForest/Mahalanobis lower everywhere;
  both score the degraded span BELOW chance (0.25/0.30).
- Why the trivial baseline wins: stage-4 features are strong and healthy-scaled,
  so failures present as one/few extreme features — exactly what max|z| catches.
  The multivariate methods hunt subtle joint patterns that aren't the signal
  here. Verified zmax is genuinely multivariate: the argmax feature varies by
  failure (F4 trips tp3_decay_slope, others differ), not one dominant feature.
- Multivariate methods lose because failures present as few-extreme-features,
  which max|z| catches directly, not as subtle whole-covariance shifts.
- Full metric scorecard adopted (ROC-AUC, PR-AUC, precision/recall at 1% FA,
  lead time) — accuracy omitted as meaningless at this imbalance.

## D31 — Two detectors are complementary: anomaly (unique failures) + supervised (shared-pattern failures)
- Leave-one-failure-out (train on 3 failures, test on held-out 4th): F3
  learnable from others (0.80), F4 NOT (0.48-0.69, near chance), F1/F2 ~chance.
- Inverts anomaly-detection difficulty: F4 is easiest-as-anomaly (unique ->
  deviates most from healthy) but hardest-to-learn-from-others (unique -> unseen
  pattern); F3 is hardest-as-anomaly but easiest-to-transfer (shared pattern).
- No supervised model beats zmax as a detector. Conclusion: anomaly detection
  catches unique failures, supervised catches shared-pattern ones — a mature
  system uses both.
- Consequence: a deployed system benefits from both — anomaly scoring for novel
  failures, supervised for recurring known-pattern failures.

## D32 — Importance must exclude infail windows (predictive vs descriptive signal)
- prefail+infail target inflated oil importance via the trivially-extreme
  during-failure state. prefail-ONLY importance: oil_std/oil_median stay top but
  reduced, tp3_decay_slope RISES to #3 — the true precursor surfaces once
  descriptive leakage is removed.
- Physical reading: both top features detect the SAME air leak — decay slope
  directly (idle air-pressure fall), oil indirectly (compressor overwork → heat).
  Consistent with F4's oil being workload-driven (oil_residual ~0, stage 3).
- oil_residual ranks low: trees learn the duty-correction from raw features, so
  the pre-computed residual is redundant to XGBoost (valuable for interpretation,
  not for tree models).

## D33 — Forecast-residual converges to zmax (failures are level not trajectory)
- Two variants tested: lag-forecaster (predicts window from its own recent past)
  adapts to the degradation — a smooth ramp is predictable from recent values,
  so residual stays small; near-random. Healthy-anchored variant (predict signal
  from other features via healthy-trained GBM) reduces to zmax on 2 signals —
  matches zmax only on F4 (where those signals carry the failure), loses
  elsewhere (zmax watches all 23 features).
- Both confirm failures are LEVEL-departures (which zmax measures over all
  features), not trajectory anomalies. Healthy has no temporal structure (D22)
  for forecasting to exploit. LSTM/GBM not pursued — a better forecaster
  predicts smooth ramps better, shrinking the residual, worsening detection.
  
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

### OQ3 — Are F1 and F2 one failure? (reading the source's #1,#1,#3,#4)
- Hypothesis: the duplicate "#1" is deliberate — F1.a (Apr 18) and
  F1.b (May 29) are episodes of ONE failure; row 2's "maintenance 30 Apr"
  is then the response to episode a, and the recurrence means the fix was
  incomplete. Explains both source quirks with zero typos.
- Counter-point: the sequence then has no failure #2 (undocumented event?
  the Apr 6 episode?), and episode b's resolution is unlogged.
- Tests (all in 03):
  (t1) Does Apr 30 ~12:00 have a maintenance footprint in the data
       (gap / depressurization / test-cycling like Apr 6)?
  (t2) Is Apr 18-30 measurably unhealthy (duty/decay elevated vs early
       April) — a continuing leak — or does it look fully healthy?
  (t3) Is the ~May 31 TP3/oil plunge a repair signature (what ended b)?
- STAKES: labeling. If Apr 18-30 was degraded, it must NOT be labeled
  healthy in group comparisons / any future training reference; failure
  count for evaluation may be 3 distinct failures, not 4.

### OQ4 — August Oil_level regime (unresolved, scoped)
- Observation: 7 of Oil_level's 8 long inactive episodes fall Jul 30-Aug 28
  (median ~20h, longest 9.5 days), while the machine runs normally
  (duty 0.158 inside vs 0.161 outside) and far from documented events
  (12% within 3 days vs 31% expected by chance; median distance 34.8 d).
  The 8th (Apr 20) is attributed to the known instrument-fault window.
- Candidate explanations, by parsimony: (a) sensor degradation after ~6
  months of service; (b) a real oil-circuit condition; (c) changed
  operation/maintenance practice. NOT resolvable from this data — all
  documented failures are air leaks and the record ends Sep 1 with no
  outcome to validate against.
- Why it is logged: MODELING consequence. Any detector trained on the full
  record will flag August heavily; this entry documents the cause so the
  decision (signal vs noise vs exclusion) is deliberate.

## OQ5 — Is F3 detectable at all, or under-examined?
- F3 showed no 48h precursor in the 4 core sensors at the median level.
  Before concluding "undetectable", widen: all analog sensors + LPS rate
  (completeness pass, this section), and — in stage 4 — variability /
  rate-of-change / cycle-structure features and change-point vs the machine's
  OWN baseline (documented window may be logged late).
- If still silent: F3 is a genuinely sudden failure. Stating that honestly is
  a finding, not a gap.

### OQ6 — Irregular day-of-week duty elevation
- Wed/Sat/Sun show ~0.19 duty vs ~0.14 other days (35% higher), not a
  weekly rhythm. Candidate causes: scheduled operations/tests on those days,
  or the failure/degraded periods happening to fall on them (Apr 18-30 degraded
  span would weight certain weekdays). Check: recompute day-of-week duty on
  HEALTHY-only rows — if the elevation vanishes, it's the degraded/failure
  periods leaking in, not a real weekly effect.
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

### L05 — A far-away artist ruins the view
- matplotlib autoscales to include EVERYTHING drawn; one off-window marker
  compressed 5 days of data into blocks. Robust plot functions clamp
  xlim to the data and treat markers as guests, not owners.
- Bonus: the "bug" was the config's flagged source oddity resurfacing —
  the paper trail worked in reverse (yaml comment -> visual anomaly).

### L06 — A proximity claim without a base rate is unfalsifiable (OQ2)
- Exhibit A was vivid: F3's 1,288-min gap starts 11 minutes before its
  failure window ends and recording resumes hours before the logged
  maintenance. It suggested a general rule — "recording stops around
  repairs" — that was about to become a feature-engineering warm-up rule.
- The base-rate control refuted it: with 331 gaps over 175 days (~1.9/day),
  every event has gaps nearby by construction. Gaps near failures are in
  fact SMALLER than typical (median 16.6 min vs 62.1 min elsewhere), and
  only 15% of the twenty largest gaps sit near any known event; the top-15
  gaps are mostly 5–49 days from the nearest failure/maintenance.
- Habit: when claiming "X happens near Y", always ask what "near Y" looks
  like by chance. State the base rate, then compare magnitude, not just
  presence. A single dramatic case is an anecdote until the control runs.
- Related: L03/L04 (recipes and tools carry assumptions) — same family of
  error, different disguise: here the assumption was that co-occurrence is
  evidence.

### L07 — Passing tests are not correctness
- find_state_episodes passed five synthetic edge-case tests, then produced
  a reversed episode on real data: overlapping refinement brackets allowed
  non-monotonic flip times. My tests modeled clean single transitions;T
  real data is ragged.
- Fixes: clamp refined flips to be monotonic; drop degenerate (zero-length)
  episodes with a printed count; and make the function VALIDATE ITS OWN
  OUTPUT (raise on end <= start) — the run_checks philosophy applied to my
  own code.
- Habit: model the messiness in tests, and let functions refuse to return
  nonsense.

### L08 — The base-rate control keeps paying (OQ1, after OQ2)
- Oil_level episodes near events: 12% within 3 days vs 31% expected by
  chance — not just "no clustering" but LESS than chance. Without the
  control, "one episode is 1.2 days from F1" could have been written up as
  a link; it is in fact the known Apr 20 instrument fault.

### L09 — Anchor an investigation to the phenomenon's extent, not the first thing noticed
- The chatter investigation was initially scoped around ~13:00 — where a
  zoomed view happened to open — and characterized the interior of an
  already-frozen episode as if it were the start. The hourly variance probe
  located the true onset at ~05:00 (motor_var 4.5→0.0, antiphase 0.86→0.0),
  ~8h earlier. The stats (toggle rate, antiphase) were arithmetically valid
  but measured over the wrong window.
- Habit: establish an episode's BOUNDARIES before characterizing its
  interior. The analysis window must be wider than the phenomenon, or you
  measure its middle and call it its edge. find_state_episodes exists
  precisely to find boundaries by measurement, not eyeball.

### L10 — Effect-size metric must match the feature's type
- The robust effect size (median shift / IQR) is built for continuous
  variables. Applied to a binary signal it returns NaN — a binary's healthy
  IQR is 0, so the division is undefined. DV_eletric's NaN in the first
  effect table was NOT missing signal; it was the wrong metric silently
  failing.
- Fix: a single type-aware effect_sizes() that dispatches by feature —
  continuous -> median-shift/IQR (robust, D12-friendly for state mixtures);
  binary -> activation-rate difference (mean_grp - mean_ref) + ratio. A
  `method` column records which was used, since effects are sortable within
  a type but not strictly comparable across.
- This replaced two patch functions (prefail_effect_by_failure_old,
  duty_shift_by_failure) that existed only to work around the NaN — one
  type-aware function subsumed both. Verified the binary row matched the old
  duty helper before deleting.
- Habit (same family as L03/L04): a statistic that "runs" is not a statistic
  that "applies" — check the metric's assumptions against the variable's
  type, not just whether it returns a number.

### L11 — "Redundant for describing state" ≠ "redundant for predicting failure"
- The 4-sensor / central-tendency scope was a reasonable DEFAULT (EDA showed
  TP3≈Reservoirs, H1/TP2 antiphase views of load state), but redundancy in
  healthy operation doesn't imply redundancy before a fault — a sensor
  uninformative about normal state can still diverge uniquely pre-failure.
- Central tendency is the FLOOR of the comparison, not the ceiling; failures
  often show first in variability/dynamics, which belong to engineered
  features (stage 4).
- Habit: check completeness explicitly before concluding a failure is
  undetectable; scope choices are defaults to revisit, not conclusions.

### L12 — Match the transform to what the data can resolve
- "Signal analysis → Fourier" is a reflex; the discipline is asking what
  frequencies survive the sampling. Vibration spectra need kHz; a 10s grid
  supports only sub-0.05 Hz phenomena. The right spectral target here is the
  slow cycle rhythm, and the right tool is Lomb-Scargle (not FFT) because the
  sampling is uneven. Same family as L02/L04: transform assumptions vs data
  reality.

### L13 — Build features one family at a time, inspecting each before the next
- Interleave construction and inspection: build a feature family, sanity-check
  its new columns (describe + distribution glance) immediately, THEN build the
  next. A degenerate family (all-NaN, wrong scale, constant) is caught at
  once, not buried among dozens of columns later.
- Keep this workflow discipline separate from state mechanics: do NOT chain
  mutating copies (df→df2→df3...) — that couples cells to run-order and
  corrupts silently on re-run. Families are pure functions returning columns
  concatenated onto one grid; the ordering discipline is about WHEN you look,
  not about mutating shared state.

### L14 — Assume nothing about digital polarity; verify against a physical anchor
- Three signals now confirmed inverted vs their documented/assumed meaning
  (Pressure_switch, Oil_level, COMP). The reflex "0=off, 1=on" is unreliable
  in this dataset. Verify every digital's meaning against a physical anchor
  (e.g. Motor_current level) before using it in a mask or feature.

### L15 — Aggregate at the right granularity, or signal averages to zero
- tp3_decay_slope fit ONE line across a whole 1h window — but a window holds
  ~40 idle stretches separated by refills; the fit measured the sawtooth's
  overall tilt (≈0), not the decay within each tooth. Effect sizes confirmed
  it: −0.00 to −0.12, dead. Fix: fit each idle stretch separately, take the
  median. The leak lives at per-stretch granularity; window-wide aggregation
  destroyed it.
- Caught by the per-family effect-size validation — a plausible-looking
  feature (sensible histogram) that separated nothing. Validation, not
  inspection, caught it.
  
### L16 — Build the feature, let the effect sizes assign it — don't assume which failure it catches
- Predicted longest_load_stretch would catch F3 (lock-up hypothesis). It caught
  F4 instead (+1.83); F3 was the OPPOSITE mechanism (short-cycling, caught by
  cycle_dur_cv/trend). The feature was worth building — just not for the failure
  predicted. Assign features to failures by measurement, not expectation.

### L17 — Always run the trivial baseline; it may win
- A 3-line max|z| detector beat IsolationForest, Mahalanobis, all seven
  supervised models, and two forecast-residual variants. Reaching for
  sophistication first would have produced a worse, less interpretable result.
- The baseline wins when the features already encode the signal — check before
  adding complexity.

### L18 — A singular covariance silently inverts a distance metric
- Mahalanobis scored healthy above failures because structural collinearity
  (occupancy fractions, one-hots, _missing flags) made the covariance singular
  (cond ~1e21); pinv returned garbage. Ledoit-Wolf shrinkage fixed the
  conditioning (cond ~660) and un-inverted it. Verify a distance metric's
  covariance is well-conditioned before trusting — and don't judge a method on a
  numerically broken run.
