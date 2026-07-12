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
  My earlier "June oil leak" label came from unverified secondary memory.
- Choice: failure_windows now transcribed from the source table into config
  (with maintenance dates from the Report column). Source numbering reads
  #1,#1,#3,#4 — second row assumed a typo for #2.
- Consequence: OQ1's test redesigned (no oil-leak window exists to test against).

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


## Open questions
## OQ1 (redesigned) — Digital polarity vs docs
- Oil_level 0.904 active / Pressure_switch 0.991 active contradict their
  documented meanings. Test: when does the INACTIVE time occur — clustered
  (failures/maintenance/gaps) or scattered? Run after step 5 context exists.

## OQ2 — Post-gap behavior
- The one investigated anomaly began 9 min after a 1h50m gap ended. Are
  anomalies clustered near gaps? If yes, post-gap minutes may need a
  warm-up/exclusion rule in feature building.

# Lessons 

## L01 
- IPython magics take no trailing comments
- autoreload refreshes code but not variables already in memory (re-run cells that created data after editing functions)
- "Python 3.11" in VSC's picker can be several different environments.

## L02
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

## L03 (from step 3)

- **A tool classifying my own processed data is not independent validation**
  — AutoViz "confirming" 7 numeric + 8 boolean columns was reading back my
  D04 dtypes. Real validation = the full-data n_other check.
- **Auto-EDA tools are a 10-minute skim, not an artifact pipeline** —
  AutoViz on pandas 3 needed a shim, is time-blind and sampled, and had
  flaky export. Timebox convenience tools; cut losses.
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
