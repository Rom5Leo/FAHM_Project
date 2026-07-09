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