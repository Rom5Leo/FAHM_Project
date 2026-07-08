# Workflow Mapping — Football Preprocessing Notebook → MetroPT-3 Pipeline

Your football notebook, stage by stage, translated to this dataset. The
column on the right is YOUR to-do list. Where a stage says "N/A", the reason
is the structural difference: football data is relational (7 tables joined by
IDs); MetroPT is one time-series table where TIME plays the role IDs played.

| # | Football notebook stage | What it really does | MetroPT-3 equivalent |
|---|---|---|---|
| 1 | imports, warnings, pickle | environment setup | same (pandas, matplotlib, yaml; parquet instead of pickle — see note A) |
| 2 | `kagglehub.dataset_download("technika148/football-database")` | reproducible data fetch | `kagglehub.dataset_download("joebeachcapital/metropt-3-dataset")` — same trick works |
| 3 | `os.listdir(path)` | discover files | same (you'll find one CSV) |
| 4 | loop CSVs → `dfs` dict | ingestion | load ONE csv, but *typed*: parse timestamp, choose dtypes (analog vs digital), handle the unnamed index column |
| 5 | `head()` / `info()` per table | first structural look | same, plus: time range, `describe()`, memory usage before/after dtype choice |
| 6 | `value_counts()` / histograms | distributions | analog sensors → hist/describe; digital signals → `value_counts()` (should be ~{0,1}; anything else = a finding) |
| 7 | recode substituteIn/Out → binary `subIn`/`subOut` | fix a bad encoding | digital signals arrive as floats 0.0/1.0 → cast to int8; decide how to mark missing; note polarity suspicions (Oil_level!) |
| 8 | reorder columns around `time` | canonical schema | define YOUR canonical column order (timestamp first, then analog, then digital) and enforce it in one place |
| 9 | split odds/probabilities into `df_games_odds` | separate non-modeling data from modeling data | separate GROUND TRUTH from sensor data: build `failure_windows` (from the paper) as its own table — it is evaluation data, never features |
| 10 | matching-features scan across table pairs | find join keys | N/A as-is (one table). The equivalent concept: **timestamp is your join key** — failure windows, future feature tables, and alerts all align on time |
| 11 | merge home/away stats into `df_combined` | build the modeling table | aggregate over time instead of merging over IDs: raw 10s grid → your chosen base grid (you decide the frequency and defend it) |
| 12 | `determine_result()` → `gameresult` target | derive the target | derive EVALUATION labels: e.g. `in_failure_window`, `hours_to_next_failure`. Critical difference from football: these are for evaluation only, never model input (4 failures ≠ a target you can train on) |
| 13 | `assign_teams()` + goal verification | domain-logic consistency checks | physics consistency checks — the fun part. Ideas: timestamps strictly increasing? modal interval = 10s? TP3 ≈ Reservoirs (they're supposed to track)? COMP and DV_eletric in antiphase? values inside plausible physical ranges? |
| 14 | duplicate / discrepancy checks (one team per player-game) | data validation | duplicate timestamps? recording gaps (find them, size them)? missingness per column? |
| 15 | aggregated player/team performance tables | feature tables | per-window aggregates and (later) per-compressor-cycle tables — that's your feature engineering stage, after preprocessing |
| 16 | (pickle at top, presumably saved at end) | persist results | save the clean typed table + the failure windows + a written data profile. Parquet, not pickle (note A) |

**Note A — why parquet over pickle:** pickle is Python-only, version-fragile,
unsafe to load from untrusted sources, and row-oriented. Parquet is columnar
(fast partial reads: `columns=[...]`), compressed, typed, and readable from
any language/tool. For DataFrames you'll reload many times, parquet wins.

**The two football-specific things with NO equivalent** (skip entirely):
team/player assignment logic and the multi-table relationship scan. Their
*spirit* — "verify your assumptions with checks the data must pass" — is the
thing to keep, pointed at physics instead of football rules.

---

## The professional pattern: notebook + src + config

Three layers, one rule each:

```
notebooks/01_preprocessing.ipynb   ← the STORY: markdown, calls, displays, plots
src/fahm_project/…                 ← the WORK: every function that transforms data
configs/config.yaml                ← the NUMBERS: every path, threshold, choice
```

- **Rule for the notebook:** no logic. If a cell is more than ~5 lines of
  code, that code belongs in `src/` and the cell becomes one function call
  plus a `display()`. Your football notebook's `assign_teams()` is exactly the
  kind of function that should have lived in a module.
- **Rule for src:** no hardcoded values. Anything tunable comes in as a
  function argument, fed from the config.
- **Rule for config:** if you change an experiment, you should only touch this
  file.

### Making `from fahm_project import …` work in the notebook

One-time setup — tell poetry where the package lives (`pyproject.toml`):

```toml
[tool.poetry]
packages = [{ include = "fahm_project", from = "src" }]
```

then `poetry install` once. In the notebook's first cell:

```python
%load_ext autoreload
%autoreload 2          # re-imports your src modules on every cell run —
                       # edit the .py, rerun the cell, no kernel restart

import yaml
from fahm_project import preprocessing as pp

with open("../configs/config.yaml") as f:
    cfg = yaml.safe_load(f)
```

and from then on notebook cells look like:

```python
df = pp.load_raw(cfg)              # work happens in src
pp.profile(df)                     # returns/prints the profile
df.head()                          # notebook does the LOOKING
```

`%autoreload 2` is the magic that makes this workflow pleasant: you develop
functions in VSC in the .py file, and the notebook always runs the latest
version.

### Suggested build order (mirrors your notebook's order)

1. config.yaml with just the paths → `load_raw()` → `head/info` in notebook
2. dtype + timestamp decisions → typed loading → memory before/after
3. distributions per sensor (notebook plots calling a src plotting helper)
4. consistency checks (stage 13-14) → a `run_checks(df)` that PRINTS pass/fail
5. failure_windows table (stage 9/12)
6. save to parquet + write your data profile document

Each step = one commit.
