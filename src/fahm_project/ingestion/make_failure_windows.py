"""Write failure_windows.csv — the evaluation ground truth for A1.

MetroPT-3 has NO label column. The documented failure events come from the
dataset paper (Veloso et al., "The MetroPT dataset for predictive
maintenance", Scientific Data, 2022) and the maintenance reports it cites.

!!! IMPORTANT !!!
The timestamps below are the commonly cited windows for this train's APU, but
you MUST verify them against Table/failure-report section of the paper before
trusting any evaluation number. If you change them, note it in decision_log.md.

These windows are used ONLY for evaluation (early-warning time, false alarms
in healthy periods) — never as training labels. With ~4 events, supervised
training on them would be meaningless.

Run:
    poetry run python -m fahm_project.ingestion.make_failure_windows --config configs/metropt.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

# VERIFY AGAINST THE PAPER BEFORE FINAL EVALUATION.
FAILURE_WINDOWS = [
    # (failure_id, fault_type, start, end, source note)
    ("F1", "air_leak", "2020-04-18 00:00:00", "2020-04-18 23:59:00", "verify vs paper"),
    ("F2", "air_leak", "2020-05-29 23:30:00", "2020-05-30 06:00:00", "verify vs paper"),
    ("F3", "oil_leak", "2020-06-05 10:00:00", "2020-06-07 14:30:00", "verify vs paper"),
    ("F4", "air_leak", "2020-07-15 14:30:00", "2020-07-15 19:00:00", "verify vs paper"),
]


def build_frame(asset_id: str) -> pd.DataFrame:
    df = pd.DataFrame(
        FAILURE_WINDOWS,
        columns=["failure_id", "fault_type", "start", "end", "source_note"],
    )
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df.insert(1, "asset_id", asset_id)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/metropt.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out = Path(cfg["paths"]["failure_windows"])
    out.parent.mkdir(parents=True, exist_ok=True)

    df = build_frame(cfg["ingestion"]["asset_id"])
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} failure windows -> {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
