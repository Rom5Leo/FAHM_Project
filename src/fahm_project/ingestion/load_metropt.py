"""Ingest the raw MetroPT-3 CSV into a typed, compressed parquet file.

The raw file is ~15M rows at 1 Hz. We never want to touch the CSV again after
this step: everything downstream (profiling, features, models) reads
``data/processed/sensor_readings.parquet``.

Design decisions (log these in decision_log.md):
  * Stream the CSV in chunks -> constant memory regardless of machine size.
  * Explicit dtypes: float32 for analog sensors, int8 for digital signals.
    Halves memory vs float64 with no meaningful precision loss at sensor scale.
  * We do NOT sort across chunks. The file is chronological at source; the
    profiler verifies monotonicity instead of paying an expensive global sort.
  * ``asset_id`` is added even though there is one asset, so the A2 simulator
    can emit multiple assets into the exact same schema later.

Run from the repo root:
    poetry run python -m fahm_project.ingestion.load_metropt --config configs/metropt.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# --- Schema ------------------------------------------------------------------

ANALOG_SENSORS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

DIGITAL_SENSORS = [
    "COMP",
    "DV_eletric",  # (sic) spelled this way in the source data
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulsion",
]

TIMESTAMP_COL = "timestamp"

# Different releases of MetroPT spell some columns differently.
# Map every known variant -> our canonical name (None = drop the column).
COLUMN_ALIASES = {
    "Caudal_impulses": "Caudal_impulsion",
    "Flowmeter": "Caudal_impulsion",
    "Timestamp": TIMESTAMP_COL,
    "gpsSpeed": None,  # present in some MetroPT releases, not useful here
}

# Digital signals are stored as 0.0/1.0 floats in the CSV; read as float32
# first, cast to int8 after (direct int parsing fails on "1.0" strings).
# Include alias spellings so they get the compact dtype at read time too.
CSV_DTYPES = {c: "float32" for c in ANALOG_SENSORS + DIGITAL_SENSORS}
CSV_DTYPES.update(
    {
        alias: "float32"
        for alias, target in COLUMN_ALIASES.items()
        if target is not None and target != TIMESTAMP_COL
    }
)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def clean_chunk(chunk: pd.DataFrame, asset_id: str) -> pd.DataFrame:
    """Normalize one raw chunk: drop index column, parse time, cast digitals."""
    # Kaggle CSV ships with an unnamed leading index column -> drop it.
    junk = [c for c in chunk.columns if c.lower().startswith("unnamed")]
    chunk = chunk.drop(columns=junk)

    # Normalize known column-name variants, drop known-useless columns.
    drop_cols = [c for c in chunk.columns if COLUMN_ALIASES.get(c, "keep") is None]
    chunk = chunk.drop(columns=drop_cols)
    rename = {c: COLUMN_ALIASES[c] for c in chunk.columns if COLUMN_ALIASES.get(c)}
    chunk = chunk.rename(columns=rename)

    # Fail loudly and helpfully if the schema still doesn't match.
    expected = {TIMESTAMP_COL, *ANALOG_SENSORS, *DIGITAL_SENSORS}
    missing = expected - set(chunk.columns)
    if missing:
        raise ValueError(
            f"CSV schema mismatch. Missing canonical columns: {sorted(missing)}.\n"
            f"Columns actually found in the file: {sorted(chunk.columns)}.\n"
            "Fix: add the real column name to COLUMN_ALIASES in load_metropt.py."
        )

    # Fast path with explicit format; falls back to flexible parsing if the
    # format ever differs (e.g. fractional seconds).
    try:
        chunk[TIMESTAMP_COL] = pd.to_datetime(
            chunk[TIMESTAMP_COL], format="%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        chunk[TIMESTAMP_COL] = pd.to_datetime(chunk[TIMESTAMP_COL])

    for col in DIGITAL_SENSORS:
        chunk[col] = chunk[col].fillna(-1).astype("int8")  # -1 marks missing

    chunk["asset_id"] = asset_id
    ordered = [TIMESTAMP_COL, "asset_id"] + ANALOG_SENSORS + DIGITAL_SENSORS
    return chunk[ordered]


def ingest(raw_csv: Path, out_parquet: Path, chunksize: int, asset_id: str) -> None:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    t0 = time.time()

    reader = pd.read_csv(
        raw_csv,
        chunksize=chunksize,
        dtype=CSV_DTYPES,
    )

    try:
        for i, chunk in enumerate(reader):
            chunk = clean_chunk(chunk, asset_id)
            table = pa.Table.from_pandas(chunk, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    out_parquet, table.schema, compression="snappy"
                )
            writer.write_table(table)

            total_rows += len(chunk)
            print(
                f"chunk {i + 1}: +{len(chunk):,} rows "
                f"(total {total_rows:,}, {time.time() - t0:.0f}s elapsed)"
            )
    finally:
        if writer is not None:
            writer.close()

    print(f"\nDone. Wrote {total_rows:,} rows -> {out_parquet}")
    size_mb = out_parquet.stat().st_size / 1e6
    print(f"Parquet size: {size_mb:.0f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/metropt.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ingest(
        raw_csv=Path(cfg["paths"]["raw_csv"]),
        out_parquet=Path(cfg["paths"]["processed_parquet"]),
        chunksize=int(cfg["ingestion"]["chunksize"]),
        asset_id=str(cfg["ingestion"]["asset_id"]),
    )


if __name__ == "__main__":
    main()
