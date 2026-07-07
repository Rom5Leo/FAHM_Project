# Domain Primer — Understanding the APU and the ML Around It

Read this alongside the MetroPT paper (Veloso et al., *The MetroPT dataset for
predictive maintenance*, Scientific Data, 2022). The paper gives the official
sensor descriptions and failure reports; this primer gives the working
intuition our pipeline is built on, tied to what we actually observed in the
data.

---

## 1. What the machine does

The **APU (Air Production Unit)** is the air compressor system of a metro
train. Trains are pneumatic animals: compressed air operates the **brakes**,
the **suspension**, doors and auxiliary systems. The APU's job is to keep the
air reservoirs pressurized between ~8.2 and ~10 bar.

It works like a refrigerator: in bursts.

1. **Idle:** compressor off. The train consumes air (braking, suspension
   corrections), so reservoir pressure slowly falls. In our plots: the
   descending slope of the Reservoirs/TP3 sawtooth, Motor_current ≈ 0.
2. **Load:** when pressure drops to ~8.2 bar, the compressor kicks in
   (start spike ~6 A, sustained ~3.8 A in this dataset), pressure climbs back
   to ~10 bar, compressor unloads. The ascending edge of the sawtooth.

The fraction of time spent loading is the **duty cycle** — healthy ≈ 0.07–0.16
in this dataset depending on the operating regime. Duty is the machine's
"heart rate": almost every compressed-air problem shows up in it.

Two more subsystems appear in the sensors:
- **Air-drying towers** (Towers, Pressure_switch, DV_pressure, H1): compressed
  air carries moisture; two towers alternate between drying the air and
  purging their own humidity.
- **Oil circuit** (Oil_temperature, Oil_level): the compressor is oil-
  lubricated and oil-cooled. Oil problems show up as temperature problems.

## 2. Sensor-by-sensor intuition (as observed in THIS dataset)

| Sensor | What it physically is | What to watch |
|---|---|---|
| TP2 | Pressure at the compressor | ~0 idle, snaps to ~10 under load. A load-state indicator in analog form. |
| TP3 | Pressure at the pneumatic panel | The sawtooth. Its **idle decay slope** is the leak meter. |
| Reservoirs | Downstream reservoir pressure | Tracks TP3 almost exactly; redundancy/sanity check. |
| H1 | Pressure drop at the cyclonic separator discharge | Filter/valve behavior. |
| DV_pressure | Pressure drop when towers discharge | ~0 under load; spikes at tower discharge. |
| Oil_temperature | Compressor oil temp | ~50–70 °C cycling normally. Sustained high = compressor overworking or oil problem. |
| Motor_current | One motor phase | 0 idle / ~6 A start / ~3.8 A load. **Pinned ≈5.7 A continuously = compressor can't keep up.** |
| COMP / DV_eletric | Intake / outlet valve signals | Perfect antiphase. DV_eletric=1 defines LOAD in our pipeline. |
| Towers / Pressure_switch | Tower alternation / discharge detect | Polarity per paper; verify empirically. |
| MPG | Starts compressor when p < 8.2 bar | The thermostat of the system. |
| LPS | Low-pressure switch, p < 7 bar | Active only 0.34% of the time — rare, and rare = informative. |
| Oil_level | Oil level switch | "Active" 90% of time contradicts the paper's "active = low oil". Polarity SUSPECT — check around F3. |
| Caudal_impulsion | Airflow at compressor output | Flow confirmation. |

## 3. Failure physics — what each fault does to the signals

### Air leak (F1, F2, F4)
Air escapes somewhere in the circuit, so the system loses pressure faster than
it should. The causal chain:

leak → idle pressure decays faster (TP3 decay rate more negative)
     → compressor must run sooner and longer (duty ↑, load duration ↑)
     → if the leak outgrows compressor capacity: compressor NEVER unloads
       (motor current pinned, TP3 plateaus below cutoff, duty = 1.0)
     → more running → oil temperature ↑ as a side effect.

Two regimes, and they need different features:
- **Gradual leak:** cycle features shine (decay rate, duty drift). F4 is the
  textbook case: duty 0.62, decay −0.58 bar/min at failure.
- **Catastrophic leak:** no cycles complete, so cycle features go SILENT
  (the F1 lesson). Coverage comes from grid-based features: rolling duty and
  `pinned_load_2h` (2-hour minimum of duty ≈ 1.0 ⇒ never unloaded).

### Oil leak (F3)
Oil, not air, escapes: lubrication and cooling degrade. Expected signature:
Oil_temperature drifting up for the same workload, possibly Oil_level state
change, little effect on pressure dynamics — which is why our air-centric
cycle features barely reacted to F3. Detecting F3 well likely needs
oil-specific features (e.g., oil temp residual vs duty, temperature rise per
load cycle). This is a known open item, not an oversight.

## 4. The ML concepts we chose, and why

**Anomaly detection, not classification.** 4 failure events cannot train a
classifier. Instead: learn what NORMAL looks like from healthy months, score
deviation from normal, and use the 4 events only to *evaluate*.

**Isolation Forest (the baseline).** Builds many random trees; points that get
isolated in few splits are "easy to separate" = anomalous. Chosen because it
is fast, needs no distributional assumptions, handles dozens of features, and
gives a continuous score. It is the dumb-but-honest baseline the workbook
demands before anything fancier.

**Rolling features.** The model sees one minute at a time, but health is a
trend, not an instant. Rolling means/stds over 30 min / 4 h / 24 h give each
minute a memory. `duty_trend` (4 h vs 24 h duty) asks: "is the recent duty
above its own recent baseline?" — a leak developing.

**Gap-aware segmentation.** 38 days of the record are missing across 331
gaps. A rolling window that bridges a 2-day hole compares apples to
yesterday's oranges. `segment_id` guarantees windows live inside contiguous
recordings only.

**Threshold + persistence.** Score > threshold for one minute is noise; for
30 of the last 60 minutes it is a condition. Persistence trades ~30 min of
detection latency for a large false-alarm reduction — the right trade for
leak-class faults, which develop over hours to days.

**Early warning + false alarms as THE metrics.** An engineer needs to know:
"how much notice do I get?" and "how often does it cry wolf?" — these are
`early_warning_h` per failure and `false_alarms_per_week` in healthy time.
AUC does not answer either question.

## 5. Vocabulary for interviews

- **Duty cycle:** fraction of time under load. The APU's vital sign.
- **TP3 idle decay rate:** bar/min lost while compressor is off. Direct leak
  measurement.
- **Pinned load:** compressor continuously loading (never reaching cutoff) —
  the catastrophic-leak signature.
- **Segment:** a contiguous stretch of recording between gaps.
- **Persistence rule:** k-of-n minutes above threshold before alerting.
- **Early-warning time:** fault_start minus first alert (positive = warned
  before).
- **Regime shift:** the ~Mar 1 change in operating pattern (duty 0.07→0.12) —
  normal behavior itself changed, so "normal" must be defined per regime.

## 6. Reading list, in order of value

1. MetroPT paper — sections on the APU description, sensor list, and the
   failure reports (this is also where the failure timestamps must be
   verified from).
2. `docs/data_profile.md` — what OUR copy of the data actually looks like.
3. `docs/figures/cycle_health_overview.png` — seven months of machine health
   in one image; every finding in the decision log is visible here.
4. scikit-learn Isolation Forest user guide — 10 minutes, enough to defend
   the baseline choice.
