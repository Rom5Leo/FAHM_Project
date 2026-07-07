# EDA Deep-Dive — guided reading

Each figure answers a specific open question. Write your conclusion under each
one; unanswered boxes become error_analysis.md / decision_log.md entries.

## 1. Event zooms (`figures/eda/event_*.png`)
For every documented failure and the undocumented March event: raw TP3,
Motor_current, Oil_temperature, DV_eletric with the window shaded.
**Questions:** What does each failure look like hours before the window —
gradual drift or sudden break? Is M10_undocumented shaped like F1
(continuous load) → likely an undocumented leak, or like a maintenance test?

> My conclusion (M10): ...
> My conclusion (per failure): ...

## 2. F3 oil story (`f3_oil_story.png`)
Oil temperature (hourly), oil temp per unit duty (cooling-efficiency proxy),
Oil_level fraction, ±10 days around F3.
**Questions:** Does any oil signal move BEFORE the window? Does Oil_level
flip state — and which direction, settling the polarity question (D15)?

> My conclusion: ...

## 3. LPS activations (`lps_events.png`)
Every minute the low-pressure switch fired, across 7 months.
**Question:** Do LPS events cluster at failures (→ strong cheap feature /
threshold-rule baseline) or scatter randomly (→ noise)?

> My conclusion: ...

## 4. Healthy vs pre-failure distributions (`dist_healthy_vs_prefail.png`)
**Question:** Which features separate 48h-pre-failure from healthy — duty?
oil temp? If none separate, early warning at 48h is NOT learnable from these
features and expectations must be reset honestly.

> My conclusion: ...

## 5. Weekly duty (`weekly_duty.png`)
**Questions:** Exactly when does the regime shift happen? Is there slow drift
that could cause false alarms months after training?

> My conclusion: ...
