# Baseline evaluation — Isolation Forest

Train periods: [['2020-02-01', '2020-02-28'], ['2020-04-05', '2020-04-14']]

Threshold: 0.779 (q=0.995 of training scores), persistence 30/60 min.

## Per-failure

| failure_id   | detected   | first_alert         |   early_warning_h |
|:-------------|:-----------|:--------------------|------------------:|
| F1           | True       | 2020-04-12 17:12:00 |             126.8 |
| F2           | False      | NaT                 |             nan   |
| F3           | True       | 2020-06-05 14:24:00 |              -4.4 |
| F4           | False      | NaT                 |             nan   |

## False alarms

- alert_episodes_total: 8
- false_alarm_episodes: 4
- healthy_weeks: 13.1
- false_alarms_per_week: 0.31

> early_warning_h > 0 means the first alert fired BEFORE the documented fault start; negative means detection during the window.
