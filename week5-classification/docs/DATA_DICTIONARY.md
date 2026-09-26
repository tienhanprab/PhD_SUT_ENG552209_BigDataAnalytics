# Source and schema reconciliation

Source: https://www.kaggle.com/datasets/isurumy93/solar-pv-anomaly-detection-dataset/versions/2

The version-2 CSV has 152,973 rows and 16 columns. The generic six-column description does not match the file's exact schema. The five selected predictors are:

| CSV column | Unit / role | Interpretation |
|---|---|---|
| light_lux | lux; predictor | Measured illuminance; NOT irradiance in W/m² |
| panel_temperature_c | °C; predictor | Measured module temperature |
| voltage_v | V; predictor | Measured panel voltage |
| current_a | A; predictor | Measured panel current |
| power_w | W; predictor | Supplied power channel; approximately V×I, with residual noise |
| fault | target | 0 Normal, 1 Fault |
| timestamp_s, day, seconds_of_day | time | Split/audit only, never predictors |
| event_id | event identifier | Audit event separation only; 0 is shared normal background |
| fault_type | label detail | Audit only; excluded to prevent target leakage |
| true_irradiance_wm2 | W/m² | Simulator ground truth; excluded from observable-input experiment |
| ambient_temperature_c | °C | Additional simulated context; excluded to retain five predictors |
| true_panel_temperature_c, true_voltage_v, true_current_a | simulator truth | Excluded |

Units are inferred from explicit CSV names and reconciled with the publisher's generic description; no lux-to-irradiance conversion is assumed. `Medium` voltage means the middle training quantile, not a nominal safe range. Discrete saturation at 65,535 lux collapses the upper tertile, giving light two effective bins.

Grain: one irregularly sampled simulated time-series row, unique `timestamp_s`, over three day indices. Metrics are per row, not time-weighted or event-level. Normal background event 0 appears in all days; no nonzero fault event crosses the split.

Source limitations: simulated dynamics, no real field measurements, only three days. Power and voltage/current are redundant. Description licensing prose suggests CC BY/MIT while metadata says CC0; raw CSV is excluded from Git and this project does not resolve the inconsistency.

Provenance: `docs/kaggle_metadata.json` preserves publisher metadata; `data/raw/provenance.json` records pinned version, URL, bytes and SHA-256. A pinned version-2 download was checked against the original download and matched exactly.
