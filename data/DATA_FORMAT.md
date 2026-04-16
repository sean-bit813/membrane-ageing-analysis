# Expected Data Formats

## data/raw/
Place raw DAQ CSV files here. Expected format:
```
Time, Voltage - Dev1_ai0, ..., Voltage - Dev1_ai7
3.725e9, 0.000662, -0.001320, 0.022417, ...
```
- `Time`: LabVIEW epoch timestamp (seconds)
- `ai0`, `ai3`: TMP pressure transducer channels
- `ai2`: Stable reference channel (temperature or reference pressure)
- `ai1`, `ai4–ai7`: Auxiliary sensors

## data/processed/
Pipeline outputs are written here automatically:
- `cycle_features.csv`: Per-cycle extracted features
- `aging_data.csv`: Processed resistance vs. cumulative dose

## Aging data CSV format (for modeling stage):
```
cumulative_dose_ppmh, condition, resistance_normalized, resistance_error, protein_loading
0, Lab-HP, 1.0, 0.02, 0.244
100000, Lab-HP, 1.05, 0.03, 0.244
...
```
