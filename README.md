# Membrane Ageing Analysis

From raw signals (Pressure transducer, contact angle, FTIR) to degradation models — a data processing
and modeling pipeline for analyzing UF membrane aging in water treatment.

## Project Structure

```
membrane-aging-prediction/
├── config/
│   └── params.yaml                       # All toggleable parameters
├── data/
│   ├── raw/                              # Place raw DAQ CSVs here
│   ├── processed/                        # Pipeline outputs
│   └── DATA_FORMAT.md                    # Expected file formats
├── src/
│   ├── processing/
│   │   ├── tmp/
│   │   │   ├── signal_processing.py      # Voltage → TMP, denoising, drift
│   │   │   ├── cycle_detection.py        # Filtration cycle segmentation
│   │   │   └── feature_extraction.py     # CWR, fouling rate, backwash η
│   │   ├── ftir/
│   │   │   └── README.md                 # → thesis Appendix D.1
│   │   ├── geometry/
│   │   │   └── README.md                 # → thesis Appendix D.3
│   │   └── contact_angle/
│   │       └── README.md                 # → thesis Section 3.3.3
│   ├── models/
│   │   ├── parametric.py                 # Power law / Exp / Weibull + AICc/BIC
│   │   ├── gp_regression.py              # GP with kernel comparison + LOO-CV
│   │   └── model_comparison.py           # Unified comparison table
│   ├── pipeline.py                       # Orchestrates all stages
│   └── utils.py                          # Config, logging, paths
├── results/
│   ├── figures/                          # Generated plots
│   └── tables/                           # Comparison CSVs
├── run.py                                # Entry point
├── requirements.txt
├── README.md
└── REFERENCES.md                         # Literature basis for all methods
```

## Pipeline

### Stage A — Signal Processing & Feature Extraction

**Input:** Raw DAQ voltage CSVs (8-channel, 10s sampling)

1. **Voltage → TMP** with transducer calibration (Thesis Eq. D.1)
2. **Denoising** — configurable: Savitzky-Golay, moving average, or median
3. **Drift correction** via stable reference channel
4. **Cycle detection** — gradient-based state machine identifies filtration,
   backwash, and CIP phases within each operational cycle
5. **CWR extraction** — locates post-CIP stable TMP regions using rolling
   variance criterion, computes R = TMP/(μ·J) (Thesis Eq. D.2)
6. **Fouling rate** — linear fit of TMP rise during filtration phase
7. **Backwash recovery** — ratio of TMP reduction after backwash

### Stage B — Degradation Modeling

**Input:** Processed aging data (resistance vs. cumulative dose per condition)

| Model | Form | Params | Selection |
|-------|------|--------|-----------|
| Power law | R/R₀ = a·D^b + c | 3 | AICc, BIC |
| Exponential | R/R₀ = a·(1−e^(−bD)) + c | 3 | AICc, BIC |
| Weibull | R/R₀ = a·(1−e^(−(D/λ)^k)) + c | 4 | AICc, BIC |
| GP (4 kernels) | non-parametric | — | LOO-RMSE, Coverage |

Parametric models include bootstrap 95% CI. GP models include calibrated
uncertainty bands and kernel comparison (RBF, Matérn-3/2, Matérn-5/2,
Rational Quadratic). See REFERENCES.md for selection rationale.

### Stage C — Cross-Condition Scaling

Demonstrates that protein-type NOM loading rate serves as a scaling factor
that approximately collapses degradation curves from different conditions.
Presented as a physically-motivated dimensional analysis validated on 6
conditions, with explicit acknowledgment of the statistical limitations.

## Configuration

All parameters live in `config/params.yaml`. Key toggles:

| Parameter | Location | Options |
|-----------|----------|---------|
| Denoising | `signal.denoise_method` | `savgol` / `moving_average` / `median` / `none` |
| Cycle detection | `cycle_detection.method` | `gradient` / `threshold` |
| Stability threshold | `feature_extraction.cwr_stability_threshold` | float |
| Parametric models | `modeling.parametric.models` | list of names |
| GP kernels | `modeling.gp.kernels` | list of names |
| Bootstrap N | `modeling.parametric.n_bootstrap` | int |

## Usage

```bash
pip install -r requirements.txt

# Full pipeline (requires data in data/raw/ and data/processed/)
python run.py

# Individual stages
python run.py --stage signal --raw-file data/raw/my_signal.csv
python run.py --stage modeling --aging-file data/processed/aging_data.csv
python run.py --stage scaling --aging-file data/processed/aging_data.csv
```

## Limitations

- **Small samples** (7–14 points per condition) constrain model complexity.
  Neural network approaches are not appropriate at this scale.
- **AICc rather than AIC** is used throughout because n/k < 40 for all fits.
- **Cross-condition scaling** is validated on 6 conditions. The protein-loading
  similarity law is a physical hypothesis, not a trained transfer model.
- **GP uncertainty calibration** should be interpreted cautiously with <15
  points — empirical coverage may not converge to nominal levels.

## Methodology References

See [REFERENCES.md](REFERENCES.md) for the literature basis of each
methodological choice, including model selection criteria, kernel selection
rationale, and degradation model functional forms. FTIR, geometry, and
contact angle processing methods are documented in the PhD thesis.
