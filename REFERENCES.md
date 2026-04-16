# References & Methodology Justification

This document cites the literature basis for each methodological choice in the pipeline.

---

## Data Processing

### TMP Signal Calibration
Voltage-to-pressure conversion follows calibrated transducer specifications with periodic calibration to account for sensor drift. The conversion equation (TMP = Voltage × calibration_factor) and resistance calculation (R = TMP / μJ) follow standard practice in membrane filtration research.
- **Thesis reference:** Appendix D, Equations D.1–D.2

### Signal Denoising (Savitzky-Golay Filter)
The Savitzky-Golay filter fits successive sub-sets of adjacent data points with a low-degree polynomial by the method of linear least squares. It preserves signal features such as relative maxima and minima better than simple moving averages, which is critical for accurate TMP peak identification during filtration cycles.
- Savitzky, A., & Golay, M. J. E. (1964). Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry*, 36(8), 1627–1639.

### FTIR Spectral Processing (EMSC)
Extended Multiplicative Signal Correction removes baseline variations and multiplicative scaling differences between spectra. It is the standard normalization method for ATR-FTIR spectra of polymer membranes, where differences in crystal–sample contact pressure can introduce non-chemical variation.
- Afseth, N. K., & Kohler, A. (2012). Extended multiplicative signal correction in vibrational spectroscopy, a tutorial. *Chemometrics and Intelligent Laboratory Systems*, 117, 92–99.

### Clean Water Resistance Extraction
The stability-based extraction method selects post-CIP measurement windows where rolling variance falls below a threshold, ensuring that only steady-state TMP values contribute to resistance calculations. This approach is more robust than simple averaging, which can be biased by transient hydraulic effects during system re-pressurization.
- **Thesis reference:** Section 3.3.4 and Appendix D.2

---

## Model Selection Criteria

### AICc (Corrected Akaike Information Criterion)
AIC estimates the relative Kullback-Leibler divergence between a fitted model and the true data-generating process. The small-sample correction (AICc) is essential when n/k < 40, which applies to all conditions in this dataset (n = 7–14, k = 3–4). Standard AIC systematically favors overly complex models in this regime.
- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716–723.
- Hurvich, C. M., & Tsai, C.-L. (1989). Regression and time series model selection in small samples. *Biometrika*, 76(2), 297–307.

### BIC (Bayesian Information Criterion)
BIC imposes a stronger complexity penalty than AIC (scaling with log(n) rather than 2). It is consistent: as n → ∞, BIC selects the true model with probability 1 if it is in the candidate set. In this project, BIC and AICc are reported together; when they agree, model selection is robust.
- Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461–464.

### Reporting Both AICc and BIC
AICc is optimized for prediction (minimizing out-of-sample error); BIC is optimized for model identification (selecting the true model). For small-sample degradation data, their agreement or disagreement is itself informative.
- Burnham, K. P., & Anderson, D. R. (2002). *Model selection and multimodel inference: A practical information-theoretic approach* (2nd ed.). Springer.

---

## Parametric Degradation Models

### Power Law: R/R₀ = a · D^b + c
The power law is the standard functional form for accelerating degradation processes in reliability engineering. The exponent b captures whether degradation is sub-linear (b < 1, decelerating) or super-linear (b > 1, accelerating). For membrane aging under radical oxidant generation, b > 1 is expected because radical production compounds with cumulative NOM-hypochlorite exposure.
- Lei, Y., Li, N., Guo, L., Li, N., Yan, T., & Lin, J. (2018). Machinery health prognostics: A systematic review from data acquisition to RUL prediction. *Mechanical Systems and Signal Processing*, 104, 799–834.

### Exponential Saturation: R/R₀ = a · (1 − exp(−bD)) + c
The exponential saturation model assumes degradation approaches an asymptotic limit. This may apply when degradable material is progressively consumed, leading to a self-limiting process. It is included to test whether the data supports saturating vs. accelerating degradation.

### Weibull: R/R₀ = a · (1 − exp(−(D/λ)^k)) + c
The Weibull model generalizes the exponential by allowing a shape parameter k that controls the degradation trajectory. k < 1 produces decelerating degradation; k > 1 produces accelerating degradation; k = 1 reduces to exponential. This flexibility makes it a strong candidate for model selection.
- Zhuang, L., et al. (2024). Remaining useful life prediction for two-phase degradation model based on reparameterized inverse Gaussian process. *European Journal of Operational Research*, 319(3), 877–890.

### Bootstrap Confidence Intervals
Nonparametric bootstrap resampling (200 iterations) provides 95% confidence intervals on both parameters and predictions without assuming a specific error distribution. This is appropriate for small samples where asymptotic normality of parameter estimates is unreliable.
- Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman and Hall/CRC.

---

## Gaussian Process Regression

### Why GP for This Problem
GP regression is the standard non-parametric Bayesian approach for small-sample regression with calibrated uncertainty quantification. It does not assume a fixed functional form, instead learning the function shape from data via a kernel (covariance function). For degradation curves with 7–14 data points, GP provides principled uncertainty estimates that parametric models cannot.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian processes for machine learning*. MIT Press.

### Kernel Selection Rationale
| Kernel | Smoothness | Why Tested |
|--------|-----------|------------|
| RBF (Squared Exponential) | C∞ (infinitely smooth) | Baseline; standard default |
| Matérn-3/2 | C¹ (once differentiable) | Degradation curves are continuous but may not be smooth — physically motivated |
| Matérn-5/2 | C² (twice differentiable) | Intermediate smoothness |
| Rational Quadratic | Mixture of scales | Captures multi-scale structure if present |

The Matérn class is preferred over RBF for physical degradation processes because RBF assumes unrealistically smooth functions. The smoothness parameter ν directly controls the differentiability of the GP sample paths.
- Rasmussen & Williams (2006), Chapter 4

### GP for Remaining Useful Life Prediction
GP regression has been applied to degradation and RUL prediction in analogous engineering contexts (bearing degradation, battery aging), demonstrating that composite kernels and proper hyperparameter optimization improve prediction accuracy and uncertainty calibration.
- Liu, K., et al. (2025). Early remaining useful life prediction for lithium-ion batteries using a Gaussian process regression model. *Batteries*, 11(6), 221.
- Wang, J., et al. (2015). Bearing remaining life prediction using Gaussian process regression with composite kernel functions. *Journal of Vibroengineering*, 17(5), 2354–2370.

---

## Cross-Validation

### Leave-One-Out Cross-Validation (LOO-CV)
LOO-CV is the standard cross-validation strategy for small datasets where k-fold CV would leave too few training samples per fold. Each data point is held out once, and prediction error is evaluated on the held-out point. For GP models, LOO-CV is computationally efficient because the LOO predictive distribution can be computed in closed form.
- Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions. *Journal of the Royal Statistical Society: Series B*, 36(2), 111–133.

### Uncertainty Calibration (Coverage Probability)
A well-calibrated 95% CI should contain ~95% of true values. We report empirical coverage at both 68% (±1σ) and 95% (±2σ) levels. Coverage < 80% at the 95% level indicates that uncertainty is underestimated — a critical diagnostic for deciding whether GP predictions are trustworthy.

---

## Cross-Condition Scaling

### Protein-Loading Dimensional Analysis
The observation that degradation curves collapse when dose is scaled by protein-type NOM loading rate represents a dimensional analysis (similarity law), analogous to Reynolds number scaling in fluid mechanics. This is presented as a physical hypothesis validated on 6 conditions, not as a statistically trained model. Formal transfer learning would require substantially more conditions.
- **Thesis reference:** Chapters 5 and 6 for the experimental evidence
