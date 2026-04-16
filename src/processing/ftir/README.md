# FTIR Spectral Processing

Processing of ATR-FTIR spectra is performed using a separate pipeline documented
in the PhD thesis (Appendix D.1). The pipeline consists of three stages:

1. **EMSC normalization** — Extended Multiplicative Signal Correction normalizes
   aged membrane spectra against a virgin membrane reference spectrum to remove
   baseline variations and scaling differences.

2. **Differential spectra generation** — Delta spectra (aged − virgin) isolate
   chemical changes due to aging. Second derivatives via Savitzky-Golay filtering
   enhance peak resolution.

3. **Peak analysis** — Automated detection of significant peaks, area calculation
   via numerical integration, and identification of reliable aging indicators
   (high area, low replicate variability) at wavenumbers 884, 1151, 1169, and
   1289 cm⁻¹.

The FTIR processing code is not included in this repository. See:
- Thesis Appendix D.1 for full code listing and method description
- Afseth & Kohler (2012) for EMSC methodology
- The processed output (FTIR peak areas vs. cumulative dose) is the input
  to the degradation modeling stage of this pipeline.
