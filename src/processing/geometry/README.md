# Membrane Surface Geometry Processing

Pore radius, surface roughness, and pore number are derived from FESEM images
using custom image processing algorithms documented in the PhD thesis
(Appendix D.3).

The processing steps include:
- Contrast stretching with percentile-based intensity rescaling
- Adaptive thresholding for pore segmentation
- Morphological filtering to remove artifacts
- Connected component labeling for individual pore identification
- Pore radius calculation assuming circular geometry: r = sqrt(A/π)
- Sensitivity analysis over processing parameters (block size, offset, etc.)

The image processing code is not included in this repository. See:
- Thesis Appendix D.3 for full methodology and sensitivity analysis
- The processed output (per-dose geometry metrics) is used as supplementary
  features in the degradation modeling stage.
