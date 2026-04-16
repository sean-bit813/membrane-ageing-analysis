"""
TMP Signal Processing
======================
Converts raw DAQ voltage signals to transmembrane pressure (TMP).
Pipeline: raw CSV → parse → calibrate → denoise → drift-correct → TMP

References:
  - Voltage-to-TMP calibration: Appendix D, Eq. D.1 of thesis
  - Savitzky-Golay filter: Savitzky & Golay (1964), Anal. Chem. 36(8)
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, medfilt
from dataclasses import dataclass, field
from ...utils import get_logger

log = get_logger(__name__)


@dataclass
class TMPSignal:
    """Processed TMP time series container."""
    time_sec: np.ndarray
    tmp_pa: np.ndarray
    tmp_raw_pa: np.ndarray
    sampling_interval: float
    channel: str
    metadata: dict = field(default_factory=dict)


def load_raw_signal(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    log.info(f"Loaded {filepath}: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def voltage_to_tmp(voltage_v, calibration_factor=200.0, psi_to_pa=6894.76):
    """TMP(Pa) = Voltage(mV) × cal_factor × psi_to_pa. See thesis Eq. D.1."""
    return voltage_v * 1000.0 * calibration_factor * psi_to_pa


def denoise(signal, method, **kwargs):
    """Denoise a 1-D signal. Methods: savgol, moving_average, median, none."""
    if method == "none":
        return signal.copy()
    if method == "savgol":
        w = min(kwargs.get("savgol_window", 15), len(signal))
        if w % 2 == 0: w += 1
        p = kwargs.get("savgol_polyorder", 3)
        if w <= p: return signal.copy()
        return savgol_filter(signal, w, p)
    if method == "moving_average":
        w = kwargs.get("window", 5)
        kernel = np.ones(w) / w
        padded = np.pad(signal, (w // 2, w // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")[:len(signal)]
    if method == "median":
        k = kwargs.get("kernel_size", 5)
        if k % 2 == 0: k += 1
        return medfilt(signal, kernel_size=k)
    raise ValueError(f"Unknown denoise method: {method}")


def correct_drift(tmp, reference_channel):
    """Subtract linear drift estimated from a stable reference channel."""
    x = np.arange(len(reference_channel))
    coeffs = np.polyfit(x, reference_channel, 1)
    drift = np.polyval(coeffs, x) - reference_channel[0]
    ref_range = np.ptp(reference_channel)
    if ref_range > 0:
        drift_normalized = drift / ref_range * np.ptp(tmp) * 0.01
    else:
        drift_normalized = 0.0
    return tmp - drift_normalized


def process_signal(df, cfg, channel="ai0"):
    """Full signal processing pipeline for one TMP channel."""
    col = f"Voltage - Dev1_{channel}"
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found. Available: {list(df.columns)}")

    voltage = df[col].values
    time_raw = df["Time"].values
    time_sec = time_raw - time_raw[0]
    dt = np.median(np.diff(time_sec))

    log.info(f"Channel {channel}: {len(voltage)} samples, dt={dt:.1f}s, "
             f"duration={time_sec[-1]/3600:.1f}h")

    tmp_raw = voltage_to_tmp(voltage, cfg.get("calibration_factor", 200.0),
                              cfg.get("psi_to_pa", 6894.76))

    method = cfg.get("denoise_method", "savgol")
    tmp_dn = denoise(tmp_raw, method,
                     savgol_window=cfg.get("savgol_window", 15),
                     savgol_polyorder=cfg.get("savgol_polyorder", 3),
                     window=int(cfg.get("ma_window_sec", 30) / max(dt, 1)))
    log.info(f"  Denoised: method={method}")

    if cfg.get("enable_drift_correction", False):
        ref_ch = cfg.get("drift_reference_channel", "ai2")
        ref_col = f"Voltage - Dev1_{ref_ch}"
        if ref_col in df.columns:
            tmp_dn = correct_drift(tmp_dn, df[ref_col].values)
            log.info(f"  Drift-corrected via {ref_ch}")

    return TMPSignal(time_sec=time_sec, tmp_pa=tmp_dn, tmp_raw_pa=tmp_raw,
                     sampling_interval=dt, channel=channel,
                     metadata={"n_samples": len(voltage),
                               "duration_h": time_sec[-1] / 3600})
