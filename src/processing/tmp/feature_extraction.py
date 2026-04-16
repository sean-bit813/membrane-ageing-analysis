"""
Feature Extraction from Filtration Cycles
==========================================
Extracts per-cycle metrics:
  - Intra-cycle fouling rate (TMP rise slope during filtration)
  - Backwash recovery efficiency

And from dedicated CWR measurement windows (every N cycles):
  - Clean water resistance: the aging experiment is paused, DI water
    is filtered through the membrane for ~30 minutes, and the
    stabilized TMP is recorded as the CWR reading.

CWR calculation: R = TMP / (mu * J)   — see thesis Eq. D.2

NOTE: CWR is NOT extracted from every cycle's CIP phase. It is a
separate measurement event at specific cumulative dose intervals.
See thesis Section 3.3.4 and Figure 3.5.
"""
import numpy as np
import pandas as pd
from typing import List
from scipy.stats import linregress
from .signal_processing import TMPSignal
from .cycle_detection import FiltrationCycle
from ...utils import get_logger

log = get_logger(__name__)


# =================================================================
# Clean Water Resistance — from dedicated measurement windows
# =================================================================

def detect_cwr_windows(signal: TMPSignal, cfg: dict,
                       cycles=None) -> List[dict]:
    """
    Detect dedicated clean water resistance measurement windows.

    CWR windows differ from normal filtration cycles:
    - Occur BETWEEN cycles (not during filtration or CIP)
    - TMP is positive but lower than feed-water peaks
    - TMP rises initially then STABILIZES (unlike filtration which
      keeps rising due to fouling)
    - Duration ~30 minutes

    Detection:
    1. Find contiguous positive-TMP regions below filtration peak level
    2. Reject segments that overlap with known cycle boundaries
    3. Verify stabilization: gradient in late portion ≈ 0
    """
    tmp = signal.tmp_pa
    dt = signal.sampling_interval
    n = len(tmp)

    cwr_dur = cfg.get("cwr_measurement_duration_sec", 1800)
    min_samples = int(cwr_dur * 0.6 / dt)
    max_samples = int(cwr_dur * 2.0 / dt)

    positive_tmp = tmp[tmp > 0]
    if len(positive_tmp) < 10:
        log.warning("Not enough positive TMP to detect CWR windows")
        return []
    filt_peak = np.percentile(positive_tmp, 90)

    cwr_ceiling = filt_peak * cfg.get("cwr_peak_fraction", 0.35)
    cwr_floor = filt_peak * 0.02

    in_cwr = (tmp > cwr_floor) & (tmp < cwr_ceiling)
    kernel = np.ones(5) / 5
    in_cwr_smooth = np.convolve(in_cwr.astype(float), kernel, mode="same") > 0.5

    # Build set of indices occupied by detected cycles (to exclude overlaps)
    cycle_indices = set()
    if cycles is not None:
        for c in cycles:
            for idx in range(c.start_idx, min(c.end_idx + 1, n)):
                cycle_indices.add(idx)

    candidates = []
    i = 0
    while i < n:
        if in_cwr_smooth[i]:
            start = i
            while i < n and in_cwr_smooth[i]:
                i += 1
            end = i
            if min_samples <= (end - start) <= max_samples:
                candidates.append((start, end))
        i += 1

    windows = []
    for start, end in candidates:
        seg_indices = set(range(start, end))

        # Reject if >20% of the segment overlaps with a detected cycle
        if cycles is not None:
            overlap = len(seg_indices & cycle_indices)
            if overlap > 0.2 * len(seg_indices):
                continue

        # Verify stabilization: late portion should be approximately flat
        seg = tmp[start:end]
        if len(seg) < 20:
            continue
        grad = np.gradient(seg)
        third = len(grad) // 3
        late_grad = np.mean(np.abs(grad[-third:]))
        # Accept if late gradient is small relative to signal magnitude
        seg_range = np.ptp(seg)
        flat_threshold = max(seg_range * 0.02, np.std(seg) * 0.5)
        if late_grad < flat_threshold:
            windows.append({
                "start_idx": start, "end_idx": end,
                "start_sec": signal.time_sec[start],
                "end_sec": signal.time_sec[min(end - 1, n - 1)],
                "duration_sec": (end - start) * dt,
            })

    log.info(f"Detected {len(windows)} CWR measurement windows "
             f"(from {len(candidates)} candidates)")
    return windows


def extract_cwr_from_window(signal: TMPSignal, window: dict, cfg: dict) -> dict:
    """
    Extract clean water resistance from one CWR measurement window.

    1. Skip initial transient (membrane re-pressurizing with DI water)
    2. Rolling-variance stability check on remainder
    3. Median of stable segment → TMP_stable
    4. R = TMP_stable / (mu * J)
    """
    tmp = signal.tmp_pa
    dt = signal.sampling_interval
    segment = tmp[window["start_idx"]:window["end_idx"]]

    if len(segment) < 10:
        return _empty_cwr(window)

    skip = int(cfg.get("cwr_stabilization_skip_sec", 300) / dt)
    if skip >= len(segment) - 5:
        skip = len(segment) // 4
    stable_seg = segment[skip:]

    if len(stable_seg) < 5:
        return _empty_cwr(window)

    win_size = max(3, int(cfg.get("cwr_stability_window_sec", 120) / dt))
    win_size = min(win_size, len(stable_seg))
    rolling_var = pd.Series(stable_seg).rolling(win_size, center=True).var().values

    threshold = cfg.get("cwr_stability_threshold", 0.0005)
    scale = max(np.std(stable_seg), 1.0)
    stable_mask = np.nan_to_num(rolling_var, nan=np.inf) < (threshold * scale ** 2)
    n_stable = int(np.sum(stable_mask))

    if n_stable > 3:
        tmp_stable = np.median(stable_seg[stable_mask])
    else:
        last = stable_seg[int(len(stable_seg) * 0.7):]
        tmp_stable = np.median(last)
        n_stable = len(last)

    flux_ms = cfg.get("permeate_flux_lmh", 50.0) / 3600.0 / 1000.0
    mu = cfg.get("water_viscosity_pa_s", 0.001002)
    resistance = abs(tmp_stable) / (mu * flux_ms) if flux_ms > 0 else np.nan

    return {
        "window_start_sec": window["start_sec"],
        "window_duration_sec": window["duration_sec"],
        "tmp_stable_pa": tmp_stable,
        "resistance": resistance,
        "n_stable_points": n_stable,
    }


def _empty_cwr(window):
    return {"window_start_sec": window["start_sec"],
            "window_duration_sec": window["duration_sec"],
            "tmp_stable_pa": np.nan, "resistance": np.nan, "n_stable_points": 0}


# =================================================================
# Per-cycle features
# =================================================================

def extract_fouling_rate(signal, cycle, cfg):
    """Linear fit of TMP rise during filtration phase."""
    filt = [p for p in cycle.phases if p.name == "filtration"]
    if not filt:
        return {"fouling_rate_pa_per_s": np.nan, "fouling_r2": np.nan}
    tmp_f = signal.tmp_pa[filt[0].start_idx:filt[0].end_idx]
    if len(tmp_f) < 5:
        return {"fouling_rate_pa_per_s": np.nan, "fouling_r2": np.nan}
    lo, hi = cfg.get("fouling_rate_fit_window_pct", [0.2, 0.9])
    n = len(tmp_f)
    seg = tmp_f[int(n * lo):int(n * hi)]
    t_seg = np.arange(len(seg)) * signal.sampling_interval
    if len(seg) < 3:
        return {"fouling_rate_pa_per_s": np.nan, "fouling_r2": np.nan}
    slope, _, r, _, _ = linregress(t_seg, seg)
    return {"fouling_rate_pa_per_s": slope, "fouling_r2": r ** 2}


def extract_backwash_efficiency(signal, cycle):
    filt = [p for p in cycle.phases if p.name == "filtration"]
    if not filt:
        return {"backwash_efficiency": np.nan}
    f = filt[0]
    end_sl = signal.tmp_pa[max(f.start_idx, f.end_idx - 10):f.end_idx]
    start_sl = signal.tmp_pa[f.start_idx:min(f.start_idx + 10, f.end_idx)]
    te = np.median(end_sl) if len(end_sl) > 0 else np.nan
    ts = np.median(start_sl) if len(start_sl) > 0 else np.nan
    return {"backwash_efficiency": (te - ts) / te if te and te != 0 else np.nan}


# =================================================================
# Orchestration
# =================================================================

def extract_all_features(signal, cycles, cfg):
    """
    Returns two DataFrames:
      - cycle_features: per-cycle fouling rate, backwash efficiency
      - cwr_features: per-CWR-window clean water resistance
    """
    cycle_records = []
    for c in cycles:
        row = {"cycle_id": c.cycle_id, "start_sec": c.start_sec,
               "duration_sec": c.duration_sec, "is_valid": c.is_valid}
        if c.is_valid:
            row.update(extract_fouling_rate(signal, c, cfg))
            row.update(extract_backwash_efficiency(signal, c))
        cycle_records.append(row)
    df_cycles = pd.DataFrame(cycle_records)

    cwr_windows = detect_cwr_windows(signal, cfg, cycles=cycles)
    cwr_records = [extract_cwr_from_window(signal, w, cfg) for w in cwr_windows]
    df_cwr = pd.DataFrame(cwr_records) if cwr_records else pd.DataFrame()

    n_valid = df_cycles["is_valid"].sum()
    log.info(f"Extracted: {n_valid} valid cycles, {len(cwr_records)} CWR measurements")
    if len(df_cwr) > 0 and "resistance" in df_cwr.columns:
        v = df_cwr.dropna(subset=["resistance"])
        if len(v) > 0:
            log.info(f"  CWR range: {v['resistance'].min():.2e} — {v['resistance'].max():.2e}")

    return df_cycles, df_cwr
