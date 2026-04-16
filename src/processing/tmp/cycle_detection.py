"""
Filtration Cycle Detection
===========================
Automatically segments TMP signal into filtration–backwash–CIP cycles.

Methods:
  - gradient: robust to varying TMP amplitude across aging stages
  - threshold: simple baseline, fails when TMP range changes
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List
from .signal_processing import TMPSignal
from ...utils import get_logger

log = get_logger(__name__)


@dataclass
class CyclePhase:
    name: str
    start_idx: int
    end_idx: int
    start_sec: float
    end_sec: float
    duration_sec: float


@dataclass
class FiltrationCycle:
    cycle_id: int
    start_idx: int
    end_idx: int
    start_sec: float
    end_sec: float
    duration_sec: float
    phases: List[CyclePhase] = field(default_factory=list)
    is_valid: bool = True
    rejection_reason: str = ""


def detect_cycles_gradient(signal: TMPSignal, cfg: dict) -> List[FiltrationCycle]:
    """Detect cycles via normalized TMP gradient. See README for details."""
    tmp = signal.tmp_pa
    dt = signal.sampling_interval
    n = len(tmp)
    min_dur = cfg.get("min_cycle_duration_sec", 2400)
    max_dur = cfg.get("max_cycle_duration_sec", 7200)

    grad = np.gradient(tmp)
    tmp_range = np.percentile(tmp, 99) - np.percentile(tmp, 1)
    if tmp_range == 0: tmp_range = 1.0
    grad_norm = grad / tmp_range

    IDLE, FILT, BW, CIP = 0, 1, 2, 3
    state = IDLE
    cycle_starts, bw_starts = [], []
    min_phase = 10

    for i in range(1, n):
        if state == IDLE:
            if grad_norm[i] > 0.005 and tmp[i] > np.percentile(tmp, 10):
                if i + 5 < n and np.mean(grad_norm[i:i+5]) > 0.002:
                    state = FILT
                    cycle_starts.append(i)
        elif state == FILT:
            if grad_norm[i] < -0.02 and i - cycle_starts[-1] > min_phase:
                state = BW
                bw_starts.append(i)
        elif state == BW:
            if abs(tmp[i]) < tmp_range * 0.05 and abs(grad_norm[i]) < 0.001:
                state = CIP
        elif state == CIP:
            state = IDLE

    log.info(f"Gradient method: {len(cycle_starts)} filtration onsets, "
             f"{len(bw_starts)} backwash onsets")

    cycles = []
    for idx, (cs, bs) in enumerate(zip(cycle_starts, bw_starts)):
        ce = cycle_starts[idx + 1] if idx + 1 < len(cycle_starts) else n - 1
        dur = (ce - cs) * dt
        t = signal.time_sec

        cycle = FiltrationCycle(
            cycle_id=idx, start_idx=cs, end_idx=ce,
            start_sec=t[cs], end_sec=t[min(ce, n-1)], duration_sec=dur)

        if dur < min_dur:
            cycle.is_valid = False
            cycle.rejection_reason = f"short ({dur:.0f}s)"
        elif dur > max_dur:
            cycle.is_valid = False
            cycle.rejection_reason = f"long ({dur:.0f}s)"

        # Segment phases
        bw_dur = cfg.get("phase_durations", {}).get("backwash", 600)
        bw_end = min(bs + int(bw_dur / dt), ce)
        cycle.phases = [
            CyclePhase("filtration", cs, bs, t[cs], t[bs], (bs-cs)*dt),
            CyclePhase("backwash", bs, bw_end, t[bs], t[min(bw_end,n-1)], (bw_end-bs)*dt),
        ]
        if bw_end < ce:
            cycle.phases.append(
                CyclePhase("cip", bw_end, ce, t[bw_end], t[min(ce,n-1)], (ce-bw_end)*dt))
        cycles.append(cycle)

    valid = sum(1 for c in cycles if c.is_valid)
    log.info(f"Built {len(cycles)} cycles ({valid} valid, {len(cycles)-valid} rejected)")
    return cycles


def detect_cycles(signal: TMPSignal, cfg: dict) -> List[FiltrationCycle]:
    method = cfg.get("method", "gradient")
    if method == "gradient":
        return detect_cycles_gradient(signal, cfg)
    raise ValueError(f"Unknown detection method: {method}")
