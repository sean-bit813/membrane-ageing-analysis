"""
Pipeline Orchestrator
======================
Stage A: Raw signal → TMP → cycle detection → feature extraction
Stage B: Processed aging data → parametric + GP modeling → comparison
Stage C: Cross-condition protein-loading scaling analysis
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from .utils import load_config, get_logger, ensure_dir
from .processing.tmp.signal_processing import load_raw_signal, process_signal
from .processing.tmp.cycle_detection import detect_cycles
from .processing.tmp.feature_extraction import extract_all_features
from .models import fit_all_parametric, fit_all_gp, build_comparison_table, select_best_models

log = get_logger("pipeline")

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 180, "savefig.bbox": "tight"})


def run_signal_stage(cfg, raw_file):
    """Stage A: signal processing + cycle detection + feature extraction."""
    log.info("=" * 55)
    log.info("STAGE A: Signal Processing & Cycle Detection")
    log.info("=" * 55)

    df = load_raw_signal(raw_file)
    signal = process_signal(df, cfg["signal"], channel="ai0")
    cycles = detect_cycles(signal, cfg["cycle_detection"])
    df_cycles, df_cwr = extract_all_features(signal, cycles, cfg["feature_extraction"])

    out = ensure_dir(cfg["paths"]["processed_data_dir"])
    df_cycles.to_csv(out / "cycle_features.csv", index=False)
    if len(df_cwr) > 0:
        df_cwr.to_csv(out / "cwr_measurements.csv", index=False)
    log.info(f"Saved → {out}")

    # Figure
    fig_dir = ensure_dir(cfg["paths"]["figures_dir"])
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    t_h = signal.time_sec / 3600
    axes[0].plot(t_h, signal.tmp_raw_pa, lw=0.3, alpha=0.4, color="#999", label="Raw")
    axes[0].plot(t_h, signal.tmp_pa, lw=0.7, color="#d62728", label="Denoised")
    for c in cycles:
        if c.is_valid:
            axes[0].axvline(c.start_sec/3600, color="#2ca02c", alpha=0.2, lw=0.5)
    # Mark CWR measurement windows
    if len(df_cwr) > 0:
        for _, row in df_cwr.iterrows():
            t0 = row["window_start_sec"] / 3600
            axes[0].axvspan(t0, t0 + row["window_duration_sec"]/3600,
                           alpha=0.15, color="#ff7f0e", label="")
        axes[0].axvspan(0, 0, alpha=0.15, color="#ff7f0e", label="CWR window")
    axes[0].set_ylabel("TMP (Pa)")
    axes[0].set_title("Signal Processing & Cycle Detection")
    axes[0].legend(fontsize=9)

    if len(df_cwr) > 0 and "resistance" in df_cwr.columns:
        valid_cwr = df_cwr.dropna(subset=["resistance"])
        axes[1].plot(valid_cwr["window_start_sec"]/3600, valid_cwr["resistance"],
                     "s-", color="#1f77b4", ms=6, label="CWR measurement")
        axes[1].set_ylabel("Clean Water Resistance (1/m)")
        axes[1].set_xlabel("Time (hours)")
        axes[1].set_title("Clean Water Resistance (from dedicated DI water windows)")
        axes[1].legend(fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "No CWR windows detected", transform=axes[1].transAxes,
                     ha="center", fontsize=12, color="#999")
        axes[1].set_xlabel("Time (hours)")

    plt.tight_layout()
    plt.savefig(fig_dir / "01_signal_processing.png")
    plt.close()
    log.info(f"Figure → {fig_dir / '01_signal_processing.png'}")
    return signal, cycles, (df_cycles, df_cwr)


def run_modeling_stage(cfg, data_file):
    """Stage B: fit parametric + GP models, compare."""
    log.info("=" * 55)
    log.info("STAGE B: Degradation Modeling")
    log.info("=" * 55)

    df = pd.read_csv(data_file)
    fig_dir = ensure_dir(cfg["paths"]["figures_dir"])
    tbl_dir = ensure_dir(cfg["paths"]["tables_dir"])
    conditions = df["condition"].unique()
    log.info(f"Loaded {len(df)} rows, {len(conditions)} conditions")

    all_comp = []
    for cond in conditions:
        sub = df[df["condition"] == cond].sort_values("cumulative_dose_ppmh")
        dose, resp = sub["cumulative_dose_ppmh"].values, sub["resistance_normalized"].values
        if len(dose) < 4:
            log.warning(f"  {cond}: {len(dose)} points — skipping")
            continue
        log.info(f"\n--- {cond} ({len(dose)} pts) ---")

        mcfg = cfg["modeling"]
        par = fit_all_parametric(dose, resp, mcfg["parametric"]["models"],
                                  mcfg["parametric"]["n_bootstrap"])
        gpr = fit_all_gp(dose, resp, mcfg["gp"]["kernels"],
                          mcfg["gp"]["normalize_x"], mcfg["gp"]["n_restarts"])
        comp = build_comparison_table(par, gpr)
        comp["condition"] = cond
        all_comp.append(comp)
        best = select_best_models(comp)

        # Figure
        _plot_fits(cond, dose, resp, par, gpr, best, fig_dir)

    if all_comp:
        full = pd.concat(all_comp, ignore_index=True)
        full.to_csv(tbl_dir / "model_comparison.csv", index=False)
        log.info(f"\nSaved table → {tbl_dir / 'model_comparison.csv'}")
        return full
    return pd.DataFrame()


def _plot_fits(cond, dose, resp, par, gpr, best, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {"power_law": "#d62728", "exponential": "#1f77b4", "weibull": "#2ca02c"}

    ax = axes[0]
    ax.scatter(dose/1e6, resp, color="black", s=50, zorder=10, edgecolors="white", lw=1.5, label="Observed")
    for name, r in par.items():
        c = colors.get(name, "#777")
        ax.plot(r.dose_grid/1e6, r.pred_grid, color=c, lw=2,
                label=f"{name} (AICc={r.aicc:.0f})", alpha=0.85)
        ax.fill_between(r.dose_grid/1e6, r.pred_ci_low, r.pred_ci_high, color=c, alpha=0.1)
    ax.set_xlabel("Cumulative Dose (×10⁶ ppm·h)")
    ax.set_ylabel("R / R₀")
    ax.set_title(f"{cond} — Parametric Models")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(dose/1e6, resp, color="black", s=50, zorder=10, edgecolors="white", lw=1.5, label="Observed")
    gp_name = best.get("best_gp", list(gpr.keys())[0]).replace("GP_", "")
    g = gpr.get(gp_name, list(gpr.values())[0])
    ax.plot(g.dose_grid/1e6, g.pred_mean, color="#ff7f0e", lw=2, label=f"GP ({g.kernel_name})")
    ax.fill_between(g.dose_grid/1e6, g.pred_mean-2*g.pred_std, g.pred_mean+2*g.pred_std,
                    color="#ff7f0e", alpha=0.15, label="95% CI")
    ax.fill_between(g.dose_grid/1e6, g.pred_mean-g.pred_std, g.pred_mean+g.pred_std,
                    color="#ff7f0e", alpha=0.25, label="68% CI")
    ax.set_xlabel("Cumulative Dose (×10⁶ ppm·h)")
    ax.set_ylabel("R / R₀")
    ax.set_title(f"{cond} — GP (LOO-RMSE={g.loo_rmse:.4f})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / f"02_models_{cond.replace('/', '_')}.png")
    plt.close()


def run_scaling_stage(cfg, data_file):
    """Stage C: protein-loading scaling analysis."""
    log.info("=" * 55)
    log.info("STAGE C: Cross-Condition Scaling Analysis")
    log.info("=" * 55)

    df = pd.read_csv(data_file)
    fig_dir = ensure_dir(cfg["paths"]["figures_dir"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    cmap = plt.cm.tab10
    conds = [c for c in df["condition"].unique()
             if df[df["condition"]==c]["protein_loading"].iloc[0] > 0]

    for i, cond in enumerate(conds):
        s = df[df["condition"]==cond].sort_values("cumulative_dose_ppmh")
        axes[0].plot(s["cumulative_dose_ppmh"]/1e6, s["resistance_normalized"],
                     "o-", color=cmap(i), label=cond, ms=5)
    axes[0].set_xlabel("Cumulative Dose (×10⁶ ppm·h)")
    axes[0].set_ylabel("R / R₀")
    axes[0].set_title("(a) Raw Dose Space")
    axes[0].legend(fontsize=8)

    from scipy.stats import pearsonr
    all_x, all_y = [], []
    for i, cond in enumerate(conds):
        s = df[df["condition"]==cond].sort_values("cumulative_dose_ppmh")
        adj = s["cumulative_dose_ppmh"].values * s["protein_loading"].iloc[0]
        axes[1].plot(adj/1e3, s["resistance_normalized"], "o-", color=cmap(i), label=cond, ms=5)
        all_x.extend(adj); all_y.extend(s["resistance_normalized"])

    ax, ay = np.array(all_x), np.array(all_y)
    mask = np.isfinite(ax) & np.isfinite(ay) & (ax > 0)
    if mask.sum() > 3:
        r, p = pearsonr(ax[mask], ay[mask])
        axes[1].text(0.05, 0.92, f"r = {r:.3f}", transform=axes[1].transAxes,
                     fontsize=11, fontweight="bold",
                     bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        log.info(f"Protein-adjusted correlation: r={r:.3f}, p={p:.2e}")

    axes[1].set_xlabel("Protein-Adjusted Dose (×10³)")
    axes[1].set_ylabel("R / R₀")
    axes[1].set_title("(b) Protein-Loading Scaled")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "03_scaling_analysis.png")
    plt.close()
    log.info(f"Figure → {fig_dir / '03_scaling_analysis.png'}")


def run_pipeline(config_path=None):
    """Run all stages. Requires data files to be present."""
    cfg = load_config(config_path)
    log.info("=" * 55)
    log.info("  Membrane Aging Prediction Pipeline")
    log.info("=" * 55)

    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    proc_dir = Path(cfg["paths"]["processed_data_dir"])

    # Stage A: look for raw signal files
    raw_csvs = sorted(raw_dir.glob("*.csv"))
    if raw_csvs:
        log.info(f"Found {len(raw_csvs)} raw file(s) in {raw_dir}")
        for f in raw_csvs:
            run_signal_stage(cfg, str(f))
    else:
        log.info(f"No raw CSVs in {raw_dir} — skipping Stage A")

    # Stage B: look for processed aging data
    aging_file = proc_dir / "aging_data.csv"
    if aging_file.exists():
        run_modeling_stage(cfg, str(aging_file))
    else:
        log.info(f"No {aging_file} — skipping Stage B")
        log.info("  (Create aging_data.csv per format in data/DATA_FORMAT.md)")

    # Stage C
    if aging_file.exists():
        run_scaling_stage(cfg, str(aging_file))

    log.info("\n" + "=" * 55)
    log.info("  Pipeline complete.")
    log.info(f"  Figures → {cfg['paths']['figures_dir']}/")
    log.info(f"  Tables  → {cfg['paths']['tables_dir']}/")
    log.info("=" * 55)
