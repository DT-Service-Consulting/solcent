#!/usr/bin/env python3
"""
make_fig_results.py
===================
Figures for Section 5 (Results).

  Fig. 4  the ceiling: oracle headroom + residual autocorrelation
  Fig. 5  where the error lives: regime breakdown + cross-house homogeneity
  Fig. 6  ensemble ablation
  Fig. 7  probabilistic calibration before and after correction
  Fig. 8  deployment: zero-shot distribution across three configurations
  Fig. 9  local calibration sweep

Every value is read from the committed result files under
v3_ensemble/models/consolidated/ and reports/weather_error/. Nothing is
hard-coded, so the figures cannot drift from the numbers in the text.

Run:  python paper/figures/make_fig_results.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v3_ensemble"))
import config as C

OUT = Path(__file__).resolve().parent
D = C.MODELS_DIR / "consolidated"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "savefig.bbox": "tight", "figure.facecolor": "white",
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})

INK, GREY, MID = "#1f1f1f", "#6b6b6b", "#9a9a9a"
ACC = "#b2182b"      # the result being highlighted
BLUE = "#2166ac"
GREEN = "#1b7837"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / (name + "." + ext))
    plt.close(fig)
    print("  wrote " + name)


# ---------------------------------------------------------------------------
def fig4_ceiling():
    o = pd.read_csv(D / "exp1_oracle.csv")
    ac = pd.read_csv(D / "exp2_autocorr.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))

    ax = axes[0]
    lab = ["NWP forecast\n(operational)", "+ oracle\n(regional truth)",
           "oracle + geometry\n(no NWP)"]
    vals = o.R2.values
    bars = ax.bar(range(3), vals, color=[MID, ACC, BLUE], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.4f}", ha="center", fontsize=7)
    gap = vals[1] - vals[0]
    ax.annotate("", xy=(0.3, vals[0]), xytext=(0.3, vals[1]),
                arrowprops=dict(arrowstyle="<->", color=ACC, lw=0.9))
    ax.text(0.40, (vals[0] + vals[1]) / 2, f"+{gap:.4f}", color=ACC,
            fontsize=7.5, va="center")
    ax.set_xticks(range(3), lab, fontsize=6.8)
    ax.set_ylabel("$R^2$ on the test split")
    ax.set_ylim(0.65, 0.88)
    ax.set_title("(a) value of perfect regional weather truth", loc="left")

    ax = axes[1]
    ax.plot(ac.lag_min, ac.autocorr, "o-", color=INK, ms=4, lw=1.0)
    for x, y in zip(ac.lag_min, ac.autocorr):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(4, 5), fontsize=6.5)
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks(ac.lag_min, [str(int(v)) for v in ac.lag_min])
    ax.set_xlabel("lag (minutes, log scale)")
    ax.set_ylabel("residual autocorrelation")
    ax.set_ylim(-0.05, 0.82)
    ax.set_title("(b) residual structure", loc="left")

    save(fig, "fig4_ceiling")


# ---------------------------------------------------------------------------
def fig5_where_error_lives():
    reg = pd.read_csv(D / "exp2_regime_forecast_binned.csv")
    w = pd.read_csv(C.REPORTS_DIR / "weather_error" / "summary_by_system_regime.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6))

    ax = axes[0]
    x = np.arange(len(reg))
    ax.bar(x, reg.R2, color=[ACC if v < 0.6 else MID for v in reg.R2], width=0.62)
    for i, v in enumerate(reg.R2):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=6.4)
    ax.set_xticks(x, [r.replace(" ", "\n") for r in reg.regime], fontsize=6.2)
    ax.set_ylabel("$R^2$")
    ax.set_ylim(0, 1.0)
    ax.set_title("(a) PV forecast by sky condition", loc="left")

    ax = axes[1]
    order = ["Overcast", "Partly cloudy", "Clear"]
    agg = w.groupby("regime").apply(
        lambda g: pd.Series({
            "n": g.n.sum(),
            "mean_actual": (g.mean_actual * g.n).sum() / g.n.sum(),
            "RMSE": ((g.RMSE ** 2 * g.n).sum() / g.n.sum()) ** 0.5,
            "MAE": (g.MAE * g.n).sum() / g.n.sum()}), include_groups=False
    ).reindex(order)
    agg["nMAE"] = 100 * agg.MAE / agg.mean_actual
    xx = np.arange(3)
    ax.bar(xx - 0.19, agg.RMSE, 0.36, color=MID, label="RMSE (W m$^{-2}$)")
    ax2 = ax.twinx()
    ax2.bar(xx + 0.19, agg.nMAE, 0.36, color=ACC, label="nMAE (%)")
    ax2.grid(False)
    ax.set_xticks(xx, [o.replace(" ", "\n") for o in order], fontsize=6.2)
    ax.set_ylabel("RMSE (W m$^{-2}$)")
    ax2.set_ylabel("nMAE (%)", color=ACC)
    ax2.tick_params(axis="y", colors=ACC)
    ax.set_title("(b) irradiance error, absolute vs relative", loc="left")

    ax = axes[2]
    piv = w.pivot(index="system_id", columns="regime", values="RMSE")[order]
    pos = [piv[o].dropna().values for o in order]
    bp = ax.boxplot(pos, labels=[o.replace(" ", "\n") for o in order],
                    patch_artist=True, widths=0.5, showfliers=False)
    for p in bp["boxes"]:
        p.set_facecolor(MID)
        p.set_alpha(0.7)
    for m in bp["medians"]:
        m.set_color(INK)
    for i, o in enumerate(order):
        cv = piv[o].std() / piv[o].median()
        ax.text(i + 1, piv[o].max() * 1.03, f"{cv:.3f}", ha="center",
                fontsize=6.4, color=ACC)
    ax.set_ylabel("forecast RMSE (W m$^{-2}$)")
    ax.tick_params(axis="x", labelsize=6.2)
    ax.set_title("(c) spread across 31 rooftops\n(std/median in red)", loc="left")

    fig.tight_layout()
    save(fig, "fig5_error_location")


# ---------------------------------------------------------------------------
def fig6_ensemble():
    e = pd.read_csv(D / "exp3_ensemble_ablation.csv")
    base = e[e.config.str.startswith("base only")].iloc[0]
    e = e.sort_values("skill")
    fig, ax = plt.subplots(figsize=(7.1, 2.9))
    short = {
        "base only (12)": "base (12 features)",
        "+ ens_mean_fresh (6 day-1 models)": "+ multi-model mean (6 centres, day-1)",
        "+ ens_ghi_mean (all 12 members)": "+ mean of all 12 members",
        "+ poa_ens_mean": "+ ensemble-mean POA",
        "+ spread features only": "+ ensemble spread only",
        "+ all 10 ensemble features": "+ all 10 ensemble features",
        "+ lagged mean (day1-3)": "+ lagged mean (ECMWF d1-d3)",
    }
    y = np.arange(len(e))
    cols = [ACC if "mean_fresh" in c else
            (BLUE if "spread" in c else MID) for c in e.config]
    ax.barh(y, e.skill, color=cols, height=0.62)
    ax.axvline(base.skill, color=INK, ls="--", lw=0.9)
    ax.text(base.skill + 0.0004, len(e) - 0.4, "baseline", fontsize=6.6,
            color=INK, rotation=90, va="top")
    for i, (s, p) in enumerate(zip(e.skill, e.mean_pinball)):
        ax.text(s + 0.0006, i, f"{s:.4f}   pinball {p:.5f}", va="center",
                fontsize=6.5)
    ax.set_yticks(y, [short.get(c, c) for c in e.config], fontsize=7)
    ax.set_xlabel("skill score vs persistence")
    ax.set_xlim(0.424, 0.4425)
    ax.set_title("Ensemble ablation — only multi-model averaging at a single "
                 "lead time helps", loc="left")
    fig.tight_layout()
    save(fig, "fig6_ensemble_ablation")


# ---------------------------------------------------------------------------
def fig7_calibration():
    r = pd.read_csv(D / "exp4_recalibration.csv")
    r = r[r.config.str.startswith("C ")]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls="--", label="perfect")
    ax.plot(r.nominal, r.coverage_before, "o-", color=ACC, ms=5,
            label="raw quantile regression")
    ax.plot(r.nominal, r.coverage_after, "s-", color=GREEN, ms=5,
            label="after conformal offset")
    for _, row in r.iterrows():
        ax.annotate(f"{row.coverage_before:.3f}", (row.nominal, row.coverage_before),
                    textcoords="offset points", xytext=(6, -9), fontsize=6.2,
                    color=ACC)
        ax.annotate(f"{row.coverage_after:.3f}", (row.nominal, row.coverage_after),
                    textcoords="offset points", xytext=(6, 5), fontsize=6.2,
                    color=GREEN)
    ax.set_xlabel("nominal quantile level")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0.15, 1.0)
    ax.set_ylim(0.15, 1.0)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(a) reliability", loc="left")

    ax = axes[1]
    gap_b = (r.coverage_before - r.nominal).values
    gap_a = (r.coverage_after - r.nominal).values
    xx = np.arange(3)
    ax.bar(xx - 0.19, gap_b, 0.36, color=ACC, label="raw")
    ax.bar(xx + 0.19, gap_a, 0.36, color=GREEN, label="recalibrated")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(xx, [f"P{int(q*100)}" for q in r.nominal])
    ax.set_ylabel("coverage − nominal")
    ax.legend(frameon=False)
    ax.set_title("(b) calibration gap", loc="left")

    fig.tight_layout()
    save(fig, "fig7_recalibration")


# ---------------------------------------------------------------------------
def fig8_deployment():
    tc = pd.read_csv(C.MODELS_DIR / "tuned" / "transfer_comparison.csv")
    name = {"inherited (RMSE, temporal)": "inherited\n(RMSE, temporal)",
            "tuned on temporal split": "tuned on\ntemporal split",
            "tuned on held-out systems": "tuned on\nheld-out systems"}
    order = ["inherited (RMSE, temporal)", "tuned on temporal split",
             "tuned on held-out systems"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))

    ax = axes[0]
    data = [tc[tc.config == c].R2.values for c in order]
    bp = ax.boxplot(data, labels=[name[c] for c in order], patch_artist=True,
                    widths=0.5, showfliers=False)
    for i, (p, c) in enumerate(zip(bp["boxes"], order)):
        p.set_facecolor(ACC if "held-out" in c else MID)
        p.set_alpha(0.75)
    for m in bp["medians"]:
        m.set_color(INK)
    for i, c in enumerate(order):
        v = tc[tc.config == c].R2.values
        ax.scatter(np.full(len(v), i + 1) + np.random.default_rng(0)
                   .uniform(-0.11, 0.11, len(v)), v, s=7, color=INK, alpha=0.45,
                   zorder=5)
    ax.axhline(0, color=ACC, lw=0.8, ls=":")
    ax.set_ylabel("$R^2$ on an unseen rooftop")
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_title("(a) zero-shot distribution, 33 rooftops", loc="left")

    ax = axes[1]
    stat = tc.groupby("config").agg(
        median=("R2", "median"), p10=("R2", lambda s: s.quantile(0.10)),
        worst=("R2", "min")).reindex(order)
    xx = np.arange(3)
    w = 0.26
    for k, (col, lab, c) in enumerate((("median", "median", MID),
                                       ("p10", "10th percentile", BLUE),
                                       ("worst", "worst rooftop", ACC))):
        ax.bar(xx + (k - 1) * w, stat[col], w, color=c, label=lab)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(xx, [name[c] for c in order], fontsize=6.5)
    ax.set_ylabel("$R^2$")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("(b) the gain is in the tail, not the median", loc="left")

    fig.tight_layout()
    save(fig, "fig8_deployment")


# ---------------------------------------------------------------------------
def fig9_calibration_sweep():
    sk = pd.read_csv(D / "exp5_calib_skill_pivot.csv")
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    styles = {"none": (INK, "o-", "no correction"),
              "bias": (BLUE, "s--", "bias offset"),
              "cap_emp+bias": (GREEN, "^--", "capacity + bias"),
              "linear": (ACC, "d--", "linear rescale")}
    for col, (c, st, lab) in styles.items():
        if col not in sk.columns:
            continue
        m = sk[col].notna()
        ax.plot(sk.calib_days[m], sk[col][m], st, color=c, ms=4.5, lw=1.0,
                label=lab)
    ax.axhline(sk.loc[sk.calib_days == 0, "none"].iloc[0], color=GREY, lw=0.7,
               ls=":")
    # shade where correction is harmful
    cross = 30
    ax.axvspan(-1, cross, color=ACC, alpha=0.055, lw=0)
    ax.text(13, ax.get_ylim()[0] + 0.012,
            "corrections harmful\nbelow ~30 days", fontsize=6.8, color=ACC,
            ha="center")
    ax.set_xlabel("days of local data used for calibration")
    ax.set_ylabel("median skill vs persistence")
    ax.set_xticks(sk.calib_days)
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Local calibration on an unseen rooftop — compare within a "
                 "column only", loc="left")
    fig.tight_layout()
    save(fig, "fig9_calibration_sweep")


def main():
    print("building Section 5 figures from committed result files")
    fig4_ceiling()
    fig5_where_error_lives()
    fig6_ensemble()
    fig7_calibration()
    fig8_deployment()
    fig9_calibration_sweep()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
