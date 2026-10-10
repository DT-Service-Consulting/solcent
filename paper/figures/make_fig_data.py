#!/usr/bin/env python3
"""
make_fig_data.py
================
Figures for Section 3 (Data and Study Area).

  Fig. 1  fleet geography and size distribution
  Fig. 2  metadata quality - tilt, azimuth, capacity ratio

Styling targets a single-column Elsevier/IEEE layout: 300 dpi, vector PDF
alongside PNG, no colour carrying information that greyscale would lose, and
type sizes that survive reduction to ~90 mm width.

Run:  python paper/figures/make_fig_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v3_ensemble"))
import config as C

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "savefig.bbox": "tight", "figure.facecolor": "white",
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})

GREY = "#4a4a4a"
MID = "#8c8c8c"
ACCENT = "#1f1f1f"
FLAG = "#b2182b"      # used only to mark implausible values


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def load():
    df = pd.read_csv(C.MODEL_READY_V3,
                     usecols=["system_id", "capacity_kwp", "tilt_deg", "azimuth_deg"])
    sysmeta = df.groupby("system_id").first().reset_index()
    raw = pd.read_csv(C.PARENT_META).drop_duplicates("system_id")
    raw = raw[raw.system_id.isin(sysmeta.system_id)]
    m = sysmeta.merge(
        raw[["system_id", "latitude", "longitude", "region", "orientation", "shade"]],
        on="system_id")
    zs = pd.read_csv(C.MODELS_DIR / "consolidated" / "exp5_loso_per_system.csv")
    m = m.merge(zs[["system_id", "cap_ratio", "cap_obs_kw"]], on="system_id", how="left")
    return m


# ---------------------------------------------------------------------------
def fig1_fleet(m):
    """Two panels: map (given room, legend outside the data) + size histogram."""
    fig = plt.figure(figsize=(7.1, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    used_cartopy = False
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        ax.remove()
        ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), lw=0.5, edgecolor=GREY)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), lw=0.5, edgecolor=GREY)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#eef2f5")
        ax.set_extent([2.3, 6.6, 49.3, 51.7], crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, lw=0.3, color=MID, alpha=0.5,
                          linestyle=":")
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 6.5}
        tf = dict(transform=ccrs.PlateCarree())
        used_cartopy = True
    except Exception as e:
        print("  [cartopy unavailable, plain axes]", e)
        tf = {}
        ax.set_aspect(1 / np.cos(np.radians(50.8)))
        ax.set_xlabel("longitude (deg E)")
        ax.set_ylabel("latitude (deg N)")

    ax.scatter(m.longitude, m.latitude, s=10 + 7 * m.capacity_kwp,
               facecolor="none", edgecolor=ACCENT, linewidth=0.8, zorder=5, **tf)
    ax.set_title("(a) fleet geography", loc="left")

    # size key placed OUTSIDE the map frame so it never covers a system
    handles = [Line2D([], [], marker="o", ls="", mfc="none", mec=ACCENT,
                      ms=np.sqrt(10 + 7 * c), label=f"{c:g}")
               for c in (2.5, 5, 10)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.14),
              ncol=3, frameon=False, title="rated capacity (kW$_p$)",
              title_fontsize=6.5, handletextpad=0.5, columnspacing=1.6,
              borderpad=0.2)

    # --- (b) capacity distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(m.capacity_kwp, bins=np.arange(2, 11.5, 1.0),
             color=MID, edgecolor="white", linewidth=0.6)
    med = m.capacity_kwp.median()
    ax2.axvline(med, color=ACCENT, lw=1.0, ls="--")
    ax2.text(med + 0.3, ax2.get_ylim()[1] * 0.92,
             f"median {med:.2f} kW$_p$", fontsize=6.5, color=ACCENT, va="top")
    ax2.set_xlabel("rated capacity (kW$_p$)")
    ax2.set_ylabel("systems")
    ax2.set_title("(b) system size", loc="left")

    save(fig, "fig1_fleet")


# ---------------------------------------------------------------------------
def fig2_metadata(m):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6))

    # --- (a) tilt, with the implausible group split out at 1-deg resolution
    ax = axes[0]
    bins = [0, 2] + list(range(5, 55, 5))
    n, _, patches = ax.hist(m.tilt_deg, bins=bins, color=MID,
                            edgecolor="white", linewidth=0.6)
    patches[0].set_facecolor(FLAG)
    patches[0].set_alpha(0.75)
    n_bad = int((m.tilt_deg <= 1.0).sum())
    msg = "{} of {} systems\nreport tilt $\\leq$ 1$\\degree$".format(n_bad, len(m))
    ax.annotate(msg,
                xy=(2.5, n_bad), xytext=(13, n_bad * 0.97),
                fontsize=6.5, color=FLAG, va="top",
                arrowprops=dict(arrowstyle="->", lw=0.6, color=FLAG))
    ax.set_xlabel("reported tilt (deg)")
    ax.set_ylabel("systems")
    ax.set_ylim(0, n_bad * 1.22)
    ax.set_title("(a) tilt", loc="left")

    # --- (b) azimuth
    ax = axes[1]
    vc = m.azimuth_deg.value_counts().sort_index()
    names = {135: "SE", 180: "S", 225: "SW"}
    ax.bar(range(len(vc)), vc.values, width=0.55,
           color=MID, edgecolor="white", linewidth=0.6)
    for x, v in zip(range(len(vc)), vc.values):
        ax.text(x, v + 0.4, str(v), ha="center", fontsize=6.5)
    ax.set_xticks(range(len(vc)),
                  ["{}\n{}".format(int(i), names.get(int(i), "")) for i in vc.index],
                  fontsize=6.5)
    ax.set_ylim(0, vc.max() * 1.3)
    ax.set_xlabel("reported azimuth (deg)")
    ax.set_ylabel("systems")
    ax.set_title("(b) azimuth: 3 values only", loc="left")

    # --- (c) capacity ratio
    ax = axes[2]
    cr = m.cap_ratio.dropna().sort_values().values
    y = np.arange(1, len(cr) + 1)
    ax.scatter(cr, y, s=11, facecolor="none", edgecolor=ACCENT, linewidth=0.7)
    ax.axvline(1.0, color=GREY, lw=0.8)
    ax.axvline(np.median(cr), color=FLAG, lw=0.9, ls="--")
    # headroom below the points so the span annotation never sits on the data
    ax.set_ylim(-len(cr) * 0.18, len(cr) * 1.30)
    ax.text(1.0, len(cr) * 1.19, "nameplate", fontsize=6.3, color=GREY,
            ha="center", va="bottom")
    ax.text(np.median(cr), len(cr) * 1.04, f"median {np.median(cr):.2f}",
            fontsize=6.3, color=FLAG, ha="center", va="bottom")
    ax.annotate("", xy=(cr.min(), -len(cr) * 0.06), xytext=(cr.max(), -len(cr) * 0.06),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=ACCENT))
    ax.text((cr.min() + cr.max()) / 2, -len(cr) * 0.16,
            "{:.2f}$\\times$ spread".format(cr.max() / cr.min()), fontsize=6.3,
            ha="center", color=ACCENT)
    ax.set_yticks([1, 10, 20, 30])
    ax.set_xlabel("observed $P_{99}$ / nameplate")
    ax.set_ylabel("system (ranked)")
    ax.set_title("(c) capacity ratio", loc="left")
    ax.set_xlim(0.45, 1.42)

    fig.tight_layout()
    save(fig, "fig2_metadata")


def main():
    m = load()
    print(f"loaded {len(m)} systems")
    fig1_fleet(m)
    fig2_metadata(m)
    print(f"\nfigures in {OUT}")


if __name__ == "__main__":
    main()
