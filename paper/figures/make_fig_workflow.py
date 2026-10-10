#!/usr/bin/env python3
"""
make_fig_workflow.py
====================
Fig. 3 - methodology workflow.

Shows the path from raw inputs to the three evaluation settings, and makes
visible what the text otherwise has to keep repeating: which branches are
operational, which are tested variants, and which are diagnostic instruments
that could never be deployed.

Two layout decisions worth recording, because earlier drafts got them wrong:

  * SARAH-3 does NOT feed the pvlib stage. Clear-sky irradiance and plane-of-
    array transposition are computed from solar geometry and the NWP beam /
    diffuse split; the satellite product is reference truth for verification
    only. An earlier version drew that edge and implied a dependency that does
    not exist.

  * MOS is drawn as a forward branch producing an alternative irradiance
    input, not as a feedback loop from the evaluation stage. Drawn as a loop
    it creates a cycle that cannot be laid out left-to-right, and it also
    misrepresents the experiment: the corrected irradiance is tested as a
    feature substitution, not fed back from a result.

Hand-laid rather than produced by a graph library, because the point of the
figure is the reading order and a layout engine will not respect it.

Run:  python paper/figures/make_fig_workflow.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 7.2, "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "savefig.bbox": "tight", "figure.facecolor": "white",
})

INK = "#1f1f1f"
GREY = "#6b6b6b"
FILL_DATA = "#eef1f4"
FILL_PROC = "#ffffff"
FILL_EVAL = "#e8eee8"
FILL_DIAG = "#fbeaea"
EDGE_DIAG = "#b2182b"


def box(ax, x, y, w, h, text, fill=FILL_PROC, edge=INK, ls="-", fs=7.0, lw=0.7):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw, edgecolor=edge, facecolor=fill, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, zorder=3, linespacing=1.3)
    return (x, y, w, h)


def arrow(ax, p0, p1, color=GREY, ls="-", lw=0.8, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=7, linewidth=lw,
        color=color, linestyle=ls, zorder=1,
        connectionstyle="arc3,rad=%s" % rad))


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def top(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


def main():
    fig, ax = plt.subplots(figsize=(7.1, 4.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    for x, lab in ((0.95, "INPUTS"), (3.15, "PROCESSING"),
                   (5.90, "MODEL"), (8.60, "EVALUATION")):
        ax.text(x, 5.78, lab, ha="center", fontsize=6.6, color=GREY,
                weight="bold")

    # ---- inputs
    pv = box(ax, 0.1, 4.75, 1.7, 0.68, "PVOutput\n33 rooftops, 5 min", FILL_DATA)
    nwp = box(ax, 0.1, 3.55, 1.7, 0.90,
              "NWP day-ahead\nECMWF + 5 centres\nlagged runs d1-d7", FILL_DATA)
    sat = box(ax, 0.1, 2.40, 1.7, 0.68, "SARAH-3\nsatellite GHI", FILL_DATA)
    era = box(ax, 0.1, 1.45, 1.7, 0.68, "ERA5\n(control only)", FILL_DATA)

    # ---- processing
    res = box(ax, 2.30, 4.75, 1.7, 0.68, "15-min means\ndaylight only")
    kt = box(ax, 2.30, 3.80, 1.7, 0.68,
             "$K_t$ interpolation\nhourly $\\rightarrow$ 15 min")
    geo = box(ax, 2.30, 2.85, 1.7, 0.68, "pvlib\nclear-sky, POA")
    mos = box(ax, 2.30, 1.45, 1.7, 0.90,
              "MOS correction\ntested as an\nirradiance substitute", ls="--")

    # ---- model column
    feat = box(ax, 4.95, 3.70, 1.9, 1.15,
               "13 features\nweather · geometry\nmetadata · PR lag", lw=1.0)
    model = box(ax, 4.95, 2.35, 1.9, 1.00,
                "LightGBM\nP25 / P50 / P90\npinball objective", lw=1.0)
    recal = box(ax, 4.95, 1.35, 1.9, 0.68, "conformal\nrecalibration (val)")
    oracle = box(ax, 4.95, 0.30, 1.9, 0.72,
                 "Fleet oracle\nleave-one-out fleet truth",
                 FILL_DIAG, EDGE_DIAG, ls="--")

    # ---- evaluation
    temporal = box(ax, 7.35, 4.55, 2.5, 0.88,
                   "Chronological split\nknown rooftop\n"
                   "$\\rightarrow$ upper bound", FILL_EVAL)
    loso = box(ax, 7.35, 3.30, 2.5, 0.98,
               "Leave-one-system-out\nunseen rooftop, 33 folds\n"
               "+ calibration 0-60 d", FILL_EVAL)
    irr = box(ax, 7.35, 1.95, 2.5, 1.05,
              "Irradiance verification\nforecast vs satellite\nERA5 as control",
              FILL_EVAL)

    # ---- feature pipeline
    arrow(ax, right(pv), left(res))
    arrow(ax, right(nwp), left(kt))
    arrow(ax, right(nwp), left(geo), rad=0.10)
    arrow(ax, right(res), (feat[0], feat[1] + feat[3] * 0.80), rad=-0.10)
    arrow(ax, right(kt), (feat[0], feat[1] + feat[3] * 0.50))
    arrow(ax, right(geo), (feat[0], feat[1] + feat[3] * 0.20), rad=0.10)

    # MOS: consumes NWP and satellite truth, offers a substitute input
    arrow(ax, bottom(nwp), (mos[0] + mos[2] * 0.35, mos[1] + mos[3]))
    arrow(ax, right(sat), (mos[0] + mos[2] * 0.78, mos[1] + mos[3]), rad=-0.20)
    arrow(ax, right(mos), (feat[0], feat[1] + feat[3] * 0.05), rad=-0.22, ls="--")

    # ---- model chain
    arrow(ax, bottom(feat), top(model))
    arrow(ax, bottom(model), top(recal))
    # oracle is an added FEATURE, so it enters the model - routed around the
    # recalibration box, which sits directly between the two
    arrow(ax, (oracle[0] + 0.25, oracle[1] + oracle[3]),
          (model[0] + 0.15, model[1]), color=EDGE_DIAG, ls="--", rad=-0.55)

    # ---- model to evaluation
    arrow(ax, (model[0] + model[2], model[1] + model[3] * 0.80),
          left(temporal), rad=-0.14)
    arrow(ax, (model[0] + model[2], model[1] + model[3] * 0.45),
          left(loso), rad=-0.06)
    arrow(ax, (recal[0] + recal[2], recal[1] + recal[3] * 0.60),
          (temporal[0], temporal[1] + temporal[3] * 0.22), rad=-0.26, ls=":")
    arrow(ax, (recal[0] + recal[2], recal[1] + recal[3] * 0.30),
          (loso[0], loso[1] + loso[3] * 0.22), rad=-0.14, ls=":")

    # ---- irradiance verification
    arrow(ax, right(sat), (irr[0], irr[1] + irr[3] * 0.68), rad=-0.05)
    # bowed well below so it passes under the MOS box rather than through it
    arrow(ax, bottom(era), (irr[0], irr[1] + irr[3] * 0.18), rad=0.30, ls=":")
    arrow(ax, bottom(nwp), (irr[0], irr[1] + irr[3] * 0.92), rad=-0.42)

    # ---- legend
    ax.add_patch(FancyBboxPatch((0.1, 0.30), 1.5, 0.56,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=0.7, edgecolor=EDGE_DIAG,
                                facecolor=FILL_DIAG, linestyle="--"))
    ax.text(0.85, 0.58, "diagnostic only\nnot deployable", ha="center",
            va="center", fontsize=6.3, color=EDGE_DIAG)
    ax.text(1.75, 0.58,
            "solid: operational\ndashed: tested variant\ndotted: control",
            fontsize=6.1, color=GREY, va="center", ha="left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / ("fig3_workflow." + ext))
    plt.close(fig)
    print("wrote fig3_workflow.pdf / .png in " + str(OUT))


if __name__ == "__main__":
    main()
