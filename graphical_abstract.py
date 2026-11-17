#!/usr/bin/env python3
"""Graphical abstract (Elsevier: landscape, at least 1328 x 531 px). Writes ../graphical_abstract.{pdf,png}."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
HERE = os.path.dirname(os.path.abspath(__file__))
C1, C2, INK, MUTE = "#0072B2", "#D55E00", "#222222", "#777777"
fig = plt.figure(figsize=(13.28 / 2, 5.31 / 2)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13.28); ax.set_ylim(0, 5.31); ax.axis("off")
ax.text(0.3, 4.95, "Pooled assay channels on a solvent-extraction cascade", fontsize=9.5, weight="bold", color=INK, va="top")
# cascade
n, x0, w, gap, y = 10, 0.35, 0.50, 0.21, 3.05
pooled = {1, 3, 5, 7, 9}
for i in range(n):
    x = x0 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, y), w, 0.62, boxstyle="round,pad=0.02,rounding_size=0.06", fc=C1 if i in pooled else "white", ec=INK, lw=0.8))
    ax.text(x + w / 2, y + 0.31, str(i + 1), ha="center", va="center", fontsize=7, color="white" if i in pooled else INK)
    if i < n - 1:
        ax.add_patch(FancyArrowPatch((x + w + 0.01, y + 0.42), (x + w + gap - 0.01, y + 0.42), arrowstyle="-|>", mutation_scale=5, lw=0.7, color=MUTE))
        ax.add_patch(FancyArrowPatch((x + w + gap - 0.01, y + 0.20), (x + w + 0.01, y + 0.20), arrowstyle="-|>", mutation_scale=5, lw=0.7, color=MUTE))
        ax.plot([x + w + gap / 2] * 2, [y - 0.16, y + 0.78], color=C2, lw=0.9, ls=(0, (2, 1.5)))
xe = x0 + n * (w + gap) - gap
ax.text(x0, y + 0.92, "aqueous", fontsize=6.5, color=MUTE, va="bottom"); ax.text(xe, y + 0.92, "organic", fontsize=6.5, color=MUTE, ha="right", va="bottom")
ax.text(x0, y - 0.30, "blue: stages pooled into one composite sample", fontsize=6.8, color=C1, va="top")
ax.text(x0, y - 0.64, "dashed: inter-stage transfers the pooled assay responds to", fontsize=6.8, color=C2, va="top")
ax.text(x0, 1.50, "A pooled assay reads the net flux across the cut\nbetween pooled and unpooled stages, and nothing else.", fontsize=7.6, color=INK, va="top", linespacing=1.35)
ax.text(x0, 0.62, "Flux circulating in a recycle loop changes no inventory:\nit needs a direct flow-and-assay channel.", fontsize=7.0, color=MUTE, va="top", linespacing=1.35)
# design laws
xr = 7.75
ax.plot([xr - 0.25] * 2, [0.3, 4.55], color="#cccccc", lw=0.8)
ax.text(xr, 4.45, "Design laws", fontsize=8.5, weight="bold", color=INK, va="top")
laws = [("Detect any single fault", r"$\lceil (N-1)/(2p)\rceil$ channels", "N stages, p sampler ports"),
        ("Isolate s simultaneous faults", r"$2s$ channels, if $2s<$ girth", "shortest unmeasured recycle loop"),
        ("Price of pooling (relative assay error)", r"worst-edge sensitivity $\leq k/M^*$", "M*: lightest inventory covering every edge")]
for j, (a, b, c) in enumerate(laws):
    yy = 3.75 - j * 1.22
    ax.text(xr, yy, a, fontsize=7.6, color=INK, va="top", weight="bold")
    ax.text(xr, yy - 0.36, b, fontsize=7.6, color=C1, va="top")
    ax.text(xr, yy - 0.72, c, fontsize=6.5, color=MUTE, va="top")
fig.savefig(os.path.join(HERE, "..", "graphical_abstract.pdf")); fig.savefig(os.path.join(HERE, "..", "graphical_abstract.png"), dpi=400)
