#!/usr/bin/env python3
"""
Fifth-revision experiments: a complete readiness walk-through, IR-0 to IR-4, for the closed-loop
circuit (benchmark R) with declared sampler and analyser constraints, and the wet-chemistry path
(sample-and-hold, unknown fault age). Simulation of a published laboratory flowsheet with assumed
hydraulics and ASSUMED hardware figures; not plant data.
Run after the other scripts. Writes ../generated/numbers_rev5.tex and results_rev5.json.
"""
import json, math, os
import numpy as np
from scipy.stats import norm
from benchmarks import build_ring

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "generated"); DES = os.path.join(HERE, "..", "designs")
rng = np.random.default_rng(20261120)
alpha, beta, eps = 0.01, 0.05, 0.05
Q = norm.sf
Rg = build_ring()
loaded = [e for e in range(Rg.E) if Rg.traffic[e] > 1e-2 * Rg.traffic.max()]
HRb = np.loadtxt(os.path.join(DES, "H_binary_ring_n18.csv"), delimiter=",")
pts = [int(v) for v in np.round(np.linspace(0, 14, 6))]
HRp = np.zeros((6, Rg.N)); HRp[np.arange(6), pts] = 1.0
DESIGNS = [("four binary pools", HRb), ("six single-stage analysers", HRp)]

# ---- declared hardware (assumed for the walk-through) ----------------------------------------
PORTS_MAX = 6                       # ports per sample-and-hold manifold
PREP_MIN, ASSAY_MIN = 30.0, 15.0    # sample preparation per batch; one composite assay
WINDOWS_H = (0.6, 1.2, 2.0, 4.0, 8.0, 12.0)          # windows offered to the audit
CHAIN = dict(s0=1e-3 * Rg.mbar.mean(), rsd=1e-3, gain=1e-2, pip=2e-3)   # floor, assay, common gain, aliquot
GBAR = 3 * CHAIN["gain"]
LOSSES = (0.1, 0.2, 0.5, 1.0)       # acceptance sets: one declared loss each


def chain_sd(H):
    y = H @ Rg.mbar
    return np.sqrt(CHAIN["s0"] ** 2 + (CHAIN["rsd"] * y) ** 2 + (H ** 2) @ (Rg.mbar ** 2 * CHAIN["pip"] ** 2))


def bank(H, D, sdD, yh):
    S = Rg.S(H, D) / sdD[:, None]; S = S - np.outer(yh, yh @ S)
    n = np.linalg.norm(S, axis=0); ok = n > 1e-12 * n.max()
    return np.where(ok, S / np.where(ok, n, 1.0), 0.0)


def audit_levels(H):
    """Per loss and edge: smallest window (h) certifying detection (IR-2), detection and isolation
    (IR-3), and the same among hardware-feasible windows (IR-4). Exact-response audit, projected residual."""
    k = H.shape[0]; sdD = math.sqrt(2) * chain_sd(H); z = norm.isf(alpha / (2 * k))
    yh = (H @ Rg.mbar) / sdD; yh /= np.linalg.norm(yh); pcc = np.sqrt(np.maximum(1 - yh ** 2, 1e-12))
    turn_h = (PREP_MIN + ASSAY_MIN * k) / 60.0
    out = {}
    for L in LOSSES:
        for e in loaded:
            d2 = d3 = d4 = None
            for Dh in WINDOWS_H:
                D = 60.0 * Dh; U = bank(H, D, sdD, yh)
                r = (H @ Rg.response(e, -L, D)[0]) / sdD; r = (1 - GBAR) * (r - yh * (yh @ r))
                pm = Q((np.abs(r) / pcc).max() - z); s0 = np.sign(U[:, e] @ r); pb = 0.0
                for sg in (1.0, -1.0):
                    Dm = s0 * U[:, [e]] - sg * np.delete(U, e, axis=1)
                    pb += np.sum(Q((r @ Dm) / np.maximum(np.linalg.norm(Dm, axis=0), 1e-300)))
                if pm <= beta and d2 is None: d2 = Dh
                if pm <= beta and pb <= eps:
                    if d3 is None: d3 = Dh
                    if Dh >= turn_h and d4 is None: d4 = Dh
            out[(L, e)] = (d2, d3, d4)
    return out, turn_h


def monitor(H, e, L, Dh, unknown_age, trials=2000):
    """Complete monitor with a projector built from one noisy baseline. Known age: one reading at age D.
    Unknown age: readings every D, onset uniform within an interval, decision at the second reading
    after onset (age uniform in [D, 2D]); the decoder searches a bank of signatures over five ages."""
    k = H.shape[0]; y = H @ Rg.mbar; sd1 = chain_sd(H); sdD = math.sqrt(2) * sd1; z = norm.isf(alpha / (2 * k))
    D = 60.0 * Dh; yh0 = y / sdD; yh0 /= np.linalg.norm(yh0)
    ages = np.linspace(D, 2 * D, 5) if unknown_age else np.array([D])
    banks = [bank(H, a, sdD, yh0) for a in ages]
    hit = 0
    for _ in range(trials):
        a = rng.uniform(D, 2 * D) if unknown_age else D
        r = H @ Rg.response(e, -L, a)[0] if L > 0 else np.zeros(k)
        g = CHAIN["gain"] * rng.standard_normal()
        base = y + sd1 * rng.standard_normal(k); read = (1 + g) * (y + r) + sd1 * rng.standard_normal(k)
        rho = (read - base) / sdD; yh = base / sdD; yh /= np.linalg.norm(yh)
        rho = rho - yh * (yh @ rho); alarm = (np.abs(rho) / np.sqrt(np.maximum(1 - yh ** 2, 1e-12))).max() > z
        if L == 0: hit += alarm; continue
        sc = np.max([np.abs(U.T @ rho) for U in banks], axis=0)
        hit += alarm and int(np.argmax(sc)) == e
    return hit / trials


M, R, rows = {}, {}, []
for name, H in DESIGNS:
    lev, turn = audit_levels(H); R[name] = {f"{L}:{e}": v for (L, e), v in lev.items()}
    for L in LOSSES:
        c = [sum(1 for e in loaded if lev[(L, e)][j] is not None) for j in range(3)]
        rows.append(f"{name} & {100 * L:g} & {c[0]} & {c[1]} & {c[2]} \\\\")
    tag = "Bin" if "binary" in name else "Point"
    M[f"nWalkTurn{tag}"] = f"{turn:.1f}"; M[f"nWalkPorts{tag}"] = int((H > 0).sum(1).max())
    # wet-chemistry path: edges at IR-4 for a 20 % loss, at their certified feasible window
    ready = [(e, lev[(0.2, e)][2]) for e in loaded if lev[(0.2, e)][2] is not None]
    known = [monitor(H, e, 0.2, Dh, False) for e, Dh in ready]
    unknown = [monitor(H, e, 0.2, Dh, True) for e, Dh in ready]
    fa = monitor(H, 0, 0.0, 4.0, False, trials=20000)
    R[name + " ages"] = dict(edges=[e for e, _ in ready], window_h=[d for _, d in ready], known=known, unknown=unknown, false_alarm=fa)
    M[f"nWalkReady{tag}"] = len(ready)
    M[f"nWalkKnownMin{tag}"] = f"{100 * min(known):.0f}"; M[f"nWalkUnknownMin{tag}"] = f"{100 * min(unknown):.0f}"
    M[f"nWalkUnknownMed{tag}"] = f"{100 * float(np.median(unknown)):.0f}"; M[f"nWalkFa{tag}"] = f"{100 * fa:.1f}"
    M[f"nWalkWinMin{tag}"] = f"{min(d for _, d in ready):g}"; M[f"nWalkWinMax{tag}"] = f"{max(d for _, d in ready):g}"
M["nWalkLoaded"] = len(loaded); M["nWalkPortsMax"] = PORTS_MAX
M["nWalkPrep"] = f"{PREP_MIN:.0f}"; M["nWalkAssay"] = f"{ASSAY_MIN:.0f}"
M["nWalkWindows"] = ", ".join(f"{w:g}" for w in WINDOWS_H)
M["nWalkGirth"] = Rg.E

with open(os.path.join(OUT, "numbers_rev5.tex"), "w") as fh:
    fh.write("% Generated by code/revision5_experiments.py -- do not edit by hand\n")
    for k_, v_ in M.items(): fh.write(f"\\newcommand{{\\{k_}}}{{{v_}}}\n")
    fh.write("\\newcommand{\\walkrows}{%\n" + "\n".join(rows) + "\n}\n")
json.dump(dict(R=R, M=M, hardware=dict(ports=PORTS_MAX, prep_min=PREP_MIN, assay_min=ASSAY_MIN, windows_h=WINDOWS_H, chain=CHAIN)),
          open(os.path.join(OUT, "results_rev5.json"), "w"), indent=1)
print(open(os.path.join(OUT, "numbers_rev5.tex")).read())

# ---- design flowchart (condensed from the table of design decisions in the supplement) ----------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42, "mathtext.fontset": "cm"})
INK, MUTE, C1, C2 = "#222222", "#666666", "#0072B2", "#D55E00"
STEPS = [
    ("1  Draw the graph", "stages = nodes,\nunmetered transfers = edges", "structural", "IR-0"),
    ("2  Meter the loops", "one flow-and-assay point per\nindependent recycle loop", "structural", "IR-1"),
    ("3  Count channels", r"detect: $\lceil (N-1)/(2p)\rceil$ (cascade)" "\n" r"identify $s$: $2s$, if $2s<$ girth", "structural", "IR-1"),
    ("4  Size the pools", r"standing load $\leq y^*=\sigma_0/\varsigma$;" "\nabove it use single stages", "chain", ""),
    ("5  Choose sampling mode", "settled phases where\n" r"$|\zeta_i|\sigma_\upsilon$ exceeds assay error", "chain", ""),
    ("6  Remove common drift", "project the residual off the\nstanding loads (one dimension)", "chain", ""),
    ("7  Audit every edge", "declared fault, window,\n" r"miss $\leq\beta$, wrong edge $\leq\varepsilon$", "probabilistic", "IR-2, IR-3"),
    ("8  Check the hardware", "window $\\geq$ assay turnaround;\naliquots drawn together", "hardware", "IR-4"),
]
COL = {"structural": C1, "chain": MUTE, "probabilistic": C2, "hardware": INK}
fig, ax = plt.subplots(figsize=(7.2, 2.45)); ax.set_xlim(0, 4); ax.set_ylim(0, 2); ax.axis("off")
W, Hh = 0.90, 0.78
for i, (ttl, body, kind, lev) in enumerate(STEPS):
    cx = 0.5 + i % 4; cy = 1.52 - (i // 4) * 1.0
    ax.add_patch(FancyBboxPatch((cx - W / 2, cy - Hh / 2), W, Hh, boxstyle="round,pad=0.01,rounding_size=0.04",
                                fc="white", ec=COL[kind], lw=1.1))
    ax.text(cx, cy + 0.24, ttl, ha="center", va="center", fontsize=6.8, fontweight="bold", color=INK)
    ax.text(cx, cy - 0.02, body, ha="center", va="center", fontsize=6.1, color=INK, linespacing=1.25)
    ax.text(cx, cy - 0.29, kind + (" · " + lev if lev else ""), ha="center", va="center", fontsize=6.2, color=COL[kind])
    if i % 4 < 3:
        ax.annotate("", xy=(cx + 0.55, cy), xytext=(cx + W / 2, cy), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
ymid = 1.02
ax.plot([3.5, 3.5, 0.5], [1.52 - Hh / 2, ymid, ymid], color=INK, lw=0.8)
ax.annotate("", xy=(0.5, 0.52 + Hh / 2), xytext=(0.5, ymid), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
fig.subplots_adjust(0.005, 0.01, 0.995, 0.99); fig.savefig(os.path.join(OUT, "fig_procedure.pdf"))
