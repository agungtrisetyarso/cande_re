#!/usr/bin/env python3
"""
Fourth-revision experiments: sampling and assay uncertainty, and the scope of the guarantees.
Run after the other four scripts (reads ../generated/designs.npz and ../designs/*.csv).
Writes ../generated/numbers_rev4.tex, results_rev4.json, fig_chain.pdf and ../designs/datasheet_R_binary.csv.

  1  phase-sensitivity factor of a dispersion aliquot, stage by stage (both benchmarks)
  2  common-mode gain drift and its exact rejection by projection (identity check, benchmark P)
  3  measurement-chain scenarios on benchmark R: floor, independent relative error, common gain,
     dispersion or phase-separated sampling; raw and drift-immune monitors; pooled and single-stage
  4  exact-response audit with the chain model and the projection; soundness against simulation
  5  error in the hold-ups that convert an assay into an inventory (benchmark P)
  6  structural identifiability without readiness: the two-channel design (benchmark P)
  7  instrument specification sheet for the four binary pools of benchmark R
"""
import json, math, os
import numpy as np
from scipy.stats import norm
from benchmarks import build_path, build_ring, coh_cols

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "generated"); DES = os.path.join(HERE, "..", "designs")
rng = np.random.default_rng(20261119)
alpha, beta, eps = 0.01, 0.05, 0.05
Q = norm.sf
M, T_, R = {}, {}, {}
def pc(x, d=0): return f"{100 * x:.{d}f}"
def pct(x): return f"{100 * x:g}"

P = build_path(); Rg = build_ring()
npz = np.load(os.path.join(OUT, "designs.npz")); H8 = npz["H8"]; H2 = npz["H2"]; H4 = npz["H4"]
HR6 = np.loadtxt(os.path.join(DES, "H_real_k6_ring_n18.csv"), delimiter=",")
HRb = np.loadtxt(os.path.join(DES, "H_binary_ring_n18.csv"), delimiter=",")
sigD_P = P.traffic.min() / 40.0
def point_H(pl, idx):
    H = np.zeros((len(idx), pl.N)); H[np.arange(len(idx)), idx] = 1.0; return H

# =============================================================================
# The measurement chain
#   one reading of channel r:  (1 + g) * (y_r + signal_r) + noise,   y = H mbar,
#   noise variance  s0^2 + (rsd y_r)^2 + sum_i eta_ri^2 mbar_i^2 (pip^2 + (g_i phi)^2),
#   g = gain of the analyser run relative to the run of the baseline (common to all channels).
# =============================================================================
ZERO = dict(s0=0.0, rsd=0.0, pip=0.0, phi=0.0, gain=0.0)
def chain_sd(pl, H, c):
    y = H @ pl.mbar
    port = (H ** 2) @ (pl.mbar ** 2 * (c["pip"] ** 2 + (pl.phase_factor() * c["phi"]) ** 2))
    return np.sqrt(c["s0"] ** 2 + (c["rsd"] * y) ** 2 + port)

def bank(pl, H, D, sdD, proj):
    """Unit decoder directions for whitened residuals; projected off the standing loads if proj."""
    S = pl.S(H, D) / sdD[:, None]
    if proj:
        yh = (H @ pl.mbar) / sdD; yh /= np.linalg.norm(yh); S = S - np.outer(yh, yh @ S)
    n = np.linalg.norm(S, axis=0); ok = n > 1e-12 * n.max()
    return np.where(ok, S / np.where(ok, n, 1.0), 0.0)

def simulate(pl, H, D, c, theta, proj, trials, edges):
    """Baseline-difference monitor under the chain model. Returns false-alarm rate and, per edge,
    the probabilities of alarm, of correct edge and of both. The projector is built from the
    measured (noisy) baseline, as it would be in operation."""
    k = H.shape[0]; y = H @ pl.mbar; sd1 = chain_sd(pl, H, c); sdD = math.sqrt(2) * sd1
    z = norm.isf(alpha / (2 * k)); U = bank(pl, H, D, sdD, proj)
    def run(r):
        g = c["gain"] * rng.standard_normal(trials)
        base = y[:, None] + sd1[:, None] * rng.standard_normal((k, trials))
        read = (1 + g) * (y + r)[:, None] + sd1[:, None] * rng.standard_normal((k, trials))
        rho = (read - base) / sdD[:, None]
        if proj:
            yh = base / sdD[:, None]; yh /= np.linalg.norm(yh, axis=0)
            rho = rho - yh * np.sum(yh * rho, axis=0); scale = np.sqrt(np.maximum(1 - yh ** 2, 1e-12))
        else: scale = 1.0
        return np.abs(rho / scale).max(axis=0) > z, np.argmax(np.abs(U.T @ rho), axis=0)
    fa = float(run(np.zeros(k))[0].mean()); out = {}
    for e in edges:
        a, ehat = run(H @ pl.response(e, -theta, D)[0])
        out[e] = (float(a.mean()), float((ehat == e).mean()), float((a & (ehat == e)).mean()))
    return fa, out

def audit(pl, H, c, proj, thetas, dgrid, edges, gbar=0.0):
    """Exact-response audit (tier 3) under the chain model. With proj the residual is projected off
    the standing loads and the response is discounted by (1 - gbar). Returns, per edge, the smallest
    relative loss on the grid for which both miss <= beta and mis-isolation <= eps are certified at
    one window, and that window."""
    k = H.shape[0]; sdD = math.sqrt(2) * chain_sd(pl, H, c); z = norm.isf(alpha / (2 * k)); res = {}
    yh = (H @ pl.mbar) / sdD; yh /= np.linalg.norm(yh); pcc = np.sqrt(np.maximum(1 - yh ** 2, 1e-12)) if proj else np.ones(k)
    banks = {D: bank(pl, H, D, sdD, proj) for D in dgrid}
    for e in edges:
        res[e] = (np.inf, None)
        for th in thetas:
            done = False
            for D in dgrid:
                U = banks[D]; r = (H @ pl.response(e, -th, D)[0]) / sdD
                if proj: r = (1 - gbar) * (r - yh * (yh @ r))
                pm = Q((np.abs(r) / pcc).max() - z)
                s0 = np.sign(U[:, e] @ r); pb = 0.0
                for sg in (1.0, -1.0):
                    Dm = s0 * U[:, [e]] - sg * np.delete(U, e, axis=1); dn = np.linalg.norm(Dm, axis=0)
                    pb += np.sum(Q((r @ Dm) / np.maximum(dn, 1e-300)))
                if pm <= beta and pb <= eps: res[e] = (float(th), float(D)); done = True; break
            if done: break
    return res

TH = np.geomspace(1e-3, 1.0, 31)

# =============================================================================
# 1. Phase-sensitivity factor of a dispersion aliquot
# =============================================================================
gP, gR = P.phase_factor(), Rg.phase_factor()
loaded = [e for e in range(Rg.E) if Rg.traffic[e] > 1e-2 * Rg.traffic.max()]
st_loaded = sorted({i for e in loaded for i in Rg.edges[e]})
M["nPhaseMaxP"] = f"{np.abs(gP).max():.2f}"; M["nPhaseMedR"] = f"{np.median(np.abs(gR[st_loaded[:-1]])):.2f}"
M["nPhaseStripR"] = f"{np.abs(gR[14:]).max():.1f}"; M["nPhaseMinR"] = f"{np.abs(gR[:14]).min():.2f}"
M["nPhaseMaxScrubR"] = f"{np.abs(gR[:14]).max():.2f}"
R["phase_factor"] = dict(P=gP.tolist(), R=gR.tolist())
M["nRingConcMin"] = f"{(Rg.mbar[:15] / 600).min():.0f}"; M["nRingConcMax"] = f"{(Rg.mbar[:15] / 600).max():.0f}"
M["nRingInvMean"] = f"{Rg.mbar.mean() / 1000:.1f}"; M["nRingFloorMg"] = f"{1e-3 * Rg.mbar.mean():.1f}"
M["nRingFloorConc"] = f"{1e-3 * Rg.mbar.mean() / 0.6:.1f}"

# =============================================================================
# 2. Common-mode gain: exact rejection (identity) and effect on the monitor, benchmark P
# =============================================================================
k = 8; y8 = H8 @ P.mbar; yh = y8 / np.linalg.norm(y8); Pr = np.eye(k) - np.outer(yh, yh); worst = 0.0
for _ in range(200):
    e = int(rng.integers(P.E)); r = H8 @ P.response(e, -rng.uniform(0.05, 1), rng.choice([0.5, 1, 4]))[0]
    g = rng.uniform(-0.2, 0.2); nu = sigD_P * rng.standard_normal(k)
    rho = (1 + g) * (y8 + r) + nu - y8
    worst = max(worst, np.abs(Pr @ rho - ((1 + g) * Pr @ r + Pr @ nu)).max() / np.abs(r).max())
assert worst < 1e-9; M["nProjIdentity"] = f"{worst:.0e}".replace("e-", r"\times10^{-") + "}"
S1 = P.S(H8, 1.0); n1, Sn1, _ = coh_cols(S1); cosy = np.abs(Sn1.T @ yh)
M["nProjCosMax"] = f"{cosy.max():.2f}"; M["nProjKeepMin"] = f"{np.sqrt(1 - cosy.max() ** 2):.2f}"; M["nProjKeepMed"] = f"{np.median(np.sqrt(1 - cosy ** 2)):.2f}"
cP = dict(ZERO, s0=sigD_P / math.sqrt(2)); gain_rows = []
edgesP = list(range(P.E))
def at_ratio(pl, H, D, c, ratio, proj, trials=400):
    """Mean alarm and isolation over all edges at a fixed anomaly ratio |phi| D / sigD, and false alarms."""
    al = iso = 0.0; fa = None
    for e in range(pl.E):
        th = min(ratio * sigD_P / D / pl.traffic[e], 1.0)
        f_, o = simulate(pl, H, D, c, th, proj, trials, [e]); al += o[e][0]; iso += o[e][1]; fa = f_ if fa is None else fa
    return al / pl.E, iso / pl.E
for gl in (0.0, 1e-5, 1e-4, 1e-3, 1e-2):
    c = dict(cP, gain=gl); row = [gl, gl * y8.mean() / sigD_P]
    for proj in (False, True):
        fa = simulate(P, H8, 1.0, c, 0.0, proj, 40000, [])[0]; al, iso = at_ratio(P, H8, 1.0, c, 24.0, proj)
        row += [fa, al, iso]
    gain_rows.append(row)
R["gain_P"] = gain_rows
T_["gainrows"] = "".join(f"{pct(r[0])} & {r[1]:.2g} & {pc(r[2], 1)} & {pc(r[3])} & {pc(r[4])} & {pc(r[5], 1)} & {pc(r[6])} & {pc(r[7])} \\\\\n" for r in gain_rows)
M["nGainLoadP"] = f"{y8.mean() / P.mbar.mean():.0f}"
M["nGainFaRawOnePct"] = pc(gain_rows[-1][2]); M["nGainFaProjOnePct"] = pc(gain_rows[-1][5], 1)
M["nGainIsoRawClean"] = pc(gain_rows[0][4]); M["nGainIsoProjClean"] = pc(gain_rows[0][7])
M["nGainOneSigma"] = f"{100 * sigD_P / y8.mean():.4f}"

# =============================================================================
# 3. Measurement-chain scenarios on benchmark R (20 % loss, window of four fastest turnover times)
# =============================================================================
D4 = 4.0 / Rg.lam; s0R = 1e-3 * Rg.mbar.mean()
pts = [int(v) for v in np.round(np.linspace(0, 14, 6))]; HRp = point_H(Rg, pts)
scen = [("additive floor only", dict(ZERO, s0=s0R)),
        ("+ independent assay error 0.1\\,\\%", dict(ZERO, s0=s0R, rsd=1e-3)),
        ("+ independent assay error 1\\,\\%", dict(ZERO, s0=s0R, rsd=1e-2)),
        ("0.1\\,\\% and common gain 1\\,\\%", dict(ZERO, s0=s0R, rsd=1e-3, gain=1e-2)),
        ("0.1\\,\\% and dispersion aliquots, $\\sigma_\\upsilon=2\\,\\%$", dict(ZERO, s0=s0R, rsd=1e-3, phi=2e-2)),
        ("0.1\\,\\% and settled-phase aliquots, $0.2\\,\\%$", dict(ZERO, s0=s0R, rsd=1e-3, pip=2e-3)),
        ("0.1\\,\\%, gain 1\\,\\%, settled-phase $0.2\\,\\%$", dict(ZERO, s0=s0R, rsd=1e-3, gain=1e-2, pip=2e-3)),
        ("routine: 1\\,\\%, gain 1\\,\\%, dispersion $2\\,\\%$", dict(ZERO, s0=s0R, rsd=1e-2, gain=1e-2, phi=2e-2))]
sets = (("six pooled", HR6), ("four binary", HRb), ("six single-stage", HRp))
scen_rows = []; scen_raw = []
for name, c in scen:
    row = [name]; raw = dict(name=name, chain=c)
    for sname, H in sets:
        for proj in (False, True):
            fa, o = simulate(Rg, H, D4, c, 0.2, proj, 2000, loaded)
            nj = sum(1 for e in loaded if o[e][2] >= 0.90); row += [fa, nj]
            raw[f"{sname}/{'proj' if proj else 'raw'}"] = dict(false_alarm=fa, edges=nj, joint=[o[e][2] for e in loaded])
    scen_rows.append(row); scen_raw.append(raw)
R["scenarios_R"] = scen_raw
def cell(fa, nj): return f"{nj} & {pc(fa, 0) if fa >= 0.095 else pc(fa, 1)}"
T_["scenrows"] = "".join(r[0] + " & " + " & ".join(cell(r[1 + 2 * j], r[2 + 2 * j]) for j in range(6)) + " \\\\\n" for r in scen_rows)
M["nScenLoaded"] = len(loaded); M["nScenPoints"] = ", ".join(str(i + 1) for i in pts)
M["nScenLoadPooled"] = f"{(HR6 @ Rg.mbar).mean() / Rg.mbar.mean():.0f}"; M["nScenLoadBin"] = f"{(HRb @ Rg.mbar).mean() / Rg.mbar.mean():.1f}"
cosR = {s: np.abs(coh_cols(Rg.S(H, D4))[1][:, loaded].T @ ((H @ Rg.mbar) / np.linalg.norm(H @ Rg.mbar))) for s, H in sets}
M["nScenKeepPooledMin"] = f"{np.sqrt(1 - cosR['six pooled'].max() ** 2):.2f}"

# =============================================================================
# 4. Exact-response audit under the chain model with projection; soundness
# =============================================================================
dgR = tuple(np.array([0.5, 1, 2, 4, 8, 16, 32]) / Rg.lam)
c_best = dict(ZERO, s0=s0R, rsd=1e-3, gain=1e-2, pip=2e-3); gbar = 3 * c_best["gain"]
sets_d = dict(sets)
aud = {s: audit(Rg, H, c_best, True, TH, dgR, loaded, gbar) for s, H in sets}
R["audit_R"] = {s: {str(e): v for e, v in a.items()} for s, a in aud.items()}
def cnt(a, lim): return sum(1 for v in a.values() if v[0] <= lim)
for s, tag in (("six pooled", "Pooled"), ("four binary", "Bin"), ("six single-stage", "Point")):
    M[f"nAud{tag}Any"] = cnt(aud[s], 1.0); M[f"nAud{tag}Twenty"] = cnt(aud[s], 0.2); M[f"nAud{tag}Ten"] = cnt(aud[s], 0.1)
jmin = 1.0; ncase = 0
for s, H in sets:
    for e, (th, D) in aud[s].items():
        if D is None: continue
        _, o = simulate(Rg, H, D, c_best, th, True, 4000, [e]); jmin = min(jmin, o[e][2]); ncase += 1
c_rout = dict(ZERO, s0=s0R, rsd=1e-2, gain=1e-2, phi=2e-2); c_rout_s = dict(ZERO, s0=s0R, rsd=1e-2, gain=1e-2, pip=2e-3)
for tagc, cc in (("Rout", c_rout), ("RoutS", c_rout_s)):
    for s, tag in (("six pooled", "Pooled"), ("four binary", "Bin"), ("six single-stage", "Point")):
        a = audit(Rg, sets_d[s], cc, True, TH, dgR, loaded, 3 * cc["gain"]); R[f"audit_{tagc}_{tag}"] = {str(e): v for e, v in a.items()}
        M[f"nAud{tagc}{tag}Any"] = cnt(a, 1.0); M[f"nAud{tagc}{tag}Twenty"] = cnt(a, 0.2)
M["nAudWinMaxH"] = f"{max(dgR) / 60:.0f}"
M["nAudSoundCases"] = ncase; M["nAudSoundMin"] = pc(jmin, 1); assert jmin >= 1 - beta - eps, jmin

# =============================================================================
# 5. Error in the hold-ups that convert an assay into an inventory (benchmark P, k = 8, ratio 24)
# =============================================================================
def holdup(level, D, reps=20, trials=40):
    hit = n = 0; ratio = 0.0; _, Sn, _ = coh_cols(P.S(H8, D))
    for _ in range(reps):
        Ha = H8 * (1 + level * rng.uniform(-1, 1, P.N))[None, :]               # the channel actually realised
        dS = np.abs(P.S(Ha, D) - P.S(H8, D)).max()
        if level > 0: ratio = max(ratio, dS / (2 * level * D))
        for _ in range(trials):
            e = int(rng.integers(P.E)); th = min(24.0 * sigD_P / D / P.traffic[e], 1.0)
            rho = Ha @ P.response(e, -th, D)[0] + sigD_P * rng.standard_normal(k)
            hit += int(np.argmax(np.abs(Sn.T @ rho)) == e); n += 1
    return hit / n, ratio
hold_rows = [(lv,) + holdup(lv, 1.0) + holdup(lv, 4.0) for lv in (0.0, 0.02, 0.05, 0.10, 0.20)]
R["holdup_P"] = hold_rows
T_["holduprows"] = "".join(f"{pct(r[0])} & {pc(r[1], 1)} & {pc(r[3], 1)} \\\\\n" for r in hold_rows)
M["nHoldRatio"] = f"{max(max(r[2], r[4]) for r in hold_rows):.2f}"; assert max(max(r[2], r[4]) for r in hold_rows) <= 1 + 1e-9

# =============================================================================
# 6. Structural identifiability without readiness (benchmark P, additive noise of the benchmark)
# =============================================================================
DG = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
def emp_count(H, trials=300):
    """Edges on which some loss up to complete is isolated in 95 % of trials at some window of the grid."""
    kk = H.shape[0]; n = 0
    for e in edgesP:
        ok = False
        for D in DG:
            _, Sn, _ = coh_cols(P.S(H, D)); rho = (H @ P.response(e, -1.0, D)[0])[:, None] + sigD_P * rng.standard_normal((kk, trials))
            if np.mean(np.argmax(np.abs(Sn.T @ rho), axis=0) == e) >= 0.95: ok = True; break
        n += ok
    return n
for tag, H in (("Two", H2), ("Four", H4), ("Eight", H8)):
    a = audit(P, H, cP, False, TH, DG, edgesP); M[f"nScope{tag}Cert"] = cnt(a, 1.0); M[f"nScope{tag}CertTen"] = cnt(a, 0.1); M[f"nScope{tag}Sim"] = emp_count(H)
    T = P.T(H); _, Tn, G = coh_cols(T); M[f"nScope{tag}Mu"] = f"{G.max():.4f}"
    R[f"scope_{tag}"] = dict(certified=M[f"nScope{tag}Cert"], simulated=M[f"nScope{tag}Sim"], coherence=float(G.max()))
assert coh_cols(P.T(H2))[2].max() < 1 - 1e-9            # two channels: structurally 1-identifying
ext = json.load(open(os.path.join(OUT, "results_ext.json")))   # two-fault simulation of the earlier script
dm = ext["double_methods_k8"]; M["nScopePairRatio"] = f"{dm[-2][0]:g}"; M["nScopePairGLR"] = pc(dm[-2][3]); M["nScopePairOMP"] = pc(dm[-2][1])

# =============================================================================
# 7. Instrument specification sheet: four binary pools on benchmark R, chain c_best, drift-immune monitor
# =============================================================================
sd1 = chain_sd(Rg, HRb, c_best); yb = HRb @ Rg.mbar; zb = norm.isf(alpha / (2 * HRb.shape[0]))
comp = lambda c: chain_sd(Rg, HRb, dict(ZERO, **c))
sheet = []
for r in range(HRb.shape[0]):
    stages = [i + 1 for i in np.flatnonzero(HRb[r] > 0)]
    sheet.append(dict(channel=r + 1, stages=stages, ports=len(stages), standing_load_g=yb[r] / 1000,
                      sd_floor_mg=comp(dict(s0=s0R))[r], sd_assay_mg=comp(dict(rsd=c_best["rsd"]))[r],
                      sd_sampling_mg=comp(dict(pip=c_best["pip"]))[r], sd_reading_mg=sd1[r],
                      alarm_mg=zb * math.sqrt(2) * sd1[r], gain_tolerated="any (projected)"))
with open(os.path.join(DES, "datasheet_R_binary.csv"), "w") as fh:
    fh.write("# Benchmark R (simulation of a published laboratory flowsheet with assumed hydraulics); not plant data\n")
    fh.write("channel,stages_pooled,ports,standing_load_g_Nd,sd_floor_mg,sd_assay_mg,sd_sampling_mg,sd_reading_mg,alarm_threshold_mg\n")
    for s in sheet:
        fh.write(f"{s['channel']},{' '.join(map(str, s['stages']))},{s['ports']},{s['standing_load_g']:.2f},{s['sd_floor_mg']:.2f},"
                 f"{s['sd_assay_mg']:.2f},{s['sd_sampling_mg']:.2f},{s['sd_reading_mg']:.2f},{s['alarm_mg']:.1f}\n")
    fh.write("# edge (stage pair), smallest relative loss certified by the exact-response audit, window in min\n")
    for e, (th, D) in aud["four binary"].items():
        i, j = Rg.edges[e]; fh.write(f"edge,{i + 1}-{j + 1},{'' if D is None else f'{th:.3f}'},{'' if D is None else f'{D:.0f}'}\n")
R["datasheet_R"] = sheet
T_["sheetrows"] = "".join(f"{s['channel']} & {', '.join(map(str, s['stages']))} & {s['standing_load_g']:.1f} & {s['sd_floor_mg']:.1f} & "
                          f"{s['sd_assay_mg']:.1f} & {s['sd_sampling_mg']:.1f} & {s['sd_reading_mg']:.1f} & {s['alarm_mg']:.0f} \\\\\n" for s in sheet)
ab = [v for v in aud["four binary"].values() if v[1] is not None]
M["nSheetLossMin"] = pc(min(v[0] for v in ab)) if ab else "--"; M["nSheetLossMax"] = pc(max(v[0] for v in ab)) if ab else "--"
M["nSheetWinMin"] = f"{min(v[1] for v in ab) / 60:.1f}" if ab else "--"; M["nSheetWinMax"] = f"{max(v[1] for v in ab) / 60:.1f}" if ab else "--"
M["nSheetCrossLoad"] = f"{s0R / c_best['rsd'] / 1000:.1f}"

# =============================================================================
# Output
# =============================================================================
with open(os.path.join(OUT, "numbers_rev4.tex"), "w") as fh:
    fh.write("% Generated by code/revision4_experiments.py -- do not edit by hand\n")
    for k_, v_ in M.items(): fh.write(f"\\newcommand{{\\{k_}}}{{{v_}}}\n")
    for k_, v_ in T_.items(): fh.write(f"\\newcommand{{\\{k_}}}{{%\n{v_}}}\n")
R["macros"] = M
json.dump(R, open(os.path.join(OUT, "results_rev4.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

# figure: stage-wise phase-sensitivity factor of benchmark R and what each sampling mode costs
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
INK, MUTE, C1 = "#222222", "#666666", "#0072B2"
fig, ax = plt.subplots(figsize=(4.6, 2.5)); st = np.arange(1, Rg.N + 1)
ax.bar(st, gR, width=0.72, color=C1, edgecolor="white", linewidth=0.6); ax.axhline(0.0, color=INK, lw=0.6)
ax.axvspan(14.5, 18.5, color=MUTE, alpha=0.10, lw=0); ax.text(16.5, 0.28, "stripping", ha="center", color=MUTE, fontsize=7)
ax.text(8.5, -1.3, "extraction and scrubbing", ha="center", color=MUTE, fontsize=7)
ax.set_xlabel("stage"); ax.set_ylabel(r"phase-sensitivity factor $\zeta_i$"); ax.set_ylim(-2.2, 0.6); ax.set_xticks(st[::2]); ax.set_xlim(0.3, 18.7)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.grid(axis="y", color="#dddddd", lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_chain.pdf")); plt.close(fig)
print(open(os.path.join(OUT, "numbers_rev4.tex")).read())
