#!/usr/bin/env python3
"""
Third-revision experiments. Run after the other three scripts.
Writes ../generated/numbers_rev3.tex, results_rev3.json, fig_ladder.pdf.

  1  robustness of the decoder to error in the nominal model (signature mismatch bound and simulation)
  2  data-driven baseline: PCA with T2 and SPE statistics on the same sensors
  3  multi-fault certificate for the exhaustive decoder (sigma_2s condition) and its remainder bound
  4  worked example: how a bleed, a second recycle or a chord changes the closed-loop graph
  5  window and skew ceilings of the unknown-onset guarantee
  6  readiness ladder figure
"""
import itertools, json, math, os
import numpy as np
from scipy.linalg import expm
from scipy.stats import norm, chi2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "generated"); DES = os.path.join(HERE, "..", "designs")
rng = np.random.default_rng(20261118)
alpha, beta, eps = 0.01, 0.05, 0.05
M, T_, R = {}, {}, {}
def pc(x, d=0): return f"{100 * x:.{d}f}"

class Plant:
    def __init__(self, K, f, edges, org):
        self.K, self.f, self.edges, self.org = K, f, edges, org
        self.N = K.shape[0]; self.E = len(edges)
        self.B = np.zeros((self.N, self.E))
        for e, (i, j) in enumerate(edges): self.B[i, e] = -1.0; self.B[j, e] = 1.0
        self.mbar = np.linalg.solve(K, f); self.Kinv = np.linalg.inv(K); self.lam = K.diagonal().max()
        self.a_org = np.array([-K[snk, src] for (src, snk) in org])
        self.traffic = self.a_org * np.array([self.mbar[src] for (src, _) in org])
    def Psi(self, D): return self.Kinv @ (np.eye(self.N) - expm(-self.K * D))
    def S(self, H, D): return H @ self.Psi(D) @ self.B
    def T(self, H): return H @ self.B
    def faulty(self, faults):
        """faults: list of (edge, theta). Returns K', list of phi, list of delta-alpha."""
        K2 = self.K.copy(); phis, das = [], []
        for e, theta in faults:
            src, snk = self.org[e]; d = theta * self.a_org[e]
            K2[src, src] += d; K2[snk, src] -= d
            sign = 1.0 if (src, snk) == self.edges[e] else -1.0
            phis.append(sign * d * self.mbar[src]); das.append(abs(d))
        return K2, phis, das
    def response(self, faults, D):
        K2, phis, das = self.faulty(faults); mss = np.linalg.solve(K2, self.f)
        return mss + expm(-K2 * D) @ (self.mbar - mss) - self.mbar, phis, das
def coh_cols(S):
    n = np.linalg.norm(S, axis=0); Sn = S / n
    G = np.abs(Sn.T @ Sn); np.fill_diagonal(G, 0); return n, Sn, G
def build_path(Dc):
    N = len(Dc); kap = 1.0 + Dc; a = 1.0 / kap; b = Dc / kap
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] = a[i] + b[i]
        if i + 1 < N: K[i + 1, i] = -a[i]
        if i - 1 >= 0: K[i - 1, i] = -b[i]
    f = np.zeros(N); f[0] = 1.0
    return Plant(K, f, [(i, i + 1) for i in range(N - 1)], [(i + 1, i) for i in range(N - 1)])
D0 = np.linspace(1.30, 0.80, 40); P = build_path(D0)
npz = np.load(os.path.join(OUT, "designs.npz")); H8 = npz["H8"]; H12 = npz["H12"]; Hbin = npz["H_binary"]
sigD = P.traffic.min() / 40.0
def point_H(idx):
    H = np.zeros((len(idx), P.N)); H[np.arange(len(idx)), idx] = 1.0; return H
H_pts = point_H(np.round(np.linspace(2, P.N - 3, 8)).astype(int))

# ----------------------------------------------------------------------------
# 1. Error in the nominal model
# ----------------------------------------------------------------------------
def mismatch(level, D, snr=24.0, reps=20, trials=40):
    k = 8; hit_m = hit_s = 0; n = 0; kap_rel = []; ratio = 0.0
    _, Tn, _ = coh_cols(P.T(H8))
    for _ in range(reps):
        Pm = build_path(D0 * (1 + level * rng.uniform(-1, 1, 40)))            # the model the decoder is built from
        Sm = Pm.S(H8, D); _, Smn, _ = coh_cols(Sm); kap = np.abs(Pm.K - P.K).sum(axis=0).max(); kap_rel.append(kap / P.lam)
        if kap > 0: ratio = max(ratio, np.abs(Sm - P.S(H8, D)).max() / (kap * D * D))
        for e in rng.choice(P.E, trials, replace=True):
            th = min(snr * sigD / D / P.traffic[e], 1.0); r = H8 @ P.response([(e, -th)], D)[0]
            rho = r + sigD * rng.standard_normal(k); n += 1
            hit_m += int(np.argmax(np.abs(Smn.T @ rho)) == e); hit_s += int(np.argmax(np.abs(Tn.T @ rho)) == e)
    return float(np.mean(kap_rel)), hit_m / n, hit_s / n, ratio
mm_rows = []; worst_ratio = 0.0
for lv in (0.0, 0.05, 0.1, 0.2, 0.4):
    k1, m1, s1, r1 = mismatch(lv, 1.0); k4, m4, s4, r4 = mismatch(lv, 4.0)
    worst_ratio = max(worst_ratio, r1, r4); mm_rows.append((lv, k1, m1, s1, m4, s4))
R["mismatch"] = mm_rows; M["nMisRatio"] = f"{worst_ratio:.2f}"; assert worst_ratio <= 1 + 1e-9
T_["mismatchrows"] = "".join(f"{pc(lv)} & {k1:.3f} & {pc(m1, 1)} & {pc(s1, 1)} & {pc(m4, 1)} & {pc(s4, 1)} \\\\\n" for lv, k1, m1, s1, m4, s4 in mm_rows)

# ----------------------------------------------------------------------------
# 2. PCA baseline (T2 and SPE) on the same sensors, trained on normal operating data
# ----------------------------------------------------------------------------
A1 = expm(-P.K); G1 = P.Psi(1.0) @ P.f
def simulate(Kf, feed_rsd, steps, B, x0=None):
    A2 = expm(-Kf); G2 = np.linalg.solve(Kf, np.eye(P.N) - A2) @ P.f
    x = np.tile(P.mbar[:, None], (1, B)) if x0 is None else x0
    for _ in range(steps): x = A2 @ x + np.outer(G2, 1.0 + feed_rsd * rng.standard_normal(B))
    return x
def pca_study(H, feed_rsd, L=4, th=0.2, B=200):
    k = H.shape[0]; meas = lambda x: H @ x + sigD / math.sqrt(2) * rng.standard_normal((k, x.shape[1]))
    Xtr = meas(simulate(P.K, feed_rsd, 150, 6000)); mu = Xtr.mean(axis=1, keepdims=True); sd = Xtr.std(axis=1, keepdims=True)
    Z = (Xtr - mu) / sd; U, s, _ = np.linalg.svd(Z @ Z.T / Z.shape[1]); cum = np.cumsum(s) / s.sum()
    a = min(int(np.searchsorted(cum, 0.90) + 1), k - 1); Pc = U[:, :a]; lam_ = s[:a]
    def stats(X):
        Zx = (X - mu) / sd; t = Pc.T @ Zx
        return (t ** 2 / lam_[:, None]).sum(axis=0), ((Zx - Pc @ t) ** 2).sum(axis=0)
    t2, spe = stats(meas(simulate(P.K, feed_rsd, 150, 6000)))
    q = 1 - alpha / 2; lim_t, lim_s = np.quantile(t2, q), np.quantile(spe, q)
    t2, spe = stats(meas(simulate(P.K, feed_rsd, 150, 6000))); fa = float(np.mean((t2 > lim_t) | (spe > lim_s)))
    det = 0.0
    for e in range(P.E):
        x0 = simulate(P.K, feed_rsd, 150, B); x = simulate(P.faulty([(e, -th)])[0], feed_rsd, L, B, x0)
        t2, spe = stats(meas(x)); det += np.mean((t2 > lim_t) | (spe > lim_s))
    return fa, det / P.E, a
pca_rows = []
for fr in (0.0, 1e-3, 1e-2):
    a_ = pca_study(H8, fr); b_ = pca_study(H_pts, fr); pca_rows.append((fr, a_, b_))
R["pca"] = pca_rows
def rsdfmt(x): return "0" if x == 0 else f"{100 * x:g}"
T_["pcarows"] = "".join(f"{rsdfmt(fr)} & {pc(a_[0], 1)} & {pc(a_[1])} & {pc(b_[0], 1)} & {pc(b_[1])} \\\\\n" for fr, a_, b_ in pca_rows)

# ----------------------------------------------------------------------------
# 3. Two simultaneous faults: exhaustive decoder, sigma_{2s} condition, remainder bound
# ----------------------------------------------------------------------------
def sigma_r(S, r):
    idx = np.array(list(itertools.combinations(range(S.shape[1]), r)))
    return float(np.linalg.svd(S[:, idx].transpose(1, 0, 2), compute_uv=False)[:, -1].min())
def two_fault(H, D, n_pairs=400, trials=200):
    k = H.shape[0]; S = P.S(H, D); s4 = sigma_r(S, 4); chi = math.sqrt(chi2.ppf(1 - eps, k))
    pairs = np.array(list(itertools.combinations(range(P.E), 2)))
    Qs = np.linalg.qr(S[:, pairs].transpose(1, 0, 2))[0]                       # (npairs, k, 2)
    cert = 0; worst = 1.0; rem = 0.0
    for _ in range(n_pairs):
        e1, e2 = sorted(rng.choice(P.E, 2, replace=False)); th = 10 ** rng.uniform(-1.5, 0, 2)
        dm, phis, das = P.response([(e1, -th[0]), (e2, -th[1])], D); r = H @ dm
        Rm = r - S[:, e1] * phis[0] - S[:, e2] * phis[1]; bound = 2 * sum(das) * sum(abs(p_) for p_ in phis) * D * D
        rem = max(rem, np.abs(Rm).max() / bound)
        if s4 * min(abs(p_) for p_ in phis) > 2 * (sigD * chi + math.sqrt(k) * bound):
            cert += 1; rho = r[:, None] + sigD * rng.standard_normal((k, trials))
            en = np.einsum("pkj,kt->pjt", Qs, rho); best = np.argmax((en ** 2).sum(axis=1), axis=0)
            truth = int(np.flatnonzero((pairs[:, 0] == e1) & (pairs[:, 1] == e2))[0]); worst = min(worst, float(np.mean(best == truth)))
    return s4, cert, n_pairs, worst, rem
tf = {}
for name, H, D in (("k8", H8, 1.0), ("k12", H12, 1.0), ("k12long", H12, 4.0)):
    tf[name] = two_fault(H, D)
R["two_fault"] = tf
M["nTwoSigEight"] = f"{tf['k8'][0]:.4f}"; M["nTwoSigTwelve"] = f"{tf['k12'][0]:.4f}"
M["nTwoCertEight"] = tf["k8"][1]; M["nTwoCertTwelve"] = tf["k12"][1]; M["nTwoCertTwelveLong"] = tf["k12long"][1]; M["nTwoPairs"] = tf["k8"][2]
certified = [v for v in tf.values() if v[1] > 0]
M["nTwoWorst"] = pc(min(v[3] for v in certified), 1) if certified else "--"
M["nTwoRem"] = f"{max(v[4] for v in tf.values()):.2f}"; assert max(v[4] for v in tf.values()) <= 1 + 1e-9

# ----------------------------------------------------------------------------
# 4. Worked example: added streams on the 18-stage closed loop (node 0 = environment)
# ----------------------------------------------------------------------------
def invariants(n_nodes, edges):
    adj = {v: set() for v in range(n_nodes)}
    for i, j in edges: adj[i].add(j); adj[j].add(i)
    used = [v for v in adj if adj[v]]
    comp = 0; seen = set()
    for v in used:
        if v in seen: continue
        comp += 1; stack = [v]
        while stack:
            u = stack.pop()
            if u in seen: continue
            seen.add(u); stack.extend(adj[u] - seen)
    cyc = len(edges) - len(used) + comp
    girth = math.inf
    for s in used:
        dist = {s: 0}; par = {s: None}; queue = [s]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if v not in dist: dist[v] = dist[u] + 1; par[v] = u; queue.append(v)
                elif par[u] != v: girth = min(girth, dist[u] + dist[v] + 1)
    def colourable(c):
        col = {}
        def go(i):
            if i == len(used): return True
            v = used[i]
            for x in range(c):
                if all(col.get(u) != x for u in adj[v]):
                    col[v] = x
                    if go(i + 1): return True
                    del col[v]
            return False
        return go(0)
    chi_ = next(c for c in (1, 2, 3, 4) if colourable(c))
    return cyc, girth, chi_
n = 18; ring = [(i, i % n + 1) for i in range(1, n + 1)]            # stages 1..18, edge (18,1) is the organic return
cases = [("closed loop, boundary streams measured", ring),
         ("organic return measured", ring[:-1]),
         ("one unmeasured bleed, stage 9", ring + [(9, 0)]),
         ("two unmeasured bleeds, stages 5 and 12", ring + [(5, 0), (12, 0)]),
         ("chord between stages 3 and 14", ring + [(3, 14)]),
         ("chord between stages 14 and 16", ring + [(14, 16)])]
gr_rows = []
for name, ed in cases:
    cyc, g, ch = invariants(n + 1, ed)
    smax = "any" if g == math.inf else str(math.ceil(g / 2) - 1)
    gr_rows.append((name, len(ed), cyc, "$\\infty$" if g == math.inf else str(g), ch, math.ceil(math.log2(ch)), smax))
R["graphs"] = gr_rows
T_["graphrows"] = "".join(f"{nm} & {ne} & {cy} & {g} & {ch} & {kd} & {sm} \\\\\n" for nm, ne, cy, g, ch, kd, sm in gr_rows)

# ----------------------------------------------------------------------------
# 5. Ceilings of the unknown-onset guarantee: Delta < A_e / (4 sqrt(k) lambda), tau < Delta A_e / (2 sqrt(k))
# ----------------------------------------------------------------------------
def ceilings(H):
    k = H.shape[0]; nrm, _, G = coh_cols(P.T(H)); A = nrm * (1 - G.max(axis=0))
    return float(A.max() / (4 * math.sqrt(k) * P.lam)), float(np.median(A) / (4 * math.sqrt(k) * P.lam)), float(A.max() / (2 * math.sqrt(k)))
c8 = ceilings(H8); cb = ceilings(Hbin)
M["nCeilWinBest"] = f"{c8[0]:.3f}"; M["nCeilWinMed"] = f"{c8[1]:.3f}"; M["nCeilSkew"] = f"{c8[2]:.2f}"
M["nCeilWinBin"] = f"{cb[0]:.3f}"; M["nCeilRate"] = f"{1 / c8[0]:.0f}"

with open(os.path.join(OUT, "numbers_rev3.tex"), "w") as fh:
    fh.write("% Generated by code/revision3_experiments.py -- do not edit by hand\n")
    for k_, v_ in M.items(): fh.write(f"\\newcommand{{\\{k_}}}{{{v_}}}\n")
    for k_, v_ in T_.items(): fh.write(f"\\newcommand{{\\{k_}}}{{%\n{v_}}}\n")
json.dump(dict(R=R, M=M), open(os.path.join(OUT, "results_rev3.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

# ----------------------------------------------------------------------------
# 6. Readiness ladder
# ----------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42})
INK, MUTE, C1 = "#222222", "#666666", "#0072B2"
steps = [("IR-0  closed", "every stream crossing the battery limit is metered and assayed", "P&ID, instrument index", "process design"),
         ("IR-1  circulation-ready", "every recycle loop short enough to hide the target faults is opened", "loop list from the P&ID", "design review"),
         ("IR-2  detection-ready", "a declared fault on this stage pair raises an alarm within the window", "mass balance, assay precision", "instrument specification"),
         ("IR-3  isolation-ready", "the alarm names the right stage pair, at the same window", "channel matrix, certificate tier", "sampler design, software acceptance"),
         ("IR-4  timing-ready", "sampler and analyser deliver the readings the certificate assumes", "sampler and analyser data sheets", "detailed design, site acceptance")]
fig = plt.figure(figsize=(7.0, 3.3)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 14); ax.set_ylim(0, 6.6); ax.axis("off")
ax.text(0.25, 6.3, "level", fontsize=7, color=MUTE); ax.text(4.3, 6.3, "what it asserts", fontsize=7, color=MUTE)
ax.text(9.6, 6.3, "evidence", fontsize=7, color=MUTE); ax.text(12.0, 6.3, "project stage", fontsize=7, color=MUTE)
for i, (lvl, what, ev, stg) in enumerate(steps):
    y = 5.1 - i * 1.15; x = 0.2 + 0.28 * i
    ax.add_patch(FancyBboxPatch((x, y), 3.5, 0.85, boxstyle="round,pad=0.02,rounding_size=0.08", fc=(0.0, 0.447, 0.698, 0.10 + 0.14 * i), ec=C1, lw=0.9))
    ax.text(x + 0.15, y + 0.42, lvl, fontsize=7.6, weight="bold", color=INK, va="center")
    ax.text(4.3 + 0.0, y + 0.42, what, fontsize=6.6, color=INK, va="center", wrap=True)
    ax.text(9.6, y + 0.42, ev, fontsize=6.4, color=MUTE, va="center"); ax.text(12.0, y + 0.42, stg, fontsize=6.4, color=MUTE, va="center")
ax.text(0.25, 0.12, "structural (diagram only): IR-0, IR-1      probabilistic, single fault (model and noise): IR-2, IR-3      hardware: IR-4", fontsize=6.6, color=C1)
fig.savefig(os.path.join(OUT, "fig_ladder.pdf")); plt.close(fig)
print(open(os.path.join(OUT, "numbers_rev3.tex")).read())
