#!/usr/bin/env python3
"""
Second-revision experiments for "Cut-space instrumentation of rare-earth
solvent-extraction cascades".

Run AFTER cascade_instrumentation.py and extended_experiments.py.
Writes ../generated/numbers_rev2.tex, results_rev2.json and figures.

Sections
  A  numerical verification of every constant in the dynamic theorems, on random
     compartmental networks, and of the probabilistic bounds by Monte Carlo
  B  exact (rational arithmetic) spark computations, separated from numerical rank tests
  C  sensitivity-budget law under proportional assay error
  D  detection/isolation under a measurement-chain noise model
  E  conventional model-based baseline: Kalman-filter innovations with GLR isolation
  F  sampling skew and baseline drift, quantified
  G  flowsheet and pooled-sampler schematic
"""
import itertools, json, math, os
from fractions import Fraction
import numpy as np
from scipy.linalg import expm, solve_discrete_are, solve_discrete_lyapunov
from scipy.optimize import linprog
from scipy.stats import norm, chi2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "generated"); DES = os.path.join(HERE, "..", "designs")
rng = np.random.default_rng(20261117)
alpha, beta, eps = 0.01, 0.05, 0.05
Q = norm.sf
M, T_, R = {}, {}, {}
def sci(x):
    mant, ex = f"{x:.1e}".split("e"); return f"{mant}\\times10^{{{int(ex)}}}"

# ----------------------------------------------------------------------------
# Benchmarks (same definitions as extended_experiments.py)
# ----------------------------------------------------------------------------
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
    def faulty(self, e, theta):
        src, snk = self.org[e]; d = theta * self.a_org[e]
        K2 = self.K.copy(); K2[src, src] += d; K2[snk, src] -= d
        sign = 1.0 if (src, snk) == self.edges[e] else -1.0
        return K2, sign * d * self.mbar[src], abs(d)
    def response(self, e, theta, D):
        K2, phi, da = self.faulty(e, theta); mss = np.linalg.solve(K2, self.f)
        return mss + expm(-K2 * D) @ (self.mbar - mss) - self.mbar, phi, da
def coh_cols(S):
    n = np.linalg.norm(S, axis=0); Sn = S / n
    G = np.abs(Sn.T @ Sn); np.fill_diagonal(G, 0); return n, Sn, G
def build_path(N=40):
    Dc = np.linspace(1.30, 0.80, N); kap = 1.0 + Dc; a = 1.0 / kap; b = Dc / kap
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] = a[i] + b[i]
        if i + 1 < N: K[i + 1, i] = -a[i]
        if i - 1 >= 0: K[i - 1, i] = -b[i]
    f = np.zeros(N); f[0] = 1.0
    return Plant(K, f, [(i, i + 1) for i in range(N - 1)], [(i + 1, i) for i in range(N - 1)])
P = build_path()
npz = np.load(os.path.join(OUT, "designs.npz"))
Hd = {k: npz[f"H{k}"] for k in (2, 3, 4, 6, 8, 12)}; H8 = Hd[8]; Hbin = npz["H_binary"]
sigD_P = P.traffic.min() / 40.0

# ============================================================================
# A. Verification of the constants of the dynamic theorems
# ============================================================================
def rand_net():
    while True:
        N = int(rng.integers(4, 13)); A = np.zeros((N, N)); edges = []
        for j in range(1, N):
            i = int(rng.integers(0, j)); edges.append((i, j))
        for _ in range(int(rng.integers(0, N))):
            i, j = rng.choice(N, 2, replace=False); i, j = int(min(i, j)), int(max(i, j))
            if (i, j) not in edges: edges.append((i, j))
        for (i, j) in edges:
            A[i, j] = rng.uniform(0.1, 2.0)
            A[j, i] = 0.0 if rng.random() < 0.2 else rng.uniform(0.1, 2.0)
        a0 = np.where(rng.random(N) < 0.4, rng.uniform(0.1, 1.5, N), 0.0)
        if a0.max() == 0: a0[int(rng.integers(N))] = 1.0
        K = np.diag(A.sum(axis=1) + a0) - A.T
        if np.linalg.cond(K) < 1e8:
            f = np.where(rng.random(N) < 0.5, rng.uniform(0.2, 2.0, N), 0.0); f[int(rng.integers(N))] += 1.0
            return N, A, a0, K, f, edges
def delta_m(K2, phi, be, D):
    """phi * int_0^D exp(-K2 s) be ds, exact also when K2 is singular."""
    N = K2.shape[0]; Aug = np.zeros((N + 1, N + 1)); Aug[:N, :N] = -K2; Aug[:N, N] = phi * be
    return expm(Aug * D)[:N, N]
worst = dict(i=0.0, ii=0.0, iv=0.0, contr=0.0, psi=np.inf, om=0.0, w=0.0, i_bdry=0.0)
n_net = 400; n_cases = 0
for _ in range(n_net):
    N, A, a0, K, f, edges = rand_net(); mbar = np.linalg.solve(K, f); lam = K.diagonal().max()
    Kinv = np.linalg.inv(K)
    for _ in range(5):
        k = int(rng.integers(1, 7)); H = rng.random((k, N))
        if rng.random() < 0.3: H = (H > 0.5).astype(float)
        D = float(10 ** rng.uniform(-2, 1)); D2 = float(10 ** rng.uniform(-2, 1))
        bdry = rng.random() < 0.3 and a0.max() > 0
        K2 = K.copy(); be = np.zeros(N)
        if bdry:
            i = int(rng.choice(np.flatnonzero(a0 > 0))); d = rng.uniform(-1, 1) * a0[i]
            K2[i, i] += d; be[i] = -1.0; phi = d * mbar[i]; da = abs(d)
        else:
            i, j = edges[int(rng.integers(len(edges)))]
            dij = rng.uniform(-1, 1) * A[i, j]; dji = rng.uniform(-1, 1) * A[j, i]
            K2[i, i] += dij; K2[j, i] -= dij; K2[j, j] += dji; K2[i, j] -= dji
            be[i] = -1.0; be[j] = 1.0; phi = dij * mbar[i] - dji * mbar[j]; da = max(abs(dij), abs(dji))
        if abs(phi) < 1e-9 or da < 1e-9: continue
        n_cases += 1
        Psi = Kinv @ (np.eye(N) - expm(-K * D)); Psi2 = Kinv @ (np.eye(N) - expm(-K * D2))
        s = H @ Psi @ be; s2 = H @ Psi2 @ be; t = H @ be
        Rm = H @ delta_m(K2, phi, be, D) - phi * s
        r1 = np.abs(Rm).max() / (2 * da * abs(phi) * D * D)
        worst["i_bdry" if bdry else "i"] = max(worst["i_bdry" if bdry else "i"], r1)
        worst["ii"] = max(worst["ii"], np.abs(s - D * t).max() / (lam * D * D))
        worst["iv"] = max(worst["iv"], np.abs(s - s2).max() / (2 * abs(D - D2)))
        worst["contr"] = max(worst["contr"], np.abs(expm(-K * D)).sum(axis=0).max(), np.abs(expm(-K2 * D)).sum(axis=0).max())
        worst["psi"] = min(worst["psi"], np.abs(np.linalg.eigvals(Psi)).min() / D)
        # unknown onset / skew (Theorem onset): per-channel error terms
        a = D; tau = a * rng.uniform(0, 0.5); ages = a + tau * rng.random(k); cl = lam + 2 * da
        read = np.array([H[c] @ delta_m(K2, phi, be, ages[c]) for c in range(k)])
        worst["om"] = max(worst["om"], np.abs(read - phi * ages * t).max() / (abs(phi) * cl * (a + tau) ** 2))
        worst["w"] = max(worst["w"], np.abs(read - phi * a * t).max() / (abs(phi) * (tau + cl * (a + tau) ** 2)))
M["nVerNets"] = n_net; M["nVerCases"] = n_cases
for key, name in (("i", "VerI"), ("i_bdry", "VerIb"), ("ii", "VerII"), ("iv", "VerIV"), ("om", "VerOm"), ("w", "VerW")):
    M["n" + name] = f"{worst[key]:.3f}"
M["nVerContr"] = f"{worst['contr']:.6f}"; M["nVerPsi"] = sci(worst['psi'])
assert max(worst[k_] for k_ in ("i", "i_bdry", "ii", "iv", "om", "w")) <= 1 + 1e-9 and worst["contr"] <= 1 + 1e-9

# probabilistic bounds against Monte Carlo on benchmark P, eight channels
def prob_checks(n_case=80, trials=40000, corr=0.0):
    k = 8; viol = dict(det=0, t1=0, t2=0, t3d=0, t3i=0); used = dict(det=0, t1=0, t2=0, t3d=0, t3i=0); tight = 0.0
    C = (1 - corr) * np.eye(k) + corr * np.ones((k, k)); Lc = np.linalg.cholesky(C)
    sbar = sigD_P * math.sqrt(1 + (k - 1) * corr); thr = sbar * norm.isf(alpha / (2 * k)); chi = math.sqrt(chi2.ppf(1 - eps, k))
    for _ in range(n_case):
        e = int(rng.integers(P.E)); th = float(10 ** rng.uniform(-2.5, 0)); D = float(rng.choice([0.25, 1.0, 4.0]))
        dm, phi, da = P.response(e, -th, D); r = H8 @ dm; S = P.S(H8, D); nrm, Sn, G = coh_cols(S)
        rho = r[:, None] + sigD_P * (Lc @ rng.standard_normal((k, trials)))
        p_miss = float(np.mean(np.abs(rho).max(axis=0) <= thr)); p_err = float(np.mean(np.argmax(np.abs(Sn.T @ rho), axis=0) != e))
        aphi = abs(phi); Rb = 2 * math.sqrt(k) * da * aphi * D * D
        def chk(key, emp, bound):
            if bound >= 0.5: return
            used[key] += 1
            if emp > bound + 4 * math.sqrt(max(bound * (1 - bound), 1e-12) / trials) + 1e-12: viol[key] += 1
        chk("det", p_miss, Q((aphi * (np.abs(S[:, e]).max() - 2 * da * D * D) - thr) / sbar))
        if aphi * nrm[e] * (1 - G[:, e].max()) > 2 * sbar * chi + 2 * Rb: chk("t1", p_err, eps)
        c = np.delete(Sn.T @ Sn[:, e], e)
        b2 = float(np.sum(Q((aphi * nrm[e] * np.sqrt((1 - c) / 2) - Rb) / sbar) + Q((aphi * nrm[e] * np.sqrt((1 + c) / 2) - Rb) / sbar)))
        chk("t2", p_err, b2)
        s0 = np.sign(Sn[:, e] @ r); b3 = 0.0
        for sg in (1.0, -1.0):
            Dm = s0 * Sn[:, [e]] - sg * np.delete(Sn, e, axis=1); b3 += float(np.sum(Q((r @ Dm) / (sbar * np.linalg.norm(Dm, axis=0)))))
        chk("t3i", p_err, b3); chk("t3d", p_miss, Q((np.abs(r).max() - thr) / sbar))
        if 1e-3 < b3 < 0.5 and p_err > 0: tight = max(tight, p_err / b3)
    return used, viol, tight
u0, v0, tg0 = prob_checks(); u1, v1, tg1 = prob_checks(corr=0.6)
R["prob_checks"] = dict(indep=(u0, v0, tg0), corr=(u1, v1, tg1))
M["nProbCases"] = sum(u0.values()) + sum(u1.values()); M["nProbViol"] = sum(v0.values()) + sum(v1.values())
M["nProbTight"] = f"{max(tg0, tg1):.2f}"
assert M["nProbViol"] == 0
# Corollary (admissible region): closed-form roots against brute force
root_bad = 0; root_n = 0
for e in rng.choice(P.E, 12, replace=False):
    for D in (0.25, 1.0):
        S = P.S(H8, D); nrm, Sn, G = coh_cols(S); k = 8
        src, snk = P.org[e]; aji = P.a_org[e]; q = P.traffic[e]
        for (A_, B_, C_, strict) in ((np.abs(S[:, e]).max(), 2 * aji * D * D, sigD_P * (norm.isf(alpha / 16) + norm.isf(beta)), False),
                                     (nrm[e] * (1 - G[:, e].max()), 4 * math.sqrt(k) * aji * D * D, 2 * sigD_P * math.sqrt(chi2.ppf(1 - eps, k)), True)):
            grid = np.geomspace(1e-6, 50, 4000); ok = grid * q * (A_ - B_ * grid) - C_; ok = ok > 0 if strict else ok >= 0
            disc = A_ * A_ - 4 * B_ * C_ / q; root_n += 1
            if disc < 0: root_bad += int(ok.any()); continue
            lo, hi = (A_ - math.sqrt(disc)) / (2 * B_), (A_ + math.sqrt(disc)) / (2 * B_)
            inside = (grid >= lo * (1 + 1e-9)) & (grid <= hi * (1 - 1e-9)); outside = (grid < lo * (1 - 1e-9)) | (grid > hi * (1 + 1e-9))
            root_bad += int((~ok[inside]).any() or ok[outside].any())
M["nRootCases"] = root_n; assert root_bad == 0

# ============================================================================
# B. Exact spark computations (rational arithmetic) and numerical rank tests
# ============================================================================
def rank_exact(cols):
    Mx = [list(row) for row in zip(*cols)]; rk = 0; ncol = len(cols)
    for c in range(ncol):
        piv = next((r_ for r_ in range(rk, len(Mx)) if Mx[r_][c] != 0), None)
        if piv is None: continue
        Mx[rk], Mx[piv] = Mx[piv], Mx[rk]
        for r_ in range(len(Mx)):
            if r_ != rk and Mx[r_][c] != 0:
                fct = Mx[r_][c] / Mx[rk][c]; Mx[r_] = [x - fct * y for x, y in zip(Mx[r_], Mx[rk])]
        rk += 1
    return rk
def spark_exact(cols):
    n = len(cols)
    for r_ in range(1, n + 1):
        for Ssub in itertools.combinations(range(n), r_):
            if rank_exact([cols[i] for i in Ssub]) < r_: return r_
    return math.inf
edges9 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (4, 5), (5, 6), (6, 7), (7, 8)]
spark_rows = []; all_ok = True
for k in (2, 3, 4, 5, 6, 8):
    vals = set()
    for _ in range(5):
        H = [[Fraction(int(rng.integers(0, 98)), 97) for _ in range(9)] for _ in range(k)]
        cols = [[H[r_][j] - H[r_][i] for r_ in range(k)] for (i, j) in edges9]
        vals.add(spark_exact(cols))
    pred = min(k + 1, 5); all_ok &= (vals == {pred}); spark_rows.append((k, pred, sorted(vals)))
assert all_ok
T_["sparkrows"] = "".join(f"{k} & {p} & {', '.join(str(v) for v in vs)} \\\\\n" for k, p, vs in spark_rows)
def nonparallel_exact(Hint, edges):
    """Exact test for integer labels: differences lie in {-1,0,1}^k, where two vectors are
    parallel iff they are equal up to sign, so a set of sign-normalised tuples decides."""
    seen = set()
    for (i, j) in edges:
        c = tuple(int(Hint[r_][j]) - int(Hint[r_][i]) for r_ in range(len(Hint)))
        if not any(c): return False
        c = c if next(v for v in c if v) > 0 else tuple(-v for v in c)
        if c in seen: return False
        seen.add(c)
    return True
HRb = np.loadtxt(os.path.join(DES, "H_binary_ring_n18.csv"), delimiter=",")
exact_bin = [nonparallel_exact(Hbin.astype(int), [(i, i + 1) for i in range(39)]),
             nonparallel_exact(HRb.astype(int), [(i, (i + 1) % 18) for i in range(18)])]
for r_ in (3, 4, 5, 6):
    Hb = np.loadtxt(os.path.join(DES, f"H_BCH_r{r_}.csv"), delimiter=",").astype(int)
    L = np.hstack([np.zeros((2 * r_, 1), int), Hb]); n = L.shape[1]
    exact_bin.append(nonparallel_exact(L, list(itertools.combinations(range(n), 2))))
assert all(exact_bin)
# numerical margins for the real-weighted designs (floating point)
def num_margin(S, r_):
    n = S.shape[1]; Sn = S / np.linalg.norm(S, axis=0); idx = np.array(list(itertools.combinations(range(n), r_)))
    sv = np.linalg.svd(Sn[:, idx].transpose(1, 0, 2), compute_uv=False)[:, -1]
    return float(sv.min())
M["nNumPairStatic"] = f"{num_margin(P.T(H8), 2):.3f}"; M["nNumPairDyn"] = f"{num_margin(P.S(H8, 1.0), 2):.3f}"
M["nNumQuadStatic"] = sci(num_margin(P.T(H8), 4)); M["nNumQuadDyn"] = sci(num_margin(P.S(H8, 1.0), 4))
M["nNumQuadCount"] = math.comb(39, 4)

# ============================================================================
# C. Sensitivity-budget law under proportional assay error
# ============================================================================
def mstar(pl, w):
    A = np.zeros((pl.E, pl.N))
    for e, (i, j) in enumerate(pl.edges): A[e, i] = A[e, j] = -1.0
    res = linprog(w, A_ub=A, b_ub=-np.ones(pl.E), bounds=[(0, None)] * pl.N, method="highs")
    return res.fun, res.x
def gain(pl, H, w):
    """min over edges of max over channels of |t_re| / (eta_r . w)."""
    T = np.abs(pl.T(H)); y = H @ w; ok = y > 0
    return float((T[ok] / y[ok, None]).max(axis=0).min())
w = P.mbar; Mst, zst = mstar(P, w)
def path_cover(w):
    n = len(w); c0, c1 = 0.0, w[0]; back = []
    for i in range(1, n):
        n0, n1 = c1, min(c0, c1) + w[i]; back.append(0 if c0 <= c1 else 1); c0, c1 = n0, n1
    st = 0 if c0 <= c1 else 1; cov = []
    for i in range(n - 1, -1, -1):
        if st == 1: cov.append(i); st = back[i - 1] if i > 0 else 0
        else: st = 1
    return sorted(cov)
cover = path_cover(w); assert abs(w[cover].sum() - Mst) < 1e-8 * Mst
assert all((i in cover) or (j in cover) for (i, j) in P.edges)
def block_design(cover, w, k):
    """Split the ordered cover into k groups minimising the largest group inventory,
    never putting two adjacent stages in one group (linear-partition dynamic programme)."""
    n = len(cover); INF = float("inf"); pre = np.concatenate([[0], np.cumsum(w[cover])])
    okgrp = lambda a, b: all(cover[t + 1] - cover[t] > 1 for t in range(a, b - 1))
    best = [[INF] * (n + 1) for _ in range(k + 1)]; arg = [[-1] * (n + 1) for _ in range(k + 1)]; best[0][0] = 0.0
    for g in range(1, k + 1):
        for b in range(1, n + 1):
            for a in range(b - 1, -1, -1):
                if not okgrp(a, b): break
                v = max(best[g - 1][a], pre[b] - pre[a])
                if v < best[g][b]: best[g][b], arg[g][b] = v, a
    g = min(range(1, k + 1), key=lambda g_: best[g_][n]); assert best[g][n] < INF
    H = np.zeros((g, P.N)); b = n; row = g - 1
    while b > 0:
        a = arg[row + 1][b]; H[row, [cover[t] for t in range(a, b)]] = 1.0; b = a; row -= 1
    return H
law_rows = []; kk = list(range(1, 21)); eff_block = []
for k in kk:
    Hb = block_design(cover, w, k); assert np.all(np.abs(P.T(Hb)).max(axis=0) == 1.0)
    g = gain(P, Hb, w); assert g <= Hb.shape[0] / Mst * (1 + 1e-9); eff_block.append(g * Mst / k)
eff_real = {k: gain(P, Hd[k], w) * Mst / k for k in Hd}; eff_bin = gain(P, Hbin, w) * Mst / 4
assert max(eff_real.values()) <= 1 + 1e-9 and eff_bin <= 1 + 1e-9
Hblk8 = block_design(cover, w, 8); np.savetxt(os.path.join(DES, "H_block_k8_N40.csv"), Hblk8, fmt="%d", delimiter=",")
M["nLawMstar"] = f"{Mst:.2f}"; M["nLawMtot"] = f"{w.sum():.2f}"; M["nLawCover"] = len(cover)
M["nLawEffBlockEight"] = f"{eff_block[7]:.2f}"; M["nLawEffRealEight"] = f"{eff_real[8]:.3f}"; M["nLawEffBin"] = f"{eff_bin:.3f}"
M["nLawEffBlockMin"] = f"{min(eff_block[1:]):.2f}"; M["nLawHeavy"] = f"{w[cover].max() / Mst:.2f}"
M["nLawPortsEight"] = int(Hblk8.sum(axis=1).max())
# random designs never beat the law
gmax = 0.0
for _ in range(2000):
    k = int(rng.integers(1, 13)); H = rng.random((k, P.N)) * (rng.random((k, P.N)) < rng.uniform(0.05, 1))
    if (H @ w).min() <= 0: continue
    gmax = max(gmax, gain(P, H, w) * Mst / k)
M["nLawRandMax"] = f"{gmax:.3f}"; assert gmax <= 1 + 1e-9

# ============================================================================
# D. Measurement-chain noise: sigma_r^2 = s0^2 + (rsd * y_r)^2 + sv^2 * sum eta^2 mbar^2
# ============================================================================
s0 = sigD_P / math.sqrt(2)
def chain_sd(H, rsd):
    y = H @ P.mbar
    return math.sqrt(2) * np.sqrt(s0 ** 2 + (rsd * y) ** 2 + (0.5 * rsd) ** 2 * (H ** 2) @ (P.mbar ** 2))
def point_H(idx):
    H = np.zeros((len(idx), P.N)); H[np.arange(len(idx)), idx] = 1.0; return H
H_pts = point_H(np.round(np.linspace(2, P.N - 3, 8)).astype(int))
def chain_eval(H, rsd, D, th=0.2, trials=300):
    k = H.shape[0]; sd = chain_sd(H, rsd); S = P.S(H, D) / sd[:, None]; _, Sn, _ = coh_cols(S)
    z = norm.isf(alpha / (2 * k)); det = iso = 0.0; njoint = 0
    for e in range(P.E):
        r = (H @ P.response(e, -th, D)[0]) / sd; rho = r[:, None] + rng.standard_normal((k, trials))
        d_ = np.abs(rho).max(axis=0) > z; i_ = np.argmax(np.abs(Sn.T @ rho), axis=0) == e
        det += d_.mean(); iso += i_.mean(); njoint += int((d_ & i_).mean() >= 0.90)
    return det / P.E, iso / P.E, njoint
chain_rows = []
for rsd in (0.0, 1e-4, 1e-3, 1e-2):
    row = [rsd]
    for H in (H8, Hblk8, H_pts): row += list(chain_eval(H, rsd, 4.0))
    chain_rows.append(row)
R["chain"] = chain_rows
def pcf(x): return f"{100 * x:.0f}"
def rsdfmt(x): return "0" if x == 0 else f"{100 * x:g}"
T_["chainrows"] = "".join(f"{rsdfmt(r[0])} & {pcf(r[1])} & {pcf(r[2])} & {r[3]} & {pcf(r[4])} & {pcf(r[5])} & {r[6]} & {pcf(r[7])} & {pcf(r[8])} & {r[9]} \\\\\n" for r in chain_rows)
M["nChainCross"] = f"{100 * s0 / (H8 @ P.mbar).mean() :.4f}"          # rsd at which the two noise terms are equal for the pooled design
M["nChainPoolInv"] = f"{(H8 @ P.mbar).mean() / P.mbar.mean():.1f}"; M["nChainBlockInv"] = f"{(Hblk8 @ P.mbar).mean() / P.mbar.mean():.1f}"

# ============================================================================
# E. Conventional baseline: Kalman-filter innovations, chi-square alarm, GLR isolation
# ============================================================================
def kf_study(H, feed_rsd, L=4, th=0.2, B=200, burn=120):
    k = H.shape[0]; N = P.N; A = expm(-P.K); G = P.Psi(1.0) @ P.f           # one sampling period = one turnover time
    Qd = (feed_rsd ** 2) * np.outer(G, G) + 1e-18 * np.eye(N); Rm = (sigD_P ** 2 / 2) * np.eye(k)
    Pp = solve_discrete_are(A.T, H.T, Qd, Rm); Sinn = H @ Pp @ H.T + Rm; Kg = Pp @ H.T @ np.linalg.inv(Sinn)
    Li = np.linalg.inv(np.linalg.cholesky(Sinn)); Sx = solve_discrete_lyapunov(A, Qd)
    def run(K2, noisy=True, B_=B):
        A2 = expm(-K2); G2 = (np.linalg.solve(K2, np.eye(N) - A2)) @ P.f
        x = np.tile(P.mbar[:, None], (1, B_)); xh = x.copy(); inn = []
        for t in range(burn + L):
            Af, Gf = (A, G) if t < burn else (A2, G2)
            wn = feed_rsd * rng.standard_normal(B_) if noisy else np.zeros(B_)
            x = Af @ x + np.outer(Gf, 1.0 + wn)
            xp = A @ xh + G[:, None]
            y = H @ x + (math.sqrt(0.5) * sigD_P * rng.standard_normal((k, B_)) if noisy else 0)
            v = y - H @ xp; xh = xp + Kg @ v
            if t >= burn: inn.append(Li @ v)
        return np.vstack(inn), H @ x - (H @ P.mbar)[:, None] + (sigD_P * rng.standard_normal((k, B_)) if noisy else 0)
    # note: x <- A x + G (1 + w) keeps the nominal steady state mbar as fixed point (A mbar + G = mbar)
    sig = np.zeros((k * L, P.E)); dynS = P.S(H, float(L))
    for e in range(P.E):
        K2 = P.faulty(e, -0.05)[0]; sig[:, e] = run(K2, noisy=False, B_=1)[0][:, 0]
    sgn = sig / np.linalg.norm(sig, axis=0)
    thr_kf = chi2.ppf(1 - alpha, k * L)
    sd_tot = np.sqrt(sigD_P ** 2 + np.diag(H @ Sx @ H.T)); zb = norm.isf(alpha / (2 * k))   # baseline monitor, recalibrated
    _, Sn, _ = coh_cols(dynS / sd_tot[:, None])
    inn0, y0 = run(P.K, B_=4000)
    fa_kf = float(np.mean((inn0 ** 2).sum(axis=0) > thr_kf)); fa_bl = float(np.mean(np.abs(y0 / sd_tot[:, None]).max(axis=0) > zb))
    d_kf = i_kf = d_bl = i_bl = 0.0
    for e in range(P.E):
        inn, yb = run(P.faulty(e, -th)[0])
        d_kf += np.mean((inn ** 2).sum(axis=0) > thr_kf); i_kf += np.mean(np.argmax(np.abs(sgn.T @ inn), axis=0) == e)
        rb = yb / sd_tot[:, None]; d_bl += np.mean(np.abs(rb).max(axis=0) > zb); i_bl += np.mean(np.argmax(np.abs(Sn.T @ rb), axis=0) == e)
    E = P.E
    return dict(fa_kf=fa_kf, d_kf=d_kf / E, i_kf=i_kf / E, fa_bl=fa_bl, d_bl=d_bl / E, i_bl=i_bl / E)
kf_rows = []
for fr in (0.0, 1e-3, 1e-2):
    a_ = kf_study(H8, fr); b_ = kf_study(H_pts, fr)
    kf_rows.append((fr, a_, b_))
R["kf"] = [(fr, a_, b_) for fr, a_, b_ in kf_rows]
def p1(x): return f"{100 * x:.1f}"
T_["kfrows"] = "".join(f"{rsdfmt(fr)} & {p1(a_['fa_bl'])} & {pcf(a_['d_bl'])} & {pcf(a_['i_bl'])} & {p1(a_['fa_kf'])} & {pcf(a_['d_kf'])} & {pcf(a_['i_kf'])} & {p1(b_['fa_kf'])} & {pcf(b_['d_kf'])} & {pcf(b_['i_kf'])} \\\\\n"
                       for fr, a_, b_ in kf_rows)

# ============================================================================
# F. Sampling skew and baseline drift, quantified (k = 8, Delta = 1, ratio 24)
# ============================================================================
def skew_drift(skew=0.0, drift=0.0, widen=False, D=1.0, snr=24.0, trials=4000):
    k = 8; S = P.S(H8, D); _, Sn, _ = coh_cols(S); thr = sigD_P * (norm.isf(alpha / (2 * k)) + (drift if widen else 0.0))
    hit = det = fa = 0; cache = {}
    for _ in range(trials):
        e = int(rng.integers(P.E)); th = min(snr * sigD_P / D / P.traffic[e], 1.0)
        if skew > 0:
            K2, _, _ = P.faulty(e, -th); mss = np.linalg.solve(K2, P.f); ages = D * (1 + skew * rng.random(k))
            r = np.array([H8[c] @ (mss + expm(-K2 * ages[c]) @ (P.mbar - mss) - P.mbar) for c in range(k)])
        else:
            if e not in cache: cache[e] = H8 @ P.response(e, -th, D)[0]
            r = cache[e]
        b = drift * sigD_P * rng.choice([-1.0, 1.0], k); nu = sigD_P * rng.standard_normal(k)
        rho = r + b + nu; hit += int(np.argmax(np.abs(Sn.T @ rho)) == e); det += np.abs(rho).max() > thr; fa += np.abs(b + nu).max() > thr
    return hit / trials, det / trials, fa / trials
skew_rows = [(sk,) + skew_drift(skew=sk, trials=1500) for sk in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)]
drift_rows = [(dr,) + skew_drift(drift=dr) + skew_drift(drift=dr, widen=True) for dr in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0)]
R["skew"] = skew_rows; R["drift"] = drift_rows
T_["skewrows"] = "".join(f"{sk:g} & {p1(h)} & {p1(d)} \\\\\n" for sk, h, d, _ in skew_rows)
T_["driftrows"] = "".join(f"{r[0]:g} & {p1(r[1])} & {p1(r[2])} & {p1(r[3])} & {p1(r[4])} & {p1(r[5])} & {p1(r[6])} \\\\\n" for r in drift_rows)

# ============================================================================
# Output
# ============================================================================
with open(os.path.join(OUT, "numbers_rev2.tex"), "w") as fh:
    fh.write("% Generated by code/revision2_experiments.py -- do not edit by hand\n")
    for k_, v_ in M.items(): fh.write(f"\\newcommand{{\\{k_}}}{{{v_}}}\n")
    for k_, v_ in T_.items(): fh.write(f"\\newcommand{{\\{k_}}}{{%\n{v_}}}\n")
R["macros"] = M
json.dump(R, open(os.path.join(OUT, "results_rev2.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "lines.markersize": 5, "legend.frameon": False, "pdf.fonttype": 42})
C1, C2, C3, REF = "#0072B2", "#D55E00", "#009E73", "#555555"
fig, ax = plt.subplots(figsize=(3.6, 2.7))
ax.axhline(1.0, color=REF, ls=":", lw=1.2, label="law (upper bound)")
ax.plot(kk, eff_block, "-o", color=C1, label="balanced pools of a lean cover")
ax.plot(list(eff_real), list(eff_real.values()), "s", color=C2, label="real-weighted isolation designs")
ax.plot([4], [eff_bin], "^", color=C3, label="four binary isolation pools")
ax.set_xlabel("number of inventory channels $k$"); ax.set_ylabel(r"worst-edge sensitivity $\times\ M^*/k$")
ax.set_ylim(0, 1.12); ax.set_xticks([1, 4, 8, 12, 16, 20]); ax.legend(fontsize=6.5, loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_law.pdf")); plt.close(fig)

# G. flowsheet and pooled-sampler schematic
fig = plt.figure(figsize=(7.0, 3.5)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
INK, MUTE, AQ, ORG = "#222222", "#777777", C1, C2
def box(x, y, w_, h_, txt, fc="white", fs=7, ec=INK):
    ax.add_patch(FancyBboxPatch((x, y), w_, h_, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w_ / 2, y + h_ / 2, txt, ha="center", va="center", fontsize=fs, color=INK)
def arr(p, q, col, ls="-", lw=1.1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=lw, color=col, linestyle=ls, shrinkA=0, shrinkB=0))
ax.text(0.2, 6.75, "(a) closed-loop circuit", fontsize=8, weight="bold", va="top")
box(0.6, 4.6, 2.6, 1.0, "extraction\nstages $1\\ldots n_E$"); box(3.9, 4.6, 2.6, 1.0, "scrubbing\n$n_E{+}1\\ldots n_E{+}n_S$"); box(8.4, 4.6, 2.6, 1.0, "stripping\n$\\ldots n$")
arr((3.2, 5.35), (3.9, 5.35), ORG); arr((6.5, 5.35), (8.4, 5.35), ORG); ax.text(7.45, 5.47, "loaded organic", fontsize=6, color=ORG, ha="center")
ax.plot([11.0, 11.6, 11.6, 0.2, 0.2], [5.35, 5.35, 6.2, 6.2, 5.35], color=ORG, lw=1.1); arr((0.2, 5.35), (0.6, 5.35), ORG)
ax.text(5.9, 6.3, "stripped organic returned (ring edge: one flux channel here opens the loop)", fontsize=6, color=ORG, ha="center")
arr((3.9, 4.85), (3.2, 4.85), AQ); arr((7.4, 4.85), (6.5, 4.85), AQ); ax.text(6.6, 4.5, "scrub", fontsize=6, color=AQ, ha="left")
arr((3.55, 3.9), (3.55, 4.6), AQ); ax.text(3.65, 3.95, "feed", fontsize=6, color=AQ, ha="left")
arr((0.6, 4.85), (0.0, 4.85), AQ); ax.text(0.05, 4.5, "raffinate", fontsize=6, color=AQ, ha="left")
arr((12.2, 4.85), (11.0, 4.85), AQ); ax.text(11.75, 4.5, "strip acid", fontsize=6, color=AQ, ha="center")
arr((8.4, 4.85), (7.9, 4.85), AQ); arr((7.9, 4.85), (7.9, 4.1), AQ); ax.text(8.0, 4.1, "strip liquor (product)", fontsize=6, color=AQ, ha="left")
ax.text(12.35, 5.9, "aqueous", fontsize=6.5, color=AQ); ax.text(12.35, 5.55, "organic", fontsize=6.5, color=ORG)
ax.text(12.35, 5.2, "boundary streams:\nflow meter + assay", fontsize=6, color=MUTE, va="top")
ax.plot([0.2, 13.8], [3.55, 3.55], color="#cccccc", lw=0.8)
ax.text(0.2, 3.4, "(b) pooled sampler serving one inventory channel", fontsize=8, weight="bold", va="top")
xs = [0.7 + 1.05 * i for i in range(8)]; pooled = {1, 3, 5, 7}
for i, x in enumerate(xs):
    box(x, 1.9, 0.8, 0.7, f"M–S\n{i + 1}", fc="#dbeaf5" if i in pooled else "white", fs=6)
    if i in pooled:
        ax.plot([x + 0.4, x + 0.4], [1.9, 1.25], color=INK, lw=0.8)
        ax.plot([x + 0.28, x + 0.52], [1.55, 1.55], color=INK, lw=2.2)
ax.plot([xs[1] + 0.4, 9.6], [1.25, 1.25], color=INK, lw=0.8)
ax.text(0.7, 0.95, "sample valves on the mixers of the pooled stages,\nopened together (sample and hold);\naliquot $\\propto\\eta_i\\times$ stage hold-up", fontsize=6, color=MUTE, va="top")
box(9.6, 0.85, 1.5, 0.8, "composite\nvessel", fs=6.5); arr((11.1, 1.25), (11.7, 1.25), INK)
box(11.7, 0.85, 1.9, 0.8, "digestion +\nelemental assay", fs=6.5)
ax.text(12.65, 0.6, "reading $y_\\eta=\\eta^\\top m$ + noise", fontsize=6.5, ha="center", color=INK)
ax.text(9.6, 2.55, "error sources: aliquot volume, representativeness\nof the dispersion sample, assay repeatability,\ndrift between baseline and reading", fontsize=6, color=MUTE, va="top")
fig.savefig(os.path.join(OUT, "fig_flowsheet.pdf")); plt.close(fig)
print(open(os.path.join(OUT, "numbers_rev2.tex")).read())
