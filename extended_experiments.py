#!/usr/bin/env python3
"""
Extended numerical study for "Cut-space instrumentation of rare-earth
solvent-extraction cascades" (revision).

Run AFTER cascade_instrumentation.py (it reads ../generated/designs.npz).
Writes ../generated/numbers_ext.tex, results_ext.json, figures and the design
matrices in ../designs/.

Benchmarks
  P : synthetic 40-stage counter-current cascade (path), as in the first script.
  R : 18-stage closed-loop Pr/Nd circuit (ring) parameterised from the measured
      stage-wise distribution coefficients of Lyon (2016), Table 17.  Flows and
      phase hold-up split are NOT given in the source and are assumed (see README).
"""
import itertools, json, math, os, sys, platform
import numpy as np, scipy
from scipy.linalg import expm
from scipy.stats import norm, chi2
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "generated"); DES = os.path.join(HERE, "..", "designs")
os.makedirs(OUT, exist_ok=True); os.makedirs(DES, exist_ok=True)
rng = np.random.default_rng(20261116)
alpha, beta, eps = 0.01, 0.05, 0.05
Q = norm.sf
R = {}                                    # results
M = {}                                    # LaTeX macros

# =============================================================================
# 0. Generic tools
# =============================================================================
class Plant:
    """Compartmental plant  dm/dt = -K m + f  with unmeasured edge list `edges`
    (tail, head); `org[e]` = (source stage, sink stage) of the organic stream on e."""
    def __init__(self, K, f, edges, org):
        self.K, self.f, self.edges, self.org = K, f, edges, org
        self.N = K.shape[0]; self.E = len(edges)
        self.B = np.zeros((self.N, self.E))
        for e, (i, j) in enumerate(edges): self.B[i, e] = -1.0; self.B[j, e] = 1.0
        self.mbar = np.linalg.solve(K, f); self.Kinv = np.linalg.inv(K)
        self.lam = K.diagonal().max()
        # organic one-way rate constant and traffic on each edge
        self.a_org = np.array([-K[snk, src] for (src, snk) in org])
        self.traffic = self.a_org * np.array([self.mbar[src] for (src, _) in org])
    def Psi(self, D): return self.Kinv @ (np.eye(self.N) - expm(-self.K * D))
    def S(self, H, D): return H @ self.Psi(D) @ self.B
    def T(self, H): return H @ self.B
    def faulty(self, e, theta):
        """Organic transfer rate on edge e scaled by (1+theta). Returns K', phi, dalpha."""
        src, snk = self.org[e]; d = theta * self.a_org[e]
        K2 = self.K.copy(); K2[src, src] += d; K2[snk, src] -= d
        sign = 1.0 if (src, snk) == self.edges[e] else -1.0     # flux counted tail -> head
        return K2, sign * d * self.mbar[src], abs(d)
    def response(self, e, theta, D):
        K2, phi, da = self.faulty(e, theta)
        mss = np.linalg.solve(K2, self.f)
        return mss + expm(-K2 * D) @ (self.mbar - mss) - self.mbar, phi, da

def coh_cols(S):
    n = np.linalg.norm(S, axis=0); Sn = S / n
    G = np.abs(Sn.T @ Sn); np.fill_diagonal(G, 0); return n, Sn, G
def coherence(S): return coh_cols(S)[2].max()

def build_path(N=40):
    qa = qo = 1.0; Dc = np.linspace(1.30, 0.80, N); kap = 1.0 + Dc
    a = qa / kap; b = qo * Dc / kap
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] = a[i] + b[i]
        if i + 1 < N: K[i + 1, i] = -a[i]
        if i - 1 >= 0: K[i - 1, i] = -b[i]
    f = np.zeros(N); f[0] = 1.0
    edges = [(i, i + 1) for i in range(N - 1)]
    org = [(i + 1, i) for i in range(N - 1)]          # organic runs from i+1 to i
    return Plant(K, f, edges, org)

# Lyon (2016), Table 17: measured D_Nd in the 18-stage PC88A flowsheet test
D_LYON = [0.56, 1.24, 0.67, 0.65, 0.59, 0.57, 0.60, 0.51, 0.60, 0.57, 0.56, 0.58, 0.57, 0.62,
          0.0008, 0.002, 0.002, 0.002]      # stages 17-18 not reported: value of stage 16 assumed
def build_ring():
    n = 18; Va = Vo = 300.0                 # mL; mixer 120 + settler 480 (source); split assumed
    qo = qf = qs = qt = 5.0                 # mL/min; assumed, of the order of the one reported flow
    cf, cs = 7.5, 10.0                      # g/L Nd in feed (10 g/L didymium, 75 % Nd) and scrub
    Dc = np.array(D_LYON); kap = Va + Dc * Vo
    K = np.zeros((n, n))
    def add(i, j, rate):                    # stream from stage i to j (j = None: environment)
        K[i, i] += rate
        if j is not None: K[j, i] -= rate
    for i in range(n): add(i, (i + 1) % n, qo * Dc[i] / kap[i])      # organic 1->2->...->18->1
    add(1, 0, (qf + qs) / kap[1]); add(0, None, (qf + qs) / kap[0])   # extraction aqueous, raffinate
    for i in range(2, 14): add(i, i - 1, qs / kap[i])                 # scrub aqueous 14 -> 3 -> 2
    for i in range(15, 18): add(i, i - 1, qt / kap[i])                # strip aqueous 18 -> 15
    add(14, None, qt / kap[14])                                       # strip liquor (product)
    f = np.zeros(n); f[1] = qf * cf; f[13] = qs * cs                  # mg/min
    edges = [(i, (i + 1) % n) for i in range(n)]
    org = [(i, (i + 1) % n) for i in range(n)]
    return Plant(K, f, edges, org)

P = build_path(); Rg = build_ring()
npz = np.load(os.path.join(OUT, "designs.npz"))
H8, H4, H12, Hbin = npz["H8"], npz["H4"], npz["H12"], npz["H_binary"]
sigD_P = P.traffic.min() / 40.0            # same noise level as the first script

# ---------------- decoders ----------------
def glr(rho, Sn):                           # matched filter = GLR for one fault, white noise
    return int(np.argmax(np.abs(Sn.T @ rho)))
def lasso_cd(rho, Sn, lam, iters=200):
    n = Sn.shape[1]; x = np.zeros(n); r = rho.copy()
    for _ in range(iters):
        xo = x.copy()
        for j in range(n):
            r += Sn[:, j] * x[j]
            z = Sn[:, j] @ r
            x[j] = math.copysign(max(abs(z) - lam, 0.0), z)
            r -= Sn[:, j] * x[j]
        if np.abs(x - xo).max() < 1e-10 * (1 + np.abs(x).max()): break
    return x
def omp(rho, Sn, s):
    sel = []; r = rho.copy()
    for _ in range(s):
        c = np.abs(Sn.T @ r); c[sel] = -1; sel.append(int(np.argmax(c)))
        A = Sn[:, sel]; r = rho - A @ np.linalg.lstsq(A, rho, rcond=None)[0]
    return sel

# =============================================================================
# 1. Three certificate tiers and the empirical threshold (benchmark P, k = 8)
# =============================================================================
TH = np.geomspace(1e-3, 1.0, 31)            # common grid of relative rate losses
DG = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)

def tiers(pl, H, sigD, thetas=TH, dgrid=DG, trials=300, empirical=True):
    k = H.shape[0]; chi = math.sqrt(chi2.ppf(1 - eps, k)); E = pl.E
    thr = sigD * norm.isf(alpha / (2 * k))
    t1 = np.full(E, np.inf); t2 = np.full(E, np.inf); t3 = np.full(E, np.inf); em = np.full(E, np.inf)
    for D in dgrid:
        S = pl.S(H, D); nrm, Sn, G = coh_cols(S); mue = G.max(axis=0)
        for e in range(E):
            for th in thetas:
                if th >= min(t1[e], t2[e], t3[e], em[e]) and th >= max(t1[e], t2[e], t3[e], em[e]): break
                dm, phi, da = pl.response(e, -th, D); r = H @ dm
                aphi = abs(phi); Rb = 2 * math.sqrt(k) * da * aphi * D * D
                # tier 1: closed form, global coherence argument (Theorem certified)
                if th < t1[e] and aphi * nrm[e] * (1 - mue[e]) > 2 * sigD * chi + 2 * Rb: t1[e] = th
                # tier 2: closed form, pairwise union bound (Theorem refined)
                if th < t2[e]:
                    c = np.delete(G[:, e], e)
                    pb = np.sum(Q((aphi * nrm[e] * np.sqrt((1 - c) / 2) - Rb) / sigD)
                                + Q((aphi * nrm[e] * np.sqrt((1 + c) / 2) - Rb) / sigD))
                    if pb <= eps: t2[e] = th
                # tier 3: exact response of the hypothesised fault, pairwise union bound
                if th < t3[e]:
                    s0 = np.sign(Sn[:, e] @ r); pb = 0.0
                    for sg in (1.0, -1.0):
                        Dm = s0 * Sn[:, [e]] - sg * np.delete(Sn, e, axis=1)
                        dn = np.linalg.norm(Dm, axis=0)
                        pb += np.sum(Q((r @ Dm) / (sigD * dn)))
                    if pb <= eps: t3[e] = th
                if empirical and th < em[e]:
                    rho = r[:, None] + sigD * rng.standard_normal((k, trials))
                    if np.mean(np.argmax(np.abs(Sn.T @ rho), axis=0) == e) >= 0.95: em[e] = th
    return t1, t2, t3, em

t1, t2, t3, em = tiers(P, H8, sigD_P)
def cnt(v, lim): return int(np.sum(v <= lim))
R["tiers_P"] = dict(t1=t1.tolist(), t2=t2.tolist(), t3=t3.tolist(), em=em.tolist())
for nm, v in (("One", t1), ("Two", t2), ("Three", t3), ("Emp", em)):
    M["nTier" + nm + "Any"] = cnt(v, 1.0); M["nTier" + nm + "Ten"] = cnt(v, 0.10)
ok = np.isfinite(t3) & np.isfinite(em)
M["nTierThreeFactorMed"] = f"{np.median(t3[ok] / em[ok]):.1f}"; M["nTierThreeFactorMax"] = f"{np.max(t3[ok] / em[ok]):.1f}"
ok1 = np.isfinite(t1) & np.isfinite(em); M["nTierOneFactorMed"] = f"{np.median(t1[ok1] / em[ok1]):.1f}"
ok2 = np.isfinite(t2) & np.isfinite(em); M["nTierTwoFactorMed"] = f"{np.median(t2[ok2] / em[ok2]):.1f}"
# soundness: at each audited threshold simulate the full monitor (alarm AND correct edge)
def joint_at(pl, H, sigD, e, th, D, trials=5000):
    k = H.shape[0]; _, Sn, _ = coh_cols(pl.S(H, D)); thr = sigD * norm.isf(alpha / (2 * k))
    r = H @ pl.response(e, -th, D)[0]; rho = r[:, None] + sigD * rng.standard_normal((k, trials))
    iso = np.argmax(np.abs(Sn.T @ rho), axis=0) == e; det = np.abs(rho).max(axis=0) > thr
    return float(np.mean(iso)), float(np.mean(iso & det))
def audit_window(pl, H, sigD, e, th, dgrid=DG):
    """First window of the grid at which the exact-response audit certifies both isolation and detection."""
    k = H.shape[0]; thr = sigD * norm.isf(alpha / (2 * k))
    for D in dgrid:
        _, Sn, _ = coh_cols(pl.S(H, D)); r = H @ pl.response(e, -th, D)[0]; s0 = np.sign(Sn[:, e] @ r); pb = 0.0
        for sg in (1.0, -1.0):
            Dm = s0 * Sn[:, [e]] - sg * np.delete(Sn, e, axis=1)
            pb += np.sum(Q((r @ Dm) / (sigD * np.linalg.norm(Dm, axis=0))))
        pm = Q((np.abs(r).max() - thr) / sigD)
        if pb <= eps and pm <= beta: return D, pb, pm
    return None
iso_min = joint_min = 1.0; n_joint = 0
for e in range(P.E):
    if not np.isfinite(t3[e]): continue
    for th in (t3[e], min(1.0, 2 * t3[e])):
        aw = audit_window(P, H8, sigD_P, e, th)
        if aw is None: continue
        pi_, pj_ = joint_at(P, H8, sigD_P, e, th, aw[0]); n_joint += 1
        iso_min = min(iso_min, pi_); joint_min = min(joint_min, pj_)
M["nSoundCases"] = n_joint; M["nSoundIsoMin"] = f"{100 * iso_min:.1f}"; M["nSoundJointMin"] = f"{100 * joint_min:.1f}"
# against assay precision
prec_rows = []
mmean = P.mbar.mean()
for rel in (1e-5, 1e-4, 1e-3, 1e-2):
    a1, a2, a3, ae = tiers(P, H8, math.sqrt(2) * rel * mmean, trials=200)
    prec_rows.append((rel, cnt(a1, 1), cnt(a2, 1), cnt(a3, 1), cnt(ae, 1), cnt(a1, .1), cnt(a2, .1), cnt(a3, .1), cnt(ae, .1)))
R["precision_tiers"] = prec_rows

# =============================================================================
# 2. Window sensitivity (benchmark P, k = 8)
# =============================================================================
Dsw = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
def iso_prob(pl, H, D, phi, sigD, decoder="dynamic", trials=400, Hdec=None):
    k = H.shape[0]; Sd = pl.S(H, D) if decoder == "dynamic" else pl.T(H)
    _, Sn, _ = coh_cols(Sd); thr = sigD * norm.isf(alpha / (2 * k)); hit = det = 0; cache = {}
    for _ in range(trials):
        e = int(rng.integers(pl.E))
        if e not in cache:
            th = min(phi / pl.traffic[e], 1.0); cache[e] = H @ pl.response(e, -th, D)[0]
        rho = cache[e] + sigD * rng.standard_normal(k)
        det += np.abs(rho).max() > thr; hit += glr(rho, Sn) == e
    return det / trials, hit / trials
sweep = {}
for lvl in (4.0, 16.0, 64.0):               # anomaly rate phi = lvl * sigD per unit time
    sweep[lvl] = dict(dyn=[iso_prob(P, H8, D, lvl * sigD_P, sigD_P)[1] for D in Dsw],
                      sta=[iso_prob(P, H8, D, lvl * sigD_P, sigD_P, "static")[1] for D in Dsw])
R["window_sweep"] = {str(k): v for k, v in sweep.items()}
# exact-audit threshold against the window for three edges (high, median, low traffic)
order = np.argsort(-P.traffic); e_hi, e_md, e_lo = int(order[0]), int(order[len(order) // 2]), int(order[-1])
audit_vs_D = {}
for e in (e_hi, e_md, e_lo):
    row = []
    for D in Dsw:
        S = P.S(H8, D); _, Sn, _ = coh_cols(S); val = np.nan
        for th in np.geomspace(1e-4, 1.0, 81):
            r = H8 @ P.response(e, -th, D)[0]; s0 = np.sign(Sn[:, e] @ r); pb = 0.0
            for sg in (1.0, -1.0):
                Dm = s0 * Sn[:, [e]] - sg * np.delete(Sn, e, axis=1)
                pb += np.sum(Q((r @ Dm) / (sigD_P * np.linalg.norm(Dm, axis=0))))
            if pb <= eps: val = th; break
        row.append(val)
    audit_vs_D[e] = row
R["audit_vs_window"] = {str(k): v for k, v in audit_vs_D.items()}
best = {e: Dsw[int(np.nanargmin(v))] for e, v in audit_vs_D.items()}
M["nBestWinHi"] = f"{best[e_hi]:g}"; M["nBestWinMd"] = f"{best[e_md]:g}"; M["nBestWinLo"] = f"{best[e_lo]:g}"

# =============================================================================
# 3. Combined guarantee with unknown onset and sampling skew (static decoder)
# =============================================================================
def combined_check(pl, H, sigD, Dl, tau, trials=2000):
    k = H.shape[0]; T = pl.T(H); nrm, Tn, G = coh_cols(T); mue = G.max(axis=0)
    chi = math.sqrt(chi2.ppf(1 - eps, k)); zz = norm.isf(alpha / (2 * k)) + norm.isf(beta)
    thr = sigD * norm.isf(alpha / (2 * k)); tinf = np.abs(T).max(axis=0)
    cases = 0; worst = 1.0
    for e in range(pl.E):
        for th in np.geomspace(1e-3, 1.0, 16):
            K2, phi, da = pl.faulty(e, -th); aphi = abs(phi); cst = pl.lam + 2 * da
            okc = True
            for a in (Dl, 2 * Dl):
                c_det = aphi * (a * tinf[e] - cst * (a + tau) ** 2) >= sigD * zz
                c_iso = aphi * a * nrm[e] * (1 - mue[e]) > 2 * sigD * chi + 2 * math.sqrt(k) * aphi * (tau + cst * (a + tau) ** 2)
                okc &= bool(c_det and c_iso)
            if not okc: continue
            cases += 1
            mss = np.linalg.solve(K2, pl.f); succ = 0
            ages = Dl + Dl * rng.random(24)                       # age at the second sampling instant
            per = trials // len(ages)
            for a in ages:
                offs = tau * rng.random(k)                        # sequential reading of the channels
                r = np.array([H[c] @ (mss + expm(-K2 * (a + offs[c])) @ (pl.mbar - mss) - pl.mbar) for c in range(k)])
                rho = r[:, None] + sigD * rng.standard_normal((k, per))
                succ += np.sum((np.abs(rho).max(axis=0) > thr) & (np.argmax(np.abs(Tn.T @ rho), axis=0) == e))
            worst = min(worst, succ / (per * len(ages)))
    return cases, worst
comb = None
for scale in (1.0, 0.1, 0.01, 0.001):
    for Dl in (0.02, 0.01, 0.005):
        cases, worst = combined_check(P, H8, scale * sigD_P, Dl, Dl / 100)
        if cases >= 5 and comb is None: comb = (scale, Dl, cases, worst)
    if comb: break
R["combined"] = comb
M["nCombScale"] = f"{1 / comb[0]:g}"; M["nCombDelta"] = f"{comb[1]:g}"; M["nCombCases"] = comb[2]; M["nCombWorst"] = f"{100 * comb[3]:.1f}"
M["nCombCasesBench"] = sum(combined_check(P, H8, sigD_P, Dl, Dl / 100, trials=48)[0] for Dl in (0.02, 0.01, 0.005))

# =============================================================================
# 4. Conventional placement: point sensors against pooled channels (k = 8)
# =============================================================================
def point_H(idx):
    H = np.zeros((len(idx), P.N)); H[np.arange(len(idx)), idx] = 1.0; return H
even = np.round(np.linspace(2, P.N - 3, 8)).astype(int)
def dopt_greedy(D, k=8, frac=0.10):
    """Greedy Bayesian D-optimal choice of k stages: maximise log det(I + S W S'/sigma^2)."""
    W = np.full(P.E, (16.0 * sigD_P) ** 2); Psi_B = P.Psi(D) @ P.B; chosen = []
    for _ in range(k):
        bestv, bestj = -np.inf, None
        for j in range(P.N):
            if j in chosen: continue
            S = Psi_B[chosen + [j], :]
            v = np.linalg.slogdet(np.eye(len(chosen) + 1) + (S * W) @ S.T / sigD_P ** 2)[1]
            if v > bestv: bestv, bestj = v, j
        chosen.append(bestj)
    return sorted(chosen)
dopt1 = dopt_greedy(1.0); dopt4 = dopt_greedy(4.0)
H_even, H_d1, H_d4 = point_H(even), point_H(dopt1), point_H(dopt4)
R["point_even"] = (even + 1).tolist(); R["point_dopt1"] = [j + 1 for j in dopt1]; R["point_dopt4"] = [j + 1 for j in dopt4]
def static_cov(H):
    T = P.T(H); n = np.linalg.norm(T, axis=0); seen = int(np.sum(n > 0))
    Tn = T[:, n > 0] / n[n > 0]; G = np.abs(Tn.T @ Tn); np.fill_diagonal(G, 0)
    iso = int(np.sum(G.max(axis=0) < 1 - 1e-9)); return seen, iso
cmp_rows = []
for name, H, Dopt in (("Pooled, real weights", H8, None), ("Points, evenly spaced", H_even, None),
                      ("Points, D-optimal", H_d1, 1.0)):
    seen, iso = static_cov(H); row = [name, seen, iso]
    for D in (1.0, 4.0, 16.0):
        Hh = H if Dopt is None else point_H(dopt_greedy(D))
        d, i = iso_prob(P, Hh, D, 16.0 * sigD_P, sigD_P, trials=600); row += [d, i]
    cmp_rows.append(row)
R["placement"] = cmp_rows

# =============================================================================
# 5. Sparse fault diagnosis: GLR, LASSO, OMP, exhaustive (k = 8 and 12)
# =============================================================================
def single_fault_methods(H, D=1.0, snrs=(8, 16, 24, 40), trials=400):
    k = H.shape[0]; _, Sn, _ = coh_cols(P.S(H, D)); lam = sigD_P * math.sqrt(2 * math.log(P.E)); out = []
    for s in snrs:
        g = l = 0; cache = {}
        for _ in range(trials):
            e = int(rng.integers(P.E))
            if e not in cache: cache[e] = H @ P.response(e, -min(s * sigD_P / D / P.traffic[e], 1.0), D)[0]
            rho = cache[e] + sigD_P * rng.standard_normal(k)
            g += glr(rho, Sn) == e
            x = lasso_cd(rho, Sn, lam); l += (np.abs(x).max() > 0 and int(np.argmax(np.abs(x))) == e)
        out.append((s, g / trials, l / trials))
    return out
R["single_methods"] = single_fault_methods(H8)
def double_fault_methods(H, D=1.0, snrs=(24, 40, 80, 160), trials=300):
    k = H.shape[0]; _, Sn, _ = coh_cols(P.S(H, D)); lam = sigD_P * math.sqrt(2 * math.log(P.E))
    pairs = list(itertools.combinations(range(P.E), 2)); out = []
    Qs = [np.linalg.qr(Sn[:, list(p)])[0] for p in pairs]
    for s in snrs:
        o = l = x_ = 0
        for _ in range(trials):
            e1, e2 = rng.choice(P.E, 2, replace=False); K2 = P.K.copy()
            for e in (e1, e2):
                src, snk = P.org[e]; th = -min(s * sigD_P / D / P.traffic[e], 1.0); d = th * P.a_org[e]
                K2[src, src] += d; K2[snk, src] -= d
            mss = np.linalg.solve(K2, P.f)
            rho = H @ (mss + expm(-K2 * D) @ (P.mbar - mss) - P.mbar) + sigD_P * rng.standard_normal(k)
            truth = {int(e1), int(e2)}
            o += set(omp(rho, Sn, 2)) == truth
            xl = lasso_cd(rho, Sn, lam); l += set(np.argsort(-np.abs(xl))[:2].tolist()) == truth and np.count_nonzero(xl) >= 2
            proj = [np.sum((Qm.T @ rho) ** 2) for Qm in Qs]; x_ += set(pairs[int(np.argmax(proj))]) == truth
        out.append((s, o / trials, l / trials, x_ / trials))
    return out
R["double_methods_k8"] = double_fault_methods(H8); R["double_methods_k12"] = double_fault_methods(H12)

# =============================================================================
# 6. Correlated and biased assay noise (k = 8, Delta = 1)
# =============================================================================
def noise_study(H, D=1.0, snr=24.0, trials=3000):
    k = H.shape[0]; S = P.S(H, D); _, Sn, _ = coh_cols(S); thr = sigD_P * norm.isf(alpha / (2 * k)); rows = []
    def run(make_noise, gain=0.0, drift=0.0, whiten=None):
        hit = fa = 0; cache = {}
        Sw = Sn if whiten is None else coh_cols(whiten @ S)[1]
        for _ in range(trials):
            e = int(rng.integers(P.E))
            if e not in cache: cache[e] = H @ P.response(e, -min(snr * sigD_P / D / P.traffic[e], 1.0), D)[0]
            nu = make_noise() + drift * sigD_P * rng.choice([-1.0, 1.0], k)
            g = 1.0 + gain * (2 * rng.random(k) - 1)
            rho = g * cache[e] + nu
            if whiten is not None: rho = whiten @ rho
            hit += glr(rho, Sw) == e; fa += np.abs(nu).max() > thr
        return hit / trials, fa / trials
    iid = lambda: sigD_P * rng.standard_normal(k)
    def corr(r):
        C = (1 - r) * np.eye(k) + r * np.ones((k, k)); L = np.linalg.cholesky(C)
        return (lambda: sigD_P * (L @ rng.standard_normal(k))), np.linalg.inv(L)
    rows.append(("Independent (reference)",) + run(iid))
    for r in (0.5, 0.9):
        mk, W = corr(r)
        rows.append((f"Common-mode correlation {r:g}",) + run(mk))
        rows.append((f"Correlation {r:g}, whitened decoder",) + run(mk, whiten=W))
    rows.append(("Gain error $\\pm5$\\,\\% per channel",) + run(iid, gain=0.05))
    rows.append(("Gain error $\\pm20$\\,\\% per channel",) + run(iid, gain=0.20))
    rows.append(("Drift bias $1\\,\\sigma_\\Delta$ per window",) + run(iid, drift=1.0))
    rows.append(("Drift bias $3\\,\\sigma_\\Delta$ per window",) + run(iid, drift=3.0))
    return rows
R["noise"] = noise_study(H8)

# =============================================================================
# 7. Nonlinear equilibrium (loading-dependent distribution), benchmark P geometry
# =============================================================================
N = P.N; Dn = np.linspace(1.30, 0.80, N); csat = 1.0
def x_of_m(m):                                   # m = x + D x/(1 + x/c)  (unit hold-ups)
    bq = 1 + Dn - m / csat
    return (-bq + np.sqrt(bq * bq + 4 * m / csat)) / (2 / csat)
def y_of_x(x): return Dn * x / (1 + x / csat)
def rhs_nl(t, m, e=None, theta=0.0):
    x = x_of_m(np.maximum(m[:N], 0)); y = y_of_x(x)
    aq = x.copy(); og = y.copy()                 # unit flows: aqueous i -> i+1, organic i -> i-1
    back = og[1:].copy()
    if e is not None: back[e] *= (1 + theta)
    J = aq[:-1] - back                           # net flux on edge (i, i+1), tail -> head
    dm = np.zeros(N); dm[:-1] -= J; dm[1:] += J
    dm[0] += 1.0 - og[0]; dm[-1] -= aq[-1]
    return np.concatenate([dm, J, [og[0] - 1.0, aq[-1]]]) if len(m) > N else dm
m0 = solve_ivp(rhs_nl, (0, 4000), np.full(N, 0.5), method="LSODA", rtol=1e-10, atol=1e-12).y[:, -1]
m0 = solve_ivp(rhs_nl, (0, 4000), m0, method="LSODA", rtol=1e-11, atol=1e-13).y[:, -1]
x0 = x_of_m(m0); y0 = y_of_x(x0)
h = 1e-6; dxdm = (x_of_m(m0 + h) - x_of_m(m0 - h)) / (2 * h)
dydm = (y_of_x(x_of_m(m0 + h)) - y_of_x(x_of_m(m0 - h))) / (2 * h)
Kl = np.zeros((N, N))
for i in range(N):
    Kl[i, i] = dxdm[i] + dydm[i]
    if i + 1 < N: Kl[i + 1, i] = -dxdm[i]
    if i - 1 >= 0: Kl[i - 1, i] = -dydm[i]
Bp = P.B
def Psi_l(D): return np.linalg.inv(Kl) @ (np.eye(N) - expm(-Kl * D))
traffic_nl = y0[1:]                              # organic one-way traffic on each edge
# conservation check (Theorem carriage) on a faulty nonlinear trajectory
sol = solve_ivp(rhs_nl, (0, 2.0), np.concatenate([m0, np.zeros(N + 1)]), args=(20, -0.5), method="LSODA", rtol=1e-11, atol=1e-13)
dm_nl = sol.y[:N, -1] - m0; jint = sol.y[N:2 * N - 1, -1]; jb = sol.y[2 * N - 1:, -1]
Bfull = np.hstack([Bp, -np.eye(N)[:, [0]], -np.eye(N)[:, [N - 1]]])       # interior edges, then the two boundary edges
cons_err = np.abs(H8 @ dm_nl - (H8 @ Bfull) @ np.concatenate([jint, jb])).max() / np.abs(H8 @ dm_nl).max()
M["nNLcons"] = "10^{%d}" % math.ceil(math.log10(cons_err))
M["nNLloadMin"] = f"{(x0 / csat).min():.2f}"; M["nNLloadMax"] = f"{(x0 / csat).max():.2f}"
def nl_iso(H, D, thetas=(0.05, 0.1, 0.2, 0.4), trials=300, coarse=1.0):
    k = H.shape[0]; S = H @ Psi_l(D) @ Bp; _, Sn, _ = coh_cols(S); rows = []
    sig = coarse * sigD_P * (m0.mean() / P.mbar.mean())   # same relative assay precision
    for th in thetas:
        hit_nl = hit_lin = 0; tot = 0
        for e in range(N - 1):
            r_nl = H @ (solve_ivp(rhs_nl, (0, D), m0, args=(e, -th), method="LSODA", rtol=1e-9, atol=1e-11).y[:, -1] - m0)
            r_lin = -th * traffic_nl[e] * (-1.0) * S[:, e]     # phi = +th * traffic (less back-flow)
            for r, key in ((r_nl, "nl"), (r_lin, "lin")):
                rho = r[:, None] + sig * rng.standard_normal((k, trials))
                c = np.sum(np.argmax(np.abs(Sn.T @ rho), axis=0) == e)
                if key == "nl": hit_nl += c
                else: hit_lin += c
            tot += trials
        rows.append((th, hit_nl / tot, hit_lin / tot))
    return rows
R["nonlinear"] = nl_iso(H8, 1.0); R["nonlinear_coarse"] = nl_iso(H8, 1.0, coarse=30.0)
# deviation of the nonlinear response from the linearised signature, relative
devs = []
for th in (0.05, 0.2, 0.4):
    S = H8 @ Psi_l(1.0) @ Bp; d = []
    for e in range(0, N - 1, 4):
        r_nl = H8 @ (solve_ivp(rhs_nl, (0, 1.0), m0, args=(e, -th), method="LSODA", rtol=1e-9, atol=1e-11).y[:, -1] - m0)
        r_lin = th * traffic_nl[e] * S[:, e]; d.append(np.linalg.norm(r_nl - r_lin) / np.linalg.norm(r_lin))
    devs.append((th, float(np.median(d)), float(np.max(d))))
R["nonlinear_dev"] = devs

# =============================================================================
# 8. Boundary-edge faults and bound checks
# =============================================================================
def boundary_check(pl, H):
    worst = 0.0
    for i in range(pl.N):
        a0 = pl.K[:, i].sum()                    # rate constant to the environment
        if a0 <= 1e-14: continue
        for th in (-0.5, -0.2, 0.3):
            for D in (0.2, 1.0, 3.0):
                K2 = pl.K.copy(); K2[i, i] += th * a0
                mss = np.linalg.solve(K2, pl.f); dm = mss + expm(-K2 * D) @ (pl.mbar - mss) - pl.mbar
                phi = th * a0 * pl.mbar[i]; be = np.zeros(pl.N); be[i] = -1.0
                Rm = H @ dm - phi * (H @ pl.Psi(D) @ be)
                worst = max(worst, np.abs(Rm).max() / (2 * abs(th * a0) * abs(phi) * D * D))
    return worst
M["nBoundaryRatio"] = f"{boundary_check(P, H8):.2f}"
def interior_check(pl, H, scaleD=1.0):
    w1 = w2 = 0.0
    for e in range(pl.E):
        for th in (-0.05, -0.5, -1.0):
            for D in np.array([0.2, 1.0, 3.0]) * scaleD:
                dm, phi, da = pl.response(e, th, D)
                if abs(phi) < 1e-6 * np.abs(pl.traffic).max(): continue
                w1 = max(w1, np.abs(H @ dm - phi * pl.S(H, D)[:, e]).max() / (2 * da * abs(phi) * D * D))
                w2 = max(w2, np.abs(pl.S(H, D)[:, e] - D * pl.T(H)[:, e]).max() / (pl.lam * D * D))
    return w1, w2

# =============================================================================
# 9. Explicit binary designs: the 4 x 40 search design and the BCH construction
# =============================================================================
np.savetxt(os.path.join(DES, "H_binary_k4_N40.csv"), Hbin, fmt="%d", delimiter=",")
for k_, Hh in (("k4", H4), ("k8", H8), ("k12", H12)):
    np.savetxt(os.path.join(DES, f"H_real_{k_}_N40.csv"), Hh, fmt="%.6f", delimiter=",")
def gf_mul(a, b, r, poly):
    res = 0
    while b:
        if b & 1: res ^= a
        b >>= 1; a <<= 1
        if a >> r: a ^= poly
    return res
PRIM = {3: 0b1011, 4: 0b10011, 5: 0b100101, 6: 0b1000011}
def bch_labels(r):
    poly = PRIM[r]; n = 2 ** r - 1; lab = []; x = 1
    for i in range(n):
        x3 = gf_mul(gf_mul(x, x, r, poly), x, r, poly)
        lab.append([(x >> t) & 1 for t in range(r)] + [(x3 >> t) & 1 for t in range(r)])
        x = gf_mul(x, 2, r, poly)
    return np.array(lab).T                         # 2r x (2^r - 1)
bch_ok = {}
for r in (3, 4, 5, 6):
    Hb = bch_labels(r); L = np.hstack([np.zeros((2 * r, 1), int), Hb])   # column 0 = environment node
    diffs = {}
    good = True
    for i, j in itertools.combinations(range(L.shape[1]), 2):
        d = tuple(L[:, j] - L[:, i]); d = d if next(v for v in d if v != 0) > 0 else tuple(-v for v in d)
        if d in diffs: good = False
        diffs[d] = (i, j)
    bch_ok[r] = good
    np.savetxt(os.path.join(DES, f"H_BCH_r{r}.csv"), Hb, fmt="%d", delimiter=",")
assert all(bch_ok.values())
R["bch_ok"] = bch_ok
bch3 = bch_labels(3)

# =============================================================================
# 10. Benchmark R: 18-stage closed-loop circuit with published distribution data
# =============================================================================
n = Rg.N
ev = np.linalg.eigvals(Rg.K)
M["nRingLamInv"] = f"{1 / Rg.lam:.0f}"                          # fastest stage turnover time, min
M["nRingTauSlowH"] = f"{1 / ev.real.min() / 60:.1f}"            # slowest time constant, hours
M["nRingSettleH"] = f"{4 / ev.real.min() / 60:.0f}"             # four time constants, hours
prod = Rg.K[:, 14].sum() * Rg.mbar[14]; raff = Rg.K[:, 0].sum() * Rg.mbar[0]
M["nRingRecovery"] = f"{100 * prod / (prod + raff):.0f}"
assert abs(prod + raff - Rg.f.sum()) < 1e-8 * Rg.f.sum()
loaded = Rg.traffic > 1e-2 * Rg.traffic.max()
M["nRingLoaded"] = int(loaded.sum()); M["nRingTrafficSpan"] = f"{Rg.traffic[loaded].max() / Rg.traffic[loaded].min():.1f}"
# structure
Hone = np.zeros((1, n)); Hone[0, 1::2] = 1.0
assert np.all(np.abs(Rg.T(Hone)).max(axis=0) == 1.0)             # one alternating pool separates all 18 edges
assert np.linalg.matrix_rank(Rg.B) == n - 1                      # one circulation
circ = np.ones(n); assert np.abs(Rg.B @ circ).max() == 0         # invisible to every inventory channel
# binary closed walk: 18 pairwise non-parallel signatures with k = 4
def canon(d):
    for v in d:
        if v != 0: return d if v > 0 else tuple(-y for y in d)
def closed_walk(k, n_edges):
    verts = list(itertools.product((0, 1), repeat=k))
    nb = {v: [(w, canon(tuple(b_ - a_ for a_, b_ in zip(v, w)))) for w in verts if w != v] for v in verts}
    used = set(); path = [verts[0]]; cnt_ = [0]
    def dfs(v):
        cnt_[0] += 1
        if cnt_[0] > 2_000_000: return False
        if len(path) == n_edges:
            cl = canon(tuple(b_ - a_ for a_, b_ in zip(v, path[0])))
            return v != path[0] and cl not in used
        for w, cl in nb[v]:
            if cl in used: continue
            used.add(cl); path.append(w)
            if dfs(w): return True
            used.discard(cl); path.pop()
        return False
    return np.array(path, float).T if dfs(verts[0]) else None
kc = math.ceil(math.log(2 * n + 1, 3) - 1e-12)
HRb = None
for k in (kc, kc + 1):
    HRb = closed_walk(k, n)
    if HRb is not None: break
assert coherence(Rg.T(HRb)) < 1 - 1e-9
M["nRingKcount"] = kc; M["nRingKbin"] = HRb.shape[0]; M["nRingMuBin"] = f"{coherence(Rg.T(HRb)):.3f}"
np.savetxt(os.path.join(DES, "H_binary_ring_n18.csv"), HRb, fmt="%d", delimiter=",")
# real-weighted design by stochastic hill-climbing on the dynamic signature matrix
def hill(pl, k, D, iters=6000, seed=1):
    r_ = np.random.default_rng(seed); H = r_.random((k, pl.N))
    def obj(Hh):
        nrm, _, G = coh_cols(pl.S(Hh, D)); return (1 - G.max()) * nrm.min() / D
    best = obj(H); step = 0.3
    for it in range(iters):
        Hn = H.copy(); i = r_.integers(k); j = r_.integers(pl.N)
        Hn[i, j] = min(1.0, max(0.0, Hn[i, j] + step * r_.standard_normal()))
        v = obj(Hn)
        if v > best: H, best = Hn, v
        if it % 1500 == 1499: step *= 0.5
    return H, best
D_R = 1 / Rg.lam                                                 # window: one fastest turnover time
bestH, bestv = None, -1
for sd in range(4):
    Hc, v = hill(Rg, 6, D_R, seed=sd)
    if v > bestv: bestH, bestv = Hc, v
HR6 = bestH
np.savetxt(os.path.join(DES, "H_real_k6_ring_n18.csv"), HR6, fmt="%.6f", delimiter=",")
M["nRingMuReal"] = f"{coherence(Rg.S(HR6, D_R)):.3f}"; M["nRingMuRealStatic"] = f"{coherence(Rg.T(HR6)):.3f}"
w1, w2 = interior_check(Rg, HR6, scaleD=D_R); M["nRingRemRatio"] = f"{w1:.2f}"; M["nRingSwRatio"] = f"{w2:.2f}"
w1p, w2p = interior_check(P, H8); M["nPathRemRatio"] = f"{w1p:.2f}"; M["nPathSwRatio"] = f"{w2p:.2f}"
dgR = tuple(D_R * np.array([0.25, 0.5, 1, 2, 4, 8]))
ring_rows = []
for rel in (1e-3, 1e-2):
    sg = math.sqrt(2) * rel * Rg.mbar.mean()
    for nm, Hh in (("six real-weighted", HR6), ("four binary", HRb)):
        a1, a2, a3, ae = tiers(Rg, Hh, sg, dgrid=dgR, trials=300)
        ring_rows.append((rel, nm, cnt(a1, 1), cnt(a2, 1), cnt(a3, 1), cnt(ae, 1), cnt(a3, .1), cnt(ae, .1)))
R["ring"] = ring_rows
# detection by the single alternating pool on the ring
def det_prob(pl, H, D, th, sigD, trials=400):
    """Smallest detection probability over the edges whose organic stream carries the element."""
    k = H.shape[0]; thr = sigD * norm.isf(alpha / (2 * k)); d = []
    for e in range(pl.E):
        if pl.traffic[e] < 1e-2 * pl.traffic.max(): continue
        r = H @ pl.response(e, -th, D)[0]
        d.append(np.mean(np.abs(r[:, None] + sigD * rng.standard_normal((k, trials))).max(axis=0) > thr))
    return float(np.min(d))
sgR = math.sqrt(2) * 1e-3 * Rg.mbar.mean()
M["nRingDetOneTen"] = f"{100 * det_prob(Rg, Hone, 4 * D_R, 0.10, sgR):.0f}"
M["nRingDetOneThirty"] = f"{100 * det_prob(Rg, Hone, 4 * D_R, 0.30, sgR):.0f}"

# =============================================================================
# 11. Output: macros, tables, figures
# =============================================================================
def pc(x, d=0): return f"{100 * x:.{d}f}"
def relfmt(x): return f"{100 * x:.3f}".rstrip("0").rstrip(".")
T_ = {}
T_["tierrows"] = "".join(f"{relfmt(r[0])} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} & {r[6]} & {r[7]} & {r[8]} \\\\\n" for r in prec_rows)
T_["placementrows"] = "".join(f"{r[0]} & {r[1]} & {r[2]} & {pc(r[3])} & {pc(r[4])} & {pc(r[5])} & {pc(r[6])} & {pc(r[7])} & {pc(r[8])} \\\\\n" for r in cmp_rows)
T_["singlerows"] = "".join(f"{s:g} & {pc(g)} & {pc(l)} \\\\\n" for s, g, l in R["single_methods"])
T_["doublerows"] = "".join(f"{a[0]:g} & {pc(a[1])} & {pc(a[2])} & {pc(a[3])} & {pc(b_[1])} & {pc(b_[2])} & {pc(b_[3])} \\\\\n"
                           for a, b_ in zip(R["double_methods_k8"], R["double_methods_k12"]))
T_["noiserows"] = "".join(f"{nm} & {pc(h_, 1)} & {pc(fa, 1)} \\\\\n" for nm, h_, fa in R["noise"])
T_["nlrows"] = "".join(f"{pc(a[0])} & {pc(a[1], 1)} & {pc(a[2], 1)} & {pc(b_[1], 1)} & {pc(b_[2], 1)} \\\\\n" for a, b_ in zip(R["nonlinear"], R["nonlinear_coarse"]))
T_["ringrows"] = "".join(f"{relfmt(r[0])} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} & {r[6]} & {r[7]} \\\\\n" for r in ring_rows)
T_["windowrows"] = "".join(f"{D:g} & " + " & ".join(f"{pc(sweep[l]['dyn'][i])} & {pc(sweep[l]['sta'][i])}" for l in (4.0, 16.0, 64.0)) + " \\\\\n"
                           for i, D in enumerate(Dsw))
def bits(Hm): return "".join("\\texttt{" + "".join(str(int(v)) for v in row) + "} \\\\\n" for row in Hm)
T_["binrows"] = bits(Hbin); T_["bchrows"] = bits(bch3); T_["ringbinrows"] = bits(HRb)
M["nNLdevMedLow"] = pc(devs[0][1], 1); M["nNLdevMedHigh"] = pc(devs[-1][1], 1); M["nNLdevMaxHigh"] = pc(devs[-1][2], 1)
M["nPointEven"] = ", ".join(str(v) for v in R["point_even"]); M["nPointDopt"] = ", ".join(str(v) for v in R["point_dopt1"])
M["nDLyon"] = ", ".join(f"{v:g}" for v in D_LYON[:14])
with open(os.path.join(OUT, "numbers_ext.tex"), "w") as fh:
    fh.write("% Generated by code/extended_experiments.py -- do not edit by hand\n")
    for k_, v_ in M.items(): fh.write(f"\\newcommand{{\\{k_}}}{{{v_}}}\n")
    for k_, v_ in T_.items(): fh.write(f"\\newcommand{{\\{k_}}}{{%\n{v_}}}\n")
R["macros"] = M
R["environment"] = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
json.dump(R, open(os.path.join(OUT, "results_ext.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "lines.markersize": 5, "legend.frameon": False, "pdf.fonttype": 42})
C1, C2, C3, REF = "#0072B2", "#D55E00", "#009E73", "#555555"
# readiness map with the three certificate tiers
fig, ax = plt.subplots(figsize=(5.2, 3.0)); idx = np.arange(1, P.E + 1)
def nanify(v): return np.where(np.isfinite(v), v, np.nan)
ax.semilogy(idx, nanify(t1), "o", color=C1, label="closed form, coherence")
ax.semilogy(idx, nanify(t2), "s", color=C2, mfc="none", label="closed form, pairwise")
ax.semilogy(idx, nanify(t3), "D", color=REF, ms=4, label="exact-response audit")
ax.semilogy(idx, nanify(em), "^", color=C3, mfc="none", label="simulation, 95 % isolation")
ax.axhline(1.0, color=REF, ls=":", lw=1)
ax.set_xlabel("edge index $e$ (between stages $e$ and $e+1$)"); ax.set_ylabel(r"smallest isolable rate loss $|\theta|$")
ax.legend(fontsize=7, loc="lower right", ncol=1)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_readiness.pdf")); plt.close(fig)
# window sensitivity
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
ax = axs[0]
for lvl, col, mk in ((4.0, C1, "o"), (16.0, C2, "s"), (64.0, C3, "^")):
    ax.plot(Dsw, sweep[lvl]["dyn"], "-", marker=mk, color=col, label=rf"$|\phi|/\sigma_\Delta={lvl:g}$, dynamic")
    ax.plot(Dsw, sweep[lvl]["sta"], "--", marker=mk, color=col, mfc="none", label=rf"$|\phi|/\sigma_\Delta={lvl:g}$, static")
ax.set_xscale("log", base=2); ax.set_xlabel(r"window length $\Delta$ (units of $1/\lambda$)")
ax.set_ylabel("probability of correct isolation"); ax.set_ylim(0, 1.03); ax.legend(fontsize=6, loc="center left")
ax.set_title("(a) simulated isolation, $k=8$", fontsize=8)
ax = axs[1]
for e, col, mk, nm in ((e_hi, C1, "o", "highest traffic"), (e_md, C2, "s", "median traffic"), (e_lo, C3, "^", "lowest traffic")):
    ax.loglog(Dsw, audit_vs_D[e], "-", marker=mk, color=col, label=f"edge {e + 1} ({nm})")
ax.set_xscale("log", base=2); ax.set_xlabel(r"window length $\Delta$ (units of $1/\lambda$)")
ax.set_ylabel(r"audited threshold $|\theta|$"); ax.legend(fontsize=7)
ax.set_title("(b) exact-response audit", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_window.pdf")); plt.close(fig)
print(json.dumps({k: v for k, v in R.items() if k not in ("tiers_P",)}, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))[:9000])
print(open(os.path.join(OUT, "numbers_ext.tex")).read())
