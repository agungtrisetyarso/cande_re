#!/usr/bin/env python3
"""
Synthetic benchmark for "Cut-space instrumentation of solvent-extraction cascades".

Everything here is a SYNTHETIC illustration: the cascade parameters are chosen by
the authors and are NOT plant data. The script reproduces every number, table and
figure of the numerical section and writes them to ../generated/.

Usage:  python3 cascade_instrumentation.py
"""
import itertools, json, math, os, sys
import numpy as np
from scipy.linalg import expm
from scipy.stats import norm, chi2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generated")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261115)

# ----------------------------------------------------------------------------
# 1. Synthetic counter-current cascade  ->  dissipative exchange network
# ----------------------------------------------------------------------------
N = 40                       # mixer-settler stages
qa, qo = 1.0, 1.0            # aqueous / organic flow (volume per unit time)
Va = np.ones(N); Vo = np.ones(N)          # phase hold-ups per stage
D = np.linspace(1.30, 0.80, N)            # linearised distribution coefficients
kappa = Va + D * Vo                        # m_i = kappa_i x_i
a = qa / kappa                             # rate i -> i+1 (aqueous)
b = qo * D / kappa                         # rate i -> i-1 (organic)
pi = np.ones(N)
for i in range(N - 1):
    pi[i + 1] = pi[i] * a[i] / b[i + 1]    # detailed balance on the path
c = pi[:-1] * a[:-1]                       # edge conductances c_i = pi_i a_i
E = N - 1
Bint = np.zeros((N, E))                    # +1 at head (i+1), -1 at tail (i)
for e in range(E):
    Bint[e, e] = -1.0; Bint[e + 1, e] = 1.0
gam = np.zeros(N); gam[0] = pi[0] * b[0]; gam[-1] = pi[-1] * a[-1]
Pi = np.diag(pi); Gam = np.diag(gam)
def lap(cv): return Bint @ np.diag(cv) @ Bint.T
L = lap(c)
f = np.zeros(N); f[0] = 1.0                # unit solute feed into stage 1
ubar = np.linalg.solve(L + Gam, f)         # nominal steady state
g = ubar[:-1] - ubar[1:]                   # driving force on each edge (tail - head)
J = c * g                                  # steady through-flux (constant along path)
assert np.allclose(J, J[0]) and np.all(g > 0)
mbar = pi * ubar
deg = np.zeros(N); deg[:-1] += c; deg[1:] += c
Lam = math.sqrt(np.sum((deg + gam) ** 2 / pi))          # Lambda of Theorem (onset)
varpi = np.sqrt(1 / pi[:-1] + 1 / pi[1:])               # varpi_e
tau_slow = 1 / np.min(np.linalg.eigvals(np.linalg.solve(Pi, L + Gam)).real)

# ----------------------------------------------------------------------------
# 2. Design utilities
# ----------------------------------------------------------------------------
def signatures(H):            # T = H Bint  (k x E)
    return H @ Bint
def coherence(T):
    n = np.linalg.norm(T, axis=0)
    G = np.abs((T / n).T @ (T / n)); np.fill_diagonal(G, 0)
    return G.max()
def margins(T):
    return np.abs(T).max(axis=0).min(), np.linalg.norm(T, axis=0).min()
def welch(k, n):
    return math.sqrt(max(n - k, 0) / (k * (n - 1)))

def frame(k, n, iters=4000, restarts=6):
    """n unit vectors in R^k with small mutual coherence (Grassmannian packing)."""
    if k == 2:
        th = np.pi * np.arange(n) / n
        return np.vstack([np.cos(th), np.sin(th)])
    best, bestmu = None, 2.0
    for _ in range(restarts):
        X = rng.standard_normal((k, n)); X /= np.linalg.norm(X, axis=0)
        p, lr = 8, 0.02
        for it in range(iters):
            G = X.T @ X; np.fill_diagonal(G, 0)
            grad = X @ (np.sign(G) * np.abs(G) ** (p - 1))
            X -= lr * grad / (np.linalg.norm(grad) + 1e-12)
            X /= np.linalg.norm(X, axis=0)
            if it == iters // 2: p, lr = 24, 0.005
        mu = coherence(X)
        if mu < bestmu: best, bestmu = X.copy(), mu
    return best

def realise_on_path(X):
    """Order and sign the columns of X so partial sums stay small, integrate along
    the path, then shift/scale into the box [0,1]^{k x N}. Coherence is unchanged."""
    k, n = X.shape
    unused = list(range(n)); s = np.zeros(k); cols = []
    centre = np.zeros(k)
    while unused:
        bestv = None
        for j in unused:
            for sg in (1.0, -1.0):
                v = np.abs(s + sg * X[:, j] - centre).max()
                if bestv is None or v < bestv[0]: bestv = (v, j, sg)
        _, j, sg = bestv
        s = s + sg * X[:, j]; cols.append(sg * X[:, j]); unused.remove(j)
    T = np.array(cols).T
    P = np.hstack([np.zeros((k, 1)), np.cumsum(T, axis=1)])        # node labels
    P -= P.min(axis=1, keepdims=True)
    scale = P.max()                                                # common scale
    return P / scale

def iso_design(k):
    H = realise_on_path(frame(k, E))
    assert H.min() >= -1e-12 and H.max() <= 1 + 1e-12
    return H

# ----------------------------------------------------------------------------
# 3. Detection budgets with a port cap (Proposition: path budget)
# ----------------------------------------------------------------------------
def capped_detection_design(p):
    K = math.ceil((N - 1) / (2 * p)); H = np.zeros((K, N))
    for r in range(K):
        for j in range(1, p + 1):
            node = 2 * p * r + 2 * j                 # 1-based stage index
            if node <= N: H[r, node - 1] = 1.0
    T = signatures(H)
    assert np.all(np.abs(T).max(axis=0) == 1.0)      # every edge separated, margin 1
    assert np.all((H > 0).sum(axis=1) <= p)
    return K
budget_table = {p: capped_detection_design(p) for p in (2, 4, 8, 12, 20)}

# ----------------------------------------------------------------------------
# 4. Binary single-anomaly isolation on the path: search for a hypercube walk
#    whose N-1 steps are pairwise non-parallel vectors of {-1,0,1}^k
# ----------------------------------------------------------------------------
def canon(d):
    for x in d:
        if x != 0: return d if x > 0 else tuple(-y for y in d)
def binary_walk(k, n_edges, max_nodes=3_000_000):
    verts = list(itertools.product((0, 1), repeat=k))
    nb = {v: [(w, canon(tuple(y - x for x, y in zip(v, w)))) for w in verts if w != v] for v in verts}
    count = [0]; used = set(); path = []
    sys.setrecursionlimit(10000)
    def dfs(v):
        if len(path) == n_edges: return True
        count[0] += 1
        if count[0] > max_nodes: return False
        cand = [(w, cl) for w, cl in nb[v] if cl not in used]
        # Warnsdorff-type ordering: go where fewest unused classes remain
        cand.sort(key=lambda wc: sum(1 for _, c2 in nb[wc[0]] if c2 not in used))
        for w, cl in cand:
            used.add(cl); path.append(w)
            if dfs(w): return True
            used.discard(cl); path.pop()
        return False
    for start in verts[:1]:
        path.clear(); used.clear()
        if dfs(start): return [start] + list(path)
    return None
k_count = math.ceil(math.log(2 * E + 1, 3) - 1e-12)
binary_result = {}
for k in (k_count, k_count + 1):
    w = binary_walk(k, E)
    if w is not None:
        Hb = np.array(w, dtype=float).T
        Tb = signatures(Hb)
        nn = Tb / np.linalg.norm(Tb, axis=0)
        Gm = np.abs(nn.T @ nn); np.fill_diagonal(Gm, 0)
        assert Gm.max() < 1 - 1e-9 and np.all(np.abs(Tb).max(axis=0) == 1)
        binary_result = {"k": k, "mu": float(Gm.max())}
        H_binary = Hb
        break
k_bch = 2 * math.ceil(math.log2(N + 1))

# ----------------------------------------------------------------------------
# 5. Generic-spark check on a small graph with a cycle (Theorem: spark law)
# ----------------------------------------------------------------------------
def spark(T, tol=1e-9):
    n = T.shape[1]
    for r in range(1, n + 1):
        for S in itertools.combinations(range(n), r):
            if np.linalg.matrix_rank(T[:, S], tol=tol) < r: return r
    return math.inf
def spark_check():
    # 9 nodes: a 5-cycle (0..4) with a 4-node tail (4-5-6-7-8); girth 5
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (4, 5), (5, 6), (6, 7), (7, 8)]
    Bs = np.zeros((9, len(edges)))
    for e, (i, j) in enumerate(edges): Bs[i, e] = -1; Bs[j, e] = 1
    out = {}
    for k in (2, 3, 4, 5, 6, 8):
        vals = {spark(rng.random((k, 9)) @ Bs) for _ in range(20)}
        out[k] = sorted(vals)
    return out
spark_table = spark_check()

# ----------------------------------------------------------------------------
# 6. Compartmental form, dynamic signatures, isolation designs
# ----------------------------------------------------------------------------
K = np.zeros((N, N))
for i in range(N):
    K[i, i] = a[i] + b[i]
    if i + 1 < N: K[i + 1, i] = -a[i]
    if i - 1 >= 0: K[i - 1, i] = -b[i]
assert np.allclose(K, (L + Gam) @ np.diag(1 / pi))        # DEN <-> compartmental form
mbar = np.linalg.solve(K, f)
assert np.allclose(mbar, pi * ubar)
lam = K.diagonal().max()                                  # fastest stage turnover rate
traffic = b[1:] * mbar[1:]                                # one-way back-traffic on each edge
Kinv = np.linalg.inv(K)
def Psi(Delta): return Kinv @ (np.eye(N) - expm(-K * Delta))
def dyn_signatures(H, Delta): return H @ Psi(Delta) @ Bint

ks = [2, 3, 4, 6, 8, 12]
designs, design_rows = {}, []
Delta_op = 1.0
for k in ks:
    H = iso_design(k); T = signatures(H); S = dyn_signatures(H, Delta_op)
    ginf, g2 = margins(T); mu = coherence(T)
    sinf, s2 = margins(S); muS = coherence(S)
    designs[k] = H
    design_rows.append(dict(k=k, mu=mu, welch=welch(k, E), g_inf=ginf, g_2=g2, fom=g2 * (1 - mu),
                            muS=muS, s_inf=sinf, s_2=s2, fomS=s2 * (1 - muS)))
mu2_bound = math.cos(math.pi / E)

# short-window bound  || s_e(Delta) - Delta t_e ||_inf <= lam Delta^2
sw_ratio = 0.0
for k in ks:
    for Dl in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0):
        dev = np.abs(dyn_signatures(designs[k], Dl) - Dl * signatures(designs[k])).max()
        sw_ratio = max(sw_ratio, dev / (lam * Dl ** 2))

alpha, beta, eps = 0.01, 0.05, 0.05
sigD = traffic.min() / 40.0            # increment noise (inventory units)
sigma = sigD / math.sqrt(2)

# ----------------------------------------------------------------------------
# 7. Monte Carlo: single-edge rate fault  b_{i+1} -> (1+theta) b_{i+1}
# ----------------------------------------------------------------------------
def response(e, theta, Delta):
    """Exact inventory deviation m(Delta) - mbar; returns it with phi, delta-alpha."""
    K2 = K.copy(); d = theta * b[e + 1]
    K2[e + 1, e + 1] += d; K2[e, e + 1] -= d
    mss = np.linalg.solve(K2, f)
    m = mss + expm(-K2 * Delta) @ (mbar - mss)
    return m - mbar, -d * mbar[e + 1], abs(d)

def run_mc(H, Delta, snrs, decoder="dynamic", trials=600):
    k = H.shape[0]
    D_ = dyn_signatures(H, Delta) if decoder == "dynamic" else signatures(H)
    Dn = D_ / np.linalg.norm(D_, axis=0)
    thr = sigD * norm.ppf(1 - alpha / (2 * k))
    det, iso = [], []
    for snr in snrs:
        phi = snr * sigD / Delta
        d = i = 0
        resp = {}
        for _ in range(trials):
            e = int(rng.integers(E))
            if e not in resp:
                theta = -phi / traffic[e]; assert theta >= -1 - 1e-12
                resp[e] = H @ response(e, theta, Delta)[0]
            rho = resp[e] + sigD * rng.standard_normal(k)
            if np.abs(rho).max() > thr: d += 1
            if np.argmax(np.abs(Dn.T @ rho)) == e: i += 1
        det.append(d / trials); iso.append(i / trials)
    return np.array(det), np.array(iso)

def false_alarm(H, trials=40000):
    k = H.shape[0]; thr = sigD * norm.ppf(1 - alpha / (2 * k))
    return float(np.mean(np.abs(sigD * rng.standard_normal((trials, k))).max(axis=1) > thr))

def remainder_check(H):
    """max over edges / faults / windows of ||R2||_inf / (2 dalpha |phi| Delta^2)."""
    worst = 0.0
    for e in range(E):
        for th in (-0.05, -0.2, -0.5, -1.0):
            for Dl in (0.2, 1.0, 3.0, 10.0):
                dm, phi, da = response(e, th, Dl)
                R2 = H @ dm - phi * dyn_signatures(H, Dl)[:, e]
                worst = max(worst, np.abs(R2).max() / (2 * da * abs(phi) * Dl ** 2))
    return worst

def certificate(H, Delta):
    """Smallest SNR = |phi| Delta / sigD for which Theorem (certified isolation) holds on
    EVERY edge, and the largest SNR up to which it holds (second-order remainder)."""
    k = H.shape[0]; S = dyn_signatures(H, Delta)
    Fm = margins(S)[1] * (1 - coherence(S)); chi = math.sqrt(chi2.ppf(1 - eps, k))
    lo, hi = 0.0, math.inf
    for e in range(E):
        # phi*Fm - 4 sqrt(k) Delta^2 phi^2 / mbar_head > 2 sigD chi
        A2 = 4 * math.sqrt(k) * Delta ** 2 / mbar[e + 1]
        disc = Fm ** 2 - 8 * A2 * sigD * chi
        if disc <= 0: return None
        r1 = (Fm - math.sqrt(disc)) / (2 * A2); r2 = (Fm + math.sqrt(disc)) / (2 * A2)
        lo = max(lo, r1); hi = min(hi, r2, traffic[e])
    if lo >= hi: return None
    return lo * Delta / sigD, hi * Delta / sigD

snrs = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40], dtype=float)
mc = {}
for k in (2, 4, 8):
    det, iso = run_mc(designs[k], Delta_op, snrs, "dynamic")
    _, iso_s = run_mc(designs[k], Delta_op, snrs, "static")
    mc[k] = dict(det=det.tolist(), iso=iso.tolist(), iso_static=iso_s.tolist(),
                 fa=false_alarm(designs[k]), rem=remainder_check(designs[k]))
H1 = np.zeros((1, N)); H1[0, 1::2] = 1.0                    # one alternating pool
det1, _ = run_mc(H1, Delta_op, snrs)
detb, isob = run_mc(H_binary, Delta_op, snrs, "dynamic") if binary_result else (None, None)

# window sweep at fixed anomaly rate (SNR 16 at Delta = 1), k = 8, both decoders
Deltas = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
sweep_dyn, sweep_sta = [], []
for Dl in Deltas:
    s_ = np.array([16.0 * Dl])                 # same phi, SNR scales with Delta
    sweep_dyn.append(float(run_mc(designs[8], Dl, s_, "dynamic")[1][0]))
    sweep_sta.append(float(run_mc(designs[8], Dl, s_, "static")[1][0]))

cert = {}
for k in ks:
    best = None
    for Dl in (0.125, 0.25, 0.5, 1.0, 2.0):
        cband = certificate(designs[k], Dl)
        if cband and (best is None or cband[0] < best[1]): best = (Dl, cband[0], cband[1])
    c1 = certificate(designs[k], Delta_op)
    cert[k] = dict(at_op=c1, best=best)
def snr95(iso):          # first grid SNR with empirical isolation >= 0.95
    for s_, v in zip(snrs, iso):
        if v >= 0.95: return float(s_)
    return None
for k in mc: mc[k]["snr95"] = snr95(mc[k]["iso"])
amin = {k: float(norm.ppf(1 - alpha / (2 * k)) + norm.ppf(1 - beta)) for k in (1, 2, 3, 5, 10)}

results = dict(N=N, E=E, J=float(J[0]), lam=float(lam), tau_slow=float(tau_slow),
               traffic_min=float(traffic.min()), traffic_max=float(traffic.max()),
               mbar_min=float(mbar.min()), mbar_max=float(mbar.max()), mbar_mean=float(mbar.mean()),
               sigma=sigma, sigD=sigD, sigma_rel=sigma / mbar.mean(),
               budget_table=budget_table, k_count=k_count, k_bch=k_bch, binary=binary_result,
               spark_table={str(k): [str(v) for v in vs] for k, vs in spark_table.items()},
               designs=design_rows, mu2_bound=mu2_bound, sw_ratio=sw_ratio, mc=mc,
               det1=det1.tolist(), det_bin=None if detb is None else detb.tolist(),
               iso_bin=None if isob is None else isob.tolist(),
               snrs=snrs.tolist(), Deltas=Deltas.tolist(), sweep_dyn=sweep_dyn, sweep_sta=sweep_sta,
               cert=cert, amin=amin, Delta_op=Delta_op, alpha=alpha, beta=beta, eps=eps)
json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)
np.savez(os.path.join(OUT, "designs.npz"), H_binary=H_binary, **{f"H{k}": designs[k] for k in ks})
print(json.dumps(results, indent=1))

# ----------------------------------------------------------------------------
# 8. Per-edge readiness certificate (Theorem: certified isolation, edge-wise form)
# ----------------------------------------------------------------------------
def edge_certificate(H, grid=(0.0625, 0.125, 0.25, 0.5, 1.0, 2.0)):
    """Smallest relative rate loss |theta| on each edge for which isolation is
    certified with probability >= 1-eps (inf if not certifiable for |theta|<=1)."""
    k = H.shape[0]; chi = math.sqrt(chi2.ppf(1 - eps, k))
    best = np.full(E, np.inf); bestD = np.zeros(E)
    for Dl in grid:
        S = dyn_signatures(H, Dl); nrm = np.linalg.norm(S, axis=0); Sn = S / nrm
        G = np.abs(Sn.T @ Sn); np.fill_diagonal(G, 0); mue = G.max(axis=0)
        A = nrm * (1 - mue)
        for e in range(E):
            qa_ = 4 * math.sqrt(k) * b[e + 1] * traffic[e] * Dl ** 2
            qb_ = traffic[e] * A[e]; qc_ = 2 * sigD * chi
            disc = qb_ ** 2 - 4 * qa_ * qc_
            if disc > 0:
                th = (qb_ - math.sqrt(disc)) / (2 * qa_)
                if th <= 1 and th < best[e]: best[e], bestD[e] = th, Dl
    return best, bestD
edge_cert = {}
for k in (4, 8, 12):
    th, dl = edge_certificate(designs[k])
    edge_cert[k] = dict(theta=[None if not np.isfinite(x) else float(x) for x in th], Delta=dl.tolist(),
                        n10=int(np.sum(th <= 0.10)), n50=int(np.sum(th <= 0.50)), n100=int(np.sum(th <= 1.0)))
# empirical per-edge 95 % isolation threshold for k = 8 (dynamic decoder, Delta = 1)
def empirical_edge_threshold(H, Delta=1.0, trials=300):
    k = H.shape[0]; S = dyn_signatures(H, Delta); Sn = S / np.linalg.norm(S, axis=0)
    grid = np.array([0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]); out = []
    for e in range(E):
        val = None
        for th in grid:
            r0 = H @ response(e, -th, Delta)[0]
            rho = r0[:, None] + sigD * rng.standard_normal((k, trials))
            if np.mean(np.argmax(np.abs(Sn.T @ rho), axis=0) == e) >= 0.95: val = float(th); break
        out.append(val)
    return out
emp8 = empirical_edge_threshold(designs[8])
results.update(edge_cert=edge_cert, emp8=emp8, traffic=traffic.tolist())
json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)

# ----------------------------------------------------------------------------
# 9. Figures
# ----------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.6, "lines.linewidth": 1.6, "lines.markersize": 5,
                     "legend.frameon": False, "pdf.fonttype": 42})
C1, C2, C3, REF = "#0072B2", "#D55E00", "#009E73", "#555555"

# Fig. 1: coherence against channel budget
fig, ax = plt.subplots(figsize=(3.5, 2.7))
kk = [d["k"] for d in design_rows]
ax.plot(kk, [d["mu"] for d in design_rows], "-o", color=C1, label=r"design, static $\mu(T)$")
ax.plot(kk, [d["muS"] for d in design_rows], "--s", color=C2, label=r"design, dynamic $\mu(S(\Delta))$, $\Delta=1$")
kfine = np.arange(2, 13)
ax.plot(kfine, [welch(k, E) for k in kfine], ":", color=REF, label="Welch lower bound")
ax.plot([2], [mu2_bound], marker="x", color=REF, ls="none", ms=7, label=r"$\cos(\pi/|E'|)$ bound, $k=2$")
ax.set_xlabel("number of inventory channels $k$"); ax.set_ylabel("mutual coherence"); ax.set_ylim(0, 1.05)
ax.set_xticks(kk); ax.legend(fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_coherence.pdf")); plt.close(fig)

# Fig. 2: isolation performance
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
ax = axs[0]
for k, col, mk, ls in ((2, C1, "o", "-"), (4, C2, "s", "--"), (8, C3, "^", "-.")):
    ax.plot(snrs, mc[k]["iso"], ls, marker=mk, color=col, label=f"$k={k}$")
ax.plot(snrs, results["iso_bin"], ":", marker="d", color=REF, label="$k=4$, binary pools")
ax.set_xscale("log"); ax.set_xlabel(r"anomaly signal-to-noise ratio $|\phi|\Delta/\sigma_\Delta$")
ax.set_ylabel("probability of correct isolation"); ax.set_ylim(0, 1.03); ax.legend(fontsize=7, loc="upper left")
ax.set_title("(a) isolation against anomaly size, $\\Delta=1$", fontsize=8)
ax = axs[1]
ax.plot(Deltas, sweep_dyn, "-^", color=C3, label="dynamic signatures $S(\\Delta)$")
ax.plot(Deltas, sweep_sta, "--v", color=C2, label="static signatures $T$ (model-free)")
ax.set_xscale("log", base=2); ax.set_xlabel(r"window length $\Delta$ (units of $1/\lambda$)")
ax.set_ylabel("probability of correct isolation"); ax.set_ylim(0, 1.03); ax.legend(fontsize=7, loc="lower center")
ax.set_title("(b) effect of the window, $k=8$, fixed $\\phi$", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_isolation.pdf")); plt.close(fig)

# Fig. 3: readiness map (certified vs empirical per-edge thresholds, k = 8)
fig, ax = plt.subplots(figsize=(3.5, 2.7))
idx = np.arange(1, E + 1)
th8 = np.array([np.nan if x is None else x for x in edge_cert[8]["theta"]])
em8 = np.array([np.nan if x is None else x for x in emp8])
ax.semilogy(idx, th8, "o", color=C1, label="certified")
ax.semilogy(idx, em8, "^", color=C3, mfc="none", label="empirical, 95 % isolation")
ax.axhline(1.0, color=REF, ls=":", lw=1); ax.text(1, 1.15, "complete loss", fontsize=7, color=REF)
ax.set_xlabel("edge index $e$ (between stages $e$ and $e+1$)")
ax.set_ylabel(r"smallest isolable rate loss $|\theta|$"); ax.legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_readiness.pdf")); plt.close(fig)
print("EDGE CERT", {k: (v["n10"], v["n50"], v["n100"]) for k, v in edge_cert.items()})
print("EMP8", emp8)
print("CERT8", [None if x is None else round(x, 4) for x in edge_cert[8]["theta"]])

# ----------------------------------------------------------------------------
# 10. LaTeX macros: every instance-dependent number in the paper comes from here
# ----------------------------------------------------------------------------
def pct(x, d=1): return f"{100 * x:.{d}f}"
M = {
 "nN": N, "nE": E, "nLam": f"{lam:.2f}", "nTauSlow": f"{tau_slow:.0f}",
 "nSigRel": f"{100 * sigma / mbar.mean():.3f}", "nTrafficSpan": f"{traffic.max() / traffic.min():.0f}",
 "nKcount": k_count, "nKbch": k_bch, "nKbin": binary_result.get("k", "--"),
 "nMuBin": f"{binary_result.get('mu', float('nan')):.3f}",
 "nMuTwoBound": f"{mu2_bound:.4f}", "nSwRatio": f"{sw_ratio:.2f}",
 "nRemMax": f"{max(v['rem'] for v in mc.values()):.2f}",
 "nFAmin": pct(min(v['fa'] for v in mc.values()), 2), "nFAmax": pct(max(v['fa'] for v in mc.values()), 2),
 "nDetOneAtTwelve": pct(det1[list(snrs).index(12.0)]),
 "nIsoTwoAtForty": pct(mc[2]["iso"][-1], 0),
 "nSnrFour": f"{mc[4]['snr95']:.0f}", "nSnrEight": f"{mc[8]['snr95']:.0f}",
 "nSnrBin": f"{snr95(results['iso_bin']):.0f}",
 "nSweepStaPeak": pct(max(sweep_sta), 0), "nSweepStaEnd": pct(sweep_sta[-1], 0),
 "nSweepDynEnd": pct(sweep_dyn[-1], 0), "nSweepDynOne": pct(sweep_dyn[list(Deltas).index(1.0)], 0),
 "nCertFour": edge_cert[4]["n100"], "nCertEight": edge_cert[8]["n100"], "nCertTwelve": edge_cert[12]["n100"],
 "nCertEightTen": edge_cert[8]["n10"],
 "nEmpWorst": pct(max(x for x in emp8 if x is not None), 0),
 "nEmpAll": sum(x is not None for x in emp8),
 "nConsMin": f"{min(c_ / e_ for c_, e_ in zip(edge_cert[8]['theta'], emp8) if c_ is not None):.1f}",
 "nConsMax": f"{max(c_ / e_ for c_, e_ in zip(edge_cert[8]['theta'], emp8) if c_ is not None):.1f}",
 "nAminOne": f"{amin[1]:.2f}", "nAminTen": f"{amin[10]:.2f}",
}
rows = "".join(f"{d['k']} & {d['mu']:.3f} & {d['welch']:.3f} & {d['g_inf']:.3f} & {d['g_2']:.3f} & "
               f"{d['muS']:.3f} & {d['s_2']:.3f} \\\\\n" for d in design_rows)
brow = " & ".join(str(p) for p in budget_table) + " \\\\\n$k_{\\det}(p)$ & " + " & ".join(str(v) for v in budget_table.values()) + " \\\\\n"
with open(os.path.join(OUT, "numbers.tex"), "w") as fh:
    fh.write("% Generated by code/cascade_instrumentation.py -- do not edit by hand\n")
    for k_, v_ in M.items(): fh.write(f"\\newcommand{{\\{k_}}}{{{v_}}}\n")
    fh.write("\\newcommand{\\designrows}{%\n" + rows + "}\n")
    fh.write("\\newcommand{\\budgetrows}{%\n$p$ & " + brow + "}\n")
print(open(os.path.join(OUT, "numbers.tex")).read())

# ----------------------------------------------------------------------------
# 11. Requirements by inversion: readiness against assay precision (k = 8)
# ----------------------------------------------------------------------------
sigD_ref = sigD
def sigma_star(H, theta, grid=(0.0625, 0.125, 0.25, 0.5, 1.0, 2.0)):
    """Largest increment noise for which isolation of a relative rate loss theta is
    certified on each edge (Corollary: requirements by inversion); 0 if never."""
    k = H.shape[0]; chi = math.sqrt(chi2.ppf(1 - eps, k)); best = np.zeros(E)
    for Dl in grid:
        S = dyn_signatures(H, Dl); nrm = np.linalg.norm(S, axis=0); Sn = S / nrm
        G = np.abs(Sn.T @ Sn); np.fill_diagonal(G, 0); mue = G.max(axis=0)
        phi = theta * traffic; da = theta * b[1:]
        val = phi * (nrm * (1 - mue) - 4 * math.sqrt(k) * da * Dl ** 2) / (2 * chi)
        best = np.maximum(best, val)
    return best
H8 = designs[8]
sstar10 = sigma_star(H8, 0.10) / math.sqrt(2) / mbar.mean()      # assay sd, relative
prec_rows, prec_levels = [], [1e-5, 1e-4, 1e-3, 1e-2]
for rel in prec_levels:
    sigD = math.sqrt(2) * rel * mbar.mean()
    th, _ = edge_certificate(H8)
    e1 = empirical_edge_threshold(H8, 1.0, trials=200); e4 = empirical_edge_threshold(H8, 4.0, trials=200)
    emp = [min([x for x in (u, v) if x is not None], default=None) for u, v in zip(e1, e4)]
    prec_rows.append(dict(rel=rel, cert10=int(np.sum(th <= 0.10)), cert=int(np.sum(th <= 1.0)),
                          emp10=sum(x is not None and x <= 0.10 for x in emp),
                          emp=sum(x is not None for x in emp)))
sigD = sigD_ref

# ring archetype: verify the port-limited detection budget on cycles C_n
def ring_budget(n, p):
    base = math.ceil(n / (2 * p)); return base if n % 2 == 0 else max(2, base)
def ring_design(n, p):
    k = ring_budget(n, p); H = np.zeros((k, n))
    if n % 2 == 0:
        cover = list(range(1, n, 2))                       # alternate stages
    else:
        cover = list(range(0, n, 2))                       # stages 1,3,...,n (0-based 0,2,..,n-1)
    per = max(1, math.ceil(len(cover) / k))
    assert per <= p
    for idx, node in enumerate(cover): H[min(idx // per, k - 1), node] = 1.0
    return H
def ring_ok(H):
    n = H.shape[1]
    return all(np.any(H[:, i] != H[:, (i + 1) % n]) for i in range(n))
ring_checked = 0
for n in range(3, 41):
    for p in range(1, 13):
        H = ring_design(n, p); assert ring_ok(H) and (H.sum(axis=1) <= p).all(), (n, p)
        ring_checked += 1
# exhaustive lower-bound check with binary pools on small rings
def ring_min_binary(n, p):
    pools = [np.array(v) for v in itertools.product((0, 1), repeat=n) if 0 < sum(v) <= p]
    for k in range(1, n + 1):
        for comb in itertools.combinations(pools, k):
            if ring_ok(np.array(comb)): return k
ring_exh = all(ring_min_binary(n, p) == ring_budget(n, p) for n in range(3, 9) for p in range(1, 4))
assert ring_exh
results.update(precision=prec_rows, sstar10=sstar10.tolist(), ring_checked=ring_checked, ring_exhaustive=ring_exh)
json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)
def pfmt(x):
    s_ = f"{100 * x:.3f}".rstrip("0").rstrip("."); return s_
prow = "".join(f"{pfmt(r['rel'])} & {r['cert10']} & {r['cert']} & {r['emp10']} & {r['emp']} \\\\\n" for r in prec_rows)
with open(os.path.join(OUT, "numbers.tex"), "a") as fh:
    fh.write("\\newcommand{\\precisionrows}{%\n" + prow + "}\n")
    fh.write(f"\\newcommand{{\\nSstarMax}}{{{100 * sstar10.max():.3f}}}\n")
    fh.write(f"\\newcommand{{\\nSstarMed}}{{{100 * np.median(sstar10):.5f}}}\n")
    fh.write(f"\\newcommand{{\\nSstarPos}}{{{int(np.sum(sstar10 > 0))}}}\n")
    fh.write(f"\\newcommand{{\\nSstarMin}}{{{100 * sstar10.min():.4f}}}\n")
    fh.write(f"\\newcommand{{\\nRingChecked}}{{{ring_checked}}}\n")
print("PRECISION", prec_rows); print("SSTAR10 %", np.round(100 * sstar10, 5)); print("ring", ring_checked, ring_exh)
