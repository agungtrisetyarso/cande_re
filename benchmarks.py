#!/usr/bin/env python3
"""
Benchmark definitions, in one importable place.

  P  synthetic 40-stage counter-current cascade (all parameters chosen by the authors)
  R  18-stage closed-loop Pr/Nd circuit: stage counts, vessel volumes and distribution
     coefficients from K. L. Lyon, M.S. thesis, University of Idaho, 2016 (Tables 3 and 17);
     flows, hold-up split, feed concentrations and the closed organic loop are ASSUMED.

The four earlier scripts carry their own copies of these builders (kept so that each
script can be read on its own). `export_benchmarks.py` writes the plants built here to
benchmarks/*.json and `verify_package.py` checks that the numbers of the paper are
recovered from those files alone, so a drift between the copies cannot go unnoticed.
"""
import numpy as np
from scipy.linalg import expm


class Plant:
    """Compartmental plant  dm/dt = -K m + f.

    edges[e] = (tail, head) of unmeasured net-transfer edge e;
    org[e]   = (source stage, sink stage) of the organic stream carried by e;
    Dc[i]    = distribution coefficient of stage i;  phi_org[i] = organic volume fraction."""

    def __init__(self, K, f, edges, org, Dc, phi_org, units):
        self.K, self.f, self.edges, self.org = K, f, list(edges), list(org)
        self.Dc = np.asarray(Dc, float); self.phi_org = np.asarray(phi_org, float); self.units = units
        self.N = K.shape[0]; self.E = len(self.edges)
        self.B = np.zeros((self.N, self.E))
        for e, (i, j) in enumerate(self.edges): self.B[i, e] = -1.0; self.B[j, e] = 1.0
        self.mbar = np.linalg.solve(K, f); self.Kinv = np.linalg.inv(K); self.lam = K.diagonal().max()
        self.a_org = np.array([-K[snk, src] for (src, snk) in self.org])
        self.traffic = self.a_org * np.array([self.mbar[src] for (src, _) in self.org])

    def Psi(self, D): return self.Kinv @ (np.eye(self.N) - expm(-self.K * D))
    def S(self, H, D): return H @ self.Psi(D) @ self.B
    def T(self, H): return H @ self.B

    def faulty(self, e, theta):
        """Organic transfer rate on edge e scaled by (1 + theta). Returns K', phi, |delta alpha|."""
        src, snk = self.org[e]; d = theta * self.a_org[e]
        K2 = self.K.copy(); K2[src, src] += d; K2[snk, src] -= d
        sign = 1.0 if (src, snk) == self.edges[e] else -1.0
        return K2, sign * d * self.mbar[src], abs(d)

    def response(self, e, theta, D):
        """Exact inventory deviation m(t0 + D) - mbar after the fault of `faulty`."""
        K2, phi, da = self.faulty(e, theta); mss = np.linalg.solve(K2, self.f)
        return mss + expm(-K2 * D) @ (self.mbar - mss) - self.mbar, phi, da

    def phase_factor(self):
        """g_i = (D_i - 1) / (1 + phi_i (D_i - 1)): relative error of the inferred stage inventory
        per unit error of the organic volume fraction of a dispersion aliquot."""
        return (self.Dc - 1.0) / (1.0 + self.phi_org * (self.Dc - 1.0))


def build_path(N=40, D_first=1.30, D_last=0.80):
    """Benchmark P. Unit phase hold-ups and unit phase flows; time in stage turnover times."""
    Dc = np.linspace(D_first, D_last, N); kap = 1.0 + Dc; a = 1.0 / kap; b = Dc / kap
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] = a[i] + b[i]
        if i + 1 < N: K[i + 1, i] = -a[i]
        if i - 1 >= 0: K[i - 1, i] = -b[i]
    f = np.zeros(N); f[0] = 1.0
    return Plant(K, f, [(i, i + 1) for i in range(N - 1)], [(i + 1, i) for i in range(N - 1)],
                 Dc, np.full(N, 0.5), dict(time="stage turnover time", inventory="inventory unit"))


# Lyon (2016), Table 17: measured D_Nd, stages 1-16. Stages 17-18 are not reported; value of stage 16 assumed.
D_LYON = [0.56, 1.24, 0.67, 0.65, 0.59, 0.57, 0.60, 0.51, 0.60, 0.57, 0.56, 0.58, 0.57, 0.62,
          0.0008, 0.002, 0.002, 0.002]
RING = dict(n=18, V_aq_mL=300.0, V_org_mL=300.0,           # mixer 120 mL + settler 480 mL (source); equal split ASSUMED
            q_mL_min=5.0,                                  # organic, feed, scrub and strip flows: ASSUMED
            c_feed_g_L=7.5, c_scrub_g_L=10.0)              # Nd in feed and scrub: ASSUMED


def build_ring():
    """Benchmark R. Time in minutes, inventory in mg of Nd."""
    n = RING["n"]; Va, Vo = RING["V_aq_mL"], RING["V_org_mL"]; q = RING["q_mL_min"]
    Dc = np.array(D_LYON); kap = Va + Dc * Vo; K = np.zeros((n, n))
    def add(i, j, rate):
        K[i, i] += rate
        if j is not None: K[j, i] -= rate
    for i in range(n): add(i, (i + 1) % n, q * Dc[i] / kap[i])       # organic 1 -> 2 -> ... -> 18 -> 1
    add(1, 0, 2 * q / kap[1]); add(0, None, 2 * q / kap[0])          # extraction aqueous (feed + scrub), raffinate
    for i in range(2, 14): add(i, i - 1, q / kap[i])                 # scrub aqueous 14 -> 3
    for i in range(15, 18): add(i, i - 1, q / kap[i])                # strip aqueous 18 -> 15
    add(14, None, q / kap[14])                                       # strip liquor (product)
    f = np.zeros(n); f[1] = q * RING["c_feed_g_L"]; f[13] = q * RING["c_scrub_g_L"]      # mg/min
    ring = [(i, (i + 1) % n) for i in range(n)]
    return Plant(K, f, ring, ring, Dc, np.full(n, Vo / (Va + Vo)), dict(time="min", inventory="mg Nd"))


def coh_cols(S):
    """Column norms, normalised columns and the matrix of absolute cosines (zero diagonal)."""
    n = np.linalg.norm(S, axis=0); Sn = S / n
    G = np.abs(Sn.T @ Sn); np.fill_diagonal(G, 0.0); return n, Sn, G
