"""Industry feature families as audit targets, and family-generic versions of the tests.

Families (item features; customer features are the mean over the customer's usable edges):
  A1 counts   typed interaction counts + unique customers            (aggregation, exact)
  A2 rates    conversion share (purchase-like edges / all) + mean type (aggregation, exact)
  A3 decay    time-decayed interaction count, half-life 2 days       (aggregation, real data only)
  R1 mf       rank-16 randomized SVD of the type-weighted matrix     (matrix factorization)
  R2 item2vec rank-16 SVD of the PPMI item co-occurrence matrix      (word2vec ~ PMI factorization)
  R3 lightgcn 0.5*R1 + 0.5*mean R1 of co-interacted items            (graph propagation)
  R4 sage     mean intrinsic attributes z of neighbouring customers  (GraphSAGE-style, user attributes)

In a 'leak' run, every family is computed from E_in plus the leaked opted-out edges (and, for R4,
the attributes of the customers behind them). Tests look at one family at a time.

Customer attributes z (4-d: gender, birth year, region x/y). In the simulator they correlate with
taste (derived from preferences); on real data they are random (no attributes are public).

Usage: python3 families.py synthetic|real SEEDS OUT.json MODE RHO PATTERN [DAYS]
"""
import sys
import json
import numpy as np
import scipy.sparse as sp
from scipy import stats
from sklearn.utils.extmath import randomized_svd
import audit_sim as a
import robust

EXACT = {"A1": True, "A2": True, "A3": False, "R1": False, "R2": False, "R3": False, "R4": True}
REP = ["R1", "R2", "R3", "R4"]
WIDE = {"R2"}   # receptive field beyond 2 hops (PMI normalization): adjust T2 for 3-hop churn


def attributes(src, world, seed):
    rng = np.random.default_rng(seed + 12345)
    if src == "synthetic":
        p = world.pref
        g = np.sign(p[:, 0] + 0.5 * rng.normal(size=a.U))
        return np.column_stack([g, p[:, 1] + 0.5 * rng.normal(size=a.U),
                                p[:, 2:4] + 0.5 * rng.normal(size=(a.U, 2))])
    return np.column_stack([np.sign(rng.normal(size=a.U)), rng.normal(size=(a.U, 3))])


def _mean_over_edges(u, i, M, n_rows, by="cust"):
    """mean of M[i] over edges grouped by customer (by='cust') or M[u] grouped by item."""
    if by == "cust":
        S = np.zeros((n_rows, M.shape[1])); np.add.at(S, u, M[i]); c = np.bincount(u, minlength=n_rows)
    else:
        S = np.zeros((n_rows, M.shape[1])); np.add.at(S, i, M[u]); c = np.bincount(i, minlength=n_rows)
    return S / np.maximum(c, 1)[:, None]


def compute(snap, mode, rho, day_seed, z, src):
    u, i, t, out = snap["u"], snap["i"], snap["t"], snap["out"]
    use = ~out
    if mode == "leak":
        use = use | (out & a.leaked_mask(u, i, rho))
    u, i, t = u[use], i[use], t[use]
    hi = 2                                                    # purchase-like types: >= 2
    F = {}
    # A1 counts
    cnt = np.zeros((a.I, 4)); np.add.at(cnt, (i, np.minimum(t, 3)), 1)
    A1 = np.hstack([cnt, cnt.sum(1, keepdims=True)])
    F["A1"] = (A1, _mean_over_edges(u, i, A1, a.U))
    # A2 rates
    n = np.bincount(i, minlength=a.I).astype(float)
    conv = np.bincount(i, weights=(t >= hi).astype(float), minlength=a.I) / np.maximum(n, 1)
    mt = np.bincount(i, weights=t.astype(float), minlength=a.I) / np.maximum(n, 1)
    A2 = np.column_stack([conv, mt])
    F["A2"] = (A2, _mean_over_edges(u, i, A2, a.U))
    # A3 decayed counts (needs interaction days)
    if "eday" in snap:
        ed = snap["eday"][use]
        w = 0.5 ** ((snap["day"] - ed) / 2.0)
        A3 = np.bincount(i, weights=w, minlength=a.I)[:, None]
        F["A3"] = (A3, _mean_over_edges(u, i, A3, a.U))
    # R1 matrix factorization
    A = sp.csr_matrix((a.TYPE_W[np.minimum(t, 3)], (u, i)), shape=(a.U, a.I))
    _, S, Vt = randomized_svd(A, a.K, n_iter=2, random_state=day_seed)
    R1 = Vt.T * np.sqrt(S)
    F["R1"] = (R1, _mean_over_edges(u, i, R1, a.U))
    # co-occurrence (binary), used by R2 and R3
    B = sp.csr_matrix((np.ones(len(u)), (u, i)), shape=(a.U, a.I)); B.data[:] = 1
    C = (B.T @ B).tocsr(); C.setdiag(0); C.eliminate_zeros()
    # R2 item2vec ~ PPMI factorization
    tot = C.sum()
    if tot > 0:
        rs = np.asarray(C.sum(1)).ravel()
        Cc = C.tocoo()
        pmi = np.log(Cc.data * tot / (rs[Cc.row] * rs[Cc.col]))
        keep = pmi > 0
        P = sp.csr_matrix((pmi[keep], (Cc.row[keep], Cc.col[keep])), shape=(a.I, a.I))
        Ub, Sb, _ = randomized_svd(P, a.K, n_iter=2, random_state=day_seed + 7)
        R2 = Ub * np.sqrt(Sb)
    else:
        R2 = np.zeros((a.I, a.K))
    F["R2"] = (R2, _mean_over_edges(u, i, R2, a.U))
    # R3 LightGCN-style propagation of R1
    deg = np.asarray(C.sum(1)).ravel()
    R3 = 0.5 * R1 + 0.5 * (C @ R1) / np.maximum(deg, 1)[:, None]
    F["R3"] = (R3, _mean_over_edges(u, i, R3, a.U))
    # R4 GraphSAGE-style aggregation of customer attributes
    R4 = _mean_over_edges(u, i, z, a.I, by="item")
    F["R4"] = (R4, _mean_over_edges(u, i, R4, a.U))
    return F


# ------------------------------------------------------------------ family-generic tests
def t1g(snap, Xi, cmax):
    i, out = snap["i"], snap["out"]
    deg_in = np.bincount(i[~out], minlength=a.I); deg_out = np.bincount(i[out], minlength=a.I)
    canary = np.where((deg_in == 0) & (deg_out > 0))[0]
    cold = np.where((deg_in == 0) & (deg_out == 0))[0]
    if len(cold) > 5000:
        cold = np.random.default_rng(len(cold)).choice(cold, 5000, replace=False)
    nrm = np.linalg.norm(Xi, axis=1)
    if not len(canary) or not len(cold):
        p_item = 1.0
    elif np.ptp(Xi[cold], axis=0).max() < 1e-9:
        # cold items share one default output (deterministic readout): exact check
        p_item = 0.0 if np.abs(Xi[canary] - Xi[cold[0]]).max() > 1e-9 else 1.0
    else:
        p_item = a.perm_test_mean_diff(nrm[canary], nrm[cold])
    p_item = max(p_item, 1e-6)
    p_cust = 0.0 if cmax > 1e-9 else 1.0           # exact: opted-out customers must have no signal
    return min(1.0, 2 * min(max(p_cust, 1e-6), p_item))


def _typed(snap, mask):
    by_item, by_cust = {}, {}
    for uu, ii, tt in zip(snap["u"][mask].tolist(), snap["i"][mask].tolist(), snap["t"][mask].tolist()):
        by_item.setdefault(ii, set()).add((uu, tt)); by_cust.setdefault(uu, set()).add((ii, tt))
    return by_item, by_cust


def t2g(s0, s1, X0, X1, exact, churn=False):
    in0, c0 = _typed(s0, ~s0["out"]); in1, c1 = _typed(s1, ~s1["out"])
    out0, _ = _typed(s0, s0["out"]); out1, _ = _typed(s1, s1["out"])
    treat, ctrl, deg = [], [], {}
    for k in in0:
        if in0[k] != in1.get(k, set()):
            continue
        changed = out0.get(k, set()) != out1.get(k, set())
        if not exact and not all(c0.get(uu) == c1.get(uu) for uu, _ in in0[k]):
            continue                                          # 2-hop invariance for non-local families
        (treat if changed else ctrl).append(k); deg[k] = len(in0[k])
    if exact:
        if not treat:
            return 1.0
        return 0.0 if np.abs(X1[treat] - X0[treat]).max() > 1e-9 else 1.0
    if len(treat) < 3 or len(ctrl) < 3:
        return 1.0
    if X0.shape[1] > 1:
        act = np.zeros(a.I, bool); act[list(set(in0) & set(in1))] = True
        X1 = a.procrustes(X0, X1, act)
    d = np.log(np.linalg.norm(X1 - X0, axis=1) / (np.linalg.norm(X0, axis=1) + 1e-9) + 1e-9)
    idx = np.array(treat + ctrl); g = np.r_[np.ones(len(treat)), np.zeros(len(ctrl))]
    cols = [np.ones(len(idx)), g, np.log1p([deg[k] for k in idx])]
    if churn:
        # churn just outside the checked receptive field: opted-in changes at the items that
        # share a customer with k (3 hops). Wide-field features (e.g. PMI) respond to it.
        ch = np.zeros(a.I)
        for k in set(in0) | set(in1):
            ch[k] = float(in0.get(k, set()) != in1.get(k, set()))
        m = ~s0["out"]
        B = sp.csr_matrix((np.ones(m.sum()), (s0["u"][m], s0["i"][m])), shape=(a.U, a.I)); B.data[:] = 1
        nb = np.asarray((B.T @ (B @ ch))).ravel()
        cols.append(np.log1p(nb[idx]))
    Xr = np.column_stack(cols)
    y = d[idx]; beta, *_ = np.linalg.lstsq(Xr, y, rcond=None); r = y - Xr @ beta
    cov = np.linalg.pinv(Xr.T @ Xr) * (r @ r) / (len(y) - Xr.shape[1])
    return float(stats.norm.sf(beta[1] / np.sqrt(cov[1, 1])))


def _as_F(X):
    return {"item": np.hstack([np.zeros((X.shape[0], 5)), X])}


def run(src, seed, mode, rho, days):
    a.LEAK_SALT = seed
    if src == "real":
        import real_data
        real_data.load(); w = real_data.RealWorld(seed)
    else:
        w = a.World(seed)
    z = attributes(src, w, seed)
    rng = np.random.default_rng(seed + 21)
    S, F = {}, {}

    def make(d):
        s = w.snapshot(); S[d] = s
        fam = compute(s, mode, rho, seed * 1000 + d, z, src)
        oc = np.where(s["optout"])[0]
        # keep item features; for customers keep only the largest norm among opted-out customers
        F[d] = {k: (Xi, float(np.linalg.norm(Xc[oc], axis=1).max()) if len(oc) else 0.0)
                for k, (Xi, Xc) in fam.items()}
        w.step()

    make(0); make(1)
    rec = {}
    for d in range(1, days + 1):
        make(d + 1)
        sp_, s0, s1 = S[d - 1], S[d], S[d + 1]
        for name, (Xi, cmax) in F[d].items():
            rec.setdefault(f"{name}_T1", []).append(t1g(s0, Xi, cmax))
            rec.setdefault(f"{name}_T2", []).append(
                t2g(s0, s1, Xi, F[d + 1][name][0], EXACT[name], churn=name in WIDE))
            if name in WIDE:   # also record T2 without the 3-hop churn adjustment
                rec.setdefault(f"{name}_T2n", []).append(t2g(s0, s1, Xi, F[d + 1][name][0], False))
            if name in REP:
                rec.setdefault(f"{name}_T5", []).append(robust.t5b(sp_, s0, s1, _as_F(Xi), rng))
                rec.setdefault(f"{name}_T6", []).append(robust.t6b(sp_, s0, s1, _as_F(Xi)))
        del S[d - 1], F[d]
    return rec


if __name__ == "__main__":
    src, seeds, out = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
    mode, rho, pattern = sys.argv[4], float(sys.argv[5]), sys.argv[6]
    days = int(sys.argv[7]) if len(sys.argv) > 7 else 10
    a.LEAK_PATTERN = pattern
    res = []
    for seed in seeds:
        rec = run(src, seed, mode, rho, days)
        res.append(dict(src=src, seed=seed, mode=mode, rho=rho, pattern=pattern, p=rec))
        print(src, mode, rho, pattern, seed, {k: round(float(np.mean(np.array(v) < .05)), 2) for k, v in rec.items()}, flush=True)
        json.dump(res, open(out, "w"))
