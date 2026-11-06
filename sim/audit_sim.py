"""Synthetic study: auditing opt-out compliance of a black-box graph feature pipeline.

Generates a daily customer-item-merchant graph, runs a clean / locally-leaking /
globally-leaking feature pipeline, and applies four black-box audit tests:
  T1 natural canaries, T2 zero-difference, T4 residual regression, T5 co-occurrence MI
  with temporal placebo. Per-day p-values are converted to e-values and multiplied.
"""
import json
import sys
import numpy as np
import scipy.sparse as sp
from scipy import stats
from sklearn.utils.extmath import randomized_svd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

U, I, M, D_X, K = 3000, 2000, 100, 16, 16
TYPE_P = np.array([0.60, 0.15, 0.17, 0.08])   # view, save, purchase, review
TYPE_W = np.array([1.0, 2.0, 3.0, 3.0])
RESAMPLE_P = 0.2
P_IN2OUT, P_OUT2IN = 0.005, 0.045


class World:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        r = self.rng
        self.x = r.normal(size=(I, D_X))
        ranks = r.permutation(I) + 1
        self.w = 1.0 / ranks ** 1.1
        self.merchant = np.minimum((r.pareto(1.2, I) * 3).astype(int), M - 1)
        self.pref = r.normal(size=(U, D_X))
        self.optout = r.random(U) < 0.10
        self.edges = [self._sample_user(u) for u in range(U)]

    def _sample_user(self, u):
        r = self.rng
        n = 1 + r.poisson(7)
        logits = np.log(self.w) + 2.0 * self.x @ self.pref[u] / np.sqrt(D_X)
        p = np.exp(logits - logits.max()); p /= p.sum()
        items = r.choice(I, size=n, replace=False, p=p)
        types = r.choice(4, size=n, p=TYPE_P)
        return items, types

    def step(self):
        r = self.rng
        for u in range(U):
            if r.random() < RESAMPLE_P:
                self.edges[u] = self._sample_user(u)
        flip = np.where(self.optout, r.random(U) < P_OUT2IN, r.random(U) < P_IN2OUT)
        self.optout = self.optout ^ flip

    def snapshot(self):
        us, its, ts = [], [], []
        for u, (items, types) in enumerate(self.edges):
            us.append(np.full(len(items), u)); its.append(items); ts.append(types)
        u = np.concatenate(us); i = np.concatenate(its); t = np.concatenate(ts)
        out = self.optout[u]
        return dict(u=u, i=i, t=t, out=out, optout=self.optout.copy(), x=self.x)


LEAK_PATTERN = "edge"   # "edge": each opted-out edge leaks independently;
                        # "user": a fraction rho of opted-out customers leak all their edges
LEAK_SALT = 0           # varies which customers leak across seeds (per-customer pattern)


def leaked_mask(u, i, rho):
    """Deterministic hash so the same opted-out edges (or users) leak every day."""
    if LEAK_PATTERN == "user":
        h = ((u.astype(np.int64) + 7919 * LEAK_SALT) * 2654435761) % 100003
    else:
        h = (u.astype(np.int64) * 1000003 + i.astype(np.int64) * 7919) % 100003
    return h / 100003.0 < rho


def bip(u, i, t, weighted=True):
    v = TYPE_W[t] if weighted else np.ones(len(u))
    return sp.csr_matrix((v, (u, i)), shape=(U, I))


def propagate(u, i, x):
    """Item-item feature propagation (SIGN/LightGCN-style): mean intrinsic embedding of
    items co-interacted with through shared customers; zero for isolated items."""
    A = sp.csr_matrix((np.ones(len(u)), (u, i)), shape=(U, I)); A.data[:] = 1
    C = (A.T @ A).tolil(); C.setdiag(0); C = C.tocsr()
    deg = np.asarray(C.sum(1)).ravel()
    return (C @ x) / np.maximum(deg, 1)[:, None]


def pipeline(snap, mode, rho, day_seed):
    """Black-box feature pipeline. mode in {clean, local, global}."""
    u, i, t, out = snap["u"], snap["i"], snap["t"], snap["out"]
    inm = ~out
    leak = out & leaked_mask(u, i, rho) if mode != "clean" else np.zeros_like(out)
    agg_mask = inm | (leak if mode == "local" else False)
    trn_mask = inm | (leak if mode == "global" else False)
    # trained component: item embeddings from SVD of weighted interaction matrix
    A_trn = bip(u[trn_mask], i[trn_mask], t[trn_mask])
    _, S, Vt = randomized_svd(A_trn, K, n_iter=2, random_state=day_seed)
    item_emb = np.hstack([Vt.T * np.sqrt(S), propagate(u[trn_mask], i[trn_mask], snap["x"])])
    # aggregation component
    ua, ia, ta = u[agg_mask], i[agg_mask], t[agg_mask]
    item_cnt = np.zeros((I, 4)); np.add.at(item_cnt, (ia, ta), 1)
    item_uniq = np.bincount(ia, minlength=I)[:, None].astype(float)
    item_F = np.hstack([item_cnt, item_uniq, item_emb])
    cust_cnt = np.zeros((U, 4)); np.add.at(cust_cnt, (ua, ta), 1)
    cust_emb = np.zeros((U, item_emb.shape[1])); np.add.at(cust_emb, ua, item_emb[ia])
    deg = cust_cnt.sum(1, keepdims=True)
    cust_F = np.hstack([cust_cnt, cust_emb / np.maximum(deg, 1)])
    return dict(item=item_F, cust=cust_F)


# --------------------------------------------------------------------------- tests
def neigh_sets(snap):
    u, i, out = snap["u"], snap["i"], snap["out"]
    key = (u.astype(np.int64) * I + i) * 4 + snap["t"]   # typed edges
    return set(key[~out].tolist()), set(key[out].tolist())


def item_neigh(keys):
    d = {}
    for k in keys:
        d.setdefault((k // 4) % I, set()).add(k)
    return d


def perm_test_mean_diff(a, b, n_perm=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); na = len(a)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        cnt += (pool[:na].mean() - pool[na:].mean()) >= obs
    return (cnt + 1) / (n_perm + 1)


def t1_canary(snap, F):
    """Items whose every edge is opt-out vs. items with no edges at all; opted-out customers
    vs. customers with no edges (none exist, so compare to the zero/default vector)."""
    i, out = snap["i"], snap["out"]
    deg_in = np.bincount(i[~out], minlength=I); deg_out = np.bincount(i[out], minlength=I)
    canary = (deg_in == 0) & (deg_out > 0); cold = (deg_in == 0) & (deg_out == 0)
    nrm = np.linalg.norm(F["item"], axis=1)
    cold_idx = np.where(cold)[0]
    if len(cold_idx) > 5000:          # cap the comparison group (random, data-independent)
        cold_idx = np.random.default_rng(len(cold_idx)).choice(cold_idx, 5000, replace=False)
    p_item = perm_test_mean_diff(nrm[canary], nrm[cold_idx]) if canary.sum() and len(cold_idx) else 1.0
    # exact check: opted-out customers must sit at the pipeline's default output
    oc = snap["optout"]
    cust_dev = np.linalg.norm(F["cust"][oc], axis=1)
    p_cust = 1.0 if np.all(cust_dev < 1e-9) else 0.0  # deterministic default -> exact test
    return min(1.0, 2 * min(p_item, max(p_cust, 1e-6))), int(canary.sum()), int(cold.sum())


def procrustes(Ea, Eb, mask):
    Ua, _, Vta = np.linalg.svd(Eb[mask].T @ Ea[mask])
    return Eb @ (Ua @ Vta)


def cust_sets(snap, mask):
    d = {}
    for u, i, t in zip(snap["u"][mask].tolist(), snap["i"][mask].tolist(), snap["t"][mask].tolist()):
        d.setdefault(u, set()).add((i, t))
    return d


def t2_zero_diff(s0, s1, F0, F1):
    """Aggregates (1-hop receptive field): exact test on items whose 1-hop E_in is unchanged.
    Embeddings (2-hop receptive field): items whose 2-hop E_in neighbourhood is unchanged;
    treated = 1-hop E_out changed, control = unchanged; degree-adjusted one-sided OLS."""
    in0, out0 = neigh_sets(s0); in1, out1 = neigh_sets(s1)
    n_in0, n_in1 = item_neigh(in0), item_neigh(in1)
    n_out0, n_out1 = item_neigh(out0), item_neigh(out1)
    c0, c1 = cust_sets(s0, ~s0["out"]), cust_sets(s1, ~s1["out"])
    e0, e1 = F0["item"][:, 5:], F1["item"][:, 5:]
    active = np.array([(k in n_in0) and (k in n_in1) for k in range(I)])
    e1a = procrustes(e0, e1, active)
    agg_d = np.abs(F1["item"][:, :5] - F0["item"][:, :5]).sum(1)
    emb_d = np.log(np.linalg.norm(e1a - e0, axis=1) / (np.linalg.norm(e0, axis=1) + 1e-9) + 1e-9)
    agg_treat, treat, ctrl, deg = [], [], [], {}
    for k in range(I):
        if k not in n_in0 or n_in0[k] != n_in1.get(k, set()):
            continue
        changed = n_out0.get(k, set()) != n_out1.get(k, set())
        if changed:
            agg_treat.append(k)
        if all(c0.get(kk // 4 // I) == c1.get(kk // 4 // I) for kk in n_in0[k]):
            (treat if changed else ctrl).append(k); deg[k] = len(n_in0[k])
    p_agg = 1.0
    if agg_treat:
        p_agg = 0.0 if agg_d[np.array(agg_treat)].max() > 1e-9 else 1.0
    p_emb = 1.0
    if len(treat) >= 3 and len(ctrl) >= 3:
        idx = np.array(treat + ctrl); g = np.r_[np.ones(len(treat)), np.zeros(len(ctrl))]
        X = np.column_stack([np.ones(len(idx)), g, np.log1p([deg[k] for k in idx])])
        y = emb_d[idx]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ beta
        cov = np.linalg.inv(X.T @ X) * (r @ r) / (len(y) - 3)
        p_emb = stats.norm.sf(beta[1] / np.sqrt(cov[1, 1]))
    return min(1.0, 2 * min(max(p_agg, 1e-6), p_emb)), len(treat), len(ctrl)


def surrogate_controls(snap, world, seed):
    u, i, t, out = snap["u"], snap["i"], snap["t"], snap["out"]
    A = bip(u[~out], i[~out], t[~out])
    _, S, Vt = randomized_svd(A, 32, n_iter=4, random_state=seed + 99)
    cnt = np.zeros((I, 4)); np.add.at(cnt, (i[~out], t[~out]), 1)
    return np.hstack([np.log1p(cnt), world.x, Vt.T * np.sqrt(S), propagate(u[~out], i[~out], world.x)])


def t4_residual(snap, F, Z):
    i, out = snap["i"], snap["out"]
    deg_in = np.bincount(i[~out], minlength=I)
    m = deg_in > 0
    y = np.log1p(np.bincount(i[out], minlength=I))[m]
    Fi = F["item"][m]; Fi = np.hstack([np.log1p(np.abs(Fi[:, :5])), Fi[:, 5:]])
    Zm = Z[m]
    d = np.zeros(m.sum())
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Zm):
        a = Ridge(1.0).fit(Zm[tr], y[tr]); b = Ridge(1.0).fit(np.hstack([Zm, Fi])[tr], y[tr])
        d[te] = (y[te] - a.predict(Zm[te])) ** 2 - (y[te] - b.predict(np.hstack([Zm, Fi])[te])) ** 2
    return stats.ttest_1samp(d, 0, alternative="greater").pvalue


def cooc(u, i):
    A = sp.csr_matrix((np.ones(len(u)), (u, i)), shape=(U, I))
    A.data[:] = 1
    return (A.T @ A).tocsr()


def t5_cooc(s0, s1, F0, rng):
    """Membership-inference co-occurrence test with a symmetric temporal placebo.
    Real pairs: co-occur via today's opt-out edges only (not via E_in today, not via tomorrow's new edges).
    Placebo pairs: co-occur via tomorrow's new edges only (not via E_in today, not via today's opt-out).
    Under H0 both groups are exchangeable draws from the same behaviour model."""
    u0, i0, o0 = s0["u"], s0["i"], s0["out"]
    u1, i1 = s1["u"], s1["i"]
    old = {}
    for u, i in zip(u0.tolist(), i0.tolist()):
        old.setdefault(u, set()).add(i)
    nw = {}
    for u, i in zip(u1.tolist(), i1.tolist()):
        nw.setdefault(u, set()).add(i)
    changed = np.array([old.get(u) != nw.get(u) for u in range(U)])
    new = changed[u1]
    C_in = cooc(u0[~o0], i0[~o0]); C_out = cooc(u0[o0], i0[o0]); C_new = cooc(u1[new], i1[new])
    deg_in = np.bincount(i0[~o0], minlength=I)
    ok = deg_in > 0

    def pairs(C, excl1, excl2):
        C = sp.triu(C, k=1).tocoo()
        keep = ok[C.row] & ok[C.col]
        r, c = C.row[keep], C.col[keep]
        ex = (np.asarray(excl1[r, c]).ravel() == 0) & (np.asarray(excl2[r, c]).ravel() == 0)
        return r[ex], c[ex]
    pr, pc = pairs(C_out, C_in, C_new)
    qr, qc = pairs(C_new, C_in, C_out)
    pr, pc = disjoint(pr, pc, rng); qr, qc = disjoint(qr, qc, rng)
    if len(pr) < 30 or len(qr) < 30:
        return 1.0, len(pr), len(qr)
    E = F0["item"][:, 5:]; En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    def feats(r, c):
        return (En[r] * En[c]).sum(1), np.log1p(deg_in[r]), np.log1p(deg_in[c])
    sp_, a1, a2 = feats(pr, pc); sq, b1, b2 = feats(qr, qc)
    y = np.concatenate([sp_, sq]); g = np.r_[np.ones(len(sp_)), np.zeros(len(sq))]
    X = np.column_stack([np.ones_like(y), g, np.r_[a1, b1], np.r_[a2, b2], np.r_[a1 * a2, b1 * b2]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    cov = np.linalg.inv(X.T @ X) * (res @ res) / (len(y) - X.shape[1])
    z = beta[1] / np.sqrt(cov[1, 1])
    return stats.norm.sf(z), len(pr), len(qr)


def disjoint(r, c, rng):
    """Greedy item-disjoint subsample so that pair-level statistics are independent."""
    used = set(); keep = []
    for k in rng.permutation(len(r)):
        if r[k] not in used and c[k] not in used:
            used.add(r[k]); used.add(c[k]); keep.append(k)
    keep = np.array(keep, int)
    return r[keep], c[keep]


def p_to_e(p, kappa=0.5):
    return kappa * max(p, 1e-12) ** (kappa - 1)


def run(seed, mode, rho, days=10):
    world = World(seed)
    rng = np.random.default_rng(seed + 1)
    snaps, feats = [], []
    for d in range(days + 1):
        s = world.snapshot(); snaps.append(s)
        feats.append(pipeline(s, mode, rho, day_seed=seed * 1000 + d))
        world.step()
    rec = {k: [] for k in ["T1", "T2", "T4", "T5"]}
    meta = {k: [] for k in ["canary", "cold", "t2_treat", "t2_ctrl", "t5_real", "t5_placebo"]}
    for d in range(days):
        s0, s1, F0, F1 = snaps[d], snaps[d + 1], feats[d], feats[d + 1]
        p1, nc, nk = t1_canary(s0, F0)
        p2, nt, nn = t2_zero_diff(s0, s1, F0, F1)
        p4 = t4_residual(s0, F0, surrogate_controls(s0, world, seed * 1000 + d))
        p5, n5a, n5b = t5_cooc(s0, s1, F0, rng)
        for k, p in zip(["T1", "T2", "T4", "T5"], [p1, p2, p4, p5]):
            rec[k].append(p)
        for k, v in zip(meta, [nc, nk, nt, nn, n5a, n5b]):
            meta[k].append(v)
    return rec, meta


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1].split(",")]
    configs = [("clean", 0.0)] + [(m, r) for m in ["local", "global"] for r in [0.02, 0.05, 0.1, 0.25, 1.0]]
    results = []
    for mode, rho in configs:
        for seed in seeds:
            rec, meta = run(seed, mode, rho)
            results.append(dict(mode=mode, rho=rho, seed=seed, p=rec,
                                meta={k: float(np.mean(v)) for k, v in meta.items()}))
            print(mode, rho, seed, {k: np.round(np.mean(np.array(v) < 0.05), 2) for k, v in rec.items()},
                  flush=True)
    json.dump(results, open(sys.argv[2], "w"))
