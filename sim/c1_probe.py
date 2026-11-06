"""How much does 'only co-purchased by opted-out customers' fix the C1 naive test?

On a CLEAN pipeline (no leak), compare cosine similarity in F of:
  ANY     pairs co-purchased by an opted-out customer (naive test)
  ONLY    pairs co-purchased only by opted-out customers (never together in E_in)
    ONLY-2HOP   ... that are still linked by a 2-hop opted-in path i - k - j
    ONLY-NOHOP  ... with no opted-in path of length <= 2 in the item graph
  RANDOM  random item pairs never co-purchased by anyone (baseline)
  PLACEBO T5's temporal placebo pairs
All groups are restricted to items with at least one opted-in edge and subsampled to be
item-disjoint. Reports the mean similarity gap to RANDOM and the share of days on which
a one-sided test (vs RANDOM, degree-adjusted) rejects at 0.05.
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
import audit_sim as a


def pairs_from(C, ok, excl=None):
    C = sp.triu(C, k=1).tocoo()
    keep = ok[C.row] & ok[C.col]
    r, c = C.row[keep], C.col[keep]
    if excl is not None:
        m = np.asarray(excl[r, c]).ravel() == 0
        r, c = r[m], c[m]
    return r, c


def test_vs(sim_a, deg_a, sim_b, deg_b):
    y = np.r_[sim_a, sim_b]; g = np.r_[np.ones(len(sim_a)), np.zeros(len(sim_b))]
    X = np.column_stack([np.ones_like(y), g, np.r_[deg_a, deg_b]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ beta
    cov = np.linalg.inv(X.T @ X) * (res @ res) / (len(y) - 3)
    return beta[1], stats.norm.sf(beta[1] / np.sqrt(cov[1, 1]))


def day(s0, s1, F0, rng):
    u0, i0, o0 = s0["u"], s0["i"], s0["out"]
    C_in = a.cooc(u0[~o0], i0[~o0]); C_out = a.cooc(u0[o0], i0[o0]); C_all = a.cooc(u0, i0)
    deg_in = np.bincount(i0[~o0], minlength=a.I); ok = deg_in > 0
    # 2-hop opted-in reachability in the item graph: (C_in @ C_in) > 0
    Cb = C_in.copy(); Cb.data[:] = 1; Cb.setdiag(0); Cb.eliminate_zeros()
    two = (Cb @ Cb).tocsr()
    E = F0["item"][:, 5:]; En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    ld = np.log1p(deg_in)

    groups = {}
    r, c = pairs_from(C_out, ok); groups["ANY"] = (r, c)
    r, c = pairs_from(C_out, ok, C_in); groups["ONLY"] = (r, c)
    h = (np.asarray(two[r, c]).ravel() > 0)
    groups["ONLY-2HOP"] = (r[h], c[h]); groups["ONLY-NOHOP"] = (r[~h], c[~h])
    cand = np.where(ok)[0]
    rr = rng.choice(cand, 20000); cc = rng.choice(cand, 20000)
    m = (rr != cc) & (np.asarray(C_all[rr, cc]).ravel() == 0)
    groups["RANDOM"] = (rr[m], cc[m])
    # T5 placebo (same construction as audit_sim.t5_cooc)
    old = {}; nw = {}
    for u, i in zip(u0.tolist(), i0.tolist()): old.setdefault(u, set()).add(i)
    for u, i in zip(s1["u"].tolist(), s1["i"].tolist()): nw.setdefault(u, set()).add(i)
    changed = np.array([old.get(u) != nw.get(u) for u in range(a.U)])
    new = changed[s1["u"]]
    C_new = a.cooc(s1["u"][new], s1["i"][new])
    rp, cp = pairs_from(C_new, ok, C_in)
    mm = np.asarray(C_out[rp, cp]).ravel() == 0
    groups["PLACEBO"] = (rp[mm], cp[mm])
    rr_, cc_ = pairs_from(C_out, ok, C_in)
    mm = np.asarray(C_new[rr_, cc_]).ravel() == 0
    groups["ONLY(sym)"] = (rr_[mm], cc_[mm])

    out = {}
    for k, (r, c) in groups.items():
        r, c = a.disjoint(r, c, rng)
        out[k] = ((En[r] * En[c]).sum(1), ld[r] + ld[c], len(r))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    acc = {}
    for seed in [0, 1, 2]:
        w = a.World(seed); S = []; F = []
        for d in range(6):
            s = w.snapshot(); S.append(s); F.append(a.pipeline(s, "clean", 0.0, seed * 1000 + d)); w.step()
        for d in range(5):
            o = day(S[d], S[d + 1], F[d], rng)
            ref = o["RANDOM"]
            for k, v in o.items():
                if k == "RANDOM":
                    acc.setdefault(k, []).append((v[0].mean(), np.nan, np.nan, v[2])); continue
                refk = o["PLACEBO"] if k == "ONLY(sym)" else ref
                gap, p = test_vs(v[0], v[1], refk[0], refk[1])
                acc.setdefault(k, []).append((v[0].mean(), gap, p, v[2]))
    print(f"{'group':12s} {'mean cos':>9s} {'gap':>7s} {'reject':>7s} {'pairs/day':>9s}")
    for k in ["RANDOM", "ANY", "ONLY", "ONLY-2HOP", "ONLY-NOHOP", "PLACEBO", "ONLY(sym)"]:
        v = np.array(acc[k], dtype=float)
        print(f"{k:12s} {np.nanmean(v[:,0]):9.3f} {np.nanmean(v[:,1]):7.3f} "
              f"{np.nanmean(v[:,2] < 0.05) if k!='RANDOM' else float('nan'):7.2f} {v[:,3].mean():9.0f}")
