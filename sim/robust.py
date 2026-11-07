"""Cluster-aware, time-bracketed variants of T5 and T6 (motivated by the real-data null study).

Real behavior differs from the stationary simulator in two ways that break the original
T5/T6 calibration:
  * dependence: one visitor contributes many pairs/edges from the same session;
  * drift: the window [t-W+1, t] and the next day t+1 are not exchangeable samples.
Fixes:
  * T5b: at most one pair per customer (and item-disjoint); the placebo brackets the window
         in time: pairs that co-occur only through interactions that LEFT the window on day
         t (expired) or ENTER it on day t+1 (fresh); symmetric exclusions across all sources.
  * T6b: reference edges are the customer's expired and fresh edges (bracketing); on real
         data, contexts exclude items from the same day as the scored edge (session
         coherence). Test unit is the customer (paired).
"""
import numpy as np
from scipy import stats
import audit_sim as a


def _keys(s):
    return s["u"].astype(np.int64) * a.I + s["i"]


def _baskets(u, i):
    d = {}
    for uu, ii in zip(u.tolist(), i.tolist()):
        d.setdefault(uu, []).append(ii)
    return d


def _delta_edges(s_from, s_ref):
    """edges of s_from not present in s_ref (customer, item)."""
    kref = set(_keys(s_ref).tolist())
    k = _keys(s_from)
    m = np.array([x not in kref for x in k.tolist()], bool) if len(k) else np.zeros(0, bool)
    return s_from["u"][m], s_from["i"][m]


def t5b(s_prev, s0, s1, F0, rng, max_pairs_per_cust=1):
    u0, i0, o0 = s0["u"], s0["i"], s0["out"]
    ue, ie = _delta_edges(s_prev, s0)      # expired
    uf, if_ = _delta_edges(s1, s0)         # fresh
    C_in = a.cooc(u0[~o0], i0[~o0]); C_out = a.cooc(u0[o0], i0[o0])
    C_exp = a.cooc(ue, ie); C_new = a.cooc(uf, if_)
    ok = np.bincount(i0[~o0], minlength=a.I) > 0
    E = F0["item"][:, 5:]; En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    ld = np.log1p(np.bincount(i0[~o0], minlength=a.I))

    def draw(u, i, excl):
        out = []
        for c, items in _baskets(u, i).items():
            items = [x for x in set(items) if ok[x]]
            if len(items) < 2:
                continue
            cand = []
            for _ in range(10):
                p, q = rng.choice(items, 2, replace=False)
                if all(M[p, q] == 0 for M in excl):
                    cand.append((p, q))
                    if len(cand) >= max_pairs_per_cust:
                        break
            out += cand
        if not out:
            return np.zeros(0, int), np.zeros(0, int)
        r, c = np.array(out).T
        return a.disjoint(r, c, rng)

    rr, rc = draw(u0[o0], i0[o0], [C_in, C_exp, C_new])
    pr1, pc1 = draw(ue, ie, [C_in, C_out, C_new])
    pr2, pc2 = draw(uf, if_, [C_in, C_out, C_exp])
    pr, pc = np.r_[pr1, pr2], np.r_[pc1, pc2]
    if len(rr) < 30 or len(pr) < 30:
        return 1.0
    sim = lambda r, c: (En[r] * En[c]).sum(1)
    y = np.r_[sim(rr, rc), sim(pr, pc)]; g = np.r_[np.ones(len(rr)), np.zeros(len(pr))]
    X = np.column_stack([np.ones_like(y), g, np.r_[ld[rr], ld[pr]], np.r_[ld[rc], ld[pc]]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ beta
    cov = np.linalg.inv(X.T @ X) * (res @ res) / (len(y) - X.shape[1])
    return float(stats.norm.sf(beta[1] / np.sqrt(cov[1, 1])))


def t6b(s_prev, s0, s1, F0):
    """Paired per-customer test: OUT edges vs the same customer's expired + fresh edges."""
    E = F0["item"][:, 5:]; En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    oo = s0["optout"]
    has_day = "eday" in s0
    cur = {}
    for k, (u, i) in enumerate(zip(s0["u"].tolist(), s0["i"].tolist())):
        cur.setdefault(u, []).append((i, int(s0["eday"][k]) if has_day else 0))
    ue, ie = _delta_edges(s_prev, s0); uf, if_ = _delta_edges(s1, s0)
    ref = {}
    for u, i in zip(np.r_[ue, uf].tolist(), np.r_[ie, if_].tolist()):
        if oo[u]:
            ref.setdefault(u, []).append(i)
    # symmetric conditioning (simulator, where a basket is redrawn as a whole): reference items are
    # by construction NOT in today's basket, so OUT items are restricted to those NOT in the
    # customer's previous/next basket. On real data the windows slide by one day, the
    # neighbouring windows contain almost all of today's basket, and references are bracketed by
    # day instead, so no restriction is applied.
    other = {}
    for s in ((s_prev, s1) if not has_day else ()):
        for u, i in zip(s["u"].tolist(), s["i"].tolist()):
            if u in ref:
                other.setdefault(u, set()).add(i)

    def score(i, ctx):
        if not ctx:
            return None
        m = En[ctx].mean(0); n = np.linalg.norm(m)
        return float(En[i] @ m / n) if n > 0 else None

    diffs = []
    for u, refs in ref.items():
        items = cur.get(u, [])
        if len(items) < 2:
            continue
        s_out = []
        for i, d in items:
            if i in other.get(u, ()):
                continue
            ctx = [j for j, dj in items if j != i and (not has_day or dj != d)]
            sc = score(i, ctx)
            if sc is not None:
                s_out.append(sc)
        s_ref = []
        all_ctx = [j for j, _ in items]
        for i in refs:
            ctx = [j for j in all_ctx if j != i]
            if not has_day and len(ctx) == len(all_ctx) and len(ctx) > 1:
                ctx = ctx[:-1]                 # same context size as OUT edges (simulator)
            sc = score(i, ctx)
            if sc is not None:
                s_ref.append(sc)
        if s_out and s_ref:
            diffs.append(np.mean(s_out) - np.mean(s_ref))
    if len(diffs) < 3:
        return 1.0
    return float(stats.ttest_1samp(diffs, 0, alternative="greater").pvalue)
