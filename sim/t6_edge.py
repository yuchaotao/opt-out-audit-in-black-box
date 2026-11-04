"""T6: edge-level membership inference with a same-user fresh-edge reference.

For an edge (u, i) on day t, the auditor builds a context vector for u from F alone:
the mean embedding of u's other day-t items (leave-one-out). Four groups of edges:
  IN    opted-in edges on day t                       (members)
  INF   day-(t+1) basket of opted-in customers whose basket changed (reference)
  OUT   opted-out edges on day t                      (suspects)
  FRESH day-(t+1) basket of the same opted-out customers, if it changed (reference)
Scorers:
  cosine  : cos(F_i, context)
  learned : logistic regression trained on opted-in customers to separate IN from INF
            (labels are free), then applied to opted-out customers.
Test: per-customer paired difference mean(OUT) - mean(FRESH) > 0, one-sided t-test.
Effect size: lambda = (OUT - FRESH) / (IN - INF), cross-fitted for the learned scorer.
"""
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
import audit_sim as a


def _baskets(snap):
    d = {}
    for u, i in zip(snap["u"].tolist(), snap["i"].tolist()):
        d.setdefault(u, []).append(i)
    return d


def _collect(s0, s1, F0):
    """Return per-edge feature rows with group labels and customer ids."""
    E = F0["item"][:, 5:]
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    b0, b1 = _baskets(s0), _baskets(s1)
    rng = np.random.default_rng(0)
    oo = s0["optout"]
    rows, grp, cust = [], [], []
    for u, items0 in b0.items():
        # reference: the customer's whole day-(t+1) basket, if it changed. Keeping only the
        # new edges would bias the reference away from the customer's strongest preferences.
        nxt = b1.get(u, [])
        new = nxt if set(nxt) != set(items0) else []
        for kind, items in (("cur", items0), ("new", new)):
            for i in items:
                # context = u's other day-t items; new edges drop one random item so that
                # both kinds of edge use contexts of the same size (n - 1)
                if kind == "cur" or i in items0:
                    ctx = [j for j in items0 if j != i]
                else:
                    ctx = list(items0)
                    ctx.pop(rng.integers(len(ctx)))
                if not ctx:
                    continue
                m = En[ctx].mean(0)
                nm = np.linalg.norm(m)
                if nm == 0:
                    continue
                m = m / nm
                # only F-derived features: anything computed from the edge lists could encode
                # the group by construction (e.g. an item's degree counts the edge itself)
                rows.append(np.concatenate([[En[i] @ m], En[i] * m]))
                g = ("OUT" if kind == "cur" else "FRESH") if oo[u] else ("IN" if kind == "cur" else "INF")
                grp.append(g); cust.append(u)
    return np.array(rows), np.array(grp), np.array(cust)


def _paired_p(score, grp, cust):
    d = {}
    for sc, g, u in zip(score, grp, cust):
        if g in ("OUT", "FRESH"):
            d.setdefault(u, {"OUT": [], "FRESH": []})[g].append(sc)
    diffs = [np.mean(v["OUT"]) - np.mean(v["FRESH"]) for v in d.values() if v["OUT"] and v["FRESH"]]
    if len(diffs) < 3:
        return 1.0
    return stats.ttest_1samp(diffs, 0, alternative="greater").pvalue


def _lam(score, grp):
    m = {g: score[grp == g].mean() if (grp == g).any() else np.nan for g in ("IN", "INF", "OUT", "FRESH")}
    den = m["IN"] - m["INF"]
    return (m["OUT"] - m["FRESH"]) / den if den != 0 else np.nan


def t6_edge(s0, s1, F0, seed=0):
    X, grp, cust = _collect(s0, s1, F0)
    if len(X) == 0:
        return dict(p_cos=1.0, p_learn=1.0, lam_cos=np.nan, lam_learn=np.nan)
    cos = X[:, 0]
    out = dict(p_cos=_paired_p(cos, grp, cust), lam_cos=_lam(cos, grp))
    # learned scorer: train on opted-in customers only (IN = 1, INF = 0), 2-fold by customer
    tr = np.isin(grp, ["IN", "INF"])
    score = np.zeros(len(X))
    folds = cust % 2
    for f in (0, 1):
        fit = tr & (folds != f)
        y = (grp[fit] == "IN").astype(int)
        # balance classes
        rng = np.random.default_rng(seed + f)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        k = min(len(pos), len(neg))
        idx = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])
        clf = LogisticRegression(C=1.0, max_iter=300).fit(X[fit][idx], y[idx])
        apply = (folds == f)
        score[apply] = clf.decision_function(X[apply])
    out.update(p_learn=_paired_p(score, grp, cust), lam_learn=_lam(score, grp))
    return out
