"""Naive tests of Section 4 versus their fixes, on CLEAN pipelines (every rejection is a
false alarm).

  C1  naive: pairs co-purchased by an opted-out customer vs random pairs
      naive': pairs co-purchased ONLY by opted-out customers vs random pairs
      fix   : T5 (only-pairs vs temporal placebo)
  C2  naive: some item whose opted-out edges changed also changed its feature
      fix   : T2 (treated vs control)
  C3  naive: >5% of input-unchanged items move by >25% (relative), raw coordinates
      fix   : same rule after orthogonal Procrustes alignment
      (the embedding block is randomly rotated each day to emulate retraining from scratch)
  C4  naive: T2 embedding test with 1-hop invariance        fix: 2-hop (both typed)
  C5  naive: T2 test with untyped invariance               fix: typed (both 2-hop / 1-hop exact)
  C6  naive: retroactive auditor vs prospective pipeline (real data only, needs history)
      fix   : auditor uses the agreed (prospective) policy

Usage: python3 naive.py synthetic|real SEEDS OUT.json [DAYS]
"""
import sys
import json
import numpy as np
import scipy.sparse as sp
from scipy import stats
import audit_sim as a
import c1_probe


# ---------------------------------------------------------------- helpers
def nbr_sets(snap, mask, typed):
    """item -> set of (customer[, type]) keys; customer -> set of (item[, type])."""
    by_item, by_cust = {}, {}
    for u, i, t in zip(snap["u"][mask].tolist(), snap["i"][mask].tolist(), snap["t"][mask].tolist()):
        ki = (u, t) if typed else u
        kc = (i, t) if typed else i
        by_item.setdefault(i, set()).add(ki)
        by_cust.setdefault(u, set()).add(kc)
    return by_item, by_cust


def t2_variant(s0, s1, F0, F1, typed, hops):
    """T2 with configurable edge identity (typed or not) and receptive field (1 or 2 hops)."""
    in0, c0 = nbr_sets(s0, ~s0["out"], typed); in1, c1 = nbr_sets(s1, ~s1["out"], typed)
    out0, _ = nbr_sets(s0, s0["out"], typed); out1, _ = nbr_sets(s1, s1["out"], typed)
    e0, e1 = F0["item"][:, 5:], F1["item"][:, 5:]
    active = np.zeros(a.I, bool)
    for k in set(in0) & set(in1):
        active[k] = True
    e1a = a.procrustes(e0, e1, active)
    emb_d = np.log(np.linalg.norm(e1a - e0, axis=1) / (np.linalg.norm(e0, axis=1) + 1e-9) + 1e-9)
    agg_d = np.abs(F1["item"][:, :5] - F0["item"][:, :5]).sum(1)
    agg_treat, treat, ctrl, deg = [], [], [], {}
    for k in in0:
        if in0[k] != in1.get(k, set()):
            continue
        changed = out0.get(k, set()) != out1.get(k, set())
        if changed:
            agg_treat.append(k)
        cust = [kk[0] if typed else kk for kk in in0[k]]
        if hops == 1 or all(c0.get(u) == c1.get(u) for u in cust):
            (treat if changed else ctrl).append(k); deg[k] = len(in0[k])
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
    return p_agg, p_emb, treat, ctrl


def naive_c2(s0, s1, F0, F1):
    """Alarm if any item whose typed opted-out neighbourhood changed also changed its feature."""
    out0, _ = nbr_sets(s0, s0["out"], True); out1, _ = nbr_sets(s1, s1["out"], True)
    items = [k for k in set(out0) | set(out1) if out0.get(k, set()) != out1.get(k, set())]
    if not items:
        return 1.0
    d = np.linalg.norm(F1["item"][items] - F0["item"][items], axis=1)
    return 0.0 if d.max() > 1e-9 else 1.0


def c3_rule(s0, s1, F0, F1, aligned, tau=0.25, frac=0.05):
    """Input-unchanged items (typed 2-hop opted-in AND 1-hop opted-out unchanged)."""
    _, _, treat, ctrl = t2_variant(s0, s1, F0, F1, typed=True, hops=2)
    idx = np.array(ctrl, int)
    if len(idx) == 0:
        return 1.0, np.nan
    e0, e1 = F0["item"][:, 5:], F1["item"][:, 5:]
    if aligned:
        act = np.zeros(a.I, bool); act[np.unique(s0["i"][~s0["out"]])] = True
        e1 = a.procrustes(e0, e1, act)
    rel = np.linalg.norm(e1[idx] - e0[idx], axis=1) / (np.linalg.norm(e0[idx], axis=1) + 1e-9)
    return (0.0 if (rel > tau).mean() > frac else 1.0), float(np.median(rel))


def c1_tests(s0, s1, F0, rng):
    o = c1_probe.day(s0, s1, F0, rng)
    ref = o["RANDOM"]
    _, p_any = c1_probe.test_vs(o["ANY"][0], o["ANY"][1], ref[0], ref[1])
    _, p_only = c1_probe.test_vs(o["ONLY"][0], o["ONLY"][1], ref[0], ref[1])
    return p_any, p_only


# ---------------------------------------------------------------- C6 (real data only)
def c6_tests(world_cls, seed, days):
    """Prospective pipeline (edges made while opted in stay usable) audited by a retroactive
    auditor (naive) or a prospective auditor (fix). Customer-level exact check of T1."""
    w = world_cls(seed)
    hist = {}
    for d in range(w.day - 30, w.day + 1):   # status before the audit = initial status
        hist[d] = w.optout.copy()
    out = []
    for _ in range(days):
        s = w.snapshot()
        eday = s["eday"]
        consented = np.array([not hist[d][u] for u, d in zip(s["u"].tolist(), eday.tolist())])
        s_pro = dict(s, out=~consented)
        F = a.pipeline(s_pro, "clean", 0.0, day_seed=seed * 1000 + w.day)
        cur_out = np.where(s["optout"])[0]
        dev = np.linalg.norm(F["cust"][cur_out], axis=1) > 1e-9
        naive_p = 0.0 if dev.any() else 1.0
        has_consented = np.zeros(a.U, bool); has_consented[s["u"][consented]] = True
        must_default = cur_out[~has_consented[cur_out]]
        fix_p = 0.0 if (np.linalg.norm(F["cust"][must_default], axis=1) > 1e-9).any() else 1.0
        out.append((naive_p, fix_p, int(dev.sum())))
        w.step(); hist[w.day] = w.optout.copy()
    return out


def rotated(F, seed):
    """Emulate retraining from a random initialization: rotate the embedding block by a random
    orthogonal matrix. Norms, cosines, aligned differences and ridge fits are unchanged."""
    k = F["item"].shape[1] - 5
    Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(k, k)))
    return dict(item=np.hstack([F["item"][:, :5], F["item"][:, 5:] @ Q]),
                cust=np.hstack([F["cust"][:, :4], F["cust"][:, 4:] @ Q]))


# ---------------------------------------------------------------- driver
def run(world_cls, seed, days):
    w = world_cls(seed)
    rng = np.random.default_rng(seed + 9)
    S, F = [], []
    for d in range(days + 1):
        s = w.snapshot(); S.append(s); F.append(a.pipeline(s, "clean", 0.0, day_seed=seed * 1000 + d)); w.step()
    rec = {}
    for d in range(days):
        s0, s1, F0, F1 = S[d], S[d + 1], F[d], F[d + 1]
        p_any, p_only = c1_tests(s0, s1, F0, rng)
        rec.setdefault("C1_naive_any", []).append(p_any)
        rec.setdefault("C1_naive_only", []).append(p_only)
        rec.setdefault("C1_fix_T5", []).append(a.t5_cooc(s0, s1, F0, rng)[0])
        rec.setdefault("C2_naive", []).append(naive_c2(s0, s1, F0, F1))
        rec.setdefault("C2_fix_T2", []).append(a.t2_zero_diff(s0, s1, F0, F1)[0])
        R0, R1 = rotated(F0, seed * 7919 + d), rotated(F1, seed * 7919 + d + 1)
        p3n, m3n = c3_rule(s0, s1, R0, R1, aligned=False)
        p3f, m3f = c3_rule(s0, s1, R0, R1, aligned=True)
        rec.setdefault("C3_naive", []).append(p3n); rec.setdefault("C3_fix", []).append(p3f)
        rec.setdefault("C3_med_raw", []).append(m3n); rec.setdefault("C3_med_aligned", []).append(m3f)
        _, pe1, _, _ = t2_variant(s0, s1, F0, F1, typed=True, hops=1)
        _, pe2, _, _ = t2_variant(s0, s1, F0, F1, typed=True, hops=2)
        pa_u, pe2u, _, _ = t2_variant(s0, s1, F0, F1, typed=False, hops=2)
        pa_t, _, _, _ = t2_variant(s0, s1, F0, F1, typed=True, hops=2)
        rec.setdefault("C4_naive_1hop", []).append(pe1)
        rec.setdefault("C4_fix_2hop", []).append(pe2)
        rec.setdefault("C5_naive_untyped_agg", []).append(pa_u)
        rec.setdefault("C5_naive_untyped_emb", []).append(pe2u)
        rec.setdefault("C5_fix_typed_agg", []).append(pa_t)
    return rec


if __name__ == "__main__":
    src, seeds, out = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
    days = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    if src == "real":
        import real_data
        real_data.load(); W = real_data.RealWorld
    else:
        W = a.World
    res = []
    for seed in seeds:
        rec = run(W, seed, days)
        if src == "real":
            c6 = c6_tests(W, seed, days)
            rec["C6_naive"] = [x[0] for x in c6]; rec["C6_fix"] = [x[1] for x in c6]
            rec["C6_n_flagged"] = [x[2] for x in c6]
        res.append(dict(seed=seed, src=src, p=rec))
        print(src, seed, {k: round(float(np.nanmean(np.array(v) < .05)), 2) for k, v in rec.items()
                          if not k.startswith(("C3_med", "C6_n"))}, flush=True)
        json.dump(res, open(out, "w"), default=float)
