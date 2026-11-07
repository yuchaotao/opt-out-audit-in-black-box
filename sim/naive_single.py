"""Pitfall 1: the restricted single-node naive test on a COMPLIANT pipeline.

Naive test: take items whose typed opted-in neighbourhood (1-hop) is identical on days d and d+1
but whose opted-out edges changed; alarm if the item's feature changed. Under a compliant pipeline
every alarm is false. Configurations isolate the failure modes:
  counts            A1 counts (exact, local)                         -> should never alarm
  prop_fixed        deterministic local propagation of intrinsic x   -> fails: neighbours' opted-in changes (P1a)
                    (0.5 x_i + 0.5 mean x_j over co-interacted j)
  prop_fixed_2hop   same, restricted to 2-hop invariant items        -> fixed
  mf_random_raw     rank-16 SVD, fresh random seed every day         -> fails (P1a, P1b, P1c)
  mf_fixed_raw      rank-16 SVD, fixed seed                          -> still fails (P1a, P1b)
  mf_fixed_aligned  fixed seed + Procrustes alignment                -> still fails (P1a)
  mf_T2             T2 (treated vs control, 2-hop invariance, aligned regression), random seed -> calibrated
Usage: python3 naive_single.py synthetic|real SEEDS OUT.json [DAYS]
"""
import sys
import json
import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
import audit_sim as a
import families as fam


def feats(s, day_seed, fixed):
    use = ~s["out"]; u, i, t = s["u"][use], s["i"][use], s["t"][use]
    cnt = np.zeros((a.I, 4)); np.add.at(cnt, (i, np.minimum(t, 3)), 1)
    B = sp.csr_matrix((np.ones(len(u)), (u, i)), shape=(a.U, a.I)); B.data[:] = 1
    C = (B.T @ B).tocsr(); C.setdiag(0); C.eliminate_zeros()
    deg = np.asarray(C.sum(1)).ravel()
    P = 0.5 * s["x"] + 0.5 * (C @ s["x"]) / np.maximum(deg, 1)[:, None]
    A = sp.csr_matrix((a.TYPE_W[np.minimum(t, 3)], (u, i)), shape=(a.U, a.I))
    _, S, Vt = randomized_svd(A, a.K, n_iter=2, random_state=0 if fixed else day_seed)
    return {"counts": cnt, "prop": P, "mf": Vt.T * np.sqrt(S)}


def groups(s0, s1):
    in0, c0 = fam._typed(s0, ~s0["out"]); in1, c1 = fam._typed(s1, ~s1["out"])
    out0, _ = fam._typed(s0, s0["out"]); out1, _ = fam._typed(s1, s1["out"])
    t1, t2 = [], []
    for k in in0:
        if in0[k] != in1.get(k, set()) or out0.get(k, set()) == out1.get(k, set()):
            continue
        t1.append(k)
        if all(c0.get(uu) == c1.get(uu) for uu, _ in in0[k]):
            t2.append(k)
    return t1, t2


def changed(X0, X1, idx):
    if not idx:
        return False
    d = np.linalg.norm(X1[idx] - X0[idx], axis=1) / (np.linalg.norm(X0[idx], axis=1) + 1e-12)
    return bool(d.max() > 1e-6)


def run(src, seed, days):
    a.LEAK_SALT = seed
    if src == "real":
        import real_data
        real_data.load(); w = real_data.RealWorld(seed)
    else:
        w = a.World(seed)
    S = []
    for d in range(days + 1):
        S.append(w.snapshot()); w.step()
    rec = {k: [] for k in ["counts", "prop_fixed", "prop_fixed_2hop", "mf_random_raw",
                           "mf_fixed_raw", "mf_fixed_aligned", "mf_T2", "n_treated", "n_treated_2hop"]}
    prev = None
    for d in range(days + 1):
        cur = (feats(S[d], seed * 1000 + d, False), feats(S[d], 0, True))
        if prev is not None:
            s0, s1 = S[d - 1], S[d]
            (r0, f0), (r1, f1) = prev, cur
            t1, t2 = groups(s0, s1)
            rec["n_treated"].append(len(t1)); rec["n_treated_2hop"].append(len(t2))
            rec["counts"].append(changed(f0["counts"], f1["counts"], t1))
            rec["prop_fixed"].append(changed(f0["prop"], f1["prop"], t1))
            rec["prop_fixed_2hop"].append(changed(f0["prop"], f1["prop"], t2))
            rec["mf_random_raw"].append(changed(r0["mf"], r1["mf"], t1))
            rec["mf_fixed_raw"].append(changed(f0["mf"], f1["mf"], t1))
            act = np.ones(a.I, bool)
            rec["mf_fixed_aligned"].append(changed(f0["mf"], a.procrustes(f0["mf"], f1["mf"], act), t1))
            rec["mf_T2"].append(bool(fam.t2g(s0, s1, r0["mf"], r1["mf"], False) < 0.05))
        prev = cur
    return rec


if __name__ == "__main__":
    src, seeds, out = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
    days = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    res = []
    for seed in seeds:
        rec = run(src, seed, days)
        res.append(dict(src=src, seed=seed, rec=rec))
        print(seed, {k: round(float(np.mean(v)), 2) for k, v in rec.items()}, flush=True)
        json.dump(res, open(out, "w"))
