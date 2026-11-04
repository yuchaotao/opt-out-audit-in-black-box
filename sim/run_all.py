"""Run passive tests under a given leak pattern.

Usage:
  python3 run_all.py PATTERN TESTS SEEDS OUT.json [null]
    PATTERN: edge | user        (how leaked opted-out edges are chosen)
    TESTS:   all | t6            (all = T1, T2, T4, T5, T6; t6 = edge-membership test only)
    null:    run only the clean pipeline
"""
import sys
import json
import numpy as np
import audit_sim as a
import t6_edge

CONFIGS = [(m, r) for m in ["local", "global"] for r in [0.02, 0.05, 0.1, 0.25, 1.0]]


def run(seed, mode, rho, tests, days=10):
    a.LEAK_SALT = seed
    w = a.World(seed)
    snaps, feats = [], []
    for d in range(days + 1):
        s = w.snapshot(); snaps.append(s)
        feats.append(a.pipeline(s, mode, rho, day_seed=seed * 1000 + d)); w.step()
    rng = np.random.default_rng(seed + 1)
    rec = {}
    for d in range(days):
        s0, s1, F0, F1 = snaps[d], snaps[d + 1], feats[d], feats[d + 1]
        if tests == "all":
            rec.setdefault("T1", []).append(a.t1_canary(s0, F0)[0])
            rec.setdefault("T2", []).append(a.t2_zero_diff(s0, s1, F0, F1)[0])
            rec.setdefault("T4", []).append(a.t4_residual(s0, F0, a.surrogate_controls(s0, w, seed * 1000 + d)))
            rec.setdefault("T5", []).append(a.t5_cooc(s0, s1, F0, rng)[0])
        r6 = t6_edge.t6_edge(s0, s1, F0, seed)
        rec.setdefault("T6", []).append(r6["p_cos"])
        rec.setdefault("T6learn", []).append(r6["p_learn"])
        rec.setdefault("lam", []).append(r6["lam_cos"])
    return rec


if __name__ == "__main__":
    pattern, tests, seeds, out = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")], sys.argv[4]
    a.LEAK_PATTERN = pattern
    configs = [("clean", 0.0)] if (len(sys.argv) > 5 and sys.argv[5] == "null") else CONFIGS
    res = []
    for mode, rho in configs:
        for seed in seeds:
            rec = run(seed, mode, rho, tests)
            res.append(dict(mode=mode, rho=rho, seed=seed, pattern=pattern, p=rec))
            print(pattern, mode, rho, seed, {k: round(float(np.mean(np.array(v) < .05)), 2)
                                             for k, v in rec.items() if k != "lam"}, flush=True)
    json.dump(res, open(out, "w"), default=float)
