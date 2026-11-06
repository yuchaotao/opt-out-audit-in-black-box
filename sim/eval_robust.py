"""Calibration/power of T5b, T6b (and original T5, T6) on real or synthetic data.
Usage: python3 eval_robust.py real|synthetic MODE RHO PATTERN SEEDS OUT.json [DAYS]"""
import sys, json, numpy as np
import audit_sim as a, robust, t6_edge
src, mode, rho, pattern = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
seeds, out = [int(x) for x in sys.argv[5].split(",")], sys.argv[6]
days = int(sys.argv[7]) if len(sys.argv) > 7 else 14
if src == "real":
    import real_data; real_data.load(); W = real_data.RealWorld
else:
    W = a.World
a.LEAK_PATTERN = pattern
res = []
for seed in seeds:
    a.LEAK_SALT = seed
    w = W(seed); rng = np.random.default_rng(seed + 3)
    S, F = [], []
    for d in range(days + 2):
        s = w.snapshot(); S.append(s)
        F.append({"item": a.pipeline(s, mode, rho, day_seed=seed * 1000 + d)["item"]})  # item features only (memory)
        w.step()
    rec = {"T5": [], "T5b": [], "T6": [], "T6b": []}
    for d in range(1, days + 1):
        sp_, s0, s1, F0 = S[d - 1], S[d], S[d + 1], F[d]
        rec["T5"].append(float(a.t5_cooc(s0, s1, F0, rng)[0]))
        rec["T5b"].append(robust.t5b(sp_, s0, s1, F0, rng))
        rec["T6"].append(float(t6_edge.t6_edge(s0, s1, F0, seed)["p_cos"]))
        rec["T6b"].append(robust.t6b(sp_, s0, s1, F0))
    res.append(dict(seed=seed, mode=mode, rho=rho, pattern=pattern, src=src, p=rec))
    print(src, mode, rho, pattern, seed, {k: round(float(np.mean(np.array(v) < .05)), 2) for k, v in rec.items()}, flush=True)
    json.dump(res, open(out, "w"))
