"""Run the passive tests (T1, T2, T4, T5, T6) on RetailRocket snapshots.
Usage: python3 run_real.py PATTERN SEEDS OUT.json [null] [DAYS]
"""
import sys
import json
import numpy as np
import audit_sim as a
import real_data
import run_all

real_data.load()
a.World = real_data.RealWorld          # run_all.run builds worlds through a.World

if __name__ == "__main__":
    pattern, seeds, out = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
    null = len(sys.argv) > 4 and sys.argv[4] == "null"
    days = int(sys.argv[5]) if len(sys.argv) > 5 else 14
    a.LEAK_PATTERN = pattern
    configs = [("clean", 0.0)] if null else [(m, r) for m in ["local", "global"] for r in [0.1, 1.0]]
    res = []
    for mode, rho in configs:
        for seed in seeds:
            rec = run_all.run(seed, mode, rho, "all", days=days)
            res.append(dict(mode=mode, rho=rho, seed=seed, pattern=pattern, p=rec))
            print(pattern, mode, rho, seed, {k: round(float(np.mean(np.array(v) < .05)), 2)
                                             for k, v in rec.items() if k != "lam"}, flush=True)
            json.dump(res, open(out, "w"), default=float)
