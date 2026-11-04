"""T6 active-canary reference test.

Adds C synthetic opted-out accounts. Each day a random set of 2*C*m items with no organic
interactions that day is drawn and split uniformly at random into treated and control halves; canary account k
purchases treated items k*m..(k+1)*m-1. Under H0 the pipeline never reads these edges,
so treated and control items are exchangeable and a label-permutation test is exact
(its validity rests only on the auditor's own randomization).

Usage: python3 t6_canary.py SEEDS OUT.json [null]
"""
import json
import sys
import numpy as np
import audit_sim as a

U0 = a.U
C, M_PER = 20, 4
a.U = U0 + C            # matrices must have room for the canary accounts


class CanaryWorld(a.World):
    def __init__(self, seed):
        super().__init__(seed)
        self.arng = np.random.default_rng(seed + 777)      # auditor's private coin flips
        self.can = np.arange(U0, U0 + C)
        self._assign()

    def _assign(self):
        # candidates: items with no organic interactions today (depends only on organic data)
        org = np.zeros(a.I, int)
        for u in range(U0):
            org[self.edges[u][0]] += 1
        pool = np.where(org == 0)[0]
        cand = self.arng.choice(pool, min(2 * C * M_PER, len(pool) // 2 * 2), replace=False)
        half = len(cand) // 2
        self.treated, self.control = cand[:half], cand[half:]
        for k, u in enumerate(self.can):
            items = self.treated[k * M_PER:(k + 1) * M_PER]
            self.edges[u] = (items, np.full(M_PER, 2))      # purchases
        self.optout[self.can] = True

    def step(self):
        super().step()
        self._assign()

    def snapshot(self):
        s = super().snapshot()
        s["treated"], s["control"] = self.treated.copy(), self.control.copy()
        return s


def perm_p(stat, labels_idx, n_t, n_perm=2000, rng=None):
    obs = stat[:n_t].mean() - stat[n_t:].mean()
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(len(stat))
        cnt += (stat[p[:n_t]].mean() - stat[p[n_t:]].mean()) >= obs
    return (cnt + 1) / (n_perm + 1)


def t6_test(snap, F, rng):
    """(i) Exact check: canary accounts' own features must equal the default (zero) output.
    (ii) Randomization test on candidate items with no organic edges today (a set that does
    not depend on the assignment): treated vs control feature norms."""
    can = np.arange(U0, U0 + C)
    p_cust = 0.0 if np.linalg.norm(F["cust"][can], axis=1).max() > 1e-9 else 1.0
    organic = snap["u"] < U0
    deg_org = np.bincount(snap["i"][organic], minlength=a.I)   # organic edges, in or out
    tr = snap["treated"][deg_org[snap["treated"]] == 0]
    co = snap["control"][deg_org[snap["control"]] == 0]
    stat = np.linalg.norm(F["item"][np.concatenate([tr, co])], axis=1)
    p_item = perm_p(stat, None, len(tr), rng=rng) if len(tr) and len(co) else 1.0
    return min(1.0, 2 * min(max(p_cust, 1e-6), p_item))


def run(seed, mode, rho, days=10):
    w = CanaryWorld(seed)
    rng = np.random.default_rng(seed + 5)
    ps = []
    for d in range(days):
        s = w.snapshot()
        F = a.pipeline(s, mode, rho, day_seed=seed * 1000 + d)
        ps.append(t6_test(s, F, rng))
        w.step()
    return ps


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1].split(",")]
    null = len(sys.argv) > 3 and sys.argv[3] == "null"
    configs = [("clean", 0.0)] if null else \
        [(m, r) for m in ["local", "global"] for r in [0.02, 0.05, 0.1, 0.25, 1.0]]
    out = []
    for mode, rho in configs:
        for seed in seeds:
            ps = run(seed, mode, rho)
            out.append(dict(mode=mode, rho=rho, seed=seed, p=ps))
            print(mode, rho, seed, np.round(np.mean(np.array(ps) < 0.05), 2), flush=True)
    json.dump(out, open(sys.argv[2], "w"))
