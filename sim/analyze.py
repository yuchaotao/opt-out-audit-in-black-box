"""Summaries and the power figure.

Inputs (produced by audit_sim.py, run_all.py, t7_active.py):
  null.json          clean pipeline, T1 T2 T4 T5          (30 audits)
  t6_null.json       clean pipeline, T6                   (30 audits)
  t7_null.json       clean pipeline, T7                   (30 audits)
  main.json          per-edge leaks, T1 T2 T4 T5
  t6_edge_sweep.json per-edge leaks, T6
  t7_main_edge.json  per-edge leaks, T7
  all_user.json      per-user leaks, T1 T2 T4 T5 T6
  t7_main_user.json  per-user leaks, T7
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit_sim import p_to_e

RHOS = [0.02, 0.05, 0.1, 0.25, 1.0]
PASSIVE = ["T1", "T2", "T4", "T5", "T6"]


def load(f):
    return json.load(open(f))


def merge(*sources):
    """Merge per-(mode, rho, seed) records from several files into one dict of test -> p-values."""
    out = {}
    for src, rename in sources:
        for r in src:
            key = (r.get("mode", "clean"), r.get("rho", 0.0), r["seed"])
            rec = out.setdefault(key, {})
            if isinstance(r["p"], dict):
                for k, v in r["p"].items():
                    rec[rename.get(k, k)] = v
            else:
                rec[rename["_list"]] = r["p"]
    return out


def eproc(ps):
    return np.cumprod([p_to_e(p) for p in ps])


def detect(e):
    return (int(np.argmax(e >= 20)) + 1) if (e >= 20).any() else None


def summarize(recs, tests):
    """recs: list of dict test -> daily p-values."""
    s = {}
    for t in tests:
        P = np.array([r[t] for r in recs if t in r])
        s[t] = dict(rej=float((P < 0.05).mean()))
    passive = [t for t in PASSIVE if t in tests]
    days = [detect(np.mean([eproc(r[t]) for t in passive], axis=0)) for r in recs]
    s["passive"] = dict(det=sum(d is not None for d in days), n=len(days), days=days)
    if "T7" in tests:
        d7 = [detect(eproc(r["T7"])) for r in recs]
        s["T7"]["det"], s["T7"]["n"], s["T7"]["days"] = sum(d is not None for d in d7), len(d7), d7
    return s


if __name__ == "__main__":
    null = merge((load("null.json"), {}), (load("t6_null.json"), {}),
                 (load("t7_null.json"), {"_list": "T7"}))
    edge = merge((load("main.json"), {}), (load("t6_edge_sweep.json"), {}),
                 (load("t7_main_edge.json"), {"_list": "T7"}))
    user = merge((load("all_user.json"), {}), (load("t7_main_user.json"), {"_list": "T7"}))
    tests = PASSIVE + ["T6learn", "T7"]

    res = {"null": summarize(list(null.values()), tests)}
    for pat, src in (("edge", edge), ("user", user)):
        for mode in ("local", "global"):
            for rho in RHOS:
                recs = [v for (m, r, s), v in src.items() if m == mode and r == rho]
                res[f"{pat}|{mode}|{rho}"] = summarize(recs, tests)
    # lambda (T6 effect size) under per-user leaks, from all_user.json
    lam = {}
    for r in load("all_user.json"):
        lam.setdefault((r["mode"], r["rho"]), []).extend(r["p"]["lam"])
    lam_null = [x for r in load("t6_null.json") for x in r["p"]["lam"]]
    res["lambda_user_global"] = {str(rho): float(np.nanmedian(lam[("global", rho)])) for rho in RHOS}
    res["lambda_null"] = float(np.nanmedian(lam_null))
    json.dump(res, open("summary.json", "w"), indent=1, default=float)

    for k, v in res.items():
        if isinstance(v, dict) and "passive" in v:
            print(k, {t: round(v[t]["rej"], 2) for t in tests if t in v},
                  "passive", f'{v["passive"]["det"]}/{v["passive"]["n"]}', v["passive"]["days"],
                  "T7", f'{v["T7"]["det"]}/{v["T7"]["n"]}', v["T7"]["days"])
    print("lambda", res["lambda_user_global"], "null", res["lambda_null"])

    # ---- figure: rows = leak pattern, cols = leakage channel
    names = {"T1": "T1 canary", "T2": "T2 zero-diff", "T4": "T4 residual", "T5": "T5 pairs",
             "T6": "T6 edges", "T7": "T7 active (ref.)"}
    mk = {"T1": "o", "T2": "s", "T4": "^", "T5": "D", "T6": "v"}
    cols = {"T1": "#1f4e79", "T2": "#c55a11", "T4": "#548235", "T5": "#7030a0", "T6": "#2e9c9c"}
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5})
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.1), sharex=True, sharey=True)
    for i, (pat, plab) in enumerate((("edge", "per-edge leak"), ("user", "per-customer leak"))):
        for j, (mode, mlab) in enumerate((("local", "aggregation"), ("global", "representation"))):
            ax = axes[i, j]
            for t in PASSIVE:
                y = [res[f"{pat}|{mode}|{r}"][t]["rej"] for r in RHOS]
                ax.plot(RHOS, y, marker=mk[t], ms=2.6, lw=0.9, color=cols[t], label=names[t])
            y7 = [res[f"{pat}|{mode}|{r}"]["T7"]["rej"] for r in RHOS]
            ax.plot(RHOS, y7, color="black", lw=1.1, ls="--", marker="x", ms=3, label=names["T7"])
            ax.axhline(0.05, color="gray", lw=0.5, ls=":")
            ax.set_xscale("log"); ax.set_xticks(RHOS); ax.set_xticklabels([".02", ".05", ".1", ".25", "1"])
            ax.set_ylim(-0.03, 1.03)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(mlab, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{plab}\nrejection rate", fontsize=7)
            if i == 1:
                ax.set_xlabel(r"leak rate $\rho$")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=6.3,
               frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=1.6, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=0.6, w_pad=0.4)
    fig.savefig("../power.pdf", bbox_inches="tight")
