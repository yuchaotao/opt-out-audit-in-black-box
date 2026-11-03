import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit_sim import p_to_e
R = json.load(open("main.json")); N = json.load(open("null.json"))
T = ["T1", "T2", "T4", "T5"]
def eproc(ps):
    return np.cumprod([p_to_e(p) for p in ps])
def summarize(rs):
    out = {}
    for k in T:
        P = np.array([r["p"][k] for r in rs])
        E = np.array([eproc(r["p"][k]) for r in rs])
        cross = [(np.argmax(e >= 20) + 1) if (e >= 20).any() else np.nan for e in E]
        out[k] = dict(rej=(P < 0.05).mean(), det=np.mean([(e >= 20).any() for e in E]),
                      days=np.nanmedian(cross) if not np.all(np.isnan(cross)) else np.nan)
    Em = np.mean([[eproc(r["p"][k]) for k in T] for r in rs], axis=1)  # mean of e-processes
    cross = [(np.argmax(e >= 20) + 1) if (e >= 20).any() else np.nan for e in Em]
    out["any"] = dict(rej=np.nan, det=np.mean([(e >= 20).any() for e in Em]),
                      days=np.nanmedian(cross) if not np.all(np.isnan(cross)) else np.nan)
    return out
res = {}
res[("null", 0.0)] = summarize(N)
for mode in ["local", "global"]:
    for rho in [0.02, 0.05, 0.1, 0.25, 1.0]:
        res[(mode, rho)] = summarize([r for r in R if r["mode"] == mode and r["rho"] == rho])
for k, v in res.items():
    print(k, {t: (round(v[t]["rej"], 2), v[t]["det"], v[t]["days"]) for t in v})
meta = {m: np.mean([r["meta"][m] for r in N]) for m in N[0]["meta"]}
print("meta", meta)
json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()}, open("summary.json", "w"), default=float)

# Figure: per-day rejection rate vs leak rate
rhos = [0.02, 0.05, 0.1, 0.25, 1.0]
names = {"T1": "T1 canary", "T2": "T2 zero-diff", "T4": "T4 residual", "T5": "T5 co-occur."}
mk = {"T1": "o", "T2": "s", "T4": "^", "T5": "D"}
cols = {"T1": "#1f4e79", "T2": "#c55a11", "T4": "#548235", "T5": "#7030a0"}
plt.rcParams.update({"font.family": "serif", "font.size": 8})
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.75), sharey=True)
for ax, mode, title in zip(axes, ["local", "global"], ["(a) aggregation leak", "(b) representation leak"]):
    for t in T:
        y = [res[(mode, r)][t]["rej"] for r in rhos]
        ax.plot(rhos, y, marker=mk[t], ms=3, lw=1, color=cols[t], label=names[t])
    ax.axhline(0.05, color="gray", lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_xticks(rhos); ax.set_xticklabels([".02", ".05", ".1", ".25", "1"])
    ax.set_title(title, fontsize=8); ax.set_xlabel(r"leak rate $\rho$")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("per-day rejection rate")
axes[0].set_ylim(-0.03, 1.03)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=6.5,
           frameon=False, bbox_to_anchor=(0.5, -0.04), handlelength=1.5, columnspacing=0.8)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig("../power.pdf", bbox_inches="tight")
