"""Summarize results/families/*.json into results/families/summary.json (rejection rates, % of days)."""
import json, glob, os
import numpy as np
R = os.path.join(os.path.dirname(__file__), "..", "results", "families")
out = {}
for f in sorted(glob.glob(os.path.join(R, "*.json"))):
    name = os.path.basename(f)[:-5]
    if name == "summary":
        continue
    d = json.load(open(f))
    if name.startswith("naive_single"):
        ks = [k for k in d[0]["rec"] if not k.startswith("n_")]
        out[name] = {k: round(100 * float(np.mean([np.mean(r["rec"][k]) for r in d])), 1) for k in ks}
        out[name]["days"] = int(sum(len(r["rec"]["counts"]) for r in d))
        out[name]["treated_per_day"] = round(float(np.mean([np.mean(r["rec"]["n_treated"]) for r in d])), 1)
    else:
        ks = d[0]["p"].keys()
        out[name] = {k: round(100 * float(np.mean(np.concatenate([np.array(r["p"][k]) < .05 for r in d]))), 1) for k in ks}
        out[name]["days"] = int(sum(len(r["p"]["A1_T1"]) for r in d))
json.dump(out, open(os.path.join(R, "summary.json"), "w"), indent=1)
for k, v in out.items():
    print(k, v)
