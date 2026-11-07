"""LaTeX rows for the feature-family table (Sec. 7) from results/families/summary.json."""
import json, os
R = os.path.join(os.path.dirname(__file__), "..", "results", "families")
S = json.load(open(os.path.join(R, "summary.json")))
cols = ["syn_clean", "syn_leak_user_0.1", "syn_leak_user_1.0", "real_clean", "real_leak_user_0.1", "real_leak_user_1.0"]
fams = [("A1", ["T1", "T2"]), ("A2", ["T1", "T2"]), ("A3", ["T1", "T2"]),
        ("R1", ["T1", "T2", "T5", "T6"]), ("R2", ["T1", "T2n", "T2", "T5", "T6"]),
        ("R3", ["T1", "T2", "T5", "T6"]), ("R4", ["T1", "T2", "T5", "T6"])]
lab = {"T2n": "T2 (2-hop)", "T2": "T2"}
out = []
for f, tests in fams:
    for j, t in enumerate(tests):
        k = f"{f}_{t}"
        vals = [("--" if k not in S.get(c, {}) else "%.0f" % S[c][k]) for c in cols]
        name = lab.get(t, t) if not (f == "R2" and t == "T2") else "T2 (+3-hop churn)"
        out.append(("%s & %s & " % (f if j == 0 else "", name)) + " & ".join(vals) + r" \\")
    out.append(r"\addlinespace[1pt]")
open(os.path.join(R, "table.tex"), "w").write("\n".join(out[:-1]) + "\n")
print("\n".join(out[:-1]))
print({c: S[c]["days"] for c in cols if c in S})
