# Trust, but Verify: Black-Box Auditing of Opt-Out Compliance in Graph Feature Pipelines

**[Download the paper (PDF)](main.pdf)** · PPAI-style (AAAI two-column) workshop draft

The pages below are rendered from `main.pdf`. To refresh them after editing, run `./build.sh`.

<img src="pages/page-01.png" width="100%" alt="Page 1">
<img src="pages/page-02.png" width="100%" alt="Page 2">
<img src="pages/page-03.png" width="100%" alt="Page 3">
<img src="pages/page-04.png" width="100%" alt="Page 4">
<img src="pages/page-05.png" width="100%" alt="Page 5">
<img src="pages/page-06.png" width="100%" alt="Page 6">
<img src="pages/page-07.png" width="100%" alt="Page 7">
<img src="pages/page-08.png" width="100%" alt="Page 8">
<img src="pages/page-09.png" width="100%" alt="Page 9">
<img src="pages/page-10.png" width="100%" alt="Page 10">

---

## Repository layout

- `main.tex`: paper source (compile with pdfLaTeX; Overleaf main file)
- `main.pdf`: compiled paper
- `pages/`: page images shown in this README
- `power.pdf`: Figure (power curves)
- `sim/audit_sim.py`: synthetic graph, clean / leaking pipelines, audit tests T1, T2, T4, T5
- `sim/t6_canary.py`: T6 active-canary reference test
- `sim/analyze.py`: summary statistics and the power figure
- `build.sh`: compiles the paper and regenerates `pages/`

## Reproduce the experiments

```
cd sim
python3 audit_sim.py 0,1,2,3,4 main.json
python3 -c "import audit_sim as a, numpy as np, json; json.dump([dict(seed=s, p=a.run(s,'clean',0.0)[0], meta={}) for s in range(100,130)], open('null.json','w'))"
python3 t6_canary.py 0,1,2,3,4 t6_main.json
python3 t6_canary.py $(seq -s, 100 129) t6_null.json null
python3 analyze.py
```
