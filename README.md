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
<img src="pages/page-11.png" width="100%" alt="Page 11">
<img src="pages/page-12.png" width="100%" alt="Page 12">
<img src="pages/page-13.png" width="100%" alt="Page 13">

---

## Repository layout

- `main.tex`: paper source (compile with pdfLaTeX; Overleaf main file)
- `main.pdf`: compiled paper
- `pages/`: page images shown in this README
- `power.pdf`: Figure (power curves)
- `sim/audit_sim.py`: synthetic graph, clean / leaking pipelines, audit tests T1, T2, T4, T5, leak patterns
- `sim/t6_edge.py`: T6 edge-level membership test
- `sim/t7_active.py`: T7 active-canary reference test
- `sim/run_all.py`: runs the passive tests under per-edge or per-customer leakage
- `sim/analyze.py`: summary statistics and the power figure
- `build.sh`: compiles the paper and regenerates `pages/`

## Reproduce the experiments

```
cd sim
python3 audit_sim.py 0,1,2,3,4 main.json                 # per-edge leaks: T1 T2 T4 T5
python3 -c "import audit_sim as a, json; json.dump([dict(seed=s, p=a.run(s,'clean',0.0)[0], meta={}) for s in range(100,130)], open('null.json','w'))"
python3 run_all.py edge t6 0,1,2,3,4 t6_edge_sweep.json   # per-edge leaks: T6
python3 run_all.py edge t6 $(seq -s, 100 129) t6_null.json null
python3 run_all.py user all 0,1,2,3,4 all_user.json      # per-customer leaks: T1 T2 T4 T5 T6
python3 t7_active.py 0,1,2,3,4 t7_main_edge.json sweep edge
python3 t7_active.py 0,1,2,3,4 t7_main_user.json sweep user
python3 t7_active.py $(seq -s, 100 129) t7_null.json null
python3 analyze.py
```
