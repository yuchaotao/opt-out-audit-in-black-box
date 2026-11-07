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
<img src="pages/page-14.png" width="100%" alt="Page 14">
<img src="pages/page-15.png" width="100%" alt="Page 15">
<img src="pages/page-16.png" width="100%" alt="Page 16">
<img src="pages/page-17.png" width="100%" alt="Page 17">
<img src="pages/page-18.png" width="100%" alt="Page 18">
<img src="pages/page-19.png" width="100%" alt="Page 19">

---

## Repository layout

- `main.tex`: paper source (compile with pdfLaTeX; Overleaf main file); `main.pdf`: compiled paper
- `pages/`: page images shown above; `power.pdf`: power figure; `build.sh`: rebuilds PDF, pages and this list
- `sim/audit_sim.py`: synthetic graph, clean / leaking pipelines, tests T1 T2 T4 T5, leak patterns
- `sim/t6_edge.py`, `sim/t7_active.py`: T6 edge-level membership test, T7 active canaries
- `sim/robust.py`: refined (cluster-aware, time-bracketed) T5 and T6 for real data
- `sim/naive.py`, `sim/c1_probe.py`: naive tests of Section 4 and their fixes
- `sim/real_data.py`, `sim/run_real.py`, `sim/eval_robust.py`: RetailRocket semi-synthetic audits
- `sim/run_all.py`, `sim/analyze.py`: synthetic sweeps and the power figure
- `data/`: dataset fetch script and notes (raw data not committed); `results/`: all experiment outputs

## Reproduce the experiments

Synthetic (from `sim/`):
```
python3 audit_sim.py 0,1,2,3,4 main.json
python3 -c "import audit_sim as a, json; json.dump([dict(seed=s, p=a.run(s,'clean',0.0)[0], meta={}) for s in range(100,130)], open('null.json','w'))"
python3 run_all.py edge t6 0,1,2,3,4 t6_edge_sweep.json
python3 run_all.py edge t6 $(seq -s, 100 129) t6_null.json null
python3 run_all.py user all 0,1,2,3,4 all_user.json
python3 t7_active.py 0,1,2,3,4 t7_main_edge.json sweep edge
python3 t7_active.py 0,1,2,3,4 t7_main_user.json sweep user
python3 t7_active.py $(seq -s, 100 129) t7_null.json null
python3 naive.py synthetic $(seq -s, 100 129) naive_synth.json 10
python3 eval_robust.py synthetic clean 0 edge $(seq -s, 100 119) rob_synth_null.json 10
python3 analyze.py
```

Real data: `data/fetch_retailrocket.sh`, then `sim/run_real_experiments.sh`.
Feature families and single-node naive tests (synthetic and real): `sim/run_families.sh`, then
`python3 sim/summarize_families.py Real data: `data/fetch_retailrocket.sh`, then `sim/run_real_experiments.sh`.Real data: `data/fetch_retailrocket.sh`, then `sim/run_real_experiments.sh`. python3 sim/make_family_table.py`.
